from __future__ import annotations

import numpy as np
import pandas as pd

from uav_vision.tracking.metrics import BBOX_COLUMNS, valid_bbox_mask


def _empty_predictions_like(gt_boxes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=gt_boxes.index, columns=BBOX_COLUMNS)


def static_first_bbox(gt_boxes: pd.DataFrame) -> pd.DataFrame:
    predictions = _empty_predictions_like(gt_boxes)
    valid = valid_bbox_mask(gt_boxes)
    if not valid.any():
        return predictions

    first_box = gt_boxes.loc[valid, BBOX_COLUMNS].iloc[0].to_numpy(dtype=float)
    predictions.loc[:, BBOX_COLUMNS] = first_box
    return predictions


def previous_bbox(gt_boxes: pd.DataFrame) -> pd.DataFrame:
    predictions = _empty_predictions_like(gt_boxes)
    last_valid_box = None

    for index, row in gt_boxes[BBOX_COLUMNS].iterrows():
        if last_valid_box is not None:
            predictions.loc[index, BBOX_COLUMNS] = last_valid_box

        values = row.to_numpy(dtype=float)
        if np.isfinite(values).all() and values[2] > 0 and values[3] > 0:
            last_valid_box = values

    return predictions


def constant_velocity_bbox(gt_boxes: pd.DataFrame) -> pd.DataFrame:
    predictions = _empty_predictions_like(gt_boxes)
    previous_valid_box = None
    last_valid_box = None

    for index, row in gt_boxes[BBOX_COLUMNS].iterrows():
        if last_valid_box is not None and previous_valid_box is not None:
            predictions.loc[index, BBOX_COLUMNS] = last_valid_box + (last_valid_box - previous_valid_box)
        elif last_valid_box is not None:
            predictions.loc[index, BBOX_COLUMNS] = last_valid_box

        values = row.to_numpy(dtype=float)
        if np.isfinite(values).all() and values[2] > 0 and values[3] > 0:
            previous_valid_box = last_valid_box
            last_valid_box = values

    return predictions


BASELINE_TRACKERS = {
    "static_first": static_first_bbox,
    "previous_bbox": previous_bbox,
    "constant_velocity": constant_velocity_bbox,
}
