from __future__ import annotations

import pandas as pd

from uav_vision.tracking.baselines import constant_velocity_bbox, previous_bbox, static_first_bbox
from uav_vision.tracking.metrics import bbox_iou, center_distance, evaluate_tracking


def _boxes(values):
    return pd.DataFrame(values, columns=["x", "y", "width", "height"])


def test_bbox_iou_identical_boxes() -> None:
    boxes = _boxes([[0, 0, 10, 10], [5, 5, 20, 10]])

    iou = bbox_iou(boxes, boxes)

    assert iou.tolist() == [1.0, 1.0]


def test_center_distance() -> None:
    gt = _boxes([[0, 0, 10, 10]])
    pred = _boxes([[3, 4, 10, 10]])

    distance = center_distance(gt, pred)

    assert distance.tolist() == [5.0]


def test_evaluate_tracking_static_sequence() -> None:
    gt = _boxes([[10, 10, 20, 20], [10, 10, 20, 20], [10, 10, 20, 20]])
    pred = static_first_bbox(gt)

    metrics = evaluate_tracking(gt, pred)

    assert metrics["mean_iou"] == 1.0
    assert metrics["precision_20"] == 1.0


def test_temporal_baselines_keep_frame_count() -> None:
    gt = _boxes([[0, 0, 10, 10], [2, 0, 10, 10], [4, 0, 10, 10]])

    assert len(previous_bbox(gt)) == 3
    assert len(constant_velocity_bbox(gt)) == 3
