from __future__ import annotations

import numpy as np
import pandas as pd


BBOX_COLUMNS = ["x", "y", "width", "height"]


def valid_bbox_mask(boxes: pd.DataFrame) -> np.ndarray:
    values = boxes[BBOX_COLUMNS].to_numpy(dtype=float)
    return np.isfinite(values).all(axis=1) & (values[:, 2] > 0) & (values[:, 3] > 0)


def bbox_iou(gt_boxes: pd.DataFrame, pred_boxes: pd.DataFrame) -> np.ndarray:
    gt = gt_boxes[BBOX_COLUMNS].to_numpy(dtype=float)
    pred = pred_boxes[BBOX_COLUMNS].to_numpy(dtype=float)
    valid = valid_bbox_mask(gt_boxes) & valid_bbox_mask(pred_boxes)

    iou = np.zeros(len(gt), dtype=float)
    if not valid.any():
        return iou

    gt_valid = gt[valid]
    pred_valid = pred[valid]

    gt_x2 = gt_valid[:, 0] + gt_valid[:, 2]
    gt_y2 = gt_valid[:, 1] + gt_valid[:, 3]
    pred_x2 = pred_valid[:, 0] + pred_valid[:, 2]
    pred_y2 = pred_valid[:, 1] + pred_valid[:, 3]

    inter_x1 = np.maximum(gt_valid[:, 0], pred_valid[:, 0])
    inter_y1 = np.maximum(gt_valid[:, 1], pred_valid[:, 1])
    inter_x2 = np.minimum(gt_x2, pred_x2)
    inter_y2 = np.minimum(gt_y2, pred_y2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    gt_area = gt_valid[:, 2] * gt_valid[:, 3]
    pred_area = pred_valid[:, 2] * pred_valid[:, 3]
    union_area = gt_area + pred_area - inter_area
    valid_union = union_area > 0

    valid_iou = np.zeros(len(gt_valid), dtype=float)
    valid_iou[valid_union] = inter_area[valid_union] / union_area[valid_union]
    iou[valid] = valid_iou
    return iou


def center_distance(gt_boxes: pd.DataFrame, pred_boxes: pd.DataFrame) -> np.ndarray:
    gt = gt_boxes[BBOX_COLUMNS].to_numpy(dtype=float)
    pred = pred_boxes[BBOX_COLUMNS].to_numpy(dtype=float)
    valid = valid_bbox_mask(gt_boxes) & valid_bbox_mask(pred_boxes)

    distance = np.full(len(gt), np.inf, dtype=float)
    if not valid.any():
        return distance

    gt_center_x = gt[valid, 0] + gt[valid, 2] / 2.0
    gt_center_y = gt[valid, 1] + gt[valid, 3] / 2.0
    pred_center_x = pred[valid, 0] + pred[valid, 2] / 2.0
    pred_center_y = pred[valid, 1] + pred[valid, 3] / 2.0
    distance[valid] = np.hypot(gt_center_x - pred_center_x, gt_center_y - pred_center_y)
    return distance


def curve_auc(thresholds: np.ndarray, values: np.ndarray) -> float:
    if len(thresholds) < 2:
        return float(values.mean()) if len(values) else 0.0
    span = thresholds[-1] - thresholds[0]
    if span <= 0:
        return 0.0
    return float(np.trapz(values, thresholds) / span)


def evaluate_tracking(gt_boxes: pd.DataFrame, pred_boxes: pd.DataFrame) -> dict:
    if len(gt_boxes) != len(pred_boxes):
        raise ValueError("Ground-truth and prediction frames must have the same length.")

    valid_gt = valid_bbox_mask(gt_boxes)
    ious = bbox_iou(gt_boxes, pred_boxes)
    distances = center_distance(gt_boxes, pred_boxes)

    frame_count = int(len(gt_boxes))
    valid_frame_count = int(valid_gt.sum())
    if valid_frame_count == 0:
        return {
            "frame_count": frame_count,
            "valid_frame_count": 0,
            "mean_iou": 0.0,
            "success_auc": 0.0,
            "precision_20": 0.0,
            "mean_center_error": 0.0,
        }

    valid_ious = ious[valid_gt]
    valid_distances = distances[valid_gt]
    finite_distances = valid_distances[np.isfinite(valid_distances)]

    success_thresholds = np.linspace(0.0, 1.0, 101)
    precision_thresholds = np.arange(0.0, 51.0, 1.0)
    success_curve = np.array([(valid_ious >= threshold).mean() for threshold in success_thresholds])
    precision_curve = np.array([(valid_distances <= threshold).mean() for threshold in precision_thresholds])

    return {
        "frame_count": frame_count,
        "valid_frame_count": valid_frame_count,
        "mean_iou": float(valid_ious.mean()),
        "success_auc": curve_auc(success_thresholds, success_curve),
        "precision_20": float((valid_distances <= 20.0).mean()),
        "mean_center_error": float(finite_distances.mean()) if len(finite_distances) else 0.0,
        "success_thresholds": success_thresholds,
        "success_curve": success_curve,
        "precision_thresholds": precision_thresholds,
        "precision_curve": precision_curve,
    }
