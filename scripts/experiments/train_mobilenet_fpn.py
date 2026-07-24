from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


VISDRONE_CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_annotation(path: Path) -> list[dict]:
    objects = []
    if not path.exists():
        return objects
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        left, top, width, height = map(float, parts[:4])
        category_id = int(float(parts[5]))
        if category_id not in VISDRONE_CATEGORIES or width <= 0 or height <= 0:
            continue
        objects.append(
            {
                "box": [left, top, left + width, top + height],
                "label": category_id,
                "area": width * height,
                "truncation": int(float(parts[6])),
                "occlusion": int(float(parts[7])),
            }
        )
    return objects


def intersect_box(box: list[float], tile: tuple[int, int, int, int], min_visible: float) -> list[float] | None:
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    visible = iw * ih
    original = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if original <= 0 or visible / original < min_visible or iw < 2 or ih < 2:
        return None
    return [ix1 - tx1, iy1 - ty1, ix2 - tx1, iy2 - ty1]


def make_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        return [(0, 0, width, height)]
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((x, y, min(width, x + tile_size), min(height, y + tile_size)))
    return tiles


class VisDroneDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        annotations_dir: Path,
        train: bool,
        limit: int = 0,
        tile_size: int = 0,
        tile_overlap: int = 160,
        min_visible: float = 0.3,
        include_empty_tiles: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.train = train
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.min_visible = min_visible
        self.include_empty_tiles = include_empty_tiles
        image_paths = sorted(images_dir.glob("*.jpg"))
        if limit > 0:
            image_paths = image_paths[:limit]
        self.samples = self._build_samples(image_paths)

    def _build_samples(self, image_paths: list[Path]) -> list[dict]:
        samples = []
        for image_path in image_paths:
            annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
            objects = read_annotation(annotation_path)
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            if self.tile_size <= 0:
                samples.append({"image": image_path, "annotation": annotation_path, "tile": None})
                continue
            for tile in make_tiles(width, height, self.tile_size, self.tile_overlap):
                tile_objects = [intersect_box(obj["box"], tile, self.min_visible) for obj in objects]
                has_objects = any(box is not None for box in tile_objects)
                if has_objects or self.include_empty_tiles:
                    samples.append({"image": image_path, "annotation": annotation_path, "tile": tile})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = read_image(sample["image"])
        objects = read_annotation(sample["annotation"])
        tile = sample["tile"]
        if tile is not None:
            x1, y1, x2, y2 = tile
            image = image[y1:y2, x1:x2]
            filtered = []
            for obj in objects:
                box = intersect_box(obj["box"], tile, self.min_visible)
                if box is not None:
                    item = dict(obj)
                    item["box"] = box
                    filtered.append(item)
            objects = filtered

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [obj["box"] for obj in objects]
        labels = [obj["label"] for obj in objects]
        height, width = image.shape[:2]

        if self.train and boxes and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes = [[width - box[2], box[1], width - box[0], box[3]] for box in boxes]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area_tensor = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp(min=0)
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp(min=0)
            if len(boxes_tensor)
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


def small_object_sampling_weights(
    dataset: VisDroneDetectionDataset,
    area_threshold: float,
    strength: float,
) -> list[float]:
    weights = []
    for sample in dataset.samples:
        objects = read_annotation(sample["annotation"])
        tile = sample["tile"]
        if tile is not None:
            objects = [obj for obj in objects if intersect_box(obj["box"], tile, dataset.min_visible) is not None]
        small_count = sum(1 for obj in objects if obj["area"] < area_threshold)
        small_fraction = small_count / max(1, len(objects))
        weights.append(1.0 + strength * small_fraction)
    return weights


def collate_fn(batch):
    return tuple(zip(*batch))


def count_training_instances(annotations_dir: Path) -> dict[int, int]:
    counts = {category_id: 0 for category_id in VISDRONE_CATEGORIES}
    for path in sorted(annotations_dir.glob("*.txt")):
        for obj in read_annotation(path):
            counts[obj["label"]] += 1
    return counts


def effective_class_weights(counts: dict[int, int], beta: float, max_weight: float) -> list[float]:
    foreground = []
    for category_id in VISDRONE_CATEGORIES:
        count = max(1, counts[category_id])
        effective_count = 1.0 - math.pow(beta, count)
        foreground.append((1.0 - beta) / effective_count)
    mean_weight = sum(foreground) / len(foreground)
    normalized = [min(max_weight, max(1.0 / max_weight, value / mean_weight)) for value in foreground]
    return [1.0, *normalized]


def balanced_small_object_roi_forward(self, features, proposals, image_shapes, targets=None):
    if self.training:
        if targets is None:
            raise ValueError("Targets are required during training.")
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    else:
        matched_idxs = None
        labels = None
        regression_targets = None

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    if not self.training:
        boxes, scores, detected_labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        results = [
            {"boxes": box, "labels": label, "scores": score}
            for box, label, score in zip(boxes, detected_labels, scores)
        ]
        return results, {}

    if labels is None or regression_targets is None or matched_idxs is None or targets is None:
        raise ValueError("Missing Faster R-CNN training targets.")

    proposal_weights = [torch.ones_like(image_labels, dtype=class_logits.dtype) for image_labels in labels]
    if abs(self.small_object_weight - 1.0) > 1e-9:
        for weights, image_labels, image_matches, target in zip(
            proposal_weights, labels, matched_idxs, targets
        ):
            positive = torch.where(image_labels > 0)[0]
            if positive.numel() > 0:
                gt_boxes = target["boxes"][image_matches[positive]]
                gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (
                    gt_boxes[:, 3] - gt_boxes[:, 1]
                ).clamp(min=0)
                weights[positive] = torch.where(
                    gt_areas < self.small_object_area,
                    torch.full_like(gt_areas, self.small_object_weight),
                    torch.ones_like(gt_areas),
                ).to(weights.dtype)

    labels_tensor = torch.cat(labels, dim=0)
    regression_tensor = torch.cat(regression_targets, dim=0)
    proposal_weight_tensor = torch.cat(proposal_weights, dim=0)
    class_weights = class_logits.new_tensor(self.balanced_class_weights)

    cross_entropy = F.cross_entropy(class_logits, labels_tensor, reduction="none")
    probabilities = F.softmax(class_logits, dim=1)
    true_probability = probabilities.gather(1, labels_tensor[:, None]).squeeze(1)
    focal_factor = (1.0 - true_probability).pow(self.focal_gamma)
    classification_weights = class_weights[labels_tensor] * proposal_weight_tensor
    classification_loss = (cross_entropy * focal_factor * classification_weights).sum()
    classification_loss = classification_loss / classification_weights.sum().clamp(min=1.0)

    positive = torch.where(labels_tensor > 0)[0]
    positive_labels = labels_tensor[positive]
    sample_count, _ = class_logits.shape
    box_regression = box_regression.reshape(sample_count, box_regression.size(-1) // 4, 4)
    box_per_coordinate = F.smooth_l1_loss(
        box_regression[positive, positive_labels],
        regression_tensor[positive],
        beta=1 / 9,
        reduction="none",
    )
    box_per_proposal = box_per_coordinate.sum(dim=1)
    box_loss = (box_per_proposal * proposal_weight_tensor[positive]).sum()
    box_loss = box_loss / max(1, labels_tensor.numel())
    return [], {"loss_classifier": classification_loss, "loss_box_reg": box_loss}


def build_model(
    num_classes: int,
    min_size: int,
    max_size: int,
    pretrained: bool,
    enhanced_loss: bool = False,
    class_weights: list[float] | None = None,
    focal_gamma: float = 2.0,
    small_object_area: float = 32.0 * 32.0,
    small_object_weight: float = 2.0,
    small_anchors: bool = False,
    detections_per_image: int = 300,
    score_threshold: float = 0.01,
):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = detections_per_image
    model.roi_heads.score_thresh = score_threshold

    if small_anchors:
        sizes = (
            (8, 16, 32, 64, 128),
            (16, 32, 64, 128, 256),
            (32, 64, 128, 256, 512),
        )
        ratios = tuple((0.5, 1.0, 2.0) for _ in sizes)
        model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

    if enhanced_loss:
        model.roi_heads.balanced_class_weights = class_weights or [1.0] * num_classes
        model.roi_heads.focal_gamma = focal_gamma
        model.roi_heads.small_object_area = small_object_area
        model.roi_heads.small_object_weight = small_object_weight
        model.roi_heads.forward = MethodType(balanced_small_object_roi_forward, model.roi_heads)
    return model


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small_lt_32x32"
    if area < 96 * 96:
        return "medium_32x32_to_96x96"
    return "large_ge_96x96"


def update_group(groups: dict[str, dict[str, int]], name: str, matched: bool) -> None:
    groups[name]["gt"] += 1
    if matched:
        groups[name]["matched"] += 1


def match_predictions(
    gt_objects: list[dict], predictions: list[dict], iou_threshold: float
) -> tuple[list[bool], list[bool]]:
    matched_gt = [False] * len(gt_objects)
    matched_predictions = [False] * len(predictions)
    prediction_order = sorted(range(len(predictions)), key=lambda index: predictions[index]["score"], reverse=True)
    for pred_index in prediction_order:
        prediction = predictions[pred_index]
        best_gt_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_objects):
            if matched_gt[gt_index] or prediction["label"] != gt["label"]:
                continue
            current_iou = iou(gt["box"], prediction["box"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt[best_gt_index] = True
            matched_predictions[pred_index] = True
    return matched_gt, matched_predictions


def finalize_group(group: dict[str, int]) -> dict[str, float | int]:
    gt = group["gt"]
    matched = group["matched"]
    return {"gt": gt, "matched": matched, "recall_at_iou": matched / gt if gt else 0.0}


def interpolated_ap(true_positive: np.ndarray, false_positive: np.ndarray, total_gt: int) -> float:
    if total_gt <= 0 or true_positive.size == 0:
        return 0.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / total_gt
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    values = []
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        candidates = precision[recall >= recall_threshold]
        values.append(float(candidates.max()) if candidates.size else 0.0)
    return float(np.mean(values))


def compute_ap_metrics(
    all_ground_truth: list[list[dict]],
    all_predictions: list[list[dict]],
    iou_thresholds: list[float],
) -> dict:
    ap_by_threshold: dict[str, float] = {}
    per_class_ap50: dict[str, float] = {}

    for threshold in iou_thresholds:
        class_aps = []
        for category_id, category_name in VISDRONE_CATEGORIES.items():
            gt_by_image = {
                image_index: [obj for obj in objects if obj["label"] == category_id]
                for image_index, objects in enumerate(all_ground_truth)
            }
            total_gt = sum(len(objects) for objects in gt_by_image.values())
            predictions = []
            for image_index, image_predictions in enumerate(all_predictions):
                predictions.extend(
                    (prediction["score"], image_index, prediction["box"])
                    for prediction in image_predictions
                    if prediction["label"] == category_id
                )
            predictions.sort(key=lambda item: item[0], reverse=True)
            used = {image_index: set() for image_index in gt_by_image}
            true_positive = np.zeros(len(predictions), dtype=np.float64)
            false_positive = np.zeros(len(predictions), dtype=np.float64)
            for prediction_index, (_, image_index, box) in enumerate(predictions):
                best_gt_index = None
                best_iou = 0.0
                for gt_index, gt in enumerate(gt_by_image[image_index]):
                    if gt_index in used[image_index]:
                        continue
                    current_iou = iou(box, gt["box"])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_index = gt_index
                if best_gt_index is not None and best_iou >= threshold:
                    used[image_index].add(best_gt_index)
                    true_positive[prediction_index] = 1.0
                else:
                    false_positive[prediction_index] = 1.0
            ap = interpolated_ap(true_positive, false_positive, total_gt)
            class_aps.append(ap)
            if abs(threshold - 0.5) < 1e-9:
                per_class_ap50[category_name] = ap
        ap_by_threshold[f"{threshold:.2f}"] = float(np.mean(class_aps)) if class_aps else 0.0

    map50 = ap_by_threshold.get("0.50", 0.0)
    map50_95 = float(np.mean(list(ap_by_threshold.values()))) if ap_by_threshold else 0.0
    return {
        "map50": map50,
        "map50_95": map50_95,
        "ap_by_iou": ap_by_threshold,
        "per_class_ap50": per_class_ap50,
    }


def evaluate(
    model,
    image_paths: list[Path],
    annotations_dir: Path,
    device: torch.device,
    conf: float,
    iou_threshold: float,
    warmup: int,
    limit: int,
) -> dict:
    model.eval()
    paths = image_paths[:limit] if limit > 0 else image_paths
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    class_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    total_predictions = 0
    total_matches = 0
    timed_seconds = 0.0
    timed_images = 0
    all_ground_truth: list[list[dict]] = []
    all_predictions: list[list[dict]] = []

    with torch.inference_mode():
        for image_path in paths[:warmup]:
            image = read_image(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
            _ = model([tensor])

        for image_path in paths:
            image = read_image(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model([tensor])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            timed_seconds += time.perf_counter() - start
            timed_images += 1

            boxes = output["boxes"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            metric_predictions = [
                {"box": [float(value) for value in box], "label": int(label), "score": float(score)}
                for box, label, score in zip(boxes, labels, scores)
            ]
            report_predictions = [prediction for prediction in metric_predictions if prediction["score"] >= conf]
            total_predictions += len(report_predictions)
            gt_objects = read_annotation(annotations_dir / f"{image_path.stem}.txt")
            matched_gt, matched_predictions = match_predictions(gt_objects, report_predictions, iou_threshold)
            total_matches += sum(matched_predictions)
            all_ground_truth.append(gt_objects)
            all_predictions.append(metric_predictions)
            for gt, is_matched in zip(gt_objects, matched_gt):
                update_group(groups, "all", is_matched)
                update_group(groups, size_bucket(gt["area"]), is_matched)
                update_group(groups, f"occlusion_{gt['occlusion']}", is_matched)
                update_group(groups, f"truncation_{gt['truncation']}", is_matched)
                update_group(class_groups, VISDRONE_CATEGORIES[gt["label"]], is_matched)

    ap_metrics = compute_ap_metrics(
        all_ground_truth,
        all_predictions,
        [round(0.5 + 0.05 * index, 2) for index in range(10)],
    )
    total_gt = sum(len(objects) for objects in all_ground_truth)
    return {
        "image_count": timed_images,
        "confidence_threshold": conf,
        "matching_iou_threshold": iou_threshold,
        "precision_at_conf": total_matches / total_predictions if total_predictions else 0.0,
        "recall_at_conf": total_matches / total_gt if total_gt else 0.0,
        "fps": timed_images / timed_seconds if timed_seconds > 0 else 0.0,
        "seconds_per_image": timed_seconds / timed_images if timed_images else 0.0,
        "total_predictions": total_predictions,
        "total_ground_truth": total_gt,
        "total_matches": total_matches,
        **ap_metrics,
        "groups": {name: finalize_group(group) for name, group in sorted(groups.items())},
        "classes": {name: finalize_group(group) for name, group in sorted(class_groups.items())},
    }


def save_training_plot(results_path: Path, output_path: Path) -> None:
    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", color="#1f5d78")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [float(row["val_map50"]) for row in rows], marker="o", label="AP50")
    axes[1].plot(epochs, [float(row["val_map50_95"]) for row in rows], marker="o", label="mAP50-95")
    axes[1].set_title("Detection accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(epochs, [float(row["val_recall"]) for row in rows], marker="o", label="Overall")
    axes[2].plot(epochs, [float(row["val_small_recall"]) for row in rows], marker="o", label="Small")
    axes[2].plot(
        epochs,
        [float(row["val_heavy_occlusion_recall"]) for row in rows],
        marker="o",
        label="Heavy occlusion",
    )
    axes[2].set_title("Recall at confidence threshold")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_prediction_grid(
    model,
    image_paths: list[Path],
    device: torch.device,
    output_path: Path,
    confidence: float,
    count: int = 6,
) -> None:
    if not image_paths:
        return
    indices = np.linspace(0, len(image_paths) - 1, min(count, len(image_paths)), dtype=int)
    fig, axes = plt.subplots(2, math.ceil(len(indices) / 2), figsize=(15, 8))
    axes = np.asarray(axes).reshape(-1)
    model.eval()
    with torch.inference_mode():
        for axis, index in zip(axes, indices):
            image_path = image_paths[int(index)]
            image = read_image(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
            output = model([tensor])[0]
            canvas = image.copy()
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                score_value = float(score.detach().cpu())
                if score_value < confidence:
                    continue
                x1, y1, x2, y2 = [int(round(value)) for value in box.detach().cpu().tolist()]
                class_id = int(label.detach().cpu())
                color = (54, 179, 126)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    canvas,
                    f"{VISDRONE_CATEGORIES.get(class_id, str(class_id))} {score_value:.2f}",
                    (x1, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            axis.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            axis.set_title(image_path.name, fontsize=9)
            axis.axis("off")
    for axis in axes[len(indices) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN MobileNetV3 FPN on VisDrone.")
    parser.add_argument("--train-images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/images"))
    parser.add_argument("--train-annotations", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/annotations"))
    parser.add_argument("--val-images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"))
    parser.add_argument("--val-annotations", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/annotations"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training/mobilenet_fpn_visdrone_e10"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=960)
    parser.add_argument("--train-tile-size", type=int, default=0)
    parser.add_argument("--train-tile-overlap", type=int, default=160)
    parser.add_argument("--min-visible", type=float, default=0.3)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize model weights without restoring optimizer state.")
    parser.add_argument("--pretraining-source", default="torchvision_coco", help="Recorded provenance for initialization weights.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--enhanced-loss", action="store_true", help="Enable balanced focal loss and area weighting.")
    parser.add_argument("--focal-loss", action="store_true", help="Enable focal modulation without class balancing.")
    parser.add_argument("--class-balanced-loss", action="store_true", help="Enable effective-number class weights.")
    parser.add_argument("--class-balance-beta", type=float, default=0.9999)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--small-object-area", type=float, default=32.0 * 32.0)
    parser.add_argument("--small-object-weight", type=float, default=1.0)
    parser.add_argument("--small-object-resample-strength", type=float, default=0.0)
    parser.add_argument("--small-anchors", action="store_true")
    parser.add_argument("--detections-per-image", type=int, default=300)
    parser.add_argument("--eval-score-threshold", type=float, default=0.01)
    parser.add_argument(
        "--selection-metric",
        choices=["map50", "map50_95", "recall", "small_recall"],
        default="map50_95",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output / "weights"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VisDroneDetectionDataset(
        args.train_images,
        args.train_annotations,
        train=True,
        limit=args.limit_train,
        tile_size=args.train_tile_size,
        tile_overlap=args.train_tile_overlap,
        min_visible=args.min_visible,
    )
    sampler = None
    if args.small_object_resample_strength > 0:
        sampling_weights = small_object_sampling_weights(
            train_dataset,
            area_threshold=args.small_object_area,
            strength=args.small_object_resample_strength,
        )
        sampler = WeightedRandomSampler(sampling_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_paths = sorted(args.val_images.glob("*.jpg"))
    class_counts = count_training_instances(args.train_annotations)
    computed_class_weights = effective_class_weights(class_counts, args.class_balance_beta, args.max_class_weight)
    use_class_balance = args.enhanced_loss or args.class_balanced_loss
    use_focal = args.enhanced_loss or args.focal_loss
    use_area_weight = args.enhanced_loss or abs(args.small_object_weight - 1.0) > 1e-9
    class_weights = computed_class_weights if use_class_balance else [1.0] * 11
    model = build_model(
        num_classes=11,
        min_size=args.min_size,
        max_size=args.max_size,
        pretrained=not args.no_pretrained and args.init_checkpoint is None,
        enhanced_loss=use_focal or use_class_balance or use_area_weight,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma if use_focal else 0.0,
        small_object_area=args.small_object_area,
        small_object_weight=args.small_object_weight if use_area_weight else 1.0,
        small_anchors=args.small_anchors,
        detections_per_image=args.detections_per_image,
        score_threshold=args.eval_score_threshold,
    )
    initialization = {
        "source": args.pretraining_source,
        "checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
    }
    if args.init_checkpoint:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_checkpoint}")
        initialization_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        initialization_state = initialization_checkpoint.get("model", initialization_checkpoint)
        model.load_state_dict(initialization_state, strict=True)
    model.to(device)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_metric = -1.0
    best_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", checkpoint.get("best_recall", best_metric)))
        best_epoch = int(checkpoint.get("best_epoch", start_epoch))

    config = vars(args).copy()
    config["device"] = str(device)
    config["train_samples"] = len(train_dataset)
    config["initialization"] = initialization
    config["class_counts"] = {VISDRONE_CATEGORIES[key]: value for key, value in class_counts.items()}
    config["class_weights"] = {
        "background": class_weights[0],
        **{VISDRONE_CATEGORIES[key]: class_weights[key] for key in VISDRONE_CATEGORIES},
    }
    config["computed_class_weights"] = {
        "background": computed_class_weights[0],
        **{VISDRONE_CATEGORIES[key]: computed_class_weights[key] for key in VISDRONE_CATEGORIES},
    }
    save_json(args.output / "args.json", config)

    result_fields = [
        "epoch",
        "train_loss",
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "lr",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
        "val_small_recall",
        "val_heavy_occlusion_recall",
        "val_fps",
        "elapsed_seconds",
    ]
    results_path = args.output / "results.csv"
    if start_epoch == 0:
        with results_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result_fields)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        component_totals: dict[str, float] = defaultdict(float)
        batch_count = 0
        start = time.perf_counter()
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(value for value in losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            for name, value in losses.items():
                component_totals[name] += float(value.detach().cpu())
            batch_count += 1
        scheduler.step()
        train_loss = epoch_loss / batch_count if batch_count else 0.0
        component_means = {
            name: total / batch_count if batch_count else 0.0 for name, total in component_totals.items()
        }

        metrics = evaluate(
            model,
            val_paths,
            args.val_annotations,
            device,
            conf=args.conf,
            iou_threshold=args.iou,
            warmup=args.warmup,
            limit=args.limit_val,
        )
        all_recall = float(metrics["groups"].get("all", {}).get("recall_at_iou", 0.0))
        small_recall = float(metrics["groups"].get("small_lt_32x32", {}).get("recall_at_iou", 0.0))
        heavy_recall = float(metrics["groups"].get("occlusion_2", {}).get("recall_at_iou", 0.0))
        selection_values = {
            "map50": float(metrics["map50"]),
            "map50_95": float(metrics["map50_95"]),
            "recall": all_recall,
            "small_recall": small_recall,
        }
        current_metric = selection_values[args.selection_metric]
        elapsed = time.perf_counter() - start

        epoch_summary = {
            "epoch": epoch + 1,
            "elapsed_seconds": elapsed,
            "train_loss": train_loss,
            "component_losses": component_means,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_metric": args.selection_metric,
            "selection_value": current_metric,
            "validation": metrics,
        }
        save_json(args.output / f"epoch_{epoch + 1:03d}_summary.json", epoch_summary)
        with results_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result_fields)
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "loss_classifier": component_means.get("loss_classifier", 0.0),
                    "loss_box_reg": component_means.get("loss_box_reg", 0.0),
                    "loss_objectness": component_means.get("loss_objectness", 0.0),
                    "loss_rpn_box_reg": component_means.get("loss_rpn_box_reg", 0.0),
                    "lr": optimizer.param_groups[0]["lr"],
                    "val_precision": metrics["precision_at_conf"],
                    "val_recall": all_recall,
                    "val_map50": metrics["map50"],
                    "val_map50_95": metrics["map50_95"],
                    "val_small_recall": small_recall,
                    "val_heavy_occlusion_recall": heavy_recall,
                    "val_fps": metrics["fps"],
                    "elapsed_seconds": elapsed,
                }
            )

        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
            "args": config,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")
        if improved:
            torch.save(checkpoint, checkpoints_dir / "best.pt")

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_precision": metrics["precision_at_conf"],
                    "val_recall": all_recall,
                    "val_map50": metrics["map50"],
                    "val_map50_95": metrics["map50_95"],
                    "val_small_recall": small_recall,
                    "val_heavy_occlusion_recall": heavy_recall,
                    "val_fps": metrics["fps"],
                    "elapsed_seconds": elapsed,
                },
                indent=2,
            )
        )

    best_checkpoint_path = checkpoints_dir / "best.pt"
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
    final_validation = evaluate(
        model,
        val_paths,
        args.val_annotations,
        device,
        conf=args.conf,
        iou_threshold=args.iou,
        warmup=args.warmup,
        limit=args.limit_val,
    )
    save_training_plot(results_path, args.output / "training_metrics.png")
    save_prediction_grid(
        model,
        val_paths[: args.limit_val] if args.limit_val > 0 else val_paths,
        device,
        args.output / "validation_predictions.jpg",
        confidence=args.conf,
    )
    final_summary = {
        "model": "fasterrcnn_mobilenet_v3_large_fpn",
        "num_classes": 11,
        "train_samples": len(train_dataset),
        "epochs": args.epochs,
        "initialization": initialization,
        "enhanced_loss": args.enhanced_loss,
        "focal_loss": use_focal,
        "class_balanced_loss": use_class_balance,
        "small_object_area_weighting": use_area_weight,
        "small_object_resample_strength": args.small_object_resample_strength,
        "small_anchors": args.small_anchors,
        "selection_metric": args.selection_metric,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(checkpoints_dir / "last.pt"),
        "class_counts": config["class_counts"],
        "class_weights": config["class_weights"],
        "final_validation": final_validation,
        "artifacts": {
            "training_metrics": str(args.output / "training_metrics.png"),
            "validation_predictions": str(args.output / "validation_predictions.jpg"),
            "results_csv": str(results_path),
        },
    }
    save_json(args.output / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
