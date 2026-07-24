from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


VISDRONE_TO_YOLO = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
}

YOLO_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_visdrone_gt(annotation_path: Path) -> list[dict]:
    objects: list[dict] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        left, top, width, height = map(float, parts[:4])
        category_id = int(float(parts[5]))
        class_id = VISDRONE_TO_YOLO.get(category_id)
        if class_id is None or width <= 0 or height <= 0:
            continue
        objects.append(
            {
                "xyxy": [left, top, left + width, top + height],
                "class_id": class_id,
                "class_name": YOLO_NAMES[class_id],
                "area": width * height,
                "truncation": int(float(parts[6])),
                "occlusion": int(float(parts[7])),
            }
        )
    return objects


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


def match_predictions(gt_objects: list[dict], predictions: list[dict], iou_threshold: float) -> list[bool]:
    matched_predictions: set[int] = set()
    matched_gt = [False] * len(gt_objects)

    for gt_index, gt in enumerate(gt_objects):
        best_index = None
        best_iou = 0.0
        for pred_index, pred in enumerate(predictions):
            if pred_index in matched_predictions:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue
            current_iou = iou(gt["xyxy"], pred["xyxy"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            matched_predictions.add(best_index)
            matched_gt[gt_index] = True

    return matched_gt


def read_training_metrics(results_csv: Path) -> dict[str, float] | None:
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
    }


def synchronize_if_needed(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector on VisDrone validation images.")
    parser.add_argument("--weights", type=Path, default=Path("outputs/training/yolov8s_visdrone_baseline_e10/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"))
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/annotations"))
    parser.add_argument("--training-results", type=Path, default=Path("outputs/training/yolov8s_visdrone_baseline_e10/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/yolov8s_visdrone_baseline_e10"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for quick debugging.")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not args.images.exists():
        raise FileNotFoundError(f"Missing images directory: {args.images}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Missing annotations directory: {args.annotations}")

    image_paths = sorted(args.images.glob("*.jpg"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    model = YOLO(str(args.weights))
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    class_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    total_predictions = 0
    timed_seconds = 0.0
    timed_images = 0

    for index, image_path in enumerate(image_paths):
        image = read_image(image_path)
        if index < args.warmup:
            _ = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            continue

        synchronize_if_needed(args.device)
        start = time.perf_counter()
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        synchronize_if_needed(args.device)
        elapsed = time.perf_counter() - start
        timed_seconds += elapsed
        timed_images += 1

        predictions: list[dict] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            for box, class_id, score in zip(boxes, classes, scores):
                predictions.append({"xyxy": box, "class_id": int(class_id), "score": float(score)})
        total_predictions += len(predictions)

        gt_objects = read_visdrone_gt(args.annotations / f"{image_path.stem}.txt")
        matched = match_predictions(gt_objects, predictions, args.iou)

        for gt, is_matched in zip(gt_objects, matched):
            update_group(groups, "all", is_matched)
            update_group(groups, size_bucket(gt["area"]), is_matched)
            update_group(groups, f"occlusion_{gt['occlusion']}", is_matched)
            update_group(groups, f"truncation_{gt['truncation']}", is_matched)
            update_group(class_groups, gt["class_name"], is_matched)

    def finalize(group: dict[str, int]) -> dict[str, float | int]:
        gt = group["gt"]
        matched = group["matched"]
        return {"gt": gt, "matched": matched, "recall_at_iou": matched / gt if gt else 0.0}

    summary = {
        "weights": str(args.weights),
        "images": str(args.images),
        "annotations": str(args.annotations),
        "image_count": len(image_paths),
        "timed_image_count": timed_images,
        "confidence_threshold": args.conf,
        "matching_iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "fps": timed_images / timed_seconds if timed_seconds > 0 else 0.0,
        "seconds_per_image": timed_seconds / timed_images if timed_images else 0.0,
        "total_predictions": total_predictions,
        "training_metrics": read_training_metrics(args.training_results),
        "groups": {name: finalize(group) for name, group in sorted(groups.items())},
        "classes": {name: finalize(group) for name, group in sorted(class_groups.items())},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.output / "groups.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "gt", "matched", "recall_at_iou"])
        for name, group in summary["groups"].items():
            writer.writerow([name, group["gt"], group["matched"], group["recall_at_iou"]])

    with (args.output / "classes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "gt", "matched", "recall_at_iou"])
        for name, group in summary["classes"].items():
            writer.writerow([name, group["gt"], group["matched"], group["recall_at_iou"]])

    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

