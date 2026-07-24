from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import batched_nms
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

YOLO_TO_VISDRONE = {value: key for key, value in VISDRONE_TO_YOLO.items()}

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

Prediction = dict[str, float | int | list[float]]


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


def finalize_group(group: dict[str, int]) -> dict[str, float | int]:
    gt = group["gt"]
    matched = group["matched"]
    return {"gt": gt, "matched": matched, "recall_at_iou": matched / gt if gt else 0.0}


def match_predictions(
    gt_objects: list[dict],
    predictions: list[Prediction],
    iou_threshold: float,
) -> tuple[list[bool], list[bool]]:
    matched_gt = [False] * len(gt_objects)
    matched_predictions = [False] * len(predictions)
    prediction_order = sorted(range(len(predictions)), key=lambda index: float(predictions[index]["score"]), reverse=True)
    for pred_index in prediction_order:
        prediction = predictions[pred_index]
        best_gt_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_objects):
            if matched_gt[gt_index] or int(prediction["label"]) != gt["label"]:
                continue
            current_iou = iou(gt["box"], prediction["box"])  # type: ignore[arg-type]
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt[best_gt_index] = True
            matched_predictions[pred_index] = True
    return matched_gt, matched_predictions


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
    all_predictions: list[list[Prediction]],
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
                    (float(prediction["score"]), image_index, prediction["box"])
                    for prediction in image_predictions
                    if int(prediction["label"]) == category_id
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
                    current_iou = iou(box, gt["box"])  # type: ignore[arg-type]
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
    return {
        "map50": ap_by_threshold.get("0.50", 0.0),
        "map50_95": float(np.mean(list(ap_by_threshold.values()))) if ap_by_threshold else 0.0,
        "ap_by_iou": ap_by_threshold,
        "per_class_ap50": per_class_ap50,
    }


def yolo_result_to_predictions(result) -> list[Prediction]:
    predictions: list[Prediction] = []
    if result.boxes is None or len(result.boxes) == 0:
        return predictions
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    for box, class_id, score in zip(boxes, classes, scores):
        label = YOLO_TO_VISDRONE.get(int(class_id))
        if label is None:
            continue
        predictions.append({"box": [float(value) for value in box], "label": label, "score": float(score)})
    return predictions


def nms_predictions(predictions: list[Prediction], nms_iou: float, max_det: int, device: str) -> list[Prediction]:
    if not predictions:
        return []
    if device != "cpu" and torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{device}" if device.isdigit() else device)
    else:
        torch_device = torch.device("cpu")
    boxes = torch.tensor([item["box"] for item in predictions], dtype=torch.float32, device=torch_device)
    scores = torch.tensor([item["score"] for item in predictions], dtype=torch.float32, device=torch_device)
    labels = torch.tensor([item["label"] for item in predictions], dtype=torch.int64, device=torch_device)
    kept = batched_nms(boxes, scores, labels, nms_iou)[:max_det].detach().cpu().tolist()
    return [predictions[index] for index in kept]


def predict_standard(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> list[Prediction]:
    result = model.predict(image, imgsz=args.imgsz, conf=args.eval_conf, iou=args.nms_iou, device=args.device, verbose=False)[0]
    return yolo_result_to_predictions(result)


def predict_multiscale(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> list[Prediction]:
    outputs: list[Prediction] = []
    for size in args.scales:
        result = model.predict(image, imgsz=size, conf=args.eval_conf, iou=args.nms_iou, device=args.device, verbose=False)[0]
        outputs.extend(yolo_result_to_predictions(result))
    return nms_predictions(outputs, args.nms_iou, args.max_det, args.device)


def make_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
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


def predict_sahi(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> list[Prediction]:
    height, width = image.shape[:2]
    overlap = int(round(args.slice_size * args.slice_overlap))
    outputs: list[Prediction] = []
    if args.sahi_standard_prediction:
        outputs.extend(predict_standard(model, image, args))
    for x1, y1, x2, y2 in make_tiles(width, height, args.slice_size, overlap):
        tile = image[y1:y2, x1:x2]
        result = model.predict(
            tile,
            imgsz=args.slice_imgsz,
            conf=args.eval_conf,
            iou=args.nms_iou,
            device=args.device,
            verbose=False,
        )[0]
        for prediction in yolo_result_to_predictions(result):
            box = prediction["box"]  # type: ignore[assignment]
            prediction["box"] = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]  # type: ignore[index]
            outputs.append(prediction)
    return nms_predictions(outputs, args.nms_iou, args.max_det, args.device)


def synchronize_if_needed(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate(model: YOLO, image_paths: list[Path], annotations_dir: Path, args: argparse.Namespace) -> dict:
    if args.mode == "standard":
        predictor = predict_standard
    elif args.mode == "multiscale":
        predictor = predict_multiscale
    else:
        predictor = predict_sahi

    for image_path in image_paths[: args.warmup]:
        _ = predictor(model, read_image(image_path), args)

    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    class_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    all_ground_truth: list[list[dict]] = []
    all_predictions: list[list[Prediction]] = []
    total_predictions = 0
    total_matches = 0
    timed_seconds = 0.0

    for image_path in image_paths:
        image = read_image(image_path)
        synchronize_if_needed(args.device)
        start = time.perf_counter()
        metric_predictions = predictor(model, image, args)
        synchronize_if_needed(args.device)
        timed_seconds += time.perf_counter() - start

        report_predictions = [item for item in metric_predictions if float(item["score"]) >= args.conf]
        gt_objects = read_annotation(annotations_dir / f"{image_path.stem}.txt")
        matched_gt, matched_predictions = match_predictions(gt_objects, report_predictions, args.iou)
        total_predictions += len(report_predictions)
        total_matches += sum(matched_predictions)
        all_ground_truth.append(gt_objects)
        all_predictions.append(metric_predictions)

        for gt, matched in zip(gt_objects, matched_gt):
            update_group(groups, "all", matched)
            update_group(groups, size_bucket(gt["area"]), matched)
            update_group(groups, f"occlusion_{gt['occlusion']}", matched)
            update_group(groups, f"truncation_{gt['truncation']}", matched)
            update_group(class_groups, VISDRONE_CATEGORIES[gt["label"]], matched)

    ap_metrics = compute_ap_metrics(
        all_ground_truth,
        all_predictions,
        [round(0.5 + 0.05 * index, 2) for index in range(10)],
    )
    total_gt = sum(len(items) for items in all_ground_truth)
    image_count = len(image_paths)
    return {
        "image_count": image_count,
        "confidence_threshold": args.conf,
        "ap_score_threshold": args.eval_conf,
        "matching_iou_threshold": args.iou,
        "precision_at_conf": total_matches / total_predictions if total_predictions else 0.0,
        "recall_at_conf": total_matches / total_gt if total_gt else 0.0,
        "fps": image_count / timed_seconds if timed_seconds else 0.0,
        "seconds_per_image": timed_seconds / image_count if image_count else 0.0,
        "total_predictions": total_predictions,
        "total_ground_truth": total_gt,
        "total_matches": total_matches,
        **ap_metrics,
        "groups": {name: finalize_group(group) for name, group in sorted(groups.items())},
        "classes": {name: finalize_group(group) for name, group in sorted(class_groups.items())},
    }


def save_prediction_grid(model: YOLO, image_paths: list[Path], output_path: Path, args: argparse.Namespace) -> None:
    if args.mode == "standard":
        predictor = predict_standard
    elif args.mode == "multiscale":
        predictor = predict_multiscale
    else:
        predictor = predict_sahi

    indices = np.linspace(0, len(image_paths) - 1, min(6, len(image_paths)), dtype=int)
    fig, axes = plt.subplots(2, math.ceil(len(indices) / 2), figsize=(15, 8))
    axes = np.asarray(axes).reshape(-1)
    for axis, index in zip(axes, indices):
        image_path = image_paths[int(index)]
        image = read_image(image_path)
        canvas = image.copy()
        for prediction in predictor(model, image, args):
            if float(prediction["score"]) < args.conf:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in prediction["box"]]  # type: ignore[arg-type]
            label = int(prediction["label"])
            score = float(prediction["score"])
            color = (54, 179, 126)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"{VISDRONE_CATEGORIES.get(label, str(label))} {score:.2f}",
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


def parse_scales(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO small-object inference ablations on VisDrone.")
    parser.add_argument("--mode", choices=["standard", "multiscale", "sahi"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"))
    parser.add_argument("--val-annotations", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/annotations"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--scales", type=parse_scales, default=parse_scales("768,960,1280"))
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--slice-imgsz", type=int, default=960)
    parser.add_argument("--slice-overlap", type=float, default=0.2)
    parser.add_argument("--sahi-standard-prediction", action="store_true")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--eval-conf", type=float, default=0.01)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not args.val_images.exists():
        raise FileNotFoundError(f"Missing validation images: {args.val_images}")
    if not args.val_annotations.exists():
        raise FileNotFoundError(f"Missing validation annotations: {args.val_annotations}")

    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.val_images.glob("*.jpg"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    model = YOLO(str(args.weights))
    metrics = evaluate(model, image_paths, args.val_annotations, args)
    summary = {
        "mode": args.mode,
        "weights": str(args.weights),
        "settings": {
            "imgsz": args.imgsz,
            "scales": args.scales,
            "slice_size": args.slice_size,
            "slice_imgsz": args.slice_imgsz,
            "slice_overlap": args.slice_overlap,
            "include_standard_prediction": args.sahi_standard_prediction,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
        },
        "metrics": metrics,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_prediction_grid(model, image_paths, args.output / "validation_predictions.jpg", args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
