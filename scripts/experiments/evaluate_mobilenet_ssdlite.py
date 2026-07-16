from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large


VISDRONE_TO_COMPATIBLE = {
    1: "person",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "car",
    6: "truck",
    9: "bus",
    10: "motorcycle",
}

COCO_TO_COMPATIBLE = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def read_ground_truth(annotation_path: Path) -> tuple[list[dict], int]:
    objects: list[dict] = []
    ignored = 0
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        left, top, width, height = map(float, parts[:4])
        category_id = int(float(parts[5]))
        class_name = VISDRONE_TO_COMPATIBLE.get(category_id)
        if class_name is None:
            ignored += 1
            continue
        if width <= 0 or height <= 0:
            ignored += 1
            continue
        objects.append(
            {
                "xyxy": [left, top, left + width, top + height],
                "class_name": class_name,
                "area": width * height,
                "truncation": int(float(parts[6])),
                "occlusion": int(float(parts[7])),
            }
        )
    return objects, ignored


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
            if pred["class_name"] != gt["class_name"]:
                continue
            current_iou = iou(gt["xyxy"], pred["xyxy"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            matched_predictions.add(best_index)
            matched_gt[gt_index] = True

    return matched_gt


def finalize_group(group: dict[str, int]) -> dict[str, float | int]:
    gt = group["gt"]
    matched = group["matched"]
    return {"gt": gt, "matched": matched, "recall_at_iou": matched / gt if gt else 0.0}


def draw_predictions(image: np.ndarray, predictions: list[dict], output_path: Path) -> None:
    canvas = image.copy()
    for pred in predictions:
        x1, y1, x2, y2 = [int(round(value)) for value in pred["xyxy"]]
        label = f"{pred['class_name']} {pred['score']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 150, 220), 2)
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 220), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate COCO-pretrained SSDLite MobileNetV3 on VisDrone-compatible classes.")
    parser.add_argument("--images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"))
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/annotations"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/mobilenet_ssdlite_coco_visdrone"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Missing images directory: {args.images}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Missing annotations directory: {args.annotations}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device)
    model.eval()

    image_paths = sorted(args.images.glob("*.jpg"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    class_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    total_predictions = 0
    ignored_gt = 0
    timed_seconds = 0.0
    visualized = False

    with torch.inference_mode():
        warmup_images = image_paths[: args.warmup]
        for image_path in warmup_images:
            image = read_image(image_path)
            _ = model([image_to_tensor(image, device)])

        for image_path in image_paths:
            image = read_image(image_path)
            tensor = image_to_tensor(image, device)

            synchronize_if_needed(device)
            start = time.perf_counter()
            output = model([tensor])[0]
            synchronize_if_needed(device)
            timed_seconds += time.perf_counter() - start

            boxes = output["boxes"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            predictions = []
            for box, label, score in zip(boxes, labels, scores):
                if score < args.conf:
                    continue
                class_name = COCO_TO_COMPATIBLE.get(int(label))
                if class_name is None:
                    continue
                predictions.append({"xyxy": [float(value) for value in box], "class_name": class_name, "score": float(score)})
            total_predictions += len(predictions)

            gt_objects, ignored = read_ground_truth(args.annotations / f"{image_path.stem}.txt")
            ignored_gt += ignored
            matched = match_predictions(gt_objects, predictions, args.iou)

            for gt, is_matched in zip(gt_objects, matched):
                update_group(groups, "all_compatible", is_matched)
                update_group(groups, size_bucket(gt["area"]), is_matched)
                update_group(groups, f"occlusion_{gt['occlusion']}", is_matched)
                update_group(groups, f"truncation_{gt['truncation']}", is_matched)
                update_group(class_groups, gt["class_name"], is_matched)

            if not visualized and predictions:
                draw_predictions(image, predictions, args.output / "mobilenet_ssdlite_predictions.jpg")
                visualized = True

    summary = {
        "model": "ssdlite320_mobilenet_v3_large",
        "weights": "COCO_V1",
        "images": str(args.images),
        "annotations": str(args.annotations),
        "image_count": len(image_paths),
        "device": str(device),
        "confidence_threshold": args.conf,
        "matching_iou_threshold": args.iou,
        "fps": len(image_paths) / timed_seconds if timed_seconds > 0 else 0.0,
        "seconds_per_image": timed_seconds / len(image_paths) if image_paths else 0.0,
        "total_predictions": total_predictions,
        "compatible_gt": groups["all_compatible"]["gt"],
        "ignored_gt": ignored_gt,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "groups": {name: finalize_group(group) for name, group in sorted(groups.items())},
        "classes": {name: finalize_group(group) for name, group in sorted(class_groups.items())},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
