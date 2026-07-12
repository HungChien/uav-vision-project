from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def synchronize_if_needed(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def export_onnx(weights: Path, output: Path, imgsz: int, opset: int, simplify: bool) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return output

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False))
    if exported.resolve() != output.resolve():
        shutil.copy2(exported, output)
    return output


def boxes_from_result(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    rows = []
    for box, confidence, class_id in zip(xyxy, confidences, classes):
        rows.append(
            {
                "xyxy": [float(value) for value in box],
                "confidence": float(confidence),
                "class_id": int(class_id),
            }
        )
    return rows


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


def match_outputs(pytorch_boxes: list[dict], onnx_boxes: list[dict], iou_threshold: float) -> dict:
    used_onnx: set[int] = set()
    matched_ious = []
    confidence_diffs = []
    for pt_box in pytorch_boxes:
        best_index = None
        best_iou = 0.0
        for onnx_index, onnx_box in enumerate(onnx_boxes):
            if onnx_index in used_onnx:
                continue
            if onnx_box["class_id"] != pt_box["class_id"]:
                continue
            current_iou = iou(pt_box["xyxy"], onnx_box["xyxy"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = onnx_index
        if best_index is not None and best_iou >= iou_threshold:
            used_onnx.add(best_index)
            matched_ious.append(best_iou)
            confidence_diffs.append(abs(pt_box["confidence"] - onnx_boxes[best_index]["confidence"]))

    return {
        "matched_count": len(matched_ious),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_confidence_abs_diff": float(np.mean(confidence_diffs)) if confidence_diffs else 0.0,
    }


def draw_boxes(image: np.ndarray, boxes: list[dict], color: tuple[int, int, int], label_prefix: str) -> np.ndarray:
    canvas = image.copy()
    for item in boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in item["xyxy"]]
        label = f"{label_prefix} c{item['class_id']} {item['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def save_visualization(image: np.ndarray, pytorch_boxes: list[dict], onnx_boxes: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    left = draw_boxes(image, pytorch_boxes, (0, 180, 0), "pt")
    right = draw_boxes(image, onnx_boxes, (0, 0, 220), "onnx")
    cv2.putText(left, "PyTorch", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 2)
    cv2.putText(right, "ONNXRuntime", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
    if width > 720:
        scale = 720 / width
        new_size = (720, int(height * scale))
        left = cv2.resize(left, new_size)
        right = cv2.resize(right, new_size)
    cv2.imwrite(str(output_path), np.concatenate([left, right], axis=1))


def timed_predict(model: YOLO, image: np.ndarray, imgsz: int, conf: float, device: str | None) -> tuple[list[dict], float]:
    if device is not None:
        synchronize_if_needed(device)
    start = time.perf_counter()
    result = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if device is not None:
        synchronize_if_needed(device)
    elapsed = time.perf_counter() - start
    return boxes_from_result(result), elapsed


def verify(args: argparse.Namespace) -> dict:
    onnx_path = export_onnx(args.weights, args.onnx, args.imgsz, args.opset, args.simplify)
    image_paths = sorted(args.images.glob("*.jpg"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    pytorch_model = YOLO(str(args.weights))
    onnx_model = YOLO(str(onnx_path))

    for image_path in image_paths[: args.warmup]:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        _ = pytorch_model.predict(image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        _ = onnx_model.predict(image, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)

    rows = []
    pytorch_seconds = 0.0
    onnx_seconds = 0.0
    visualized = False
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        pytorch_boxes, pytorch_elapsed = timed_predict(pytorch_model, image, args.imgsz, args.conf, args.device)
        onnx_boxes, onnx_elapsed = timed_predict(onnx_model, image, args.imgsz, args.conf, "cpu")
        pytorch_seconds += pytorch_elapsed
        onnx_seconds += onnx_elapsed

        match = match_outputs(pytorch_boxes, onnx_boxes, args.match_iou)
        rows.append(
            {
                "image": image_path.name,
                "pytorch_count": len(pytorch_boxes),
                "onnx_count": len(onnx_boxes),
                "count_delta": len(onnx_boxes) - len(pytorch_boxes),
                "matched_count": match["matched_count"],
                "match_rate_from_pytorch": match["matched_count"] / len(pytorch_boxes) if pytorch_boxes else 1.0,
                "mean_matched_iou": match["mean_iou"],
                "mean_confidence_abs_diff": match["mean_confidence_abs_diff"],
                "pytorch_seconds": pytorch_elapsed,
                "onnx_seconds": onnx_elapsed,
            }
        )

        if not visualized and pytorch_boxes and onnx_boxes:
            save_visualization(image, pytorch_boxes, onnx_boxes, args.output / "pytorch_vs_onnx.jpg")
            visualized = True

    results = pd.DataFrame(rows)
    results.to_csv(args.output / "per_image_comparison.csv", index=False)

    summary = {
        "weights": str(args.weights),
        "onnx": str(onnx_path),
        "images": str(args.images),
        "image_count": int(len(results)),
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "match_iou_threshold": args.match_iou,
        "pytorch_device": args.device,
        "onnx_provider": "CPUExecutionProvider",
        "pytorch_fps": float(len(results) / pytorch_seconds) if pytorch_seconds > 0 else 0.0,
        "onnx_fps": float(len(results) / onnx_seconds) if onnx_seconds > 0 else 0.0,
        "pytorch_seconds_per_image": float(pytorch_seconds / len(results)) if len(results) else 0.0,
        "onnx_seconds_per_image": float(onnx_seconds / len(results)) if len(results) else 0.0,
        "pytorch_total_detections": int(results["pytorch_count"].sum()) if len(results) else 0,
        "onnx_total_detections": int(results["onnx_count"].sum()) if len(results) else 0,
        "mean_abs_count_delta": float(results["count_delta"].abs().mean()) if len(results) else 0.0,
        "mean_match_rate_from_pytorch": float(results["match_rate_from_pytorch"].mean()) if len(results) else 0.0,
        "mean_matched_iou": float(results["mean_matched_iou"].mean()) if len(results) else 0.0,
        "mean_confidence_abs_diff": float(results["mean_confidence_abs_diff"].mean()) if len(results) else 0.0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8s to ONNX and compare PyTorch vs ONNXRuntime predictions.")
    parser.add_argument("--weights", type=Path, default=Path("outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("models/exported/yolov8s_visdrone_aug_e10.onnx"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/deployment/yolov8s_visdrone_onnx_validation"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()

    summary = verify(args)
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
