from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import batched_nms


MODULE_PATH = Path(__file__).with_name("train_mobilenet_fpn.py")
SPEC = importlib.util.spec_from_file_location("mobilenet_training", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load training utilities from {MODULE_PATH}")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)

Prediction = dict[str, float | int | list[float]]
Predictor = Callable[[np.ndarray], list[Prediction]]


def load_model(checkpoint_path: Path, device: torch.device, args: argparse.Namespace):
    model = training.build_model(
        num_classes=11,
        min_size=args.min_size,
        max_size=args.max_size,
        pretrained=False,
        detections_per_image=args.detections_per_image,
        score_threshold=args.eval_score_threshold,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device).eval()
    return model


def output_to_predictions(output: dict[str, torch.Tensor]) -> list[Prediction]:
    return [
        {
            "box": [float(value) for value in box],
            "label": int(label),
            "score": float(score),
        }
        for box, label, score in zip(
            output["boxes"].detach().cpu().tolist(),
            output["labels"].detach().cpu().tolist(),
            output["scores"].detach().cpu().tolist(),
        )
    ]


def infer_once(model, image: np.ndarray, device: torch.device) -> list[Prediction]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
    with torch.inference_mode():
        output = model([tensor])[0]
    return output_to_predictions(output)


def build_standard_predictor(model, device: torch.device) -> Predictor:
    return lambda image: infer_once(model, image, device)


def build_multiscale_predictor(
    model,
    device: torch.device,
    scales: list[tuple[int, int]],
    nms_iou: float,
    detections_per_image: int,
) -> Predictor:
    def predict(image: np.ndarray) -> list[Prediction]:
        outputs = []
        original_min_size = model.transform.min_size
        original_max_size = model.transform.max_size
        try:
            for min_size, max_size in scales:
                model.transform.min_size = (min_size,)
                model.transform.max_size = max_size
                outputs.extend(infer_once(model, image, device))
        finally:
            model.transform.min_size = original_min_size
            model.transform.max_size = original_max_size

        if not outputs:
            return []
        boxes = torch.tensor([item["box"] for item in outputs], dtype=torch.float32, device=device)
        scores = torch.tensor([item["score"] for item in outputs], dtype=torch.float32, device=device)
        labels = torch.tensor([item["label"] for item in outputs], dtype=torch.int64, device=device)
        kept = batched_nms(boxes, scores, labels, nms_iou)[:detections_per_image].detach().cpu().tolist()
        return [outputs[index] for index in kept]

    return predict


def build_sahi_predictor(model, device: torch.device, args: argparse.Namespace) -> Predictor:
    from sahi.models.torchvision import TorchVisionDetectionModel
    from sahi.predict import get_sliced_prediction

    category_mapping = {"0": "background"}
    category_mapping.update({str(key): value for key, value in training.VISDRONE_CATEGORIES.items()})
    detection_model = TorchVisionDetectionModel(
        model=model,
        device=str(device),
        confidence_threshold=args.eval_score_threshold,
        category_mapping=category_mapping,
        load_at_init=True,
    )

    def predict(image: np.ndarray) -> list[Prediction]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            rgb,
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.slice_overlap,
            overlap_width_ratio=args.slice_overlap,
            perform_standard_pred=args.sahi_standard_prediction,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=args.nms_iou,
            postprocess_class_agnostic=False,
            verbose=0,
            progress_bar=False,
            batch_size=1,
            force_postprocess_type=True,
        )
        predictions = [
            {
                "box": [float(value) for value in item.bbox.to_xyxy()],
                "label": int(item.category.id),
                "score": float(item.score.value),
            }
            for item in result.object_prediction_list
        ]
        predictions.sort(key=lambda item: float(item["score"]), reverse=True)
        return predictions[: args.detections_per_image]

    return predict


def evaluate_predictor(
    predictor: Predictor,
    image_paths: list[Path],
    annotations_dir: Path,
    device: torch.device,
    confidence: float,
    matching_iou: float,
    warmup: int,
) -> dict:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    class_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    all_ground_truth: list[list[dict]] = []
    all_predictions: list[list[Prediction]] = []
    total_predictions = 0
    total_matches = 0
    timed_seconds = 0.0

    for image_path in image_paths[:warmup]:
        predictor(training.read_image(image_path))

    for image_path in image_paths:
        image = training.read_image(image_path)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        metric_predictions = predictor(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timed_seconds += time.perf_counter() - started

        report_predictions = [item for item in metric_predictions if float(item["score"]) >= confidence]
        ground_truth = training.read_annotation(annotations_dir / f"{image_path.stem}.txt")
        matched_gt, matched_predictions = training.match_predictions(
            ground_truth,
            report_predictions,
            matching_iou,
        )
        total_predictions += len(report_predictions)
        total_matches += sum(matched_predictions)
        all_ground_truth.append(ground_truth)
        all_predictions.append(metric_predictions)

        for gt, matched in zip(ground_truth, matched_gt):
            training.update_group(groups, "all", matched)
            training.update_group(groups, training.size_bucket(gt["area"]), matched)
            training.update_group(groups, f"occlusion_{gt['occlusion']}", matched)
            training.update_group(class_groups, training.VISDRONE_CATEGORIES[gt["label"]], matched)

    ap_metrics = training.compute_ap_metrics(
        all_ground_truth,
        all_predictions,
        [round(0.5 + 0.05 * index, 2) for index in range(10)],
    )
    total_gt = sum(len(items) for items in all_ground_truth)
    image_count = len(image_paths)
    return {
        "image_count": image_count,
        "confidence_threshold": confidence,
        "matching_iou_threshold": matching_iou,
        "precision_at_conf": total_matches / total_predictions if total_predictions else 0.0,
        "recall_at_conf": total_matches / total_gt if total_gt else 0.0,
        "fps": image_count / timed_seconds if timed_seconds else 0.0,
        "seconds_per_image": timed_seconds / image_count if image_count else 0.0,
        "total_predictions": total_predictions,
        "total_ground_truth": total_gt,
        "total_matches": total_matches,
        **ap_metrics,
        "groups": {name: training.finalize_group(group) for name, group in sorted(groups.items())},
        "classes": {name: training.finalize_group(group) for name, group in sorted(class_groups.items())},
    }


def save_prediction_grid(
    predictor: Predictor,
    image_paths: list[Path],
    output_path: Path,
    confidence: float,
) -> None:
    indices = np.linspace(0, len(image_paths) - 1, min(6, len(image_paths)), dtype=int)
    fig, axes = plt.subplots(2, math.ceil(len(indices) / 2), figsize=(15, 8))
    axes = np.asarray(axes).reshape(-1)
    for axis, index in zip(axes, indices):
        image_path = image_paths[int(index)]
        image = training.read_image(image_path)
        canvas = image.copy()
        for prediction in predictor(image):
            if float(prediction["score"]) < confidence:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in prediction["box"]]
            label = int(prediction["label"])
            score = float(prediction["score"])
            color = (54, 179, 126)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"{training.VISDRONE_CATEGORIES.get(label, str(label))} {score:.2f}",
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


def parse_scales(value: str) -> list[tuple[int, int]]:
    scales = []
    for item in value.split(","):
        min_size, max_size = item.split(":")
        scales.append((int(min_size), int(max_size)))
    return scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MobileNet small-object inference ablations on VisDrone.")
    parser.add_argument("--mode", choices=["standard", "multiscale", "sahi"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--val-images",
        type=Path,
        default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"),
    )
    parser.add_argument(
        "--val-annotations",
        type=Path,
        default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/annotations"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--scales", default="640:960,800:1280,960:1536")
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--slice-overlap", type=float, default=0.2)
    parser.add_argument("--sahi-standard-prediction", action="store_true")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--detections-per-image", type=int, default=300)
    parser.add_argument("--eval-score-threshold", type=float, default=0.01)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args)

    if args.mode == "standard":
        predictor = build_standard_predictor(model, device)
        settings = {"min_size": args.min_size, "max_size": args.max_size}
    elif args.mode == "multiscale":
        scales = parse_scales(args.scales)
        predictor = build_multiscale_predictor(
            model,
            device,
            scales=scales,
            nms_iou=args.nms_iou,
            detections_per_image=args.detections_per_image,
        )
        settings = {"scales": scales, "nms_iou": args.nms_iou}
    else:
        predictor = build_sahi_predictor(model, device, args)
        settings = {
            "slice_height": args.slice_height,
            "slice_width": args.slice_width,
            "overlap": args.slice_overlap,
            "include_standard_prediction": args.sahi_standard_prediction,
            "postprocess": "class-aware NMS",
            "nms_iou": args.nms_iou,
            "model_min_size": args.min_size,
            "model_max_size": args.max_size,
        }

    image_paths = sorted(args.val_images.glob("*.jpg"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    metrics = evaluate_predictor(
        predictor,
        image_paths,
        args.val_annotations,
        device,
        confidence=args.conf,
        matching_iou=args.iou,
        warmup=min(args.warmup, len(image_paths)),
    )
    summary = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "settings": settings,
        "metrics": metrics,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_prediction_grid(
        predictor,
        image_paths,
        args.output / "validation_predictions.jpg",
        confidence=args.conf,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()