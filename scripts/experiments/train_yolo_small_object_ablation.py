from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import MethodType

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import loss as ultralytics_loss
from ultralytics.utils.loss import v8DetectionLoss


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


class FocalV8DetectionLoss(v8DetectionLoss):
    def __init__(self, model: torch.nn.Module, gamma: float) -> None:
        super().__init__(model)
        self.gamma = gamma

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = ultralytics_loss.make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        probabilities = pred_scores.sigmoid()
        p_t = target_scores * probabilities + (1.0 - target_scores) * (1.0 - probabilities)
        focal_factor = (1.0 - p_t).clamp(min=0.0, max=1.0).pow(self.gamma)
        cls_loss = bce_loss * focal_factor
        if self.class_weights is not None:
            cls_loss *= self.class_weights
        loss[1] = cls_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


def make_focal_trainer(gamma: float):
    class FocalDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

            def init_criterion(self):
                return FocalV8DetectionLoss(self, gamma=gamma)

            model.init_criterion = MethodType(init_criterion, model)
            model.criterion = None
            return model

    return FocalDetectionTrainer


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
        width, height = float(parts[2]), float(parts[3])
        category_id = int(float(parts[5]))
        if category_id not in VISDRONE_TO_YOLO or width <= 0 or height <= 0:
            continue
        objects.append({"area": width * height})
    return objects


def build_small_resample_yaml(
    source_yaml: Path,
    output_dir: Path,
    strength: float,
    area_threshold: float,
) -> Path:
    train_images = Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/images")
    train_annotations = Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/annotations")
    val_images = Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    list_path = output_dir / f"small_resample_strength{strength:g}_train.txt"
    records = []
    for image_path in sorted(train_images.glob("*.jpg")):
        objects = read_annotation(train_annotations / f"{image_path.stem}.txt")
        small_count = sum(1 for obj in objects if obj["area"] < area_threshold)
        small_fraction = small_count / max(1, len(objects))
        repeat = 1 + int(round(strength * small_fraction))
        for _ in range(max(1, repeat)):
            records.append(image_path.resolve().as_posix())
    list_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(YOLO_NAMES))
    yaml_path = output_dir / f"small_resample_strength{strength:g}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {Path('.').resolve().as_posix()}",
                f"train: {list_path.resolve().as_posix()}",
                f"val: {val_images.resolve().as_posix()}",
                "",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    metadata = {
        "source_yaml": str(source_yaml),
        "train_list": str(list_path),
        "yaml": str(yaml_path),
        "source_train_images": len(list(train_images.glob("*.jpg"))),
        "resampled_train_entries": len(records),
        "strength": strength,
        "area_threshold": area_threshold,
    }
    (output_dir / f"small_resample_strength{strength:g}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO slim small-object training ablations.")
    parser.add_argument("--weights", type=Path, default=Path("outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/visdrone_yolo/visdrone.yaml"))
    parser.add_argument("--project", type=Path, default=Path("outputs/training"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--small-object-resample-strength", type=float, default=0.0)
    parser.add_argument("--resample-output", type=Path, default=Path("outputs/ablation/yolo_slim_small_objects/resample_data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    data = args.data
    if args.small_object_resample_strength > 0:
        data = build_small_resample_yaml(
            args.data,
            args.resample_output,
            strength=args.small_object_resample_strength,
            area_threshold=32.0 * 32.0,
        )

    model = YOLO(str(args.weights))
    trainer = make_focal_trainer(args.focal_gamma) if args.focal_gamma > 0 else None
    model.train(
        trainer=trainer,
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        lr0=args.lr0,
        lrf=0.01,
        optimizer="SGD",
        cos_lr=True,
        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=0,
        perspective=0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=3,
        plots=True,
    )


if __name__ == "__main__":
    main()
