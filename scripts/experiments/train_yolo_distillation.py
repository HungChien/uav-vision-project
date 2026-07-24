from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src").resolve()))

from uav_vision.distillation import (  # noqa: E402
    DistillationConfig,
    DistillationDetectionModel,
    DistillationTrainer,
    configure_distillation,
    consume_distillation_stats,
    reset_distillation_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a slim YOLOv8 detector with a full YOLOv8s teacher.")
    parser.add_argument(
        "--student",
        type=Path,
        default=Path("outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt"),
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("outputs/training/yolov8s_visdrone_mildaug_e100/weights/best.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/visdrone_yolo/visdrone.yaml"))
    parser.add_argument("--project", type=Path, default=Path("outputs/training"))
    parser.add_argument("--name", default="yolov8s_slim04375_distilled_e20")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--box-weight", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--topk", type=int, default=1500)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sanitize_checkpoint(path: Path) -> None:
    if not path.exists():
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model")
    if isinstance(model, DistillationDetectionModel):
        model.__class__ = DetectionModel
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    for path in (args.student, args.teacher, args.data):
        if not path.exists():
            raise FileNotFoundError(path)

    project = args.project.resolve()
    output = project / args.name
    config = DistillationConfig(
        classification_weight=args.classification_weight,
        box_weight=args.box_weight,
        temperature=args.temperature,
        topk=args.topk,
    )
    teacher = YOLO(str(args.teacher)).model
    configure_distillation(teacher, config)
    reset_distillation_stats()

    student = YOLO(str(args.student))

    def on_train_start(trainer) -> None:
        configure_distillation(teacher.to(trainer.device).eval(), config)
        metadata = {
            "student": str(args.student),
            "teacher": str(args.teacher),
            "config": config.__dict__,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "fraction": args.fraction,
        }
        trainer.save_dir.mkdir(parents=True, exist_ok=True)
        (trainer.save_dir / "distillation_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def on_train_epoch_end(trainer) -> None:
        row = {"epoch": int(trainer.epoch + 1), **consume_distillation_stats()}
        with (trainer.save_dir / "distillation_metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")

    student.add_callback("on_train_start", on_train_start)
    student.add_callback("on_train_epoch_end", on_train_epoch_end)
    student.train(
        trainer=DistillationTrainer,
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=False,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        fraction=args.fraction,
        pretrained=True,
        optimizer="SGD",
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        cos_lr=True,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=min(5, args.epochs),
        amp=True,
        plots=True,
    )

    for checkpoint_name in ("best.pt", "last.pt"):
        sanitize_checkpoint(output / "weights" / checkpoint_name)
    print(json.dumps({"output": str(output), "best": str(output / "weights" / "best.pt")}, indent=2))


if __name__ == "__main__":
    main()
