from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8s on the VisDrone detection dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/visdrone_yolo/visdrone.yaml"))
    parser.add_argument("--model", default="yolov8s.pt", help="Ultralytics model checkpoint or model YAML.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="outputs/training")
    parser.add_argument("--name", default="yolov8s_visdrone")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--degrees", type=float, default=0.0, help="Random rotation range in degrees.")
    parser.add_argument("--translate", type=float, default=0.1, help="Random translation fraction.")
    parser.add_argument("--scale", type=float, default=0.5, help="Random scale gain.")
    parser.add_argument("--shear", type=float, default=0.0, help="Random shear range in degrees.")
    parser.add_argument("--perspective", type=float, default=0.0, help="Random perspective transform fraction.")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue augmentation gain.")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation augmentation gain.")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV value augmentation gain.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp augmentation probability.")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation probability.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for the last N epochs.")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}. Run scripts/convert_visdrone_to_yolo.py first.")

    model = YOLO(args.model)
    project_dir = Path(args.project).resolve()
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        pretrained=True,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
        cos_lr=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
