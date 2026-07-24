from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one TensorRT detection engine on VisDrone validation.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/visdrone_yolo/visdrone.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/quantization/scene_calibration"),
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--validation-images", type=int, default=548)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.engine.exists():
        raise FileNotFoundError(args.engine)
    output_dir = args.output.resolve()
    run_dir = output_dir / args.name
    results = YOLO(str(args.engine), task="detect").val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    inference_ms = float(results.speed["inference"])
    summary = {
        "name": args.name,
        "engine": str(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "data": str(args.data),
        "validation_images": args.validation_images,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "preprocess_ms": float(results.speed["preprocess"]),
        "inference_ms": inference_ms,
        "postprocess_ms": float(results.speed["postprocess"]),
        "fps": 1000.0 / inference_ms,
        "results_dir": str(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
