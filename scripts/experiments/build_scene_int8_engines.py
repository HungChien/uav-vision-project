from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT INT8 engine from one scene-specific calibration set.")
    parser.add_argument("--scene", choices=("bright", "dark", "dense"), required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt"),
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("data/processed/visdrone_yolo/int8_calibration"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/exported"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workspace", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_yaml = args.calibration_dir / f"{args.scene}.yaml"
    if not calibration_yaml.exists():
        raise FileNotFoundError(calibration_yaml)

    started = time.perf_counter()
    exported = Path(
        YOLO(str(args.weights)).export(
            format="engine",
            imgsz=args.imgsz,
            int8=True,
            data=str(calibration_yaml),
            batch=1,
            device=args.device,
            workspace=args.workspace,
            simplify=True,
        )
    )
    source_int8_onnx = args.weights.with_suffix(".int8.onnx")
    stem = f"yolov8s_slim04375_visdrone_e100_int8_{args.scene}"
    engine_path = args.output_dir / f"{stem}.engine"
    onnx_path = args.output_dir / f"{stem}.onnx"
    shutil.copy2(exported, engine_path)
    shutil.copy2(source_int8_onnx, onnx_path)

    summary = {
        "scene": args.scene,
        "weights": str(args.weights),
        "calibration_yaml": str(calibration_yaml),
        "imgsz": args.imgsz,
        "engine": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "quantized_onnx": str(onnx_path),
        "quantized_onnx_bytes": onnx_path.stat().st_size,
        "build_seconds": time.perf_counter() - started,
    }
    summary_path = args.calibration_dir / f"build_{args.scene}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
