from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
from modelopt.onnx.quantization import quantize
from ultralytics.utils.export.engine import onnx2engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed INT8/FP16 TensorRT layer-sensitivity engine.")
    parser.add_argument("--group", choices=("backbone", "neck", "head"), required=True)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/exported/yolov8s_slim04375_visdrone_e100_fp32.onnx"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/visdrone_yolo/int8_calibration/bright.txt"),
    )
    parser.add_argument(
        "--metadata-engine",
        type=Path,
        default=Path("models/exported/yolov8s_slim04375_visdrone_e100_fp16.engine"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/exported"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--workspace", type=float, default=4.0)
    return parser.parse_args()


def model_index(node_name: str) -> int | None:
    match = re.search(r"/model\.(\d+)(?:/|$)", node_name)
    return int(match.group(1)) if match else None


def in_group(node_name: str, group: str) -> bool:
    index = model_index(node_name)
    if index is None:
        return False
    if group == "backbone":
        return index <= 9
    if group == "neck":
        return 10 <= index <= 21
    return index == 22


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    width_padding = size - resized_width
    height_padding = size - resized_height
    left = int(round(width_padding / 2 - 0.1))
    right = int(round(width_padding / 2 + 0.1))
    top = int(round(height_padding / 2 - 0.1))
    bottom = int(round(height_padding / 2 + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


def calibration_array(manifest: Path, size: int) -> np.ndarray:
    paths = [Path(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = np.empty((len(paths), 3, size, size), dtype=np.float32)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read calibration image: {path}")
        data[index] = letterbox(image, size)
    return data


def read_engine_metadata(path: Path) -> dict:
    with path.open("rb") as file:
        length = int.from_bytes(file.read(4), byteorder="little", signed=True)
        if length <= 0 or length > 1_000_000:
            raise ValueError(f"No Ultralytics metadata header found in {path}")
        return json.loads(file.read(length).decode("utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.load(str(args.onnx), load_external_data=False)
    excluded = [node.name for node in source.graph.node if in_group(node.name, args.group)]
    excluded_ops = Counter(node.op_type for node in source.graph.node if node.name in set(excluded))
    input_name = source.graph.input[0].name
    stem = f"yolov8s_slim04375_visdrone_e100_int8_bright_{args.group}_fp16"
    quantized_path = args.output_dir / f"{stem}.onnx"
    engine_path = args.output_dir / f"{stem}.engine"

    started = time.perf_counter()
    calibration = calibration_array(args.manifest, args.imgsz)
    quantize(
        str(args.onnx),
        quantize_mode="int8",
        calibration_data={input_name: calibration},
        calibration_method="max",
        calibration_eps=["cpu"],
        nodes_to_exclude=excluded,
        high_precision_dtype="fp16",
        output_path=str(quantized_path),
    )
    del calibration

    quantized = onnx.load(str(quantized_path), load_external_data=False)
    node_counts = Counter(node.op_type for node in quantized.graph.node)
    metadata = read_engine_metadata(args.metadata_engine)
    onnx2engine(
        str(quantized_path),
        engine_path,
        workspace=args.workspace,
        shape=(1, 3, args.imgsz, args.imgsz),
        metadata=metadata,
        prefix=f"Sensitivity {args.group}: ",
    )
    summary = {
        "group_restored_to_fp16": args.group,
        "source_onnx": str(args.onnx),
        "calibration_manifest": str(args.manifest),
        "calibration_images": sum(1 for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()),
        "excluded_nodes": len(excluded),
        "excluded_operation_types": dict(sorted(excluded_ops.items())),
        "quantize_linear_nodes": node_counts["QuantizeLinear"],
        "dequantize_linear_nodes": node_counts["DequantizeLinear"],
        "quantized_onnx": str(quantized_path),
        "engine": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "build_seconds": time.perf_counter() - started,
    }
    summary_path = args.output_dir / f"{stem}_build.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
