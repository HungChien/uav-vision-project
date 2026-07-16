from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized_h) // 2
    left = (size - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def preprocess(image: np.ndarray, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
    image = letterbox(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1).astype(dtype) / 255.0
    return np.expand_dims(image, axis=0)


def load_inputs(image_paths: list[Path], size: int, dtype: np.dtype = np.float32) -> list[np.ndarray]:
    inputs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        inputs.append(preprocess(image, size, dtype))
    return inputs


def count_readable_images(image_paths: list[Path]) -> int:
    count = 0
    for path in image_paths:
        if cv2.imread(str(path)) is not None:
            count += 1
    return count


def create_session(model_path: Path, provider: str) -> tuple[ort.InferenceSession | None, str | None]:
    try:
        providers = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider]
        session = ort.InferenceSession(str(model_path), providers=providers)
        active_providers = session.get_providers()
        if provider not in active_providers:
            return session, f"Requested {provider}, active providers: {active_providers}"
        return session, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def benchmark_session(session: ort.InferenceSession, inputs: list[np.ndarray], warmup: int) -> dict:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for input_tensor in inputs[:warmup]:
        session.run([output_name], {input_name: input_tensor})

    elapsed_values = []
    output_shapes = []
    for input_tensor in inputs:
        start = time.perf_counter()
        outputs = session.run([output_name], {input_name: input_tensor})
        elapsed_values.append(time.perf_counter() - start)
        output_shapes.append(list(outputs[0].shape))

    total = float(sum(elapsed_values))
    return {
        "image_count": len(elapsed_values),
        "fps": len(elapsed_values) / total if total > 0 else 0.0,
        "seconds_per_image": total / len(elapsed_values) if elapsed_values else 0.0,
        "min_seconds": float(np.min(elapsed_values)) if elapsed_values else 0.0,
        "median_seconds": float(np.median(elapsed_values)) if elapsed_values else 0.0,
        "max_seconds": float(np.max(elapsed_values)) if elapsed_values else 0.0,
        "output_shape": output_shapes[0] if output_shapes else None,
    }


def input_dtype_for_session(session: ort.InferenceSession) -> np.dtype:
    input_type = session.get_inputs()[0].type
    if input_type == "tensor(float16)":
        return np.float16
    return np.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNXRuntime execution providers on preprocessed image tensors.")
    parser.add_argument("--model", type=Path, default=Path("models/exported/yolov8s_visdrone_aug_e10.onnx"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/deployment/yolov8s_onnxruntime_backend_benchmark"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider", "CUDAExecutionProvider"])
    parser.add_argument("--preload-cuda-dlls", action="store_true")
    args = parser.parse_args()

    if args.preload_cuda_dlls and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    image_paths = sorted(args.images.glob("*.jpg"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model),
        "images": str(args.images),
        "imgsz": args.imgsz,
        "requested_image_count": len(image_paths),
        "readable_image_count": count_readable_images(image_paths),
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "preload_cuda_dlls": bool(args.preload_cuda_dlls),
        "providers": {},
    }

    for provider in args.providers:
        session, error = create_session(args.model, provider)
        if session is None:
            report["providers"][provider] = {"available": False, "error": error}
            continue
        inputs = load_inputs(image_paths, args.imgsz, input_dtype_for_session(session))
        if not inputs:
            raise FileNotFoundError(f"No readable images found in {args.images}")
        metrics = benchmark_session(session, inputs, args.warmup)
        metrics["available"] = error is None
        metrics["active_providers"] = session.get_providers()
        metrics["input_type"] = session.get_inputs()[0].type
        if error:
            metrics["warning"] = error
        report["providers"][provider] = metrics

    (args.output / "summary.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(report, indent=2, default=_json_default))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
