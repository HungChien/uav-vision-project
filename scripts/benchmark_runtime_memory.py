from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "data" / "raw" / "VisDrone" / "VisDrone2019-DET" / "VisDrone2019-DET-val" / "images"
DEFAULT_OUTPUT = ROOT / "outputs" / "deployment" / "runtime_memory_benchmark"
DEFAULT_ASSETS = ROOT / "docs" / "assets" / "runtime_memory_benchmark"


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    backend: str
    model: str
    mode: str = "detect"
    imgsz: int = 960
    half: bool = False


SCENARIOS = [
    Scenario(
        name="full_pt_fp32_960",
        label="Full YOLOv8s\nPyTorch FP32 960",
        backend="pytorch",
        model="outputs/training/yolov8s_visdrone_mildaug_e100/weights/best.pt",
    ),
    Scenario(
        name="slim_pt_fp32_960",
        label="Slim 0.4375\nPyTorch FP32 960",
        backend="pytorch",
        model="outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt",
    ),
    Scenario(
        name="slim_pt_fp16_960",
        label="Slim 0.4375\nPyTorch FP16 960",
        backend="pytorch",
        model="outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt",
        half=True,
    ),
    Scenario(
        name="slim_pt_fp16_640",
        label="Slim 0.4375\nPyTorch FP16 640",
        backend="pytorch",
        model="outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt",
        imgsz=640,
        half=True,
    ),
    Scenario(
        name="slim_onnx_fp16_960",
        label="Slim 0.4375\nONNX FP16 960",
        backend="onnx",
        model="models/exported/yolov8s_slim04375_visdrone_e100_fp16.onnx",
        half=True,
    ),
    Scenario(
        name="slim_trt_fp16_960",
        label="Slim 0.4375\nTensorRT FP16 960",
        backend="tensorrt",
        model="models/exported/yolov8s_slim04375_visdrone_e100_fp16.engine",
        half=True,
    ),
    Scenario(
        name="slim_trt_int8_960",
        label="Slim 0.4375\nTensorRT INT8 960",
        backend="tensorrt",
        model="models/exported/yolov8s_slim04375_visdrone_e100_int8_bright.engine",
    ),
    Scenario(
        name="slim_trt_fp16_960_track",
        label="Slim 0.4375\nTensorRT FP16 + ByteTrack",
        backend="tensorrt",
        model="models/exported/yolov8s_slim04375_visdrone_e100_fp16.engine",
        mode="track",
        half=True,
    ),
]


def mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


class RuntimeSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.peak_rss_bytes = 0
        self.peak_gpu_process_bytes: int | None = None
        self.peak_gpu_device_bytes: int | None = None
        self._nvml = None
        self._handle = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._handle = None

    def close(self) -> None:
        self.stop()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def sample(self) -> dict[str, int | None]:
        rss = int(self.process.memory_info().rss)
        gpu_process = self._gpu_process_bytes()
        gpu_device = self._gpu_device_bytes()
        with self._lock:
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            if gpu_process is not None:
                self.peak_gpu_process_bytes = max(self.peak_gpu_process_bytes or 0, gpu_process)
            if gpu_device is not None:
                self.peak_gpu_device_bytes = max(self.peak_gpu_device_bytes or 0, gpu_device)
        return {
            "rss_bytes": rss,
            "gpu_process_bytes": gpu_process,
            "gpu_device_bytes": gpu_device,
        }

    def _gpu_process_bytes(self) -> int | None:
        if self._nvml is None or self._handle is None:
            return None
        processes = []
        for function_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            function = getattr(self._nvml, function_name, None)
            if function is None:
                continue
            try:
                processes.extend(function(self._handle))
            except Exception:
                continue
        values = [
            int(process.usedGpuMemory)
            for process in processes
            if int(process.pid) == os.getpid() and getattr(process, "usedGpuMemory", None) is not None
        ]
        return max(values) if values else 0

    def _gpu_device_bytes(self) -> int | None:
        if self._nvml is None or self._handle is None:
            return None
        try:
            return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        except Exception:
            return None

    def reset_peaks(self) -> None:
        current = self.sample()
        with self._lock:
            self.peak_rss_bytes = int(current["rss_bytes"] or 0)
            self.peak_gpu_process_bytes = current["gpu_process_bytes"]
            self.peak_gpu_device_bytes = current["gpu_device_bytes"]

    def peaks(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "rss_bytes": self.peak_rss_bytes,
                "gpu_process_bytes": self.peak_gpu_process_bytes,
                "gpu_device_bytes": self.peak_gpu_device_bytes,
            }

    def start(self) -> None:
        self.reset_peaks()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 5))
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()


def snapshot_mb(sample: dict[str, int | None]) -> dict[str, float | None]:
    return {
        "rss_mb": mib(sample["rss_bytes"]),
        "gpu_process_mb": mib(sample["gpu_process_bytes"]),
        "gpu_device_mb": mib(sample["gpu_device_bytes"]),
    }


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def run_prediction(model, image: np.ndarray, args: argparse.Namespace):
    common = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": 0,
        "verbose": False,
    }
    if args.half:
        common["half"] = True
    if args.mode == "track":
        return model.track(image, persist=True, tracker="bytetrack.yaml", **common)[0]
    return model.predict(image, **common)[0]


def run_worker(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model_path = Path(args.model).resolve()
    image_paths = sorted(Path(args.images).glob("*.jpg"))
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    sampler = RuntimeSampler(args.sample_interval)
    baseline = sampler.sample()
    sampler.start()

    try:
        model = YOLO(str(model_path), task="detect")
        after_constructor = sampler.sample()

        for index in range(args.warmup):
            image = cv2.imread(str(image_paths[index % len(image_paths)]))
            if image is None:
                raise ValueError(f"Could not read {image_paths[index % len(image_paths)]}")
            result = run_prediction(model, image, args)
            _ = len(result.boxes) if result.boxes is not None else 0
        sync_cuda()
        after_warmup = sampler.sample()
        startup_peaks = sampler.peaks()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            torch = None

        sampler.reset_peaks()
        latencies = []
        detections = 0
        for index in range(args.iterations):
            image = cv2.imread(str(image_paths[(index + args.warmup) % len(image_paths)]))
            if image is None:
                continue
            sync_cuda()
            start = time.perf_counter()
            result = run_prediction(model, image, args)
            sync_cuda()
            latencies.append(time.perf_counter() - start)
            detections += len(result.boxes) if result.boxes is not None else 0

        after_inference = sampler.sample()
        runtime_peaks = sampler.peaks()
        torch_allocated = None
        torch_reserved = None
        try:
            import torch

            if torch.cuda.is_available():
                torch_allocated = int(torch.cuda.max_memory_allocated())
                torch_reserved = int(torch.cuda.max_memory_reserved())
        except Exception:
            pass
    finally:
        sampler.close()

    baseline_mb = snapshot_mb(baseline)
    startup_peak_mb = snapshot_mb(startup_peaks)
    runtime_peak_mb = snapshot_mb(runtime_peaks)
    latency_array = np.asarray(latencies, dtype=np.float64)
    result = {
        "name": args.name,
        "label": args.label,
        "backend": args.backend,
        "mode": args.mode,
        "model": str(model_path),
        "model_size_mb": model_path.stat().st_size / (1024.0 * 1024.0),
        "imgsz": args.imgsz,
        "half": bool(args.half),
        "warmup_iterations": args.warmup,
        "measured_iterations": len(latencies),
        "total_detections": detections,
        "baseline": baseline_mb,
        "after_constructor": snapshot_mb(after_constructor),
        "after_warmup": snapshot_mb(after_warmup),
        "after_inference": snapshot_mb(after_inference),
        "startup_peak": startup_peak_mb,
        "runtime_peak": runtime_peak_mb,
        "runtime_increment": {
            "rss_mb": runtime_peak_mb["rss_mb"] - baseline_mb["rss_mb"],
            "gpu_process_mb": (
                runtime_peak_mb["gpu_process_mb"] - baseline_mb["gpu_process_mb"]
                if runtime_peak_mb["gpu_process_mb"] is not None and baseline_mb["gpu_process_mb"] is not None
                else None
            ),
            "gpu_device_mb": (
                runtime_peak_mb["gpu_device_mb"] - baseline_mb["gpu_device_mb"]
                if runtime_peak_mb["gpu_device_mb"] is not None and baseline_mb["gpu_device_mb"] is not None
                else None
            ),
        },
        "torch_peak_allocated_mb": mib(torch_allocated),
        "torch_peak_reserved_mb": mib(torch_reserved),
        "latency_ms": {
            "mean": float(latency_array.mean() * 1000.0),
            "median": float(np.median(latency_array) * 1000.0),
            "p95": float(np.percentile(latency_array, 95) * 1000.0),
            "min": float(latency_array.min() * 1000.0),
            "max": float(latency_array.max() * 1000.0),
        },
        "fps_from_mean_latency": float(1.0 / latency_array.mean()),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def worker_command(scenario: Scenario, args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--name",
        scenario.name,
        "--label",
        scenario.label,
        "--backend",
        scenario.backend,
        "--model",
        str(ROOT / scenario.model),
        "--mode",
        scenario.mode,
        "--imgsz",
        str(scenario.imgsz),
        "--images",
        str(args.images),
        "--output",
        str(output),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--sample-interval",
        str(args.sample_interval),
    ]
    if scenario.half:
        command.append("--half")
    return command


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_ram_gb": psutil.virtual_memory().total / (1024.0**3),
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        if len(values) >= 3:
            info.update({"gpu": values[0], "gpu_memory_mb": float(values[1]), "driver": values[2]})
    except Exception:
        pass
    return info


def flatten_result(result: dict) -> dict:
    process_gpu_peak = result["runtime_peak"]["gpu_process_mb"]
    if process_gpu_peak is not None and process_gpu_peak > 0:
        runtime_gpu_memory = result["runtime_increment"]["gpu_process_mb"]
        gpu_memory_source = "NVML per-process delta"
    else:
        runtime_gpu_memory = result["runtime_increment"]["gpu_device_mb"]
        gpu_memory_source = "NVML device delta (WDDM fallback)"
    return {
        "name": result["name"],
        "label": result["label"].replace("\n", " "),
        "backend": result["backend"],
        "mode": result["mode"],
        "precision": "FP16" if result["half"] or "fp16" in result["model"].lower() else (
            "INT8" if "int8" in result["model"].lower() else "FP32"
        ),
        "imgsz": result["imgsz"],
        "model_size_mb": result["model_size_mb"],
        "baseline_rss_mb": result["baseline"]["rss_mb"],
        "startup_peak_rss_mb": result["startup_peak"]["rss_mb"],
        "runtime_peak_rss_mb": result["runtime_peak"]["rss_mb"],
        "runtime_increment_rss_mb": result["runtime_increment"]["rss_mb"],
        "runtime_peak_gpu_process_mb": result["runtime_peak"]["gpu_process_mb"],
        "runtime_increment_gpu_process_mb": result["runtime_increment"]["gpu_process_mb"],
        "runtime_peak_gpu_device_mb": result["runtime_peak"]["gpu_device_mb"],
        "runtime_gpu_memory_mb": runtime_gpu_memory,
        "gpu_memory_source": gpu_memory_source,
        "torch_peak_allocated_mb": result["torch_peak_allocated_mb"],
        "torch_peak_reserved_mb": result["torch_peak_reserved_mb"],
        "mean_latency_ms": result["latency_ms"]["mean"],
        "median_latency_ms": result["latency_ms"]["median"],
        "p95_latency_ms": result["latency_ms"]["p95"],
        "fps": result["fps_from_mean_latency"],
    }


def save_chart(rows: list[dict], output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    rss = [row["runtime_peak_rss_mb"] for row in rows]
    gpu = [row["runtime_gpu_memory_mb"] or 0.0 for row in rows]
    latency = [row["mean_latency_ms"] for row in rows]
    colors = ["#2F6690", "#3A7D44", "#57A773", "#8FCB9B", "#5B8E7D", "#E09F3E", "#C44536", "#6D597A"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
    y = np.arange(len(rows))
    for axis, values, title, xlabel in (
        (axes[0], rss, "Peak process RAM", "MiB"),
        (axes[1], gpu, "Runtime GPU memory increase", "MiB"),
        (axes[2], latency, "End-to-end latency", "ms / frame"),
    ):
        bars = axis.barh(y, values, color=colors[: len(rows)])
        axis.set_yticks(y, labels if axis is axes[0] else [""] * len(rows))
        axis.invert_yaxis()
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}", va="center", fontsize=8)
    fig.suptitle("UAV Runtime Memory and Latency Benchmark (batch=1)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_suite(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for scenario in SCENARIOS:
        scenario_output = args.output / scenario.name
        scenario_output.mkdir(parents=True, exist_ok=True)
        command = worker_command(scenario, args, scenario_output)
        print(f"[run] {scenario.name}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (scenario_output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (scenario_output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        summary_path = scenario_output / "summary.json"
        if completed.returncode != 0 or not summary_path.exists():
            failures.append(
                {
                    "name": scenario.name,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                }
            )
            print(f"[failed] {scenario.name}", flush=True)
            continue
        results.append(json.loads(summary_path.read_text(encoding="utf-8")))
        print(f"[done] {scenario.name}", flush=True)

    rows = [flatten_result(result) for result in results]
    if not rows:
        raise RuntimeError(f"All memory benchmark scenarios failed: {failures}")

    fieldnames = list(rows[0])
    summary_csv = args.output / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "protocol": {
            "batch_size": 1,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "sample_interval_seconds": args.sample_interval,
            "ram_metric": "Peak resident set size (RSS) of the isolated inference process.",
            "gpu_metric": "Peak per-process GPU memory sampled with NVML.",
            "gpu_fallback": "When Windows WDDM does not expose per-process memory, device-memory delta is reported.",
            "latency_metric": "Ultralytics end-to-end predict/track call, excluding image disk read.",
        },
        "hardware": hardware_info(),
        "results": rows,
        "failures": failures,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_chart(rows, args.output / "runtime_memory_benchmark.png")

    args.assets.mkdir(parents=True, exist_ok=True)
    (args.assets / "summary.csv").write_text(summary_csv.read_text(encoding="utf-8"), encoding="utf-8")
    (args.assets / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_chart(rows, args.assets / "runtime_memory_benchmark.png")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark isolated runtime RAM, GPU memory, and latency.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--name", default="custom")
    parser.add_argument("--label", default="Custom")
    parser.add_argument("--backend", choices=["pytorch", "onnx", "tensorrt"], default="pytorch")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mode", choices=["detect", "track"], default="detect")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        if args.model is None:
            raise ValueError("--model is required in worker mode")
        run_worker(args)
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
