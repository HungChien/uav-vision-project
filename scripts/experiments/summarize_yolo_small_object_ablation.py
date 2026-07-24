"""Summarize YOLO slim small-object ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("full_yolov8s_standard_ref", "Full YOLOv8s standard ref", "full_yolov8s_standard_ref"),
    ("standard", "Standard slim", "standard"),
    ("multiscale_768_960_1280", "Multi-scale 768/960/1280", "multiscale_768_960_1280"),
    ("sahi_640_overlap020", "SAHI 640 overlap 0.20", "sahi_640_overlap020"),
    ("control_finetune_e6_eval", "Matched fine-tune e6", "control_finetune_e6_eval"),
    ("focal_gamma2_e6_eval", "Focal gamma 2 e6", "focal_gamma2_e6_eval"),
    (
        "small_resample_strength2_e6_eval",
        "Small-object resample e6",
        "small_resample_strength2_e6_eval",
    ),
]


def load_row(root: Path, key: str, name: str, folder: str) -> dict[str, float | int | str]:
    summary_path = root / folder / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = summary["metrics"]
    groups = metrics["groups"]
    return {
        "key": key,
        "method": name,
        "images": metrics["image_count"],
        "precision_at_conf": metrics["precision_at_conf"],
        "recall_at_conf": metrics["recall_at_conf"],
        "small_recall": groups["small_lt_32x32"]["recall_at_iou"],
        "medium_recall": groups["medium_32x32_to_96x96"]["recall_at_iou"],
        "large_recall": groups["large_ge_96x96"]["recall_at_iou"],
        "heavy_occlusion_recall": groups["occlusion_2"]["recall_at_iou"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "fps": metrics["fps"],
    }


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["method"]) for row in rows]
    small = [float(row["small_recall"]) for row in rows]
    map50 = [float(row["map50"]) for row in rows]
    fps = [float(row["fps"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

    for ax, values, title, ylabel in [
        (axes[0], small, "Small-object recall", "Recall @ IoU 0.5"),
        (axes[1], map50, "mAP50", "AP @ IoU 0.5"),
        (axes[2], fps, "Throughput", "FPS"),
    ]:
        ax.bar(range(len(rows)), values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ablation/yolo_slim_small_objects"),
        help="Root directory containing per-experiment summary.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ablation/yolo_slim_small_objects/summary.csv"),
        help="CSV summary path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("outputs/ablation/yolo_slim_small_objects/summary.png"),
        help="Summary plot path.",
    )
    parser.add_argument(
        "--docs-assets",
        type=Path,
        default=Path("docs/assets/yolo_small_object_ablation"),
        help="Directory to mirror the report CSV and plot.",
    )
    args = parser.parse_args()

    rows = [load_row(args.root, *experiment) for experiment in EXPERIMENTS]
    write_csv(rows, args.output)
    plot_rows(rows, args.plot)

    docs_csv = args.docs_assets / "summary.csv"
    docs_plot = args.docs_assets / "summary.png"
    write_csv(rows, docs_csv)
    plot_rows(rows, docs_plot)

    print(json.dumps({"rows": rows, "csv": str(args.output), "plot": str(args.plot)}, indent=2))


if __name__ == "__main__":
    main()
