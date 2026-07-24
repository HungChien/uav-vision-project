from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("outputs/quantization/scene_calibration")
SCENE_RUNS = (
    ("FP16", "fp16_reference"),
    ("INT8 dark", "int8_dark"),
    ("INT8 bright", "int8_bright"),
    ("INT8 dense", "int8_dense"),
)
SENSITIVITY_RUNS = (
    ("INT8 bright", "int8_bright"),
    ("Backbone FP16", "sensitivity_backbone_fp16"),
    ("Neck FP16", "sensitivity_neck_fp16"),
    ("Head FP16", "sensitivity_head_fp16"),
    ("FP16", "fp16_reference"),
)


def load_rows(specification: tuple[tuple[str, str], ...]) -> list[dict]:
    rows = []
    for label, directory in specification:
        summary = json.loads((ROOT / directory / "summary.json").read_text(encoding="utf-8"))
        rows.append({"method": label, **{key: summary[key] for key in ("precision", "recall", "map50", "map50_95", "inference_ms", "fps", "engine_bytes")}})
    return rows


def add_deltas(rows: list[dict], reference: dict) -> None:
    for row in rows:
        for metric in ("precision", "recall", "map50", "map50_95", "fps"):
            row[f"delta_{metric}"] = row[metric] - reference[metric]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_bars(axis, rows: list[dict], metric: str, title: str, ylim: tuple[float, float], colors: list[str]) -> None:
    labels = [row["method"] for row in rows]
    values = [row[metric] for row in rows]
    bars = axis.bar(range(len(rows)), values, color=colors[: len(rows)])
    axis.set_title(title, fontweight="bold")
    axis.set_ylim(*ylim)
    axis.set_xticks(range(len(rows)), labels, rotation=18, ha="right")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        text = f"{value:.1f}" if metric == "fps" else f"{value:.4f}"
        axis.text(bar.get_x() + bar.get_width() / 2, value, text, ha="center", va="bottom", fontsize=8)


def plot(path: Path, scene_rows: list[dict], sensitivity_rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    scene_colors = ["#4C78A8", "#596F9B", "#F2A93B", "#E15759"]
    sensitivity_colors = ["#F2A93B", "#59A14F", "#B279A2", "#E15759", "#4C78A8"]
    add_bars(axes[0, 0], scene_rows, "map50", "Calibration-set mAP50", (0.40, 0.47), scene_colors)
    add_bars(axes[0, 1], scene_rows, "map50_95", "Calibration-set mAP50-95", (0.23, 0.28), scene_colors)
    add_bars(axes[1, 0], sensitivity_rows, "map50_95", "Layer sensitivity: mAP50-95", (0.23, 0.28), sensitivity_colors)
    add_bars(axes[1, 1], sensitivity_rows, "fps", "Layer sensitivity: FPS", (300, 500), sensitivity_colors)
    fig.suptitle("TensorRT INT8 Calibration and Mixed-Precision Sensitivity", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scene_rows = load_rows(SCENE_RUNS)
    fp16 = scene_rows[0]
    add_deltas(scene_rows, fp16)
    sensitivity_rows = load_rows(SENSITIVITY_RUNS)
    add_deltas(sensitivity_rows, fp16)
    write_csv(ROOT / "calibration_results.csv", scene_rows)
    write_csv(ROOT / "sensitivity_results.csv", sensitivity_rows)
    plot(ROOT / "int8_calibration_ablation.png", scene_rows, sensitivity_rows)
    payload = {"scene_calibration": scene_rows, "layer_sensitivity": sensitivity_rows}
    (ROOT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
