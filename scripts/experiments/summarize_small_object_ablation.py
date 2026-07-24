from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RUNS = (
    ("Original E10", Path("outputs/evaluation/mobilenet_fpn_aerial_e10_complete/summary.json"), "final_validation"),
    ("Matched control", Path("outputs/ablation/mobilenet_small_objects/standard_finetune_e6_eval800/summary.json"), "metrics"),
    ("Multiscale", Path("outputs/ablation/mobilenet_small_objects/multiscale_640_800_960/summary.json"), "metrics"),
    ("SAHI", Path("outputs/ablation/mobilenet_small_objects/sahi_512_overlap020/summary.json"), "metrics"),
    ("Focal Loss", Path("outputs/ablation/mobilenet_small_objects/focal_gamma2_e6_eval800/summary.json"), "metrics"),
    ("Small-object resampling", Path("outputs/ablation/mobilenet_small_objects/small_resample_strength2_e6_eval800/summary.json"), "metrics"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the MobileNet small-object ablation runs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ablation/mobilenet_small_objects"),
    )
    return parser.parse_args()


def load_rows() -> list[dict[str, float | int | str]]:
    rows = []
    for method, path, root in DEFAULT_RUNS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload[root]
        rows.append(
            {
                "method": method,
                "images": metrics["image_count"],
                "small_object_recall": metrics["groups"]["small_lt_32x32"]["recall_at_iou"],
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "fps": metrics["fps"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    labels = [str(row["method"]) for row in rows]
    colors = ["#4C78A8", "#9D9D9D", "#59A14F", "#E15759", "#B279A2", "#F28E2B"]
    panels = (
        ("small_object_recall", "Small-object recall", 0.0, 0.45),
        ("map50", "mAP50", 0.0, 0.25),
        ("map50_95", "mAP50-95", 0.0, 0.13),
        ("fps", "FPS", 0.0, 85.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, (field, title, lower, upper) in zip(axes.flat, panels):
        values = [float(row[field]) for row in rows]
        bars = axis.bar(range(len(labels)), values, color=colors)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_ylim(lower, upper)
        axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            label = f"{value:.2f}" if field == "fps" else f"{value:.3f}"
            axis.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=9)
    fig.suptitle("MobileNet FPN Small-Object Ablation on VisDrone Validation", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    csv_path = args.output / "summary.csv"
    figure_path = args.output / "small_object_ablation.png"
    write_csv(csv_path, rows)
    plot(figure_path, rows)
    print(json.dumps({"csv": str(csv_path), "figure": str(figure_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
