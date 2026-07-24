from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = (
    (
        "YOLOv8s teacher",
        Path("outputs/evaluation/yolov8s_visdrone_mildaug_e100/summary.json"),
    ),
    (
        "Slim baseline",
        Path("outputs/evaluation/yolov8s_slim04375_visdrone_e100/summary.json"),
    ),
    (
        "Distilled slim",
        Path("outputs/evaluation/yolov8s_slim04375_distilled_e20/summary.json"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize teacher, slim, and distilled VisDrone evaluations.")
    parser.add_argument("--output", type=Path, default=Path("outputs/distillation/yolov8s_slim04375_e20"))
    return parser.parse_args()


def load_row(name: str, path: Path) -> dict[str, float | str]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["training_metrics"]
    groups = data["groups"]
    return {
        "model": name,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "small_recall": groups["small_lt_32x32"]["recall_at_iou"],
        "heavy_occlusion_recall": groups["occlusion_2"]["recall_at_iou"],
        "all_recall_at_iou": groups["all"]["recall_at_iou"],
        "fps": data["fps"],
    }


def recovery(distilled: float, baseline: float, teacher: float) -> float | None:
    gap = teacher - baseline
    return (distilled - baseline) / gap if gap > 0 else None


def create_plot(rows: list[dict[str, float | str]], output: Path) -> None:
    names = [str(row["model"]) for row in rows]
    colors = ("#2563eb", "#64748b", "#16a34a")
    panels = (
        ("mAP50-95", "map50_95", (0.0, 0.35)),
        ("Small-object recall", "small_recall", (0.0, 0.6)),
        ("Heavy-occlusion recall", "heavy_occlusion_recall", (0.0, 0.45)),
        ("Inference speed", "fps", (0.0, max(float(row["fps"]) for row in rows) * 1.2)),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (title, key, limits) in zip(axes.flat, panels):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(names, values, color=colors, width=0.64)
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            label = f"{value:.3f}" if key != "fps" else f"{value:.1f}"
            axis.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom")
    figure.suptitle("YOLOv8s Slim Knowledge Distillation on VisDrone")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [load_row(name, path) for name, path in MODELS]
    fieldnames = list(rows[0])
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    teacher, baseline, distilled = rows
    recovered = {}
    for key in ("map50_95", "small_recall", "heavy_occlusion_recall", "all_recall_at_iou"):
        recovered[key] = recovery(float(distilled[key]), float(baseline[key]), float(teacher[key]))
    payload = {"models": rows, "teacher_gap_recovery": recovered}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    create_plot(rows, args.output / "distillation_comparison.png")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
