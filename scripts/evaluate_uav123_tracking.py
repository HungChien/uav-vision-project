from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from uav_vision.data.uav123 import read_tracking_annotation
from uav_vision.tracking.baselines import BASELINE_TRACKERS
from uav_vision.tracking.metrics import evaluate_tracking


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _plot_curves(curves: dict, threshold_key: str, value_key: str, title: str, xlabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, metrics in curves.items():
        plt.plot(metrics[threshold_key], metrics[value_key], label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _aggregate_curves(sequence_results: pd.DataFrame, curve_column: str, threshold_column: str) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.asarray(sequence_results[threshold_column].iloc[0], dtype=float)
    curves = np.vstack(sequence_results[curve_column].map(lambda values: np.asarray(values, dtype=float)).to_list())
    return thresholds, curves.mean(axis=0)


def evaluate_annotations(annotations_dir: Path, output_dir: Path) -> dict:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_files = sorted(path for path in annotations_dir.glob("*.txt") if path.is_file())

    rows = []
    aggregate_curves = {}
    for tracker_name, tracker_fn in BASELINE_TRACKERS.items():
        tracker_rows = []
        for annotation_path in annotation_files:
            gt = read_tracking_annotation(annotation_path)
            predictions = tracker_fn(gt)
            metrics = evaluate_tracking(gt, predictions)
            row = {
                "tracker": tracker_name,
                "sequence": annotation_path.stem,
                "frame_count": metrics["frame_count"],
                "valid_frame_count": metrics["valid_frame_count"],
                "mean_iou": metrics["mean_iou"],
                "success_auc": metrics["success_auc"],
                "precision_20": metrics["precision_20"],
                "mean_center_error": metrics["mean_center_error"],
                "success_thresholds": metrics.get("success_thresholds", np.linspace(0.0, 1.0, 101)),
                "success_curve": metrics.get("success_curve", np.zeros(101)),
                "precision_thresholds": metrics.get("precision_thresholds", np.arange(0.0, 51.0, 1.0)),
                "precision_curve": metrics.get("precision_curve", np.zeros(51)),
            }
            rows.append(row)
            tracker_rows.append(row)

        tracker_results = pd.DataFrame(tracker_rows)
        success_thresholds, success_curve = _aggregate_curves(tracker_results, "success_curve", "success_thresholds")
        precision_thresholds, precision_curve = _aggregate_curves(tracker_results, "precision_curve", "precision_thresholds")
        aggregate_curves[tracker_name] = {
            "success_thresholds": success_thresholds,
            "success_curve": success_curve,
            "precision_thresholds": precision_thresholds,
            "precision_curve": precision_curve,
        }

    sequence_results = pd.DataFrame(rows)
    compact_results = sequence_results.drop(
        columns=["success_thresholds", "success_curve", "precision_thresholds", "precision_curve"]
    )
    compact_results.to_csv(output_dir / "sequence_metrics.csv", index=False)

    summary_rows = []
    for tracker_name, group in compact_results.groupby("tracker"):
        summary_rows.append(
            {
                "tracker": tracker_name,
                "sequence_count": int(group["sequence"].nunique()),
                "frame_count": int(group["frame_count"].sum()),
                "valid_frame_count": int(group["valid_frame_count"].sum()),
                "mean_iou": float(group["mean_iou"].mean()),
                "success_auc": float(group["success_auc"].mean()),
                "precision_20": float(group["precision_20"].mean()),
                "mean_center_error": float(group["mean_center_error"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("success_auc", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    _plot_curves(
        aggregate_curves,
        "success_thresholds",
        "success_curve",
        "UAV123 Annotation Baseline Success Curves",
        "IoU threshold",
        output_dir / "success_curves.png",
    )
    _plot_curves(
        aggregate_curves,
        "precision_thresholds",
        "precision_curve",
        "UAV123 Annotation Baseline Precision Curves",
        "center error threshold (pixels)",
        output_dir / "precision_curves.png",
    )

    report = {
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "annotation_file_count": len(annotation_files),
        "trackers": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate annotation-only UAV123 tracking baselines.")
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/UAV123/anno"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracking/uav123_annotation_baselines"))
    args = parser.parse_args()

    report = evaluate_annotations(args.annotations, args.output)
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
