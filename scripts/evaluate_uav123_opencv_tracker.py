from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from uav_vision.data.uav123 import read_tracking_annotation
from uav_vision.tracking.metrics import BBOX_COLUMNS, evaluate_tracking, valid_bbox_mask


TRACKER_FACTORIES = {
    "KCF": lambda: cv2.TrackerKCF_create(),
    "CSRT": lambda: cv2.TrackerCSRT_create(),
    "MIL": lambda: cv2.TrackerMIL_create(),
}

DASIAMRPN_MODEL_FILES = {
    "model": "dasiamrpn_model.onnx",
    "kernel_cls1": "dasiamrpn_kernel_cls1.onnx",
    "kernel_r1": "dasiamrpn_kernel_r1.onnx",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def load_manifest(path: Path, max_sequences: int | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    manifest = manifest[
        manifest["frame_dir_exists"].astype(bool)
        & manifest["first_frame_exists"].astype(bool)
        & manifest["last_frame_exists"].astype(bool)
    ].copy()
    if max_sequences is not None:
        manifest = manifest.head(max_sequences)
    return manifest


def frame_path(row: pd.Series, frame_index: int) -> Path:
    return Path(row["frame_dir"]) / f"{frame_index:0{int(row['nz'])}d}.{row['ext']}"


def draw_box(image: np.ndarray, box, color: tuple[int, int, int], label: str) -> None:
    if box is None:
        return
    x, y, w, h = [int(round(float(value))) for value in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def create_dasiamrpn_tracker(model_dir: Path):
    missing_files = [file_name for file_name in DASIAMRPN_MODEL_FILES.values() if not (model_dir / file_name).is_file()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing DaSiamRPN model files in {model_dir}: {missing}")

    params = cv2.TrackerDaSiamRPN_Params()
    params.model = str(model_dir / DASIAMRPN_MODEL_FILES["model"])
    params.kernel_cls1 = str(model_dir / DASIAMRPN_MODEL_FILES["kernel_cls1"])
    params.kernel_r1 = str(model_dir / DASIAMRPN_MODEL_FILES["kernel_r1"])
    return cv2.TrackerDaSiamRPN_create(params)


def create_tracker(name: str, dasiamrpn_model_dir: Path | None = None):
    normalized_name = name.upper()
    if normalized_name == "DASIAMRPN":
        if dasiamrpn_model_dir is None:
            raise ValueError("DASIAMRPN requires --dasiamrpn-model-dir.")
        return create_dasiamrpn_tracker(dasiamrpn_model_dir)

    factory = TRACKER_FACTORIES.get(normalized_name)
    if factory is None:
        available = sorted([*TRACKER_FACTORIES.keys(), "DASIAMRPN"])
        raise ValueError(f"Unsupported tracker: {name}. Available: {available}")
    return factory()


def run_tracker_on_sequence(
    tracker_name: str,
    row: pd.Series,
    annotations_dir: Path,
    dasiamrpn_model_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict, float]:
    gt = read_tracking_annotation(annotations_dir / f"{row['sequence']}.txt")
    predictions = pd.DataFrame(np.nan, index=gt.index, columns=BBOX_COLUMNS)
    valid_gt = valid_bbox_mask(gt)
    if not valid_gt.any():
        metrics = evaluate_tracking(gt, predictions)
        return predictions, metrics, 0.0

    first_valid_pos = int(np.flatnonzero(valid_gt)[0])
    first_frame_number = int(row["start_frame"]) + first_valid_pos
    first_image = cv2.imread(str(frame_path(row, first_frame_number)))
    if first_image is None:
        metrics = evaluate_tracking(gt, predictions)
        return predictions, metrics, 0.0

    init_values = gt.loc[first_valid_pos, BBOX_COLUMNS].to_numpy(dtype=float)
    init_box = tuple(int(round(float(value))) for value in init_values)
    tracker = create_tracker(tracker_name, dasiamrpn_model_dir=dasiamrpn_model_dir)
    tracker.init(first_image, init_box)
    predictions.loc[first_valid_pos, BBOX_COLUMNS] = init_box

    update_count = 0
    start_time = time.perf_counter()
    for local_pos in range(first_valid_pos + 1, len(gt)):
        frame_number = int(row["start_frame"]) + local_pos
        image = cv2.imread(str(frame_path(row, frame_number)))
        if image is None:
            continue
        ok, box = tracker.update(image)
        update_count += 1
        if ok:
            predictions.loc[local_pos, BBOX_COLUMNS] = [float(value) for value in box]
    elapsed = time.perf_counter() - start_time

    metrics = evaluate_tracking(gt, predictions)
    fps = update_count / elapsed if elapsed > 0 else 0.0
    return predictions, metrics, fps


def _aggregate_curves(sequence_results: pd.DataFrame, curve_column: str, threshold_column: str) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.asarray(sequence_results[threshold_column].iloc[0], dtype=float)
    curves = np.vstack(sequence_results[curve_column].map(lambda values: np.asarray(values, dtype=float)).to_list())
    return thresholds, curves.mean(axis=0)


def plot_curves(curves: dict, threshold_key: str, value_key: str, title: str, xlabel: str, output_path: Path) -> None:
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


def save_visualization(row: pd.Series, annotations_dir: Path, predictions: pd.DataFrame, output_path: Path) -> None:
    gt = read_tracking_annotation(annotations_dir / f"{row['sequence']}.txt")
    valid_gt = valid_bbox_mask(gt)
    if not valid_gt.any():
        return

    positions = [0, len(gt) // 2, len(gt) - 1]
    tiles = []
    for pos in positions:
        frame_number = int(row["start_frame"]) + pos
        image = cv2.imread(str(frame_path(row, frame_number)))
        if image is None:
            continue
        draw_box(image, gt.loc[pos, BBOX_COLUMNS], (0, 255, 0), "gt")
        if predictions.loc[pos, BBOX_COLUMNS].notna().all():
            draw_box(image, predictions.loc[pos, BBOX_COLUMNS], (0, 0, 255), "pred")
        image = cv2.resize(image, (426, 240))
        tiles.append(image)

    if not tiles:
        return
    canvas = np.concatenate(tiles, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def evaluate_trackers(
    manifest_path: Path,
    annotations_dir: Path,
    output_dir: Path,
    tracker_names: list[str],
    max_sequences: int | None,
    dasiamrpn_model_dir: Path | None = None,
) -> dict:
    manifest = load_manifest(manifest_path, max_sequences=max_sequences)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    aggregate_curves = {}
    visualized = False
    for tracker_name in tracker_names:
        tracker_rows = []
        for _, sequence_row in manifest.iterrows():
            predictions, metrics, fps = run_tracker_on_sequence(
                tracker_name,
                sequence_row,
                annotations_dir,
                dasiamrpn_model_dir=dasiamrpn_model_dir,
            )
            result_row = {
                "tracker": tracker_name.upper(),
                "sequence": sequence_row["sequence"],
                "frame_count": metrics["frame_count"],
                "valid_frame_count": metrics["valid_frame_count"],
                "mean_iou": metrics["mean_iou"],
                "success_auc": metrics["success_auc"],
                "precision_20": metrics["precision_20"],
                "mean_center_error": metrics["mean_center_error"],
                "fps": fps,
                "success_thresholds": metrics.get("success_thresholds", np.linspace(0.0, 1.0, 101)),
                "success_curve": metrics.get("success_curve", np.zeros(101)),
                "precision_thresholds": metrics.get("precision_thresholds", np.arange(0.0, 51.0, 1.0)),
                "precision_curve": metrics.get("precision_curve", np.zeros(51)),
            }
            rows.append(result_row)
            tracker_rows.append(result_row)
            if not visualized:
                save_visualization(
                    sequence_row,
                    annotations_dir,
                    predictions,
                    output_dir / "visualizations" / f"{tracker_name.lower()}_{sequence_row['sequence']}.jpg",
                )
                visualized = True

        tracker_results = pd.DataFrame(tracker_rows)
        success_thresholds, success_curve = _aggregate_curves(tracker_results, "success_curve", "success_thresholds")
        precision_thresholds, precision_curve = _aggregate_curves(tracker_results, "precision_curve", "precision_thresholds")
        aggregate_curves[tracker_name.upper()] = {
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
                "fps": float(group["fps"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("success_auc", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    plot_curves(
        aggregate_curves,
        "success_thresholds",
        "success_curve",
        "UAV123 OpenCV Tracker Success Curves",
        "IoU threshold",
        output_dir / "success_curves.png",
    )
    plot_curves(
        aggregate_curves,
        "precision_thresholds",
        "precision_curve",
        "UAV123 OpenCV Tracker Precision Curves",
        "center error threshold (pixels)",
        output_dir / "precision_curves.png",
    )

    report = {
        "manifest": str(manifest_path),
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "sequence_count": int(manifest["sequence"].nunique()),
        "dasiamrpn_model_dir": str(dasiamrpn_model_dir) if dasiamrpn_model_dir is not None else None,
        "trackers": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenCV trackers on UAV123 frame sequences.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/tracking/uav123_frames/sequence_manifest.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/UAV123/anno"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracking/uav123_opencv_trackers"))
    parser.add_argument("--trackers", nargs="+", default=["KCF"], choices=sorted([*TRACKER_FACTORIES.keys(), "DASIAMRPN"]))
    parser.add_argument("--dasiamrpn-model-dir", type=Path, default=Path("models/checkpoints/dasiamrpn"))
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    report = evaluate_trackers(
        args.manifest,
        args.annotations,
        args.output,
        args.trackers,
        args.max_sequences,
        dasiamrpn_model_dir=args.dasiamrpn_model_dir,
    )
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()


