from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRACKING_COLUMNS = ["x", "y", "width", "height"]


def read_tracking_annotation(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=TRACKING_COLUMNS)

    frame = pd.read_csv(path, header=None, sep=r"[\s,]+", engine="python")
    frame = frame.iloc[:, :4]
    frame.columns = TRACKING_COLUMNS
    frame["sequence"] = path.stem
    frame["frame_index"] = range(1, len(frame) + 1)
    frame["bbox_area"] = frame["width"] * frame["height"]
    frame["center_x"] = frame["x"] + frame["width"] / 2
    frame["center_y"] = frame["y"] + frame["height"] / 2
    frame["center_dx"] = frame.groupby("sequence")["center_x"].diff().fillna(0)
    frame["center_dy"] = frame.groupby("sequence")["center_y"].diff().fillna(0)
    return frame


def plot_histogram(values: pd.Series, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    values.dropna().plot(kind="hist", bins=50)
    plt.title(title)
    plt.xlabel(values.name or "value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def analyze_uav123_annotations(annotations_dir: Path, output_dir: Path) -> dict:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")

    files = sorted(path for path in annotations_dir.glob("*.txt") if path.is_file())
    if files:
        annotations = pd.concat([read_tracking_annotation(path) for path in files], ignore_index=True)
    else:
        annotations = pd.DataFrame(columns=TRACKING_COLUMNS + ["sequence", "frame_index", "bbox_area", "center_dx", "center_dy"])

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(output_dir / "annotations.csv", index=False)

    if not annotations.empty:
        plot_histogram(annotations["bbox_area"], "UAV123 Bounding Box Area", output_dir / "bbox_area_hist.png")
        plot_histogram(annotations["center_dx"].abs() + annotations["center_dy"].abs(), "UAV123 Approximate Center Movement", output_dir / "center_movement_hist.png")

    sequence_lengths = annotations.groupby("sequence")["frame_index"].max().sort_values(ascending=False) if not annotations.empty else pd.Series(dtype="int64")

    return {
        "annotations_dir": str(annotations_dir),
        "annotation_file_count": len(files),
        "sequence_count": int(annotations["sequence"].nunique()) if not annotations.empty else 0,
        "total_annotated_frames": int(len(annotations)),
        "top_sequence_lengths": sequence_lengths.head(10).astype(int).to_dict(),
        "bbox_area": annotations["bbox_area"].describe().to_dict() if not annotations.empty else {},
    }
