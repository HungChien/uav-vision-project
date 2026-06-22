from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


VISDRONE_DET_COLUMNS = [
    "bbox_left",
    "bbox_top",
    "bbox_width",
    "bbox_height",
    "score",
    "category_id",
    "truncation",
    "occlusion",
]

VISDRONE_CATEGORIES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def read_image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def read_annotation_file(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=VISDRONE_DET_COLUMNS)
    frame = pd.read_csv(path, header=None, names=VISDRONE_DET_COLUMNS)
    frame["annotation_file"] = path.name
    frame["image_id"] = path.stem
    frame["category_name"] = frame["category_id"].map(VISDRONE_CATEGORIES).fillna("unknown")
    frame["bbox_area"] = frame["bbox_width"] * frame["bbox_height"]
    frame["bbox_aspect_ratio"] = frame["bbox_width"] / frame["bbox_height"].clip(lower=1)
    return frame


def load_annotations(annotations_dir: Path) -> pd.DataFrame:
    files = sorted(annotations_dir.glob("*.txt"))
    if not files:
        return pd.DataFrame(columns=VISDRONE_DET_COLUMNS)
    return pd.concat([read_annotation_file(path) for path in files], ignore_index=True)


def plot_series(series: pd.Series, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_histogram(values: pd.Series, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    values.dropna().plot(kind="hist", bins=50)
    plt.title(title)
    plt.xlabel(values.name or "value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def analyze_visdrone_det(images_dir: Path, annotations_dir: Path, output_dir: Path) -> dict:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")

    images = list_images(images_dir)
    image_sizes = []
    for image_path in images:
        size = read_image_size(image_path)
        if size is not None:
            width, height = size
            image_sizes.append({"image": image_path.name, "width": width, "height": height})

    image_frame = pd.DataFrame(image_sizes)
    annotations = load_annotations(annotations_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_frame.to_csv(output_dir / "image_sizes.csv", index=False)
    annotations.to_csv(output_dir / "annotations.csv", index=False)

    if not annotations.empty:
        plot_series(annotations["category_name"].value_counts(), "VisDrone Category Distribution", output_dir / "category_distribution.png")
        plot_histogram(annotations["bbox_area"], "VisDrone Bounding Box Area", output_dir / "bbox_area_hist.png")
        plot_series(annotations["occlusion"].value_counts().sort_index(), "VisDrone Occlusion Distribution", output_dir / "occlusion_distribution.png")

    size_counter = Counter((row["width"], row["height"]) for row in image_sizes)
    small_objects = int((annotations["bbox_area"] < 32 * 32).sum()) if not annotations.empty else 0

    return {
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir),
        "image_count": len(images),
        "readable_image_count": len(image_sizes),
        "annotation_file_count": len(list(annotations_dir.glob("*.txt"))),
        "object_count": int(len(annotations)),
        "category_distribution": annotations["category_name"].value_counts().to_dict() if not annotations.empty else {},
        "top_image_sizes": {f"{width}x{height}": count for (width, height), count in size_counter.most_common(10)},
        "bbox_area": annotations["bbox_area"].describe().to_dict() if not annotations.empty else {},
        "small_object_count_lt_32x32": small_objects,
        "small_object_ratio_lt_32x32": small_objects / len(annotations) if len(annotations) else 0.0,
    }

