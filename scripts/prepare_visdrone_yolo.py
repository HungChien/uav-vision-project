from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VISDRONE_TO_YOLO = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    7: 6,  # tricycle
    8: 7,  # awning-tricycle
    9: 8,  # bus
    10: 9,  # motor
}

YOLO_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def convert_annotation(annotation_path: Path, image_path: Path, output_path: Path) -> tuple[int, int]:
    width, height = read_image_size(image_path)
    kept = 0
    skipped = 0
    rows: list[str] = []

    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split(",")
        if len(parts) < 8:
            skipped += 1
            continue

        left, top, box_width, box_height = map(float, parts[:4])
        category_id = int(float(parts[5]))
        class_id = VISDRONE_TO_YOLO.get(category_id)
        if class_id is None or box_width <= 0 or box_height <= 0:
            skipped += 1
            continue

        x_center = (left + box_width / 2) / width
        y_center = (top + box_height / 2) / height
        norm_width = box_width / width
        norm_height = box_height / height

        rows.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
        kept += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return kept, skipped


def convert_split(split_dir: Path) -> dict[str, int]:
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")

    image_count = 0
    kept_count = 0
    skipped_count = 0

    for annotation_path in sorted(annotations_dir.glob("*.txt")):
        image_path = images_dir / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            skipped_count += 1
            continue
        output_path = labels_dir / annotation_path.name
        kept, skipped = convert_annotation(annotation_path, image_path, output_path)
        image_count += 1
        kept_count += kept
        skipped_count += skipped

    return {
        "images": image_count,
        "labels": len(list(labels_dir.glob("*.txt"))),
        "kept_objects": kept_count,
        "skipped_objects": skipped_count,
    }


def write_dataset_yaml(root: Path, output_path: Path) -> None:
    train_images = root / "VisDrone2019-DET-train" / "images"
    val_images = root / "VisDrone2019-DET-val" / "images"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(YOLO_NAMES))
    content = f"""path: {root.resolve().as_posix()}
train: {train_images.resolve().as_posix()}
val: {val_images.resolve().as_posix()}

names:
{names}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET annotations to YOLO format.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/VisDrone/VisDrone2019-DET"),
        help="Root directory containing VisDrone2019-DET-train and VisDrone2019-DET-val.",
    )
    parser.add_argument(
        "--yaml-output",
        type=Path,
        default=Path("data/processed/visdrone_yolo/visdrone.yaml"),
        help="Output Ultralytics dataset YAML path.",
    )
    args = parser.parse_args()

    train_stats = convert_split(args.root / "VisDrone2019-DET-train")
    val_stats = convert_split(args.root / "VisDrone2019-DET-val")
    write_dataset_yaml(args.root, args.yaml_output)

    print("VisDrone YOLO conversion complete")
    print(f"train: {train_stats}")
    print(f"val: {val_stats}")
    print(f"dataset_yaml: {args.yaml_output}")


if __name__ == "__main__":
    main()

