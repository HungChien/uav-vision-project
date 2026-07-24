from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scene-specific VisDrone INT8 calibration manifests.")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/images"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/annotations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/visdrone_yolo/int8_calibration"),
    )
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_record(image_path: Path, annotation_dir: Path) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotation_path = annotation_dir / f"{image_path.stem}.txt"
    valid_objects = 0
    if annotation_path.exists():
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            fields = line.split(",")
            if len(fields) >= 6 and fields[4] != "0" and 1 <= int(fields[5]) <= 10:
                valid_objects += 1
    return {
        "image": str(image_path.resolve()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "object_count": valid_objects,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(record["image"] for record in records) + "\n", encoding="utf-8")


def write_yaml(path: Path, manifest: Path) -> None:
    lines = [
        f"path: {Path.cwd().resolve().as_posix()}",
        f"train: {manifest.resolve().as_posix()}",
        f"val: {manifest.resolve().as_posix()}",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in NAMES.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(records: list[dict]) -> dict:
    return {
        "images": len(records),
        "brightness_mean": float(np.mean([record["brightness"] for record in records])),
        "brightness_min": float(np.min([record["brightness"] for record in records])),
        "brightness_max": float(np.max([record["brightness"] for record in records])),
        "contrast_mean": float(np.mean([record["contrast"] for record in records])),
        "object_count_mean": float(np.mean([record["object_count"] for record in records])),
        "object_count_min": int(np.min([record["object_count"] for record in records])),
        "object_count_max": int(np.max([record["object_count"] for record in records])),
    }


def save_distribution(path: Path, all_records: list[dict], groups: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist([record["brightness"] for record in all_records], bins=40, alpha=0.35, label="All train")
    axes[1].hist([record["object_count"] for record in all_records], bins=40, alpha=0.35, label="All train")
    colors = {"bright": "#ECA82C", "dark": "#4C78A8", "dense": "#E15759"}
    for name, records in groups.items():
        axes[0].hist(
            [record["brightness"] for record in records],
            bins=30,
            histtype="step",
            linewidth=2,
            color=colors[name],
            label=name.title(),
        )
        axes[1].hist(
            [record["object_count"] for record in records],
            bins=30,
            histtype="step",
            linewidth=2,
            color=colors[name],
            label=name.title(),
        )
    axes[0].set_title("Calibration-set brightness")
    axes[0].set_xlabel("Mean grayscale value")
    axes[1].set_title("Calibration-set object density")
    axes[1].set_xlabel("Valid objects per image")
    for axis in axes:
        axis.set_ylabel("Images")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_montage(path: Path, records: list[dict], title: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(records, size=min(9, len(records)), replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for axis, record in zip(axes.flat, chosen):
        image = cv2.cvtColor(cv2.imread(record["image"]), cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        axis.set_title(f"L={record['brightness']:.1f}, objects={record['object_count']}", fontsize=9)
        axis.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    records = [image_record(path, args.annotations) for path in image_paths]
    if len(records) < args.samples:
        raise ValueError(f"Requested {args.samples} images from a dataset containing {len(records)} images.")

    groups = {
        "dark": sorted(records, key=lambda record: (record["brightness"], -record["object_count"]))[: args.samples],
        "bright": sorted(records, key=lambda record: (-record["brightness"], -record["object_count"]))[: args.samples],
        "dense": sorted(records, key=lambda record: (-record["object_count"], record["brightness"]))[: args.samples],
    }

    summary = {"source_images": len(records), "samples_per_set": args.samples, "sets": {}}
    for offset, (name, selected) in enumerate(groups.items()):
        manifest = args.output / f"{name}.txt"
        yaml_path = args.output / f"{name}.yaml"
        write_manifest(manifest, selected)
        write_yaml(yaml_path, manifest)
        save_montage(args.output / f"{name}_montage.jpg", selected, f"{name.title()} calibration samples", args.seed + offset)
        summary["sets"][name] = {
            **summarize(selected),
            "manifest": str(manifest),
            "yaml": str(yaml_path),
        }

    names = list(groups)
    summary["overlap"] = {
        f"{left}_{right}": len({record["image"] for record in groups[left]} & {record["image"] for record in groups[right]})
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ["set", "images", "brightness_mean", "brightness_min", "brightness_max", "contrast_mean", "object_count_mean", "object_count_min", "object_count_max"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow({"set": name, **{field: summary["sets"][name][field] for field in fields[1:]}})
    save_distribution(args.output / "calibration_distributions.png", records, groups)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
