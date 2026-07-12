from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(sequence_dir: Path) -> int:
    return sum(1 for path in sequence_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_uav123_config(config_path: Path) -> dict:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"seqUAV123=\{(?P<body>.*?)\};", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find seqUAV123 block in {config_path}")

    body = match.group("body")
    pattern = re.compile(
        r"struct\('name','(?P<name>[^']+)','path','(?P<path>[^']+)',"
        r"'startFrame',(?P<start>\d+),'endFrame',(?P<end>\d+),"
        r"'nz',(?P<nz>\d+),'ext','(?P<ext>[^']+)'",
        flags=re.DOTALL,
    )
    mapping = {}
    for item in pattern.finditer(body):
        raw_path = item.group("path").replace("/", "\\").rstrip("\\")
        frame_dir = raw_path.split("\\")[-1]
        mapping[item.group("name")] = {
            "frame_dir": frame_dir,
            "start_frame": int(item.group("start")),
            "end_frame": int(item.group("end")),
            "nz": int(item.group("nz")),
            "ext": item.group("ext"),
        }
    return mapping


def summarize_uav123_frames(data_root: Path, annotations_dir: Path, config_path: Path, output: Path) -> dict:
    frames_root = data_root / "data_seq" / "UAV123"
    if not frames_root.exists():
        raise FileNotFoundError(f"Frame directory not found: {frames_root}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Sequence config not found: {config_path}")

    sequence_dirs = sorted(path for path in frames_root.iterdir() if path.is_dir())
    annotation_files = sorted(path for path in annotations_dir.glob("*.txt") if path.is_file())
    annotation_names = {path.stem for path in annotation_files}
    config = parse_uav123_config(config_path)

    frame_dir_rows = []
    missing_annotations = []
    for sequence_dir in sequence_dirs:
        image_count = count_images(sequence_dir)
        has_annotation = sequence_dir.name in annotation_names
        if not has_annotation:
            missing_annotations.append(sequence_dir.name)
        frame_dir_rows.append(
            {
                "sequence": sequence_dir.name,
                "image_count": image_count,
                "has_annotation": has_annotation,
            }
        )

    missing_frame_dirs = sorted(annotation_names - {path.name for path in sequence_dirs})
    total_images = sum(row["image_count"] for row in frame_dir_rows)

    manifest_rows = []
    missing_config = []
    missing_mapped_frame_dirs = []
    frame_count_mismatches = []
    for annotation_path in annotation_files:
        name = annotation_path.stem
        item = config.get(name)
        if item is None:
            missing_config.append(name)
            continue

        frame_dir = frames_root / item["frame_dir"]
        frame_dir_exists = frame_dir.exists()
        if not frame_dir_exists:
            missing_mapped_frame_dirs.append(name)

        expected_frames = item["end_frame"] - item["start_frame"] + 1
        annotation_rows = len(annotation_path.read_text(encoding="utf-8", errors="ignore").splitlines())
        if expected_frames != annotation_rows:
            frame_count_mismatches.append(
                {
                    "sequence": name,
                    "expected_frames": expected_frames,
                    "annotation_rows": annotation_rows,
                }
            )

        first_frame = frame_dir / f"{item['start_frame']:0{item['nz']}d}.{item['ext']}"
        last_frame = frame_dir / f"{item['end_frame']:0{item['nz']}d}.{item['ext']}"
        manifest_rows.append(
            {
                "sequence": name,
                "frame_dir": str(frame_dir),
                "frame_dir_name": item["frame_dir"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "frame_count": expected_frames,
                "annotation_rows": annotation_rows,
                "first_frame_exists": first_frame.exists(),
                "last_frame_exists": last_frame.exists(),
                "frame_dir_exists": frame_dir_exists,
                "nz": item["nz"],
                "ext": item["ext"],
            }
        )

    unresolved_manifest_rows = [
        row for row in manifest_rows if not row["frame_dir_exists"] or not row["first_frame_exists"] or not row["last_frame_exists"]
    ]
    summary = {
        "data_root": str(data_root),
        "frames_root": str(frames_root),
        "annotations_dir": str(annotations_dir),
        "config_path": str(config_path),
        "frame_dir_count": len(sequence_dirs),
        "annotation_file_count": len(annotation_files),
        "total_image_count": total_images,
        "configured_sequence_count": len(config),
        "manifest_sequence_count": len(manifest_rows),
        "unresolved_manifest_count": len(unresolved_manifest_rows),
        "missing_config_count": len(missing_config),
        "missing_mapped_frame_dir_count": len(missing_mapped_frame_dirs),
        "frame_count_mismatch_count": len(frame_count_mismatches),
        "missing_annotation_count": len(missing_annotations),
        "missing_frame_dir_count": len(missing_frame_dirs),
        "missing_annotations": missing_annotations,
        "missing_frame_dirs": missing_frame_dirs,
        "missing_config": missing_config,
        "missing_mapped_frame_dirs": missing_mapped_frame_dirs,
        "frame_count_mismatches": frame_count_mismatches[:20],
        "top_frame_dirs_by_images": sorted(frame_dir_rows, key=lambda item: item["image_count"], reverse=True)[:10],
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "frame_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["sequence,image_count,has_annotation"]
    lines.extend(f"{row['sequence']},{row['image_count']},{row['has_annotation']}" for row in frame_dir_rows)
    (output / "frame_sequences.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_header = [
        "sequence",
        "frame_dir",
        "frame_dir_name",
        "start_frame",
        "end_frame",
        "frame_count",
        "annotation_rows",
        "first_frame_exists",
        "last_frame_exists",
        "frame_dir_exists",
        "nz",
        "ext",
    ]
    manifest_lines = [",".join(manifest_header)]
    manifest_lines.extend(",".join(str(row[key]) for key in manifest_header) for row in manifest_rows)
    (output / "sequence_manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize extracted UAV123 frame sequences.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/UAV123"))
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/UAV123/anno"))
    parser.add_argument("--config", type=Path, default=Path("data/raw/UAV123/metadata/configSeqs.m"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracking/uav123_frames"))
    args = parser.parse_args()

    summary = summarize_uav123_frames(args.data_root, args.annotations, args.config, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
