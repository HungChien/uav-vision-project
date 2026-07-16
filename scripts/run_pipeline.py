from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.backend == "pt":
        return args.weights
    if args.backend == "onnx":
        return args.onnx
    if args.backend == "engine":
        return args.engine
    raise ValueError(f"Unsupported backend: {args.backend}")


def detect_source_type(source: Path) -> str:
    if source.is_dir():
        return "frames"
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported source type: {source}")


def image_paths_from_directory(source: Path) -> list[Path]:
    paths = [path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def color_for_id(track_id: int | None, class_id: int) -> tuple[int, int, int]:
    seed = int(track_id if track_id is not None else class_id + 1000)
    rng = np.random.default_rng(seed)
    color = rng.integers(64, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def result_rows(result, frame_index: int, frame_name: str, mode: str) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    track_ids = None
    if mode == "track" and result.boxes.id is not None:
        track_ids = result.boxes.id.detach().cpu().numpy().astype(int)

    rows = []
    for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, classes)):
        x1, y1, x2, y2 = [float(value) for value in box]
        track_id = int(track_ids[index]) if track_ids is not None else None
        rows.append(
            {
                "frame_index": frame_index,
                "frame_name": frame_name,
                "track_id": track_id,
                "class_id": int(class_id),
                "class_name": result.names.get(int(class_id), str(class_id)),
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )
    return rows


def draw_rows(image: np.ndarray, rows: list[dict], mode: str) -> np.ndarray:
    canvas = image.copy()
    for row in rows:
        x1, y1, x2, y2 = [int(round(row[key])) for key in ("x1", "y1", "x2", "y2")]
        track_id = row["track_id"]
        class_id = int(row["class_id"])
        color = color_for_id(track_id, class_id)
        if mode == "track" and track_id is not None:
            label = f"id {track_id} {row['class_name']} {row['confidence']:.2f}"
        else:
            label = f"{row['class_name']} {row['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return canvas


def predict_frame(model: YOLO, image: np.ndarray, args: argparse.Namespace):
    if args.mode == "detect":
        return model.predict(image, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
    if args.mode == "track":
        return model.track(
            image,
            persist=True,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
    raise ValueError(f"Unsupported mode: {args.mode}")


def process_frames(model: YOLO, frames: list[tuple[str, np.ndarray]], args: argparse.Namespace, source_type: str) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    processed_frames = 0
    timed_seconds = 0.0
    video_writer = None
    target_frame_count = len(frames)
    if args.max_frames > 0:
        target_frame_count = min(target_frame_count, args.max_frames)
    visualization_indices = {1, max(1, target_frame_count // 2), max(1, target_frame_count)}

    for frame_index, (frame_name, image) in enumerate(frames, start=1):
        if args.max_frames > 0 and processed_frames >= args.max_frames:
            break
        start = time.perf_counter()
        result = predict_frame(model, image, args)
        elapsed = time.perf_counter() - start
        rows = result_rows(result, frame_index, frame_name, args.mode)
        rendered = draw_rows(image, rows, args.mode)

        if source_type == "image":
            cv2.imwrite(str(args.output / f"{Path(frame_name).stem}_{args.mode}.jpg"), rendered)
        elif args.save_video:
            if video_writer is None:
                height, width = rendered.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(args.output / f"{args.mode}_visualization.mp4"), fourcc, args.video_fps, (width, height))
            video_writer.write(rendered)

        if frame_index in visualization_indices:
            cv2.imwrite(str(visual_dir / f"{frame_index:06d}_{args.mode}.jpg"), rendered)

        all_rows.extend(rows)
        processed_frames += 1
        timed_seconds += elapsed

    if video_writer is not None:
        video_writer.release()

    output_csv = args.output / ("tracks.csv" if args.mode == "track" else "detections.csv")
    fieldnames = [
        "frame_index",
        "frame_name",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    track_ids = {row["track_id"] for row in all_rows if row["track_id"] is not None}
    class_counts: dict[str, int] = {}
    for row in all_rows:
        class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1

    summary = {
        "source": str(args.source),
        "source_type": source_type,
        "mode": args.mode,
        "backend": args.backend,
        "model": str(resolve_model_path(args)),
        "tracker": args.tracker if args.mode == "track" else None,
        "processed_frames": processed_frames,
        "total_rows": len(all_rows),
        "unique_track_ids": len(track_ids),
        "fps": processed_frames / timed_seconds if timed_seconds > 0 else 0.0,
        "seconds_per_frame": timed_seconds / processed_frames if processed_frames else 0.0,
        "class_counts": class_counts,
        "output_csv": str(output_csv),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary


def load_frames_from_source(source: Path, source_type: str) -> tuple[list[tuple[str, np.ndarray]], float | None]:
    if source_type == "image":
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Could not read image: {source}")
        return [(source.name, image)], None

    if source_type == "frames":
        frames = []
        for path in image_paths_from_directory(source):
            image = cv2.imread(str(path))
            if image is not None:
                frames.append((path.name, image))
        return frames, None

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or None
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        frames.append((f"frame_{index:06d}", frame))
    capture.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a unified UAV detection or tracking demo on an image, video, or frame directory.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/deployment/uav_vision_demo"))
    parser.add_argument("--mode", choices=["detect", "track"], default="detect")
    parser.add_argument("--backend", choices=["pt", "onnx", "engine"], default="pt")
    parser.add_argument("--weights", type=Path, default=Path("outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("models/exported/yolov8s_visdrone_aug_e10.onnx"))
    parser.add_argument("--engine", type=Path, default=Path("models/exported/yolov8s_slim04375_visdrone_e100_fp16.engine"))
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=20.0)
    args = parser.parse_args()

    source_type = detect_source_type(args.source)
    model_path = resolve_model_path(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = YOLO(str(model_path), task="detect")
    frames, source_fps = load_frames_from_source(args.source, source_type)
    if not frames:
        raise ValueError(f"No frames were loaded from {args.source}")
    if source_fps and args.video_fps == 20.0:
        args.video_fps = float(source_fps)

    summary = process_frames(model, frames, args, source_type)
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
