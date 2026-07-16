from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


VISDRONE_CATEGORIES = {
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
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_annotation(path: Path) -> list[dict]:
    objects = []
    if not path.exists():
        return objects
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        left, top, width, height = map(float, parts[:4])
        category_id = int(float(parts[5]))
        if category_id not in VISDRONE_CATEGORIES or width <= 0 or height <= 0:
            continue
        objects.append(
            {
                "box": [left, top, left + width, top + height],
                "label": category_id,
                "area": width * height,
                "truncation": int(float(parts[6])),
                "occlusion": int(float(parts[7])),
            }
        )
    return objects


def intersect_box(box: list[float], tile: tuple[int, int, int, int], min_visible: float) -> list[float] | None:
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    visible = iw * ih
    original = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if original <= 0 or visible / original < min_visible or iw < 2 or ih < 2:
        return None
    return [ix1 - tx1, iy1 - ty1, ix2 - tx1, iy2 - ty1]


def make_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        return [(0, 0, width, height)]
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((x, y, min(width, x + tile_size), min(height, y + tile_size)))
    return tiles


class VisDroneDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        annotations_dir: Path,
        train: bool,
        limit: int = 0,
        tile_size: int = 0,
        tile_overlap: int = 160,
        min_visible: float = 0.3,
        include_empty_tiles: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.train = train
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.min_visible = min_visible
        self.include_empty_tiles = include_empty_tiles
        image_paths = sorted(images_dir.glob("*.jpg"))
        if limit > 0:
            image_paths = image_paths[:limit]
        self.samples = self._build_samples(image_paths)

    def _build_samples(self, image_paths: list[Path]) -> list[dict]:
        samples = []
        for image_path in image_paths:
            annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
            objects = read_annotation(annotation_path)
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            if self.tile_size <= 0:
                samples.append({"image": image_path, "annotation": annotation_path, "tile": None})
                continue
            for tile in make_tiles(width, height, self.tile_size, self.tile_overlap):
                tile_objects = [intersect_box(obj["box"], tile, self.min_visible) for obj in objects]
                has_objects = any(box is not None for box in tile_objects)
                if has_objects or self.include_empty_tiles:
                    samples.append({"image": image_path, "annotation": annotation_path, "tile": tile})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = read_image(sample["image"])
        objects = read_annotation(sample["annotation"])
        tile = sample["tile"]
        if tile is not None:
            x1, y1, x2, y2 = tile
            image = image[y1:y2, x1:x2]
            filtered = []
            for obj in objects:
                box = intersect_box(obj["box"], tile, self.min_visible)
                if box is not None:
                    item = dict(obj)
                    item["box"] = box
                    filtered.append(item)
            objects = filtered

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [obj["box"] for obj in objects]
        labels = [obj["label"] for obj in objects]
        height, width = image.shape[:2]

        if self.train and boxes and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes = [[width - box[2], box[1], width - box[0], box[3]] for box in boxes]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area_tensor = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp(min=0)
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp(min=0)
            if len(boxes_tensor)
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, min_size: int, max_size: int, pretrained: bool):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small_lt_32x32"
    if area < 96 * 96:
        return "medium_32x32_to_96x96"
    return "large_ge_96x96"


def update_group(groups: dict[str, dict[str, int]], name: str, matched: bool) -> None:
    groups[name]["gt"] += 1
    if matched:
        groups[name]["matched"] += 1


def match_predictions(gt_objects: list[dict], predictions: list[dict], iou_threshold: float) -> list[bool]:
    used_predictions: set[int] = set()
    matched_gt = [False] * len(gt_objects)
    for gt_index, gt in enumerate(gt_objects):
        best_index = None
        best_iou = 0.0
        for pred_index, pred in enumerate(predictions):
            if pred_index in used_predictions:
                continue
            if pred["label"] != gt["label"]:
                continue
            current_iou = iou(gt["box"], pred["box"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            used_predictions.add(best_index)
            matched_gt[gt_index] = True
    return matched_gt


def finalize_group(group: dict[str, int]) -> dict[str, float | int]:
    gt = group["gt"]
    matched = group["matched"]
    return {"gt": gt, "matched": matched, "recall_at_iou": matched / gt if gt else 0.0}


def evaluate(
    model,
    image_paths: list[Path],
    annotations_dir: Path,
    device: torch.device,
    conf: float,
    iou_threshold: float,
    warmup: int,
    limit: int,
) -> dict:
    model.eval()
    paths = image_paths[:limit] if limit > 0 else image_paths
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    class_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "matched": 0})
    total_predictions = 0
    timed_seconds = 0.0
    timed_images = 0

    with torch.inference_mode():
        for image_path in paths[:warmup]:
            image = read_image(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
            _ = model([tensor])

        for image_path in paths:
            image = read_image(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model([tensor])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            timed_seconds += time.perf_counter() - start
            timed_images += 1

            boxes = output["boxes"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            predictions = [
                {"box": [float(value) for value in box], "label": int(label), "score": float(score)}
                for box, label, score in zip(boxes, labels, scores)
                if score >= conf
            ]
            total_predictions += len(predictions)
            gt_objects = read_annotation(annotations_dir / f"{image_path.stem}.txt")
            matched = match_predictions(gt_objects, predictions, iou_threshold)
            for gt, is_matched in zip(gt_objects, matched):
                update_group(groups, "all", is_matched)
                update_group(groups, size_bucket(gt["area"]), is_matched)
                update_group(groups, f"occlusion_{gt['occlusion']}", is_matched)
                update_group(groups, f"truncation_{gt['truncation']}", is_matched)
                update_group(class_groups, VISDRONE_CATEGORIES[gt["label"]], is_matched)

    return {
        "image_count": timed_images,
        "confidence_threshold": conf,
        "matching_iou_threshold": iou_threshold,
        "fps": timed_images / timed_seconds if timed_seconds > 0 else 0.0,
        "seconds_per_image": timed_seconds / timed_images if timed_images else 0.0,
        "total_predictions": total_predictions,
        "groups": {name: finalize_group(group) for name, group in sorted(groups.items())},
        "classes": {name: finalize_group(group) for name, group in sorted(class_groups.items())},
    }


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN MobileNetV3 FPN on VisDrone.")
    parser.add_argument("--train-images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/images"))
    parser.add_argument("--train-annotations", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/annotations"))
    parser.add_argument("--val-images", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images"))
    parser.add_argument("--val-annotations", type=Path, default=Path("data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/annotations"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training/mobilenet_fpn_visdrone_e10"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=960)
    parser.add_argument("--train-tile-size", type=int, default=0)
    parser.add_argument("--train-tile-overlap", type=int, default=160)
    parser.add_argument("--min-visible", type=float, default=0.3)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output / "weights"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VisDroneDetectionDataset(
        args.train_images,
        args.train_annotations,
        train=True,
        limit=args.limit_train,
        tile_size=args.train_tile_size,
        tile_overlap=args.train_tile_overlap,
        min_visible=args.min_visible,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_paths = sorted(args.val_images.glob("*.jpg"))
    model = build_model(num_classes=11, min_size=args.min_size, max_size=args.max_size, pretrained=not args.no_pretrained)
    model.to(device)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_recall = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_recall = float(checkpoint.get("best_recall", best_recall))

    config = vars(args).copy()
    config["device"] = str(device)
    config["train_samples"] = len(train_dataset)
    save_json(args.output / "args.json", config)

    results_path = args.output / "results.csv"
    if start_epoch == 0:
        with results_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["epoch", "train_loss", "lr", "val_recall", "val_small_recall", "val_heavy_occlusion_recall", "val_fps"],
            )
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        start = time.perf_counter()
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(value for value in losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        scheduler.step()
        train_loss = epoch_loss / batch_count if batch_count else 0.0

        metrics = evaluate(
            model,
            val_paths,
            args.val_annotations,
            device,
            conf=args.conf,
            iou_threshold=args.iou,
            warmup=args.warmup,
            limit=args.limit_val,
        )
        all_recall = float(metrics["groups"].get("all", {}).get("recall_at_iou", 0.0))
        small_recall = float(metrics["groups"].get("small_lt_32x32", {}).get("recall_at_iou", 0.0))
        heavy_recall = float(metrics["groups"].get("occlusion_2", {}).get("recall_at_iou", 0.0))
        elapsed = time.perf_counter() - start

        epoch_summary = {
            "epoch": epoch + 1,
            "elapsed_seconds": elapsed,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "validation": metrics,
        }
        save_json(args.output / f"epoch_{epoch + 1:03d}_summary.json", epoch_summary)
        with results_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["epoch", "train_loss", "lr", "val_recall", "val_small_recall", "val_heavy_occlusion_recall", "val_fps"],
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "val_recall": all_recall,
                    "val_small_recall": small_recall,
                    "val_heavy_occlusion_recall": heavy_recall,
                    "val_fps": metrics["fps"],
                }
            )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_recall": max(best_recall, all_recall),
            "args": config,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")
        if all_recall > best_recall:
            best_recall = all_recall
            torch.save(checkpoint, checkpoints_dir / "best.pt")

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_recall": all_recall,
                    "val_small_recall": small_recall,
                    "val_heavy_occlusion_recall": heavy_recall,
                    "val_fps": metrics["fps"],
                    "elapsed_seconds": elapsed,
                },
                indent=2,
            )
        )

    final_summary = {
        "model": "fasterrcnn_mobilenet_v3_large_fpn",
        "num_classes": 11,
        "train_samples": len(train_dataset),
        "epochs": args.epochs,
        "best_recall": best_recall,
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "last_checkpoint": str(checkpoints_dir / "last.pt"),
    }
    save_json(args.output / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
