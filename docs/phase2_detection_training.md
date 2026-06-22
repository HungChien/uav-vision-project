# Phase 2: Detection Training

## Model Choice

The selected baseline detector is `YOLOv8s`.

Rationale:

- It is a one-stage detector with a strong speed and accuracy tradeoff.
- It is better suited for real-time UAV imagery than heavier two-stage models such as Faster R-CNN.
- The `s` scale provides more capacity than the smallest `n` scale while remaining practical for GPU training and later deployment.
- It supports a straightforward export path to ONNX and TensorRT for future optimization work.

## Dataset Conversion

VisDrone DET annotations use:

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,category_id,truncation,occlusion
```

YOLO training expects one label file per image:

```text
class_id x_center y_center width height
```

Run conversion:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/convert_visdrone_to_yolo.py
```

The converter excludes `ignored` and `others` categories, then maps the ten standard VisDrone object classes to YOLO class IDs.

## Training

Smoke test:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_yolo_visdrone.py --epochs 1 --imgsz 640 --batch 4 --name yolov8s_visdrone_smoke
```

Baseline training:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_yolo_visdrone.py --epochs 50 --imgsz 960 --batch 8 --name yolov8s_visdrone
```

Outputs are written under:

```text
outputs/training/
```

## Baseline Run

Completed run:

```text
model: YOLOv8s
epochs: 10
image_size: 960
batch_size: 8
run_dir: outputs/training/yolov8s_visdrone_baseline_e10
```

Final validation metrics at epoch 10:

| Metric | Value |
| --- | ---: |
| Precision | 0.54108 |
| Recall | 0.42557 |
| mAP50 | 0.42594 |
| mAP50-95 | 0.25508 |

Saved weights:

```text
outputs/training/yolov8s_visdrone_baseline_e10/weights/best.pt
outputs/training/yolov8s_visdrone_baseline_e10/weights/last.pt
```
