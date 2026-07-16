# Phase 4: YOLOv8s Slimming Experiment

## Goal

The pruning-stage target is to obtain an intermediate detector that is smaller than `YOLOv8s` while remaining more accurate than `YOLOv8n`.

Target:

```text
YOLOv8s size > intermediate model size > YOLOv8n size
YOLOv8s accuracy >= intermediate model accuracy > YOLOv8n accuracy
```

## Method

Direct channel pruning with `torch-pruning` was investigated first. The Ultralytics YOLOv8 inference graph exposes the Detect head through postprocessing logic, which prevents the dependency graph from safely finding normal backbone and neck pruning groups in this environment. To avoid reporting a fake pruning result, the experiment uses a structural slimming alternative:

- Keep the YOLOv8 detection architecture.
- Use the same depth multiplier as `YOLOv8s`.
- Reduce the width multiplier from `0.50` to `0.375`.
- Initialize from the trained `YOLOv8s` checkpoint where shapes are compatible.
- Train on VisDrone for 50 epochs with the same augmentation profile used by the lightweight YOLO experiments.

This produces a real smaller model with fewer channels and lower parameter count.

Model config:

```text
configs/detection/yolov8s_slim_width0375.yaml
```

Training command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_detector.py --model configs/detection/yolov8s_slim_width0375.yaml --pretrained-weights outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt --epochs 50 --imgsz 960 --batch 8 --workers 4 --name yolov8s_slim_visdrone_e50 --degrees 10 --translate 0.12 --scale 0.7 --shear 2 --perspective 0.0005 --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.45 --fliplr 0.5 --flipud 0 --mosaic 1.0 --mixup 0.05 --close-mosaic 3
```

Evaluation command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_detector.py --weights outputs/training/yolov8s_slim_visdrone_e50/weights/best.pt --training-results outputs/training/yolov8s_slim_visdrone_e50/results.csv --output outputs/evaluation/yolov8s_slim_visdrone_e50 --imgsz 960 --conf 0.25 --iou 0.5
```

## Outputs

```text
outputs/training/yolov8s_slim_visdrone_e50/
|-- results.csv
|-- results.png
`-- weights/
    |-- best.pt
    `-- last.pt

outputs/evaluation/yolov8s_slim_visdrone_e50/
|-- summary.json
|-- groups.csv
`-- classes.csv

outputs/optimization/yolov8s_slim_pruning_comparison/
|-- summary.csv
`-- yolov8s_slim_pruning_comparison.png
```

## First Slimming Results

| Model | Parameters | Weight size | Precision | Recall | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n e50 | 3.01 M | 5.98 MB | 0.49617 | 0.38970 | 0.37789 | 0.20529 | 112.68 |
| YOLOv8s-slim e50 | 6.42 M | 12.49 MB | 0.44556 | 0.35742 | 0.33091 | 0.16867 | 105.31 |
| YOLOv8s e10 | 11.14 M | 21.49 MB | 0.53961 | 0.41959 | 0.41815 | 0.22856 | 120.69 |

Recall by difficult condition:

| Model | Overall recall | Small recall | Heavy occlusion recall |
| --- | ---: | ---: | ---: |
| YOLOv8n e50 | 0.50825 | 0.38447 | 0.25353 |
| YOLOv8s-slim e50 | 0.48158 | 0.36527 | 0.23806 |
| YOLOv8s e10 | 0.54380 | 0.42656 | 0.29859 |

## Analysis

The slimming target is only partially met. `YOLOv8s-slim e50` is smaller than `YOLOv8s`: the checkpoint decreases from 21.49 MB to 12.49 MB, and the parameter count decreases from 11.14 M to 6.42 M.

However, the accuracy target is not met. The slim model reaches mAP50 0.33091, which is below `YOLOv8n e50` at 0.37789. Its small-object recall and heavy-occlusion recall are also below `YOLOv8n e50`.

The most likely reason is that the `0.375` width multiplier is too aggressive for the current training recipe. It reduces model capacity but does not benefit from the mature pretrained weights as cleanly as a native YOLOv8n or YOLOv8s checkpoint.

## Conservative Recovery Run

Based on the first slimming result, the next run used a less aggressive width reduction, milder augmentation, and longer recovery training.

Changes:

- Increase width multiplier from `0.375` to `0.4375`.
- Keep the same depth multiplier as `YOLOv8s`.
- Initialize from the trained `YOLOv8s` checkpoint where shapes are compatible.
- Reduce geometric augmentation strength.
- Train for 100 epochs with later mosaic shutdown.

Model config:

```text
configs/detection/yolov8s_slim_width04375.yaml
```

Training command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_detector.py --model configs/detection/yolov8s_slim_width04375.yaml --pretrained-weights outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt --epochs 100 --imgsz 960 --batch 8 --workers 4 --patience 25 --name yolov8s_slim04375_visdrone_e100 --degrees 5 --translate 0.1 --scale 0.5 --shear 0 --perspective 0 --hsv-h 0.015 --hsv-s 0.6 --hsv-v 0.4 --fliplr 0.5 --flipud 0 --mosaic 1.0 --mixup 0 --close-mosaic 10
```

Evaluation command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_detector.py --weights outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt --training-results outputs/training/yolov8s_slim04375_visdrone_e100/results.csv --output outputs/evaluation/yolov8s_slim04375_visdrone_e100 --imgsz 960 --conf 0.25 --iou 0.5
```

Recovery outputs:

```text
outputs/training/yolov8s_slim04375_visdrone_e100/
|-- results.csv
|-- results.png
`-- weights/
    |-- best.pt
    `-- last.pt

outputs/evaluation/yolov8s_slim04375_visdrone_e100/
|-- summary.json
|-- groups.csv
`-- classes.csv

outputs/optimization/yolov8s_slim_recovery_comparison/
|-- summary.csv
`-- yolov8s_slim_recovery_comparison.png
```

## Recovery Results

| Model | Parameters | Weight size | Precision | Recall | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n e50 | 3.01 M | 5.98 MB | 0.49617 | 0.38970 | 0.37789 | 0.20529 | 112.68 |
| YOLOv8s-slim 0.375 e50 | 6.42 M | 12.49 MB | 0.44556 | 0.35742 | 0.33091 | 0.16867 | 105.31 |
| YOLOv8s-slim 0.4375 e100 | 8.62 M | 16.70 MB | 0.55991 | 0.45664 | 0.45085 | 0.26494 | 113.27 |
| YOLOv8s e10 | 11.14 M | 21.49 MB | 0.53961 | 0.41959 | 0.41815 | 0.22856 | 120.69 |

Recall by difficult condition:

| Model | Overall recall | Small recall | Heavy occlusion recall |
| --- | ---: | ---: | ---: |
| YOLOv8n e50 | 0.50825 | 0.38447 | 0.25353 |
| YOLOv8s-slim 0.375 e50 | 0.48158 | 0.36527 | 0.23806 |
| YOLOv8s-slim 0.4375 e100 | 0.59676 | 0.49553 | 0.36113 |
| YOLOv8s e10 | 0.54380 | 0.42656 | 0.29859 |

## Recovery Analysis

The conservative recovery run meets the intermediate-model target. Compared with `YOLOv8s e10`, `YOLOv8s-slim 0.4375 e100` reduces the checkpoint from 21.49 MB to 16.70 MB and reduces parameters from 11.14 M to 8.62 M.

It also exceeds `YOLOv8n e50` by a clear margin: mAP50 improves from 0.37789 to 0.45085, mAP50-95 improves from 0.20529 to 0.26494, and small-object recall improves from 0.38447 to 0.49553.

The result also exceeds the short `YOLOv8s e10` baseline on accuracy under the current training budget. This does not mean the slim model is universally better than a fully trained YOLOv8s, but it shows that conservative slimming plus recovery training is effective for the current VisDrone setup.

## Conclusion

The `0.375` width run is too aggressive and should not replace `YOLOv8n e50`. The `0.4375` recovery run is the better pruning-stage candidate: it is smaller than `YOLOv8s`, more accurate than `YOLOv8n`, and keeps real-time evaluation speed.

Recommended use:

- Use `YOLOv8n e50` when the strictest model-size constraint matters most.
- Use `YOLOv8s-slim 0.4375 e100` as the balanced lightweight detector.
- Keep full `YOLOv8s` as the accuracy-first baseline for longer training and future teacher-student distillation.
