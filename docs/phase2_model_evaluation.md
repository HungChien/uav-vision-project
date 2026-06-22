# Phase 2: Model Evaluation

## Goal

Evaluate the trained YOLOv8s VisDrone detector with actual validation results.

Metrics covered:

- mAP from the validation run
- FPS from timed inference on validation images
- Recall under complex conditions using IoU matching against the original VisDrone annotations
- Small-object and occlusion analysis

## Command

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_visdrone_detector.py --weights outputs/training/yolov8s_visdrone_baseline_e10/weights/best.pt --imgsz 960 --conf 0.25 --iou 0.5
```

Outputs:

```text
outputs/evaluation/yolov8s_visdrone_baseline_e10/
|-- summary.json
|-- groups.csv
`-- classes.csv
```

## Actual Results

Evaluation target:

```text
weights: outputs/training/yolov8s_visdrone_baseline_e10/weights/best.pt
validation_images: 548
timed_images: 528
image_size: 960
confidence_threshold: 0.25
matching_iou_threshold: 0.5
```

Validation metrics from the completed YOLO run:

| Metric | Value |
| --- | ---: |
| Precision | 0.54108 |
| Recall | 0.42557 |
| mAP50 | 0.42594 |
| mAP50-95 | 0.25508 |

Measured inference speed:

| Metric | Value |
| --- | ---: |
| FPS | 125.38 |
| Seconds per image | 0.00798 |

## Complex Scene Analysis

Recall by object size at IoU 0.5:

| Size group | GT objects | Matched | Recall |
| --- | ---: | ---: | ---: |
| small, area < 32x32 | 25825 | 11041 | 0.42753 |
| medium, 32x32 to 96x96 | 10609 | 8470 | 0.79838 |
| large, area >= 96x96 | 1030 | 920 | 0.89320 |

Observation:

The detector performs much better on medium and large objects than on small objects. This is the main bottleneck for aerial imagery in the current baseline.

Recall by occlusion level at IoU 0.5:

| Occlusion | GT objects | Matched | Recall |
| --- | ---: | ---: | ---: |
| 0 | 16371 | 10744 | 0.65628 |
| 1 | 18119 | 8814 | 0.48645 |
| 2 | 2974 | 873 | 0.29354 |

Observation:

Recall decreases sharply as occlusion increases. Heavily occluded targets are the second major failure mode after small objects.

Class-level recall at IoU 0.5:

| Class | GT objects | Matched | Recall |
| --- | ---: | ---: | ---: |
| car | 13814 | 11057 | 0.80042 |
| bus | 251 | 115 | 0.45817 |
| pedestrian | 8538 | 3901 | 0.45690 |
| motor | 4541 | 2074 | 0.45673 |
| van | 1963 | 816 | 0.41569 |
| truck | 749 | 288 | 0.38451 |
| people | 4843 | 1539 | 0.31778 |
| tricycle | 1010 | 261 | 0.25842 |
| awning-tricycle | 494 | 107 | 0.21660 |
| bicycle | 1261 | 273 | 0.21649 |

Observation:

The model is strongest on cars, which are the most frequent and visually consistent class. Bicycle, awning-tricycle, and tricycle are weak classes, likely due to smaller object size, class ambiguity, and fewer examples.

## Summary

The model is fast enough for real-time use on the tested GPU, but the current 10-epoch baseline still has limited accuracy. The most important next improvements should target small objects and occluded targets through longer training, milder augmentation, higher image size, tiling, or model scaling.
