# Phase 2: Data Augmentation

## Goal

Data augmentation is used to improve generalization under UAV-specific visual conditions:

- Large viewpoint changes
- Small targets and scale variation
- Motion blur and image noise
- Lighting and weather variation
- Dense scenes with partial occlusion

## Strategy

The training pipeline uses Ultralytics YOLO augmentations. The selected augmentation profile is moderate rather than extreme, because UAV images contain many small objects and aggressive geometric transforms can make tiny boxes unusable.

Training profile:

| Augmentation | Value | Purpose |
| --- | ---: | --- |
| rotation `degrees` | 10.0 | Simulate camera roll and viewpoint changes |
| translation `translate` | 0.12 | Improve robustness to target position changes |
| scale `scale` | 0.7 | Improve handling of altitude and distance variation |
| shear `shear` | 2.0 | Add mild geometric distortion |
| perspective `perspective` | 0.0005 | Simulate aerial viewpoint changes |
| hue `hsv_h` | 0.015 | Mild color shift |
| saturation `hsv_s` | 0.7 | Lighting and weather variation |
| value `hsv_v` | 0.45 | Brightness variation |
| horizontal flip `fliplr` | 0.5 | Symmetric scene augmentation |
| vertical flip `flipud` | 0.0 | Avoid unrealistic camera geometry |
| mosaic `mosaic` | 1.0 | Improve dense and small-object training |
| mixup `mixup` | 0.05 | Mild regularization |

## Command

Short augmented experiment:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_yolo_visdrone.py --epochs 10 --imgsz 960 --batch 8 --workers 4 --name yolov8s_visdrone_aug_e10 --degrees 10 --translate 0.12 --scale 0.7 --shear 2 --perspective 0.0005 --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.45 --fliplr 0.5 --flipud 0 --mosaic 1.0 --mixup 0.05 --close-mosaic 3
```

Longer training can reuse the same profile with more epochs:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_yolo_visdrone.py --epochs 50 --imgsz 960 --batch 8 --workers 4 --name yolov8s_visdrone_aug_e50 --degrees 10 --translate 0.12 --scale 0.7 --shear 2 --perspective 0.0005 --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.45 --fliplr 0.5 --flipud 0 --mosaic 1.0 --mixup 0.05 --close-mosaic 10
```

## Experiment Result

Completed short augmented run:

```text
model: YOLOv8s
epochs: 10
image_size: 960
batch_size: 8
run_dir: outputs/training/yolov8s_visdrone_aug_e10
```

Comparison with the 10-epoch baseline:

| Run | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| baseline e10 | 0.54108 | 0.42557 | 0.42594 | 0.25508 |
| augmented e10 | 0.53961 | 0.41959 | 0.41815 | 0.22856 |

Observation:

The stronger augmentation profile did not improve the 10-epoch result. This is expected for a short run on a small-object-heavy aerial dataset: rotation, perspective, scale, and MixUp increase learning difficulty and may require longer training or a milder policy.

Next training profile:

```text
degrees: 5
translate: 0.10
scale: 0.50
shear: 0
perspective: 0
hsv_h: 0.015
hsv_s: 0.6
hsv_v: 0.4
mosaic: 1.0
mixup: 0.0
```
