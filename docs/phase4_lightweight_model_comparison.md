# Phase 4: Lightweight Model Comparison

## Goal

This experiment evaluates `YOLOv8n` as a lightweight alternative to the current `YOLOv8s` detector for UAV imagery. The comparison uses the same VisDrone conversion, image size, augmentation settings, and evaluation protocol used by the existing YOLOv8s augmented run.

## Training Command

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_detector.py --model yolov8n.pt --epochs 10 --imgsz 960 --batch 8 --workers 4 --name yolov8n_visdrone_aug_e10 --degrees 10 --translate 0.12 --scale 0.7 --shear 2 --perspective 0.0005 --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.45 --fliplr 0.5 --flipud 0 --mosaic 1.0 --mixup 0.05 --close-mosaic 3

& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_detector.py --model yolov8n.pt --epochs 50 --imgsz 960 --batch 8 --workers 4 --name yolov8n_visdrone_aug_e50 --degrees 10 --translate 0.12 --scale 0.7 --shear 2 --perspective 0.0005 --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.45 --fliplr 0.5 --flipud 0 --mosaic 1.0 --mixup 0.05 --close-mosaic 3
```

Training output:

```text
outputs/training/yolov8n_visdrone_aug_e10/
|-- results.csv
`-- weights/
    |-- best.pt
    `-- last.pt

outputs/training/yolov8n_visdrone_aug_e50/
|-- results.csv
`-- weights/
    |-- best.pt
    `-- last.pt
```

The 50-epoch run was interrupted after epoch 36 and resumed from `last.pt`:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/train_detector.py --model outputs/training/yolov8n_visdrone_aug_e50/weights/last.pt --resume
```

## Evaluation Commands

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_detector.py --weights outputs/training/yolov8n_visdrone_aug_e10/weights/best.pt --training-results outputs/training/yolov8n_visdrone_aug_e10/results.csv --output outputs/evaluation/yolov8n_visdrone_aug_e10 --imgsz 960 --conf 0.25 --iou 0.5

& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_detector.py --weights outputs/training/yolov8n_visdrone_aug_e50/weights/best.pt --training-results outputs/training/yolov8n_visdrone_aug_e50/results.csv --output outputs/evaluation/yolov8n_visdrone_aug_e50 --imgsz 960 --conf 0.25 --iou 0.5

& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_detector.py --weights outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt --training-results outputs/training/yolov8s_visdrone_aug_e10/results.csv --output outputs/evaluation/yolov8s_visdrone_aug_e10 --imgsz 960 --conf 0.25 --iou 0.5
```

Comparison output:

```text
outputs/optimization/yolov8n_vs_yolov8s/
|-- summary.csv
`-- yolov8n_vs_yolov8s.png

outputs/optimization/yolov8n_e50_comparison/
|-- summary.csv
`-- yolov8n_e50_comparison.png

outputs/evaluation/mobilenet_ssdlite_coco_visdrone/
|-- mobilenet_ssdlite_predictions.jpg
`-- summary.json

outputs/evaluation/mobilenet_ssdlite_coco_visdrone_conf005/
|-- mobilenet_ssdlite_predictions.jpg
`-- summary.json

outputs/optimization/mobilenet_ssdlite_comparison/
|-- summary.csv
`-- mobilenet_ssdlite_comparison.png

outputs/training/mobilenet_fpn_visdrone_e10_full/
|-- results.csv
|-- summary.json
`-- weights/
    |-- best.pt
    `-- last.pt

outputs/optimization/mobilenet_fpn_training/
|-- summary.csv
`-- mobilenet_fpn_training.png
```

## Actual Results

| Model | Weight size | Parameters | Precision | Recall | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n e10 | 5.97 MB | 3.01 M | 0.42048 | 0.34475 | 0.31046 | 0.16023 | 122.53 |
| YOLOv8n e50 | 5.98 MB | 3.01 M | 0.49617 | 0.38970 | 0.37789 | 0.20529 | 112.68 |
| YOLOv8s e10 | 21.49 MB | 11.14 M | 0.53961 | 0.41959 | 0.41815 | 0.22856 | 120.69 |

Recall by object condition at IoU 0.5:

| Model | Overall recall | Small recall | Medium recall | Large recall | Heavy occlusion recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n e10 | 0.44936 | 0.31884 | 0.72542 | 0.87864 | 0.20040 |
| YOLOv8n e50 | 0.50825 | 0.38447 | 0.77283 | 0.88641 | 0.25353 |
| YOLOv8s e10 | 0.54380 | 0.42656 | 0.79536 | 0.89223 | 0.29859 |

## MobileNet Baseline

A `SSDLite320 MobileNetV3-Large` detector was added as a MobileNet-family lightweight candidate. The tested model uses the official COCO-pretrained torchvision weights and is evaluated on the VisDrone validation images without VisDrone fine-tuning.

Because the COCO detector does not contain all VisDrone classes, this evaluation only counts compatible categories:

| VisDrone category | COCO-compatible class |
| --- | --- |
| `pedestrian`, `people` | `person` |
| `bicycle` | `bicycle` |
| `car`, `van` | `car` |
| `truck` | `truck` |
| `bus` | `bus` |
| `motor` | `motorcycle` |

Command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/experiments/evaluate_mobilenet_ssdlite.py --output outputs/evaluation/mobilenet_ssdlite_coco_visdrone --conf 0.25 --iou 0.5

& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/experiments/evaluate_mobilenet_ssdlite.py --output outputs/evaluation/mobilenet_ssdlite_coco_visdrone_conf005 --conf 0.05 --iou 0.5
```

Actual MobileNet results:

| Model | Scope | Parameters | Weight size | Confidence | FPS | Recall at IoU 0.5 | Small recall | Heavy occlusion recall | Predictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3-SSDLite | COCO pretrained compatible classes | 3.44 M | 13.42 MB | 0.25 | 26.68 | 0.00885 | 0.00023 | 0.00033 | 886 |
| MobileNetV3-SSDLite | COCO pretrained compatible classes | 3.44 M | 13.42 MB | 0.05 | 25.03 | 0.10373 | 0.01758 | 0.02904 | 133320 |

The COCO-pretrained MobileNet detector does not transfer well to VisDrone without fine-tuning. At the normal 0.25 confidence threshold, compatible-class recall is only 0.00885. Lowering the threshold to 0.05 raises recall to 0.10373, but it also produces 133320 predictions on 548 validation images, which indicates severe false-positive pressure.

## MobileNet FPN Fine-Tuning

The next MobileNet experiment fine-tunes `FasterRCNN MobileNetV3-Large FPN` on the full VisDrone training set. This model keeps a MobileNet backbone but adds an FPN detector head, which is more suitable for multi-scale UAV targets than the fixed-size SSDLite320 detector.

Training settings:

| Item | Value |
| --- | --- |
| Model | `fasterrcnn_mobilenet_v3_large_fpn` |
| Initialization | COCO pretrained |
| Classes | 10 VisDrone object classes plus background |
| Train images | 6471 |
| Validation images | 548 |
| Min input size | 640 |
| Max input size | 960 |
| Batch size | 2 |
| Full-image epochs | 10 |
| Tile fine-tune | 1 epoch, `640` tile, `160` overlap |
| Tile training samples | 40055 |

Commands:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/experiments/train_mobilenet_fpn.py --epochs 10 --batch-size 2 --workers 2 --min-size 640 --max-size 960 --output outputs/training/mobilenet_fpn_visdrone_e10_full

& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/experiments/train_mobilenet_fpn.py --epochs 11 --batch-size 2 --workers 2 --min-size 640 --max-size 960 --train-tile-size 640 --train-tile-overlap 160 --resume outputs/training/mobilenet_fpn_visdrone_e10_full/weights/last.pt --output outputs/training/mobilenet_fpn_visdrone_e10_full
```

Actual results:

| Model | Training stage | Weight size | Validation recall | Small recall | Heavy occlusion recall | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3-FPN | full image e10 | 144.84 MB | 0.28979 | 0.13189 | 0.12994 | 50.81 |
| MobileNetV3-FPN | tile fine-tune e11 | 144.84 MB | 0.28071 | 0.12203 | 0.12054 | 56.67 |
| YOLOv8n e50 | full image | 5.98 MB | 0.50825 | 0.38447 | 0.25353 | 112.68 |
| YOLOv8s e10 | full image | 21.49 MB | 0.54380 | 0.42656 | 0.29859 | 120.69 |

Fine-tuning improves MobileNet substantially over the off-the-shelf SSDLite baseline. The COCO-pretrained SSDLite recall was 0.00885 at confidence 0.25, while the VisDrone-trained MobileNetV3-FPN reaches 0.28979 validation recall after 10 full-image epochs.

The tile fine-tune pass reduces training loss from 0.94356 to 0.74016, but full-image validation recall drops from 0.28979 to 0.28071. This indicates that direct tile-only fine-tuning changes the training distribution and does not immediately improve full-image validation performance. The best checkpoint remains the epoch-10 full-image checkpoint.

## Analysis

`YOLOv8n` is much smaller than `YOLOv8s`: the best checkpoint is about 5.98 MB instead of 21.49 MB, and the parameter count drops from 11.14 M to 3.01 M. This is a clear model-size reduction for deployment storage and memory pressure.

Longer training closes a meaningful part of the gap. Increasing `YOLOv8n` from 10 to 50 epochs raises mAP50 from 0.31046 to 0.37789 and mAP50-95 from 0.16023 to 0.20529. Small-object recall improves from 0.31884 to 0.38447, and heavy-occlusion recall improves from 0.20040 to 0.25353.

The 50-epoch `YOLOv8n` run still trails `YOLOv8s e10` on accuracy: mAP50 is 0.37789 versus 0.41815, and mAP50-95 is 0.20529 versus 0.22856. The remaining weakness is concentrated in small and heavily occluded targets, which are common in UAV imagery.

The measured FPS remains high across all detector variants. The evaluation script reports 112.68 FPS for `YOLOv8n e50`, 122.53 FPS for `YOLOv8n e10`, and 120.69 FPS for `YOLOv8s e10`. These numbers include preprocessing and postprocessing, so raw ONNXRuntime CUDA or TensorRT latency should be used for final deployment speed decisions.

The added MobileNetV3-SSDLite baseline has similar parameter count to `YOLOv8n`, but its current COCO-pretrained detector is not a practical replacement for the VisDrone-trained YOLO models. The main reason is domain mismatch: VisDrone objects are much smaller, denser, and more heavily occluded than common COCO objects.

The MobileNetV3-FPN fine-tuning run confirms that VisDrone training is necessary. It improves recall by a large margin over SSDLite, but it still trails `YOLOv8n e50` and `YOLOv8s e10` while producing a much larger checkpoint. The likely causes are the two-stage detector overhead, small-object density, and the limited 10-epoch training budget. More careful tile mixing, longer fine-tuning, lower learning rate after tile conversion, or tile-based validation may be needed before the MobileNet-FPN route becomes competitive.

## Conclusion

`YOLOv8n e50` is a stronger lightweight candidate than the 10-epoch run and gives a practical model-size reduction with usable accuracy. For this UAV detection task, `YOLOv8s` remains the better accuracy-first baseline, while `YOLOv8n e50` is suitable when storage, memory, or edge deployment constraints are stricter. A practical deployment path is:

- Keep `YOLOv8s` for accuracy-focused GPU deployment.
- Use `YOLOv8n e50` for stricter storage or memory constraints.
- Do not use off-the-shelf COCO MobileNetV3-SSDLite directly for VisDrone detection.
- Treat MobileNetV3-FPN as a documented experiment rather than the main deployment model unless longer training and better tile mixing improve validation recall.
- Use `YOLOv8s-slim 0.4375 e100` as the balanced model after pruning-stage recovery, because it is smaller than `YOLOv8s` and more accurate than `YOLOv8n e50` in the current VisDrone evaluation.
- Continue with FP16 ONNX or TensorRT acceleration for the best deployment candidate before pursuing more complex pruning.
