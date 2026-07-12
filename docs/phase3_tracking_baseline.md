# Phase 3: Tracking Baseline

## Goal

The third stage starts with a reproducible tracking evaluation protocol for UAV123-style single-object annotations. The first baseline validates metrics directly from annotations. The next baselines run OpenCV trackers on the extracted UAV123 image frames.

## Frame Data Preparation

UAV123 frame data is extracted under:

```text
data/raw/UAV123/
|-- anno/
|-- archives/UAV123.tar.gz
|-- data_seq/UAV123/
`-- metadata/configSeqs.m
```

The extracted data contains 91 raw frame directories and 113476 image frames. The benchmark sequence configuration maps these frame directories to 123 evaluation sequences with start and end frames. The generated manifest confirms that all 123 annotation sequences have valid frame mappings and no frame-count mismatch.

Frame preparation command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/prepare_uav123_frames.py --data-root data/raw/UAV123 --annotations data/raw/UAV123/anno --config data/raw/UAV123/metadata/configSeqs.m --output outputs/tracking/uav123_frames
```

Frame preparation outputs:

```text
outputs/tracking/uav123_frames/
|-- frame_summary.json
|-- frame_sequences.csv
`-- sequence_manifest.csv
```

## Annotation-Only Baselines

The annotation-only baselines are intentionally simple:

- `static_first`: repeats the first valid target box for the full sequence.
- `previous_bbox`: predicts the previous valid target box.
- `constant_velocity`: predicts the next box from the last two valid boxes.

These baselines use ground-truth annotation history, so they are not deployable trackers. They provide a sanity check for success and precision metrics.

Command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_uav123_tracking.py --annotations data/raw/UAV123/anno --output outputs/tracking/uav123_annotation_baselines
```

Annotation-only results:

| Tracker | Success AUC | Precision@20px | Mean IoU | Mean center error |
| --- | ---: | ---: | ---: | ---: |
| constant_velocity | 0.91326 | 0.99504 | 0.91323 | 1.47 |
| previous_bbox | 0.84814 | 0.98876 | 0.84803 | 2.97 |
| static_first | 0.05428 | 0.05447 | 0.05014 | 217.07 |

The constant-velocity baseline is strongest on the annotation-only benchmark because most consecutive UAV123 boxes move smoothly. The static baseline is much weaker because it cannot follow object displacement over long sequences.

## OpenCV Tracker Baselines

The image-based tracker baselines use OpenCV KCF, CSRT, and DaSiamRPN. Each sequence is initialized from the first valid ground-truth box, then the tracker predicts boxes on the following frames.

DaSiamRPN is the Siamese tracker used for the learning-based tracking baseline. The implementation uses the OpenCV contrib tracker API with the official ONNX model files:

```text
models/checkpoints/dasiamrpn/
|-- dasiamrpn_model.onnx
|-- dasiamrpn_kernel_cls1.onnx
`-- dasiamrpn_kernel_r1.onnx
```

The model files are local runtime artifacts and are excluded from version control.

Commands:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_uav123_opencv_tracker.py --trackers KCF --output outputs/tracking/uav123_opencv_kcf
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_uav123_opencv_tracker.py --trackers CSRT --output outputs/tracking/uav123_opencv_csrt
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/evaluate_uav123_opencv_tracker.py --trackers DASIAMRPN --output outputs/tracking/uav123_opencv_dasiamrpn
```

Output files:

```text
outputs/tracking/uav123_opencv_kcf/
|-- summary.json
|-- summary.csv
|-- sequence_metrics.csv
|-- success_curves.png
|-- precision_curves.png
`-- visualizations/

outputs/tracking/uav123_opencv_csrt/
|-- summary.json
|-- summary.csv
|-- sequence_metrics.csv
|-- success_curves.png
|-- precision_curves.png
`-- visualizations/

outputs/tracking/uav123_opencv_dasiamrpn/
|-- summary.json
|-- summary.csv
|-- sequence_metrics.csv
|-- success_curves.png
|-- precision_curves.png
`-- visualizations/

outputs/tracking/uav123_opencv_tracker_comparison/
|-- summary.csv
`-- tracker_comparison.png
```

Full UAV123 result:

| Tracker | Sequences | Frames | Success AUC | Precision@20px | Mean IoU | Mean center error | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DaSiamRPN | 123 | 112578 | 0.55639 | 0.71779 | 0.55524 | 80.73 | 39.92 |
| KCF | 123 | 112578 | 0.20722 | 0.27855 | 0.20397 | 17.21 | 191.40 |
| CSRT | 123 | 112578 | 0.41732 | 0.62375 | 0.41600 | 71.10 | 54.01 |

DaSiamRPN is the strongest tracker in the full UAV123 run. Compared with CSRT, it improves Success AUC from 0.41732 to 0.55639 and Precision@20px from 0.62375 to 0.71779. The tradeoff is speed: DaSiamRPN runs at 39.92 FPS, slower than CSRT at 54.01 FPS and much slower than KCF at 191.40 FPS.

CSRT is more accurate than KCF on overlap and center precision, but it is about 3.5 times slower. KCF remains the fastest real-time baseline, while CSRT provides a stronger traditional tracker baseline for quality-focused comparison. DaSiamRPN becomes the main learning-based Siamese baseline for later improvement work.

## Metrics

The evaluator reports:

- `success_auc`: area under the IoU success curve from threshold 0.0 to 1.0.
- `precision_20`: center-location precision at 20 pixels.
- `mean_iou`: average overlap on valid annotated frames.
- `mean_center_error`: average center distance in pixels.
- `fps`: average tracker update speed measured during frame-by-frame tracking.

## Detection-Based Multi-Object Tracking

The final tracking-stage experiment connects the trained YOLOv8s VisDrone detector with ByteTrack. The local data does not include VisDrone-MOT annotations, so this run is evaluated as an engineering integration experiment with runtime, track continuity, class counts, and visual inspection instead of MOT metrics such as MOTA or IDF1.

Command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/run_uav_vision_demo.py --source data/raw/UAV123/data_seq/UAV123/group1 --mode track --backend pt --max-frames 600 --save-video --output outputs/tracking/yolov8s_bytetrack_uav123_group1_1
```

Output files:

```text
outputs/tracking/yolov8s_bytetrack_uav123_group1_1/
|-- summary.json
|-- tracks.csv
|-- track_visualization.mp4
`-- visualizations/
```

Actual run:

| Item | Value |
| --- | ---: |
| Detector weights | `outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt` |
| Tracker | `bytetrack.yaml` |
| Sequence | `group1_1` |
| Processed frames | 600 |
| Track rows | 1866 |
| Unique track IDs | 7 |
| Average tracks per frame | 3.11 |
| FPS | 56.53 |

Class counts:

| Class | Count |
| --- | ---: |
| pedestrian | 1866 |

The result confirms that the trained detector can be connected to ByteTrack for multi-object ID assignment on UAV video frames. The 600-frame run produces 1866 tracked pedestrian rows and 7 unique track IDs. The end-to-end speed is 56.53 FPS with visualization and video writing enabled, which is practical for real-time experimentation.
