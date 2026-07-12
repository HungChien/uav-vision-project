# Phase 4: ONNX Export and Runtime Verification

## Goal

The first deployment-stage task exports the trained YOLOv8s VisDrone detector to ONNX and verifies that ONNXRuntime inference remains consistent with the original PyTorch model.

## Exported Model

Source weights:

```text
outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt
```

Exported ONNX model:

```text
models/exported/yolov8s_visdrone_aug_e10.onnx
```

The exported model is a runtime artifact and is excluded from version control.

## Command

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/export_and_verify_yolo_onnx.py --limit 100 --warmup 10 --output outputs/deployment/yolov8s_visdrone_onnx_validation
```

## Output Files

```text
outputs/deployment/yolov8s_visdrone_onnx_validation/
|-- per_image_comparison.csv
|-- pytorch_vs_onnx.jpg
`-- summary.json
```

## Runtime Setup

| Item | Value |
| --- | --- |
| PyTorch device | `0` |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| ONNXRuntime provider | `CPUExecutionProvider` |
| Image size | 960 |
| Confidence threshold | 0.25 |
| Matched-box IoU threshold | 0.50 |
| Validation image count | 100 |

## Actual Results

| Metric | Value |
| --- | ---: |
| PyTorch FPS | 51.52 |
| ONNXRuntime FPS | 4.06 |
| PyTorch seconds per image | 0.01941 |
| ONNXRuntime seconds per image | 0.24608 |
| PyTorch total detections | 4336 |
| ONNX total detections | 4363 |
| Mean absolute count delta per image | 1.67 |
| Mean match rate from PyTorch | 0.94677 |
| Mean matched IoU | 0.95959 |
| Mean confidence absolute difference | 0.02158 |

## Analysis

The ONNX export is valid and produces detections that closely match the PyTorch model. The mean matched IoU is 0.95959, and the mean confidence difference is only 0.02158 across matched boxes. The ONNX model produces slightly more boxes on this 100-image subset, with an average count difference of 1.67 boxes per image.

The current ONNXRuntime test runs on CPU, so it is much slower than PyTorch running on the GPU. This result should not be treated as a final deployment-speed comparison. It confirms model export correctness and establishes the next optimization target: enabling GPU-backed ONNXRuntime, TensorRT, or OpenVINO acceleration depending on the deployment device.

## Next Step

The next deployment step should compare accelerated backends:

- ONNXRuntime CUDA or TensorRT on the local GPU.
- OpenVINO on Intel hardware if CPU deployment is required.
- A unified inference script that accepts images, videos, and frame directories, then writes detections, track IDs, and visualization outputs.

## ONNXRuntime Backend Benchmark

After installing `onnxruntime-gpu`, the environment exposes the following ONNXRuntime providers:

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

CUDA provider creation initially fell back to CPU because the CUDA and cuDNN DLLs bundled with PyTorch were not on the provider load path. The backend benchmark therefore uses ONNXRuntime DLL preloading before session creation.

Command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/benchmark_onnxruntime_backends.py --limit 100 --warmup 10 --preload-cuda-dlls --output outputs/deployment/yolov8s_onnxruntime_backend_benchmark
```

Output:

```text
outputs/deployment/yolov8s_onnxruntime_backend_benchmark/
`-- summary.json
```

Actual backend benchmark:

| Provider | Active provider | FPS | Seconds per image | Median seconds | Output shape |
| --- | --- | ---: | ---: | ---: | --- |
| CPUExecutionProvider | CPUExecutionProvider | 10.93 | 0.09148 | 0.09200 | `[1, 14, 18900]` |
| CUDAExecutionProvider | CUDAExecutionProvider + CPUExecutionProvider | 90.39 | 0.01106 | 0.01131 | `[1, 14, 18900]` |

This benchmark measures raw ONNXRuntime forward inference on preprocessed tensors. It does not include Ultralytics postprocessing, non-maximum suppression, visualization, or file I/O. Under this condition, CUDAExecutionProvider is about 8.27 times faster than CPUExecutionProvider on the tested 100-image subset.

The earlier end-to-end ONNX verification remains useful because it validates output consistency after full model postprocessing. The backend benchmark isolates inference backend speed and confirms that CUDA acceleration is available for the exported ONNX model.

## Unified Inference Demo

The deployment prototype provides one command-line entry point for image, video, and frame-directory inputs. It supports detection-only output and detection-based tracking output.

Script:

```text
scripts/run_uav_vision_demo.py
```

Supported modes:

| Mode | Description | CSV output |
| --- | --- | --- |
| `detect` | Runs YOLOv8s detection and writes bounding boxes. | `detections.csv` |
| `track` | Runs YOLOv8s detection with ByteTrack ID assignment. | `tracks.csv` |

Supported backends:

| Backend | Model |
| --- | --- |
| `pt` | `outputs/training/yolov8s_visdrone_aug_e10/weights/best.pt` |
| `onnx` | `models/exported/yolov8s_visdrone_aug_e10.onnx` |

### Image Detection

Command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/run_uav_vision_demo.py --source data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg --mode detect --backend pt --output outputs/deployment/uav_vision_demo_image_detect
```

Output:

```text
outputs/deployment/uav_vision_demo_image_detect/
|-- 0000001_02999_d_0000005_detect.jpg
|-- detections.csv
|-- summary.json
`-- visualizations/
```

Actual result:

| Metric | Value |
| --- | ---: |
| Processed frames | 1 |
| Detection rows | 62 |
| Unique track IDs | 0 |

### Frame Directory Tracking

Command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/run_uav_vision_demo.py --source data/raw/UAV123/data_seq/UAV123/group1 --mode track --backend pt --max-frames 120 --save-video --output outputs/deployment/uav_vision_demo_frames_track
```

Output:

```text
outputs/deployment/uav_vision_demo_frames_track/
|-- summary.json
|-- track_visualization.mp4
|-- tracks.csv
`-- visualizations/
```

Actual result:

| Metric | Value |
| --- | ---: |
| Processed frames | 120 |
| Track rows | 396 |
| Unique track IDs | 5 |
| FPS | 60.48 |

### Video Detection with ONNX

Command:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/run_uav_vision_demo.py --source outputs/tracking/yolov8s_bytetrack_uav123_group1_1/group1_1_bytetrack.mp4 --mode detect --backend onnx --max-frames 60 --save-video --output outputs/deployment/uav_vision_demo_video_detect_onnx
```

Output:

```text
outputs/deployment/uav_vision_demo_video_detect_onnx/
|-- detect_visualization.mp4
|-- detections.csv
|-- summary.json
`-- visualizations/
```

Actual result:

| Metric | Value |
| --- | ---: |
| Processed frames | 60 |
| Detection rows | 204 |
| Unique track IDs | 0 |
| FPS | 1.76 |

The unified demo completes the deployment-facing prototype interface: a user can pass a single image, a video, or a directory of ordered frames and receive structured CSV output plus visualized results. The ONNX video path currently uses the Ultralytics ONNX wrapper, so its end-to-end FPS includes preprocessing, postprocessing, drawing, and video writing rather than only raw ONNXRuntime forward time.
