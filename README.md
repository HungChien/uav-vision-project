# UAV Vision Project

A reproducible computer vision project for UAV imagery, focused on dataset analysis, object detection, target tracking, model optimization, and deployment-ready inference.

The repository is organized as a reproducible Python project. It separates configuration, data access, exploratory analysis, model code, scripts, documentation, and tests so experiments can be extended without mixing large local artifacts into version control.

## Project Scope

- Dataset understanding and exploratory data analysis for VisDrone and UAV123
- YOLO-based object detection training, evaluation, and hard-case analysis
- UAV123 single-object tracking baselines and Siamese tracker evaluation
- Detection and ByteTrack integration for image, video, and frame-directory inference
- Lightweight detector comparison, structural slimming, ONNX export, and TensorRT benchmarking
- Reproducible command-line workflows, documentation, and validation tests

## Repository Layout

```text
.
|-- configs/              # Training, evaluation, and deployment configuration
|-- data/                 # Local dataset mount point; large files are ignored
|-- docs/                 # Technical notes, reports, and dataset documentation
|-- models/               # Local checkpoints and exported models; large files are ignored
|-- outputs/              # Local reports, plots, logs, and analysis outputs
|-- scripts/              # Command-line entry points
|-- src/uav_vision/       # Core Python package
`-- tests/                # Unit and lightweight integration tests
```

## Status

The project includes the full pipeline from data preparation to deployment-oriented inference:

- VisDrone and UAV123 data understanding
- YOLO detector training and evaluation
- Single-object tracking evaluation
- Detection plus ByteTrack multi-object tracking
- YOLOv8s-slim deployment with ONNXRuntime and TensorRT FP16
