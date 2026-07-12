# UAV Vision Project

A research-oriented computer vision project for UAV imagery, focused on dataset analysis, object detection, target tracking, model optimization, and deployment preparation.

The repository is organized as a reproducible Python project. It separates configuration, data access, exploratory analysis, model code, scripts, documentation, and tests so experiments can be extended without mixing large local artifacts into version control.

## Project Scope

- Dataset understanding and exploratory data analysis for VisDrone and UAV123 annotations
- Object detection experiments with models such as YOLO, Faster R-CNN, or SSD
- Target tracking experiments with methods such as SORT, DeepSORT, or Siamese trackers
- Detection and tracking integration for image and video inference pipelines
- Model optimization, including lightweight backbones, pruning, quantization, ONNX export, and TensorRT preparation
- Experiment documentation, reproducible scripts, and technical reports

## Repository Layout

```text
.
|-- configs/              # Training, evaluation, and deployment configuration
|-- data/                 # Local dataset mount point; large files are ignored
|-- docs/                 # Technical notes, reports, and dataset documentation
|-- models/               # Local checkpoints and exported models; large files are ignored
|-- notebooks/            # Exploratory notebooks
|-- outputs/              # Local reports, plots, logs, and analysis outputs
|-- scripts/              # Command-line entry points
|-- src/uav_vision/       # Core Python package
`-- tests/                # Unit and lightweight integration tests
```

## Status

Dataset analysis, detection training, model evaluation, and the first tracking evaluation baseline are in place.
