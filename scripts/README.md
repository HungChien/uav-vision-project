# Scripts

Command-line entry points for repeatable project workflows.

Environment and data:

- `check_environment.py`: checks Python, PyTorch, CUDA, OpenCV, and package availability.
- `prepare_visdrone_yolo.py`: converts VisDrone detection annotations to YOLO format.
- `prepare_uav123_frames.py`: prepares UAV123 frame manifests from raw sequence data.
- `analyze_datasets.py`: runs dataset-level exploratory analysis and exports plots and tables.

Detection:

- `train_detector.py`: trains YOLO detectors, including YAML-defined slim variants initialized from compatible checkpoints.
- `evaluate_detector.py`: evaluates detector checkpoints on VisDrone with object-size and occlusion breakdowns.

Tracking:

- `evaluate_tracking_annotations.py`: evaluates annotation-only UAV123 tracking baselines.
- `evaluate_single_object_trackers.py`: evaluates OpenCV KCF, CSRT, and DaSiamRPN trackers.

Deployment:

- `export_onnx.py`: exports YOLO checkpoints to ONNX and verifies PyTorch versus ONNX outputs. Use `--half --export-only` for FP16 deployment exports.
- `benchmark_onnxruntime.py`: benchmarks ONNXRuntime CPU and CUDA providers. The input tensor type is matched to the ONNX model precision.
- `run_pipeline.py`: runs the integrated detection or detection-plus-tracking pipeline on an image, video, or frame directory with PyTorch, ONNX, or TensorRT engine backends.

Experimental comparisons:

- `experiments/evaluate_mobilenet_ssdlite.py`: evaluates COCO-pretrained SSDLite MobileNetV3 on VisDrone-compatible categories.
- `experiments/train_mobilenet_fpn.py`: fine-tunes Faster R-CNN MobileNetV3 FPN on VisDrone with optional tile-based training.
