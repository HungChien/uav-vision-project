# Experimental Scripts

Auxiliary model-comparison scripts kept for reproducibility. The main deliverable pipeline uses the YOLO detector, ByteTrack integration, ONNX export, and TensorRT FP16 deployment scripts in the parent `scripts/` directory.

- `evaluate_mobilenet_ssdlite.py`
- `train_mobilenet_fpn.py`: supports COCO or aerial-domain checkpoint initialization, class-balanced focal loss, area-weighted small-object regression, small anchors, and complete AP validation.
- `ablate_mobilenet_small_objects.py`: evaluates standard, SAHI sliced, and multiscale inference with a unified small-object metric protocol.
- `summarize_small_object_ablation.py`: reads completed run summaries and generates the unified ablation CSV and comparison chart.
- `prepare_int8_calibration_sets.py`: selects and audits bright, dark, and high-density VisDrone calibration sets.
- `build_scene_int8_engines.py`: builds one TensorRT INT8 engine from a scene-specific calibration manifest.
- `evaluate_tensorrt_engine.py`: evaluates a TensorRT engine with a unified VisDrone protocol and saves metrics and plots.
- `build_mixed_precision_sensitivity.py`: restores a logical model region to FP16 to localize INT8-sensitive layers.
- `summarize_int8_calibration.py`: generates the scene-calibration and layer-sensitivity result tables and chart.
- `train_yolo_distillation.py`: fine-tunes the slim YOLOv8s detector with full YOLOv8s output-distribution supervision.
- `ablate_yolo_small_objects.py`: evaluates standard, multi-scale, and sliced YOLO inference for small-object recall on VisDrone.
- `train_yolo_small_object_ablation.py`: fine-tunes the slim YOLOv8s checkpoint for matched-control, Focal Loss, and small-object resampling ablations.
- `summarize_yolo_small_object_ablation.py`: collects YOLO small-object ablation JSON outputs into a CSV table and chart.
- `summarize_yolo_distillation.py`: compares teacher, slim baseline, and distilled results from saved evaluation JSON files.
