# Phase 4: ONNX and TensorRT Deployment Benchmark

## Goal

This deployment experiment validates the `YOLOv8s-slim 0.4375 e100` detector across ONNXRuntime CUDA and TensorRT.

The benchmark covers:

- ONNX FP32 export.
- ONNX FP16 export.
- ONNXRuntime CUDA benchmark.
- TensorRT FP16 engine export and validation.
- TensorRT INT8 engine export, calibration, and validation.

## Model

```text
outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt
```

## Environment

```text
Python: D:\Anaconda3\envs\ml-gpu\python.exe
GPU: NVIDIA GeForce RTX 5080 Laptop GPU
TensorRT: 11.1.0.106
ONNXRuntime GPU: 1.22.0
Image size: 960
Validation images: 548
ONNXRuntime benchmark images: 200
```

Additional deployment packages installed in the environment:

```text
tensorrt==11.1.0.106
nvidia-modelopt==0.45.0
onnxruntime-gpu==1.22.0
onnxslim==0.1.94
```

## Export Commands

ONNX FP32:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/export_onnx.py --weights outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt --onnx models/exported/yolov8s_slim04375_visdrone_e100_fp32.onnx --output outputs/deployment/yolov8s_slim04375_onnx_fp32_validation --imgsz 960 --device 0 --limit 100 --warmup 10 --simplify
```

ONNX FP16:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/export_onnx.py --weights outputs/training/yolov8s_slim04375_visdrone_e100/weights/best.pt --onnx models/exported/yolov8s_slim04375_visdrone_e100_fp16.onnx --output outputs/deployment/yolov8s_slim04375_onnx_fp16_validation --imgsz 960 --device 0 --limit 100 --warmup 10 --half
```

TensorRT FP16:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" -c "from ultralytics import YOLO; YOLO(r'outputs\training\yolov8s_slim04375_visdrone_e100\weights\best.pt').export(format='engine', imgsz=960, half=True, device=0)"
```

TensorRT INT8:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" -c "from ultralytics import YOLO; YOLO(r'outputs\training\yolov8s_slim04375_visdrone_e100\weights\best.pt').export(format='engine', imgsz=960, int8=True, data=r'data\processed\visdrone_yolo\visdrone.yaml', device=0)"
```

## Outputs

```text
models/exported/yolov8s_slim04375_visdrone_e100_fp32.onnx
models/exported/yolov8s_slim04375_visdrone_e100_fp16.onnx
models/exported/yolov8s_slim04375_visdrone_e100_fp16.engine
models/exported/yolov8s_slim04375_visdrone_e100_int8.engine

outputs/deployment/yolov8s_slim04375_onnxruntime_fp32_benchmark/summary.json
outputs/deployment/yolov8s_slim04375_onnxruntime_fp16_benchmark/summary.json
outputs/deployment/yolov8s_slim04375_deployment_summary/summary.csv
outputs/deployment/yolov8s_slim04375_deployment_summary/deployment_benchmark.png
```

## Results

| Artifact | Backend | Precision | Size | FPS | Inference latency | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ONNX FP32 | ONNXRuntime CUDA | FP32 | 33.25 MB | 82.38 | 12.14 ms | - | - |
| ONNX FP16 | ONNXRuntime CUDA | FP16 | 16.65 MB | 106.36 | 9.40 ms | - | - |
| TensorRT FP16 engine | TensorRT | FP16 | 21.69 MB | 562.37 | 1.78 ms | 0.45794 | 0.26999 |
| TensorRT INT8 engine | TensorRT | INT8 | 65.96 MB | 418.46 | 2.39 ms | 0.42815 | 0.24414 |

INT8 accuracy loss relative to TensorRT FP16:

| Metric | FP16 engine | INT8 engine | Change |
| --- | ---: | ---: | ---: |
| Precision | 0.56059 | 0.54777 | -0.01282 |
| Recall | 0.46590 | 0.43413 | -0.03177 |
| mAP50 | 0.45794 | 0.42815 | -0.02979 |
| mAP50-95 | 0.26999 | 0.24414 | -0.02585 |
| Inference latency | 1.78 ms | 2.39 ms | +0.61 ms |

## Analysis

ONNX FP16 is clearly better than ONNX FP32 for CUDA inference in this setup. It reduces the exported model size by about half and improves ONNXRuntime CUDA throughput from 82.38 FPS to 106.36 FPS.

TensorRT FP16 is the strongest deployment result. It keeps validation accuracy close to the PyTorch detector and reduces engine inference latency to 1.78 ms per image under the Ultralytics validation path.

TensorRT INT8 is not beneficial in this run. It reduces mAP50 by 0.02979 and mAP50-95 by 0.02585 relative to the FP16 engine, while also increasing inference latency from 1.78 ms to 2.39 ms. The INT8 engine is also larger than the FP16 engine because the serialized TensorRT artifact includes quantized graph metadata, calibration scales, and engine-specific tactic data.

## Conclusion

The recommended deployment target for the current `YOLOv8s-slim 0.4375 e100` model is TensorRT FP16.

INT8 should not be selected for the current model and calibration setup because it is both less accurate and slower than TensorRT FP16 in the actual benchmark.

## Follow-up Study

Bright, dark, and high-density calibration sets plus backbone, neck, and detection-head mixed-precision variants were evaluated in a follow-up experiment. The expanded study confirmed TensorRT FP16 as the deployment target. See `phase4_int8_scene_calibration.md` for the complete results.
