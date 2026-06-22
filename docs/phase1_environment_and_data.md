# Phase 1: Environment And Data Understanding

## Environment

Target environment:

```powershell
D:\Anaconda3\envs\ml-gpu\python.exe
```

Observed package status:

| Tool | Version / Status |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA in PyTorch | 12.8 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| OpenCV | 4.13.0 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| Matplotlib | 3.10.9 |
| PyYAML | 6.0.3 |
| Seaborn | Missing, optional for richer plots |

Run the reproducible check:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/check_env.py
```

Optional install:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" -m pip install seaborn
```

Development tools:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" -m pip install -r requirements-dev.txt
```

## Dataset Layout

Large datasets are not tracked by Git. Put them under:

```text
data/raw/
├── VisDrone/
│   └── VisDrone2019-DET/
│       ├── VisDrone2019-DET-train/
│       │   ├── images/
│       │   └── annotations/
│       └── VisDrone2019-DET-val/
│           ├── images/
│           └── annotations/
└── UAV123/
    ├── anno/
    └── data_seq/
```

Recommended official entry points:

- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- UAV123: https://cemse.kaust.edu.sa/ivul/uav123

Because these datasets are large and mirrors change over time, keep the canonical links in documentation and download manually or use `scripts/download_datasets.py` after filling direct archive URLs.

## EDA Goals

VisDrone detection EDA:

- Image count and image size distribution
- Annotation count by category
- Bounding box width, height, area, and aspect ratio
- Small-object ratio and occlusion/truncation distribution

UAV123 tracking EDA:

- Sequence count
- Frame count inferred from annotation rows
- Bounding box size and movement statistics from tracking annotations

## Commands

Check environment:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/check_env.py
```

Run VisDrone EDA after downloading data:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/run_eda.py visdrone-det --images data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/images --annotations data/raw/VisDrone/VisDrone2019-DET/VisDrone2019-DET-train/annotations --output outputs/eda/visdrone_train
```

Run UAV123 EDA after downloading data:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/run_eda.py uav123 --annotations data/raw/UAV123/anno --output outputs/eda/uav123
```
