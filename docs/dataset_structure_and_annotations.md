# Dataset Structure And Annotation Format

## VisDrone2019-DET

Source:

- Official repository: https://github.com/VisDrone/VisDrone-Dataset
- Task used in this project: Task 1, object detection in images

Downloaded archives:

```text
data/raw/VisDrone/archives/
├── VisDrone2019-DET-train.zip
└── VisDrone2019-DET-val.zip
```

Extracted structure:

```text
data/raw/VisDrone/VisDrone2019-DET/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

Local structure check:

| Split | Images | Annotation files | Objects |
| --- | ---: | ---: | ---: |
| train | 6471 | 6471 | 353550 |
| val | 548 | 548 | 40169 |

Annotation format, one `.txt` file per image:

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,category_id,truncation,occlusion
```

Example:

```text
871,572,54,92,1,4,0,0
948,592,62,92,1,4,0,0
874,705,67,110,1,4,0,1
```

Category mapping:

| ID | Name |
| ---: | --- |
| 0 | ignored |
| 1 | pedestrian |
| 2 | people |
| 3 | bicycle |
| 4 | car |
| 5 | van |
| 6 | truck |
| 7 | tricycle |
| 8 | awning-tricycle |
| 9 | bus |
| 10 | motor |
| 11 | others |

EDA outputs:

```text
outputs/eda/visdrone_train/
outputs/eda/visdrone_val/
```

Main findings:

- Train split has 353550 annotated objects; validation split has 40169.
- Object categories are highly imbalanced. `car`, `pedestrian`, `motor`, and `people` dominate.
- Small targets are a central challenge: about 60.15% of train objects and 67.73% of val objects are smaller than `32x32` pixels.
- Image sizes vary. Train images are mostly `1400x1050`, `1400x788`, `2000x1500`, `1360x765`, and `1916x1078`; validation images are mostly `1360x765`, `960x540`, and `1920x1080`.

## UAV123

Expected role:

- Single-object tracking dataset for the third project stage.
- Useful for understanding tracking sequence layout, per-frame bounding-box annotations, success rate, and precision metrics.

Source used for current setup:

- Non-official benchmark mirror: https://github.com/Johnsonirene/tracker_benchmark_v1.1
- Downloaded only `UAV123/anno/` from the repository.
- Did not download benchmark results, trackers, figures, or `UAV123_10fps`.

Local structure:

```text
data/raw/UAV123/
└── anno/
    ├── *.txt
    └── att/
        └── *.txt
```

Local structure check:

| Type | Files |
| --- | ---: |
| sequence bbox annotations | 123 |
| attribute annotations | 123 |

Sequence bbox annotation format, one `.txt` file per sequence:

```text
x,y,width,height
```

Example:

```text
703,361,57,114
703,361,57,114
705,362,57,114
```

Some frames use missing boxes:

```text
NaN,NaN,NaN,NaN
```

These rows should be treated as frames without a valid target box, often caused by out-of-view or fully invisible target states.

Attribute annotation files under `att/` use multi-column binary flags, for example:

```text
1,1,0,1,0,0,0,0,1,1,1,1
```

Status note:

- The previous KAUST/IVUL official entry point and common historical direct URLs returned `404` during this setup.
- Current setup uses annotation files only. Original image/video frames are not included in this mirror.
- Annotation EDA found 123 sequences, 112578 frame rows, and 109895 rows with valid bbox area values.
- Run annotation EDA with:

```powershell
& "D:\Anaconda3\envs\ml-gpu\python.exe" scripts/run_eda.py uav123 --annotations data/raw/UAV123/anno --output outputs/eda/uav123
```
