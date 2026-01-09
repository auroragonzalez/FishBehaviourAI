# Data Directory Structure

This directory contains all data files for the FishBehaveAI project.

## Structure

```
data/
├── csvs/                    # Time-series behavioral data (tracked in git)
│   ├── timeVideo1.csv
│   ├── timeVideo2.csv
│   ├── timeVideo3.csv
│   └── timeVideo4.csv
│
└── datasets/                # Roboflow training datasets (ignored in git)
    ├── dataset_video1/              # Dataset from video 1
    │   ├── train/
    │   ├── valid/
    │   ├── test/
    │   └── data.yaml
    ├── dataset_Video2/              # Dataset from video 2
    ├── dataset_Video3/              # Dataset from video 3
    └── dataset_video4/              # Dataset from video 4
```

## CSV Files (csvs/)

Small time-series files containing behavioral annotations. These are tracked in git because they're small and essential for analysis.

- `timeVideo*.csv` - Temporal annotations of fish behaviors

## Datasets (datasets/)

Image frame datasets from Roboflow. **These ARE tracked in git** (~264MB total).

**Current datasets:**
- `dataset_video1/` - 3,949 files, 127.55 MB
- `dataset_Video2/` - 329 files, 15.63 MB  
- `dataset_Video3/` - 937 files, 49.07 MB
- `dataset_video4/` - 1,947 files, 72.24 MB

Each dataset contains `train/`, `valid/`, and `test/` splits.

### To download/update datasets from Roboflow:

```python
from roboflow import Roboflow
import os

os.chdir('data/datasets')
rf = Roboflow(api_key="YOUR_API_KEY")

# Video 1
project = rf.workspace("fishbehaviour").project("dataset_video1")
dataset = project.version(1).download("yolov9", location="./video1")

# Video 2
project = rf.workspace("fishbehaviour").project("prova01_train_videodos")
dataset = project.version(1).download("yolov9", location="./video2")

# Video 3
project = rf.workspace("fishbehaviour").project("dataset_video3")
dataset = project.version(1).download("yolov9", location="./video3")

# Video 4
project = rf.workspace("fishbehaviour").project("dataset_video4-laude")
dataset = project.version(1).download("yolov9", location="./video4")
```

See [setup_steps.txt](../docs/setup_steps.txt) for more details.
