# Fish Behaviour Detection Using YOLOv9

Automated detection and analysis of fish reproductive behaviour in underwater videos using deep learning and behavioral modeling.

## Overview

This project applies YOLOv9 object detection to analyze fish spawning behaviors from underwater video recordings. It includes complete pipelines for:
- Training custom YOLO models on fish detection
- Analyzing temporal patterns of fish presence
- Modeling behavioral transitions using Markov chains
- Computing time budgets and normalized transition matrices

## Paper

This repository contains the code and experiments for the paper "AI-Driven Analysis of Fish Reproductive Behaviour".

## Project Structure

```
.
├── assets/                       # Generated outputs
│   ├── images/                   # Saved images from notebooks
│   ├── plots/                    # Generated plots and figures
│   └── videos/                   # Processed video outputs
│
├── configs/                      # Configuration files
│   ├── data/                     # Dataset configs
│   ├── models/                   # Model architecture configs
│   └── training/                 # Training parameter configs
│
├── data/                         # Datasets and annotations
│   ├── timeVideo1.csv            # Behavioral time-series data (Video 1)
│   ├── timeVideo2.csv            # Behavioral time-series data (Video 2)
│   ├── timeVideo3.csv            # Behavioral time-series data (Video 3)
│   ├── timeVideo4.csv            # Behavioral time-series data (Video 4)
│   ├── csvs/                     # Behavioral time-series data (copies)
│   │   ├── timeVideo1.csv
│   │   ├── timeVideo2.csv
│   │   ├── timeVideo3.csv
│   │   └── timeVideo4.csv
│   ├── datasets/                 # YOLO training datasets (from Roboflow)
│   │   ├── dataset_video1/       # Video 1 dataset (train/valid/test splits)
│   │   ├── dataset_Video2/       # Video 2 dataset (train/valid/test splits)
│   │   ├── dataset_Video3/       # Video 3 dataset (train/valid/test splits)
│   │   └── dataset_video4/       # Video 4 dataset (train/valid/test splits)
│   └── README.md                 # Data directory documentation
│
├── docs/                         # Documentation
│   ├── USAGE.md                  # Usage examples
│   └── setup_steps.txt           # Setup instructions
│
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── analysis/                 # Behavioral analysis
│   │   ├── NormalizedTransitionsVideo1.ipynb
│   │   ├── NormalizedTransitionsVideo2.ipynb
│   │   ├── NormalizedTransitionsVideo3.ipynb
│   │   ├── NormalizedTransitionsVideo4.ipynb
│   │   ├── TimeBudgetVideo1.ipynb
│   │   ├── TimeBudgetVideo2.ipynb
│   │   ├── TimeBudgetVideo3.ipynb
│   │   └── TimeBudgetVideo4.ipynb
│   ├── markov_chain/             # Markov chain order estimation
│   │   ├── Video1MarkovChainOrderEstimation.ipynb
│   │   ├── Video2MarkovChainOrderEstimation.ipynb
│   │   ├── Video3MarkovChainOrderEstimation.ipynb
│   │   └── Video4MarkovChainOrderEstimation.ipynb
│   └── training/                 # Model training & preparation
│       ├── 00_data_preparation.ipynb
│       ├── 01_experiments.ipynb
│       └── 02_models_training.ipynb
│
├── results/                      # Experiment results
│   ├── archived_runs/            # Archived experiment runs
│   ├── final_experiments/        # Final experiments for publication
│   ├── training_runs/            # YOLO training outputs
│   └── validation_runs/          # Validation results
│
├── runs/                         # YOLO detection outputs
│   └── detect/                   # Detection results from inference
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_utils.py             # Data processing utilities (Roboflow integration)
│   └── visualization.py          # Visualization utilities (detection plots)
│
├── cnr/                          # Python virtual environment
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # License file
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── yolov9c.pt                    # Pre-trained YOLOv9 weights (download separately)
```

**Note**: The `yolov9c.pt` file is not included in the repository due to its size. See installation instructions below to download it.

## Features

- **YOLOv9-based Fish Detection**: Custom-trained models for detecting fish in underwater videos
- **Temporal Analysis**: Track fish presence and behavior patterns over time
- **Behavioral Modeling**: Markov chain analysis for behavioral state transitions
- **Time Budget Analysis**: Quantify time spent in different behavioral states
- **Transition Matrix Analysis**: Normalized transition probabilities between behaviors
- **Visualization Tools**: Generate plots and statistics for research publications

## Getting Started

### Prerequisites

- Python 3.8+ (developed with Python 3.12)
- CUDA-capable GPU (recommended for training)
- 4+ GB GPU memory for inference
- Linux/macOS/Windows

### Installation

1. Clone the repository:
```bash
git clone git@github.com:auroragonzalez/FishBehaviourAI.git
cd FishBehaviourAI
```

2. Create and activate virtual environment:
```bash
python3 -m venv cnr
source cnr/bin/activate  # On Linux/Mac
# or
cnr\Scripts\activate.bat  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained YOLOv9 weights:
```bash
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt -O yolov9c.pt
```

Alternatively, download manually from [YOLOv9 releases](https://github.com/WongKinYiu/yolov9/releases) and place the file in the project root directory as `yolov9c.pt`.

### Configuration

1. **Datasets**:
   - **All datasets are already included** in this repository under [data/datasets/](data/datasets/) (~264MB total)
   - You can use them directly without downloading from Roboflow
   - The behavioral annotation CSVs are also included in [data/csvs/](data/csvs/)

2. **Roboflow Setup** (optional - only if you want to update/re-download datasets):
   - See [docs/setup_steps.txt](docs/setup_steps.txt) for Roboflow API configuration
   - Configure your API key to download datasets from Roboflow

## Usage

### 1. Data Preparation

**The datasets are already included in the repository** under `data/datasets/`, so you can skip this step and proceed directly to training!

If you need to update or re-download datasets from Roboflow, you can use the data preparation notebook:

```bash
jupyter notebook notebooks/training/00_data_preparation.ipynb
```

Or use the Python API:

```python
from src.data_utils import download_roboflow_dataset

dataset = download_roboflow_dataset(
    api_key="YOUR_API_KEY",
    workspace="your-workspace",
    project_name="fish-detection",
    version=1,
    format="yolov9"
)
```

### 2. Training a Model

Use the training notebooks to train custom YOLOv9 models:

1. [notebooks/training/01_experiments.ipynb](notebooks/training/01_experiments.ipynb) - Initial experiments
2. [notebooks/training/02_models_training.ipynb](notebooks/training/02_models_training.ipynb) - Full training pipeline

Training results are saved to `results/training_runs/` with:
- Model weights (`best.pt`, `last.pt`)
- Training metrics (`results.csv`)
- Configuration (`args.yaml`)

They are not included in the repository given their weight but it is ready to be run


### 3. Behavioral Analysis

The project includes three types of behavioral analysis notebooks:

#### Time Budget Analysis
Analyzes the proportion of time spent in different behavioral states:
- [notebooks/analysis/TimeBudgetVideo1.ipynb](notebooks/analysis/TimeBudgetVideo1.ipynb)
- [notebooks/analysis/TimeBudgetVideo2.ipynb](notebooks/analysis/TimeBudgetVideo2.ipynb)
- [notebooks/analysis/TimeBudgetVideo3.ipynb](notebooks/analysis/TimeBudgetVideo3.ipynb)
- [notebooks/analysis/TimeBudgetVideo4.ipynb](notebooks/analysis/TimeBudgetVideo4.ipynb)

#### Transition Analysis
Computes normalized transition matrices between behavioral states:
- [notebooks/analysis/NormalizedTransitionsVideo1.ipynb](notebooks/analysis/NormalizedTransitionsVideo1.ipynb)
- [notebooks/analysis/NormalizedTransitionsVideo2.ipynb](notebooks/analysis/NormalizedTransitionsVideo2.ipynb)
- [notebooks/analysis/NormalizedTransitionsVideo3.ipynb](notebooks/analysis/NormalizedTransitionsVideo3.ipynb)
- [notebooks/analysis/NormalizedTransitionsVideo4.ipynb](notebooks/analysis/NormalizedTransitionsVideo4.ipynb)

#### Markov Chain Order Estimation
Determines the optimal order for Markov chain models of behavior:
- [notebooks/markov_chain/Video1MarkovChainOrderEstimation.ipynb](notebooks/markov_chain/Video1MarkovChainOrderEstimation.ipynb)
- [notebooks/markov_chain/Video2MarkovChainOrderEstimation.ipynb](notebooks/markov_chain/Video2MarkovChainOrderEstimation.ipynb)
- [notebooks/markov_chain/Video3MarkovChainOrderEstimation.ipynb](notebooks/markov_chain/Video3MarkovChainOrderEstimation.ipynb)
- [notebooks/markov_chain/Video4MarkovChainOrderEstimation.ipynb](notebooks/markov_chain/Video4MarkovChainOrderEstimation.ipynb)

For more detailed usage examples, see [docs/USAGE.md](docs/USAGE.md).

## Project Workflow

The typical workflow for using this project:

1. **Setup**: Install dependencies and configure Roboflow API
2. **Data Download**: Download datasets from Roboflow using `00_data_preparation.ipynb`
3. **Training**: Train YOLOv9 models on fish detection using `01_experiments.ipynb` and `02_models_training.ipynb`
4. **Inference**: Run trained models on videos to generate detections (videos not included in the project, just preprocessed frames, contact us if you want to have access to the original unprocessed videos or use your own video with our model)
5. **Behavioral Analysis**: 
   - Analyze time budgets (proportion of time in each state)
   - Compute transition matrices (probability of state changes)
   - Estimate Markov chain order (memory depth of behavior)

## Results and Outputs

### Training Results

Training outputs are stored in `results/` with the following structure:

- **training_runs/**: Individual training experiments with:
  - `weights/best.pt` - Best model checkpoint
  - `weights/last.pt` - Last model checkpoint
  - `results.csv` - Training metrics (loss, mAP, etc.)
  - `args.yaml` - Training configuration
  
- **validation_runs/**: Validation results on test sets

- **final_experiments/**: Experiments used in the research paper

### Detection Results

Detection outputs from inference are saved in `runs/detect/`:
- Annotated images/videos with bounding boxes
- Label files in YOLO format (one per frame)
- Confidence scores for each detection

### Analysis Outputs

Analysis results are saved in `assets/`:
- **plots/**: Generated figures (time budgets, transition matrices, detection plots)
- **images/**: Exported images from notebooks

## Datasets

The project uses 4 video datasets:
- **Video 1**: Dataset with 3,949 annotated frames (127.55 MB)
- **Video 2**: Dataset with 329 annotated frames (15.63 MB)
- **Video 3**: Dataset with 937 annotated frames (49.07 MB)
- **Video 4**: Dataset with 1,947 annotated frames (72.24 MB)

Each dataset includes train/valid/test splits and behavioral time-series annotations in CSV format.

See [data/README.md](data/README.md) for more details on dataset structure.

## Key Dependencies

- **ultralytics** (≥8.0.0) - YOLOv9 implementation
- **torch** (≥2.0.0) - Deep learning framework
- **opencv-python** (≥4.8.0) - Video processing
- **roboflow** (≥1.0.0) - Dataset management
- **matplotlib** (≥3.7.0) - Plotting and visualization
- **pandas** (≥2.0.0) - Data manipulation
- **jupyter** (≥1.0.0) - Interactive notebooks

See [requirements.txt](requirements.txt) for the complete list.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
<TO BE UPDATED AFTER PUBLICATION>
```

## Contact

For questions or issues:
- Open an issue in this repository
- Contact the main authors:
  - Aurora González: <aurora.gonzalez2@um.es>
  - S. Caruso: <s.caruso@um.es>

## Acknowledgments

- **YOLOv9**: State-of-the-art object detection model
- **Ultralytics**: YOLO implementation and training framework
- **Roboflow**: Dataset management and annotation platform
- **ThinInAzul Project**: Research project supporting this work

---

**Note**: This repository is part of ongoing research on AI-driven fish behavior analysis. The code and models are provided for research and educational purposes.



