# Fish Behaviour Detection Using YOLOv9

Automated detection and analysis of fish behaviour in underwater videos using deep learning.

## Paper

This repository contains the code and experiments for the paper "AI-Driven Analysis of Fish Reproductive Behaviour".

## Project Structure

```
.
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── visualization.py          # Visualization utilities
│   └── data_utils.py            # Data processing utilities
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── training/                # Model training notebooks
│   │   ├── 01_experiments.ipynb
│   │   └── 02_models_training.ipynb
│   ├── analysis/                # Behavior analysis notebooks
│   │   ├── TimeBudgetVideo1-4.ipynb
│   │   └── NormalizedTransitionsVideo1-4.ipynb
│   └── markov_chain/            # Markov chain analysis
│       └── Video1-4MarkovChainOrderEstimation.ipynb
├── scripts/                      # Standalone scripts
│   └── legacy_analysis.py       # Original analysis script
├── data/                         # Behavior definition data
│   └── timeVideo1-4.csv         # Time-coded behavior data
├── results/                      # Training and validation results
│   ├── training_runs/           # YOLO training outputs
│   ├── validation_runs/         # Validation results
│   └── final_experiments/       # Final experiments for paper
├── assets/                       # Generated outputs
│   ├── images/                  # Saved images from notebooks
│   ├── plots/                   # Generated plots
│   └── videos/                  # Processed videos
├── configs/                      # Configuration files
│   ├── models/                  # Model configurations
│   ├── training/                # Training parameters
│   └── data/                    # Data configurations
├── docs/                         # Documentation and paper
│   ├── Revision_FishBehaviour-5.pdf
│   ├── setup_steps.txt
│   └── USAGE.md                 # Usage examples
└── README.md

```

## Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Installation

1. Clone the repository:
```bash
git clone git@github.com:auroragonzalez/FishBehaviourAI.git
cd FishBehaviourAI
```

2. Create a virtual environment:
```bash
python3 -m venv cnr
source cnr/bin/activate  # On Linux/Mac
# or
cnr\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

See `docs/setup_steps.txt` for Roboflow API configuration.

## Usage

### Training a Model

See `notebooks/training/02_models_training.ipynb` for training procedures.

### Running Detection Analysis

```python
from src.visualization import plot_detections_over_time

# Analyze detections over time
plot_detections_over_time(
    video_path="/path/to/video.mp4",
    labels_dir="/path/to/labels",
    output_prefix="T1",
    tick_interval=10,
    title="Fish Detections Over Time"
)
```

### Behavior Analysis

The project includes three types of behavior analysis:

1. **Time Budget Analysis** - `notebooks/analysis/TimeBudgetVideo*.ipynb`
2. **Transition Analysis** - `notebooks/analysis/NormalizedTransitionsVideo*.ipynb`
3. **Markov Chain Order** - `notebooks/markov_chain/Video*MarkovChainOrderEstimation.ipynb`

### Using Visualization Tools

```python
from src.visualization import natural_sort_key, get_frame_number

# Sort files naturally
files = sorted(os.listdir(directory), key=natural_sort_key)
```

For more detailed usage examples, see `docs/USAGE.md`.

## 📈 Results

Training results and model weights are stored in:
- `results/training_runs/detect/` - Individual training runs
- `results/final_experiments/` - Final experiments for the paper

Each training run contains:
- `args.yaml` - Training configuration
- `results.csv` - Training metrics
- `weights/` - Model weights (best.pt, last.pt)

## 🔬 Experiments

The following experiments are documented in the notebooks:

1. **Video 1-4**: Individual video training and testing
2. **Combined Training**: Videos 1-3 combined training
3. **Temporal Analysis**: Detection patterns over time (T13, T15)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{fishbehaviour2025,
  title={Fish Behaviour Detection},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025}
}
```

## 📧 Contact

For questions or issues, please open an issue in the repository.

## 🙏 Acknowledgments

- YOLOv9 implementation
- Roboflow for dataset management
- [Add other acknowledgments]

## 📄 License

[Add your license here]
