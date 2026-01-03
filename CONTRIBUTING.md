# Contributing to Fish Behaviour Detection

## Code Organization

### Source Code (`src/`)
Reusable modules and utilities:
- `visualization.py`: Plotting and visualization functions
- `data_utils.py`: Data loading and preprocessing
- `__init__.py`: Package initialization

### Notebooks (`notebooks/`)
Numbered sequentially by workflow:
1. `01_experiments.ipynb`: Initial exploration
2. `02_models_training.ipynb`: Model training
3. `03_analysis_postprocessing.ipynb`: Results analysis
4. `04_analysis_T13.ipynb`: Specific video analysis
5. `05_analysis_T15.ipynb`: Specific video analysis
6. `06_postprocessing_functions.ipynb`: Utility functions

### Scripts (`scripts/`)
Standalone execution scripts and legacy code.

### Results (`results/`)
- `training_runs/detect/`: YOLO training outputs
- `validation_runs/`: Validation results
- `final_experiments/`: Final paper results

### Configs (`configs/`)
Configuration files for experiments and models.

### Docs (`docs/`)
- Paper PDF
- Usage documentation
- Setup instructions

## Best Practices

1. **Modular Code**: Put reusable functions in `src/` modules
2. **Documentation**: Add docstrings to all functions
3. **Notebooks**: Keep notebooks focused on specific tasks
4. **Version Control**: Don't commit large files (videos, weights)
5. **Results**: Save important results with descriptive names

## Adding New Features

1. Create new module in `src/` if needed
2. Document in notebooks for reproducibility
3. Update README.md with usage examples
4. Add dependencies to requirements.txt
