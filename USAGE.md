# Usage Guide

This document provides step-by-step instructions for running the VOTE-RAP repository.

## Prerequisites

Before starting, ensure you have:
1. Python 3.9+ installed
2. Anaconda or Miniconda installed
3. All dependencies installed (see `requirements.txt`)

## Setup

1. **Clone and navigate to the repository**:
   ```bash
   cd vote-rap
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n vote-rap python=3.9
   conda activate vote-rap
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Register Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=vote-rap --display-name="Python (vote-rap)"
   ```

## Running the Notebooks

### Option 1: Run Feature Engineering + Modeling (Full Pipeline)

If you want to regenerate all features from scratch:

1. **Start JupyterLab**:
   ```bash
   jupyter lab
   ```

2. **Run feature engineering scripts in order**:
   - Run `scripts/01-feature-engineering/Author's Popularity/authors_popularity.py`
     - This generates author popularity features
     - Output: `data/features/author_popularity.csv`
   
   - Run `scripts/01-feature-engineering/Party Popularity/party_popularity.py`
     - This generates party popularity features
     - Output: `data/features/party_popularity_best_window_last_5_sessions.csv`
   
   - Run `scripts/01-feature-engineering/Historical Approval Rate/historical_approval_rate.py`
     - This generates historical approval rate (HAR) features
     - Output: `data/features/proposition_history_predictions_historical_probability_rule.csv`

3. **Run the modeling notebook**:
   - Open `scripts/02-modeling/vote-rap-model.ipynb`
   - This loads all features and trains the final VOTE-RAP model
   - Make sure all feature CSV files are in the `data/` directory

### Option 2: Run Modeling Only (Using Pre-computed Features)

If you just want to run the final model with the provided feature files:

1. **Start JupyterLab**:
   ```bash
   jupyter lab
   ```

2. **Run the modeling notebook**:
   - Open `scripts/02-modeling/vote-rap-model.ipynb`
   - Select the `Python (vote-rap)` kernel
   - Run all cells
   - The notebook will load pre-computed features from the `data/` directory

## Expected Outputs

### Feature Engineering Scripts

- **authors_popularity.py**: Generates `data/features/author_popularity.csv`
- **party_popularity.py**: Generates `data/features/party_popularity_best_window_last_5_sessions.csv`
- **historical_approval_rate.py**: Generates `data/features/proposition_history_predictions_historical_probability_rule.csv`

### Modeling Notebook

The `vote-rap-model.ipynb` notebook will:
1. Load and merge all feature files
2. Perform data preprocessing
3. Train XGBoostClassifier with hyperparameter optimization
4. Evaluate the model and compare with baseline
5. Generate visualizations (ROC curves, confusion matrices, etc.)

## Troubleshooting

### Issue: FileNotFoundError when loading data

**Solution**: Make sure you're running scripts from the correct directory. The paths are relative to the script location:
- Feature engineering scripts: `../../data/`
- Modeling notebook: `../../data/`

### Issue: Missing dependencies

**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: Kernel not found in Jupyter

**Solution**: 
```bash
python -m ipykernel install --user --name=vote-rap --display-name="Python (vote-rap)"
```

Then restart JupyterLab and select the kernel.

### Issue: Feature engineering scripts require additional data

**Note**: Some feature engineering scripts may reference additional data files that aren't included in this repository (e.g., raw API data, intermediate processing files). In this case:
- Use the pre-computed feature files already in the `data/` directory
- Or refer to the original research repository for complete data acquisition scripts

## Data Files

The repository includes the following pre-computed data files in `data/`:
- `vote_sessions_full.csv` - Main voting sessions dataset
- `features/author_popularity.csv` - Author popularity features
- `features/party_popularity_best_window_last_5_sessions.csv` - Party popularity features
- `features/proposition_history_predictions_historical_probability_rule.csv` - Historical approval rate features
- `voting_sessions_orientations_clean.csv` - Vote orientation data

## Notes

- The scripts are designed to be run sequentially for feature engineering
- The modeling notebook can be run independently if feature files are already present
- Some cells may take several minutes to execute (especially hyperparameter optimization)
- Make sure you have sufficient RAM (recommended: 8GB+) for running the full pipeline

