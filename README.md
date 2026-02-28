# VOTE-RAP: Vote Outcome Prediction using Temporal Evidence of Related Approval Patterns

This repository contains the implementation and results for the VOTE-RAP vote-outcome prediction experiments.

## Overview

VOTE-RAP is a machine learning approach to predict whether a proposition (bill) in the Brazilian Chamber of Deputies will be **approved or rejected**, leveraging **temporal evidence** about how similar proposals and parties performed in the past.

### Key Results

- **AUROC**: 0.9108 (Baseline: 0.8599) - **+5.9 percentage points improvement**
- **F1-Score for Rejected Class**: 0.700 (Baseline: 0.637) - **+9.9 percentage points improvement**

## Repository Structure

```
vote-rap/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── data/                              # Datasets (raw + processed)
│   ├── vote_sessions_full.csv         # Main sessions table (one row per vote session)
│   ├── features/                      # Engineered feature CSVs used by the models
│   ├── voting/                        # Roll-call votes + orientations (by year)
│   ├── propositions/                  # Proposition metadata (XLSX/CSV by year)
│   ├── authors/                       # Proposition authorship tables (CSV by year)
│   └── extra/                         # Auxiliary lookup tables (e.g., deputy→party by legislature)
├── scripts/
│   ├── 00 - Data Aquisition/         # Data acquisition scripts
│   ├── 01-feature-engineering/       # Feature engineering scripts
│   │   ├── Author's Popularity/      # Author popularity feature
│   │   ├── Party Popularity/         # Party popularity feature
│   │   └── Historical Approval Rate/ # Historical approval rate (HAR) feature
│   ├── 02-modeling/                   # Modeling scripts/notebooks
│   └── 03-comparisons/                # Comparison studies (e.g., Albuquerque)
├── results/                           # Generated figures/logs (feature engineering + modeling)
└── img/                               # Example figures used in the README
    ├── AUROC_comparison.png
    └── approval_rate_theme.png
```

For step-by-step execution, see `USAGE.md`. If you run into path issues, see `SETUP_NOTES.md`.

## Features

The model uses three temporal and structural features:

### 1. Vote Orientation
Represents the coalition/ideological stance associated with each proposition, capturing how parties position themselves (government vs. opposition, left vs. right).

### 2. Party Popularity
A party-level metric indicating how successful a party has been at getting its authored propositions approved over a chosen time window. High popularity = the party's proposals are frequently approved.

### 3. Historical Approval Rate (HAR)
A feature representing the recent empirical probability that **similar propositions** were approved within time windows of 1, 2, 3, 4, 5, or 10 years. HAR reflects institutional memory and long-term tendencies.

## Installation

### Prerequisites

- Python 3.9 or higher
- Anaconda or Miniconda (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone <REPOSITORY_URL>
   cd vote-rap
   ```

2. Create a conda environment:
   ```bash
   conda create -n vote-rap python=3.9
   conda activate vote-rap
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install JupyterLab (if not already installed):
   ```bash
   conda install jupyterlab ipykernel
   python -m ipykernel install --user --name=vote-rap --display-name="Python (vote-rap)"
   ```

## Usage

### Quickstart (using pre-computed features)

The repository already includes the engineered feature CSVs under `data/features/`. You can run the main model directly:

```bash
python "scripts/02-modeling/global_votes_prediction_FULL_enhanced.py"
```

This writes figures/logs under `results/modeling/` (and prints metrics to the console).

### Regenerating features (optional)

If you want to recreate the engineered features from the raw tables in `data/`:

```bash
python "scripts/01-feature-engineering/Author's Popularity/authors_popularity.py"
python "scripts/01-feature-engineering/Party Popularity/party_popularity.py"
python "scripts/01-feature-engineering/Historical Approval Rate/historical_approval_rate.py"
```

Outputs:
- `data/features/author_popularity.csv`
- `data/features/party_popularity_best_window_last_5_sessions.csv`
- `data/features/proposition_history_predictions_historical_probability_rule.csv`

### Other experiments

- **Year-by-year moving-window evaluation**: `scripts/02-modeling/global_votes_prediction_yearly_enhanced.py`
- **Ablation study**: `scripts/02-modeling/ablation_vote_rap.py`
- **Comparisons**: `scripts/02-modeling/compare_vote_rap_vs_viola.py` and `scripts/02-modeling/compare_vote_rap_vs_albuquerque.py`

Some experiments include notebooks (e.g., `scripts/02-modeling/baselines.ipynb`).

### Workflow

The typical workflow is:

1. **Feature Engineering** (Optional): Run the feature engineering scripts to generate the three main features (author popularity, party popularity, historical approval rate).
   - **Note**: The feature engineering scripts use relative paths to the `data/` directory. Alternatively, you can use the pre-computed feature files already in `data/features/`.
2. **Modeling**: Run the modeling scripts which:
   - Loads all features
   - Performs data preprocessing
   - Trains an XGBoostClassifier with hyperparameter optimization
   - Evaluates the model and compares with baseline

## Methodology

### Data Collection

The dataset is built using official open data portals of the Brazilian Chamber of Deputies:
- Roll-call vote records
- Proposition metadata
- Deputies and party information
- Legislature and session details

### Data Preparation

- Cleaning and harmonizing identifiers
- Keeping only propositions with clear "approved" or "rejected" outcomes
- Ensuring no temporal leakage: **all features for a proposition use only past data**
- Chronological split into **80% training** and **20% testing**

### Modeling Approach

- **Algorithm**: XGBoostClassifier
- **Hyperparameter Optimization**: Two-stage approach
  1. RandomizedSearchCV for wide exploration
  2. GridSearchCV for fine-tuning
- **Evaluation Metric**: AUROC
- **Cross-Validation**: 3-fold Stratified K-Fold
- **Preprocessing**: StandardScaler applied to numeric features

## Results

The VOTE-RAP model significantly outperforms the baseline:

| Metric | Baseline | VOTE-RAP | Improvement |
|--------|----------|----------|-------------|
| AUROC | 0.8599 | 0.9108 | +5.9 pp |
| F1-Score (Rejected) | 0.637 | 0.700 | +9.9 pp |

Additional evaluation includes:
- Threshold tuning to maximize F1_rejected
- Detailed confusion matrix and PR curves
- Temporal analysis showing performance varies with political stability

## Limitations and Future Work

- Performance decreases during periods of political instability
- Future work may incorporate:
  - More advanced temporal models
  - Richer contextual features
  - Complex network-based metrics

## License

See LICENSE file for details.

