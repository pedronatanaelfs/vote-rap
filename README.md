# VOTE-RAP: Vote Outcome Prediction using Temporal Evidence of Related Approval Patterns

This repository contains the implementation and results for the research presented at the **2025 Doctoral Consortium School – ADBIS**, Tampere University.

**Authors**: Pedro N. F. da Silva and colleagues

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
├── data/                              # Data files
│   ├── vote_sessions_full.csv         # Main voting sessions dataset
│   ├── author_popularity.csv          # Author popularity features
│   ├── party_popularity_best_window_last_5_sessions.csv  # Party popularity features
│   ├── proposition_history_predictions_historical_probability_rule.csv  # Historical approval rate
│   └── voting_sessions_orientations_clean.csv  # Vote orientation data
├── notebooks/
│   ├── 01-feature-engineering/        # Feature engineering notebooks
│   │   ├── 01-vote-orientation.ipynb  # Vote orientation feature
│   │   ├── 02-party-popularity.ipynb  # Party popularity feature
│   │   └── 03-historical-approval-rate.ipynb  # Historical approval rate (HAR) feature
│   └── 02-modeling/                   # Modeling notebooks
│       └── vote-rap-model.ipynb       # Final VOTE-RAP model
└── img/                               # Result images and visualizations
    ├── AUROC_comparison.png
    └── approval_rate_theme.png
```

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
   git clone [REPOSITORY_URL]
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

### Running the Notebooks

1. Start JupyterLab:
   ```bash
   jupyter lab
   ```

2. Navigate to the notebooks directory and open the notebooks in order:
   - **Feature Engineering** (run in order):
     - `notebooks/01-feature-engineering/01-vote-orientation.ipynb`
     - `notebooks/01-feature-engineering/02-party-popularity.ipynb`
     - `notebooks/01-feature-engineering/03-historical-approval-rate.ipynb`
   
   - **Modeling**:
     - `notebooks/02-modeling/vote-rap-model.ipynb`

3. Make sure to select the `Python (vote-rap)` kernel when opening notebooks.

### Workflow

The typical workflow is:

1. **Feature Engineering** (Optional): Run the feature engineering notebooks to generate the three main features (vote orientation, party popularity, historical approval rate).
   - **Note**: The feature engineering notebooks may require path updates. They currently reference `../data/` but should reference `../../data/` since they're in subdirectories. Alternatively, you can use the pre-computed feature files already in the `data/` directory.
2. **Modeling**: Run the main modeling notebook which:
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

## Citation

If you use this work, please cite:

```
Silva, P. N. F. da, et al. (2025). Vote Outcome Prediction using Temporal Evidence of Related Approval Patterns. 
2025 Doctoral Consortium School – ADBIS, Tampere University.
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue on the repository or contact the authors.

