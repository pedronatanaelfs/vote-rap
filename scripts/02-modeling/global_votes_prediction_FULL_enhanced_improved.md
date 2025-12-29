# VOTE-RAP: Global Votes Prediction — FULL Enhanced Model

This script trains and evaluates the **VOTE-RAP “FULL Enhanced”** vote-outcome model for the Brazilian Chamber of Deputies. It predicts whether a voting session outcome (`aprovacao`) is **Approved (1)** or **Rejected (0)** using an XGBoost classifier and three engineered temporal signals (author popularity, party popularity, historical approval rate) alongside government orientation and author-count features.

This is a Python-script replication of `global_votes_prediction_FULL_enhanced.ipynb`.

---

## What the script outputs

The run generates:

1. `approval_rejection_by_year.png`
2. `vote_orientation_accuracy_by_year.png`
3. `confusion_matrix.png`
4. `roc_curve.png`
5. `precision_recall_curve.png`
6. `feature_importance_all.png`
7. `feature_importance_new_features.png`
8. `distribution_party_popularity.png`
9. `distribution_historical_approval_rate.png`
10. `correlation_matrix_new_features.png`
11. `auroc_comparison.png`
12. `f1_comparison.png`

And a full log file:

- `global_votes_prediction_FULL_enhanced_output.txt`

---

## Inputs and merges

The script reads four CSVs:

1) **Main sessions dataset** — `data/vote_sessions_full.csv`  
   Loaded columns include: `id`, `data`, `aprovacao`, `propositionID`, `siglaOrgao`, `year`, `author_type`, `num_authors`, `theme`, `legislatura`, `Governo`, `Oposição`, `GOV.`

2) **Author popularity** — `data/features/author_popularity.csv`  
   Columns used: `idVotacao`, `popularity`

3) **Party popularity** — `data/features/party_popularity_best_window_last_5_sessions.csv`  
   Columns used: `id`, `party_popularity`

4) **Historical approval rate** — `data/features/proposition_history_predictions_historical_probability_rule.csv`  
   Columns used: `id`, `historical_approval_rate`

Merge logic:
- Start from `vote_sessions_full.csv`
- Merge author popularity on `id == idVotacao`
- Merge party popularity on `id`
- Merge historical approval rate on `id`
- Drop `propositionID` and `idVotacao`
- Drop duplicate session ids (`id`)

### Observed sizes in the logged run (2025-11-28)
- Loaded `vote_sessions_full.csv`: 41,461 rows
- Loaded `author_popularity.csv`: 1,990 rows
- Loaded `party_popularity...csv`: 8,914 rows
- Loaded proposition history file: 9,260 rows
- After merging and dedup: 9,260 rows
- After preprocessing (dropping missing target): 8,914 rows

---

## Feature engineering (exact behavior)

### 1) Government Orientation (`gov_orientation`)
The script resolves inconsistencies between `GOV.` and `Governo`:

- If `GOV.` equals `Governo` → use that value
- Else if `GOV.` is not zero → use `GOV.`
- Else → use `Governo`

### 2) Author-count features
- `num_authors_trunc`: `num_authors` capped at **10**
- `has_more_than_10_authors`: boolean (`num_authors > 10`)

### 3) Missing value imputation
- Author popularity (`popularity`): fill missing with **0**
- Party popularity (`party_popularity`): fill missing with **0**
- Historical approval rate (`historical_approval_rate`): fill missing with **0.5** (neutral)

### 4) Target cleaning
- `aprovacao` is coerced to numeric Int64
- Rows with missing `aprovacao` are dropped

### 5) Theme cleaning (not used by the model)
- Missing `theme` is filled with `"Not defined"`

---

## Exploratory plots produced (before training)

1) **Approval/Rejection by year**  
`approval_rejection_by_year.png`  
Stacked bars of yearly approval vs rejection shares.

2) **Government Orientation “accuracy” by year**  
`vote_orientation_accuracy_by_year.png`  
Computed only on rows where `gov_orientation ∈ {1, -1}`, with “correct” defined as:
- `gov_orientation == 1` and `aprovacao == 1`, or
- `gov_orientation == -1` and `aprovacao == 0`.

---

## Model: features, split, scaling

### Features used (exact list)
The model trains on 6 inputs:

- `popularity` (author popularity)
- `gov_orientation`
- `num_authors_trunc`
- `has_more_than_10_authors`
- `party_popularity`
- `historical_approval_rate`

### Temporal split (80/20)
The script keeps chronological order by splitting by row position:

- Train: first 80% (7,131 rows)
- Test: last 20% (1,783 rows)

### Scaling (StandardScaler)
Only these numeric features are scaled:

- `popularity`
- `party_popularity`
- `historical_approval_rate`

Other features are left unscaled.

---

## Hyperparameter search (as implemented)

- Search method: `RandomizedSearchCV`
- Candidates: **75** random combinations
- CV: **3-fold StratifiedKFold** (shuffle=True, random_state=42)
- Metric: AUROC via `roc_auc_score` with probabilities
- Model used in search: `XGBClassifier(tree_method='hist', early_stopping_rounds=10, n_jobs=1, eval_metric='auc')`
- Early stopping uses a further **random stratified** train/validation split *inside* the training portion.

### Note about the logged run
The log prints **“Phase 1 best AUROC: nan”** even though it also prints a parameter set and successfully trains the final model. Treat this as an artifact of the search scoring in that specific run, not as the final test AUROC.

---

## Evaluation (from the execution log)

### Threshold optimization for the rejected class
The script selects a probability threshold to maximize **F1 for the Rejected class (label 0)** using the model’s `P(class=0)`:

- Best `F1_rejected = 0.706` at `threshold = 0.49`

Confusion matrix at that threshold:

```
[[ 149   95]
 [  29 1510]]
```

Classification report (same threshold), highlights:
- Rejected: precision 0.837, recall 0.611, F1 0.706
- Approved: precision 0.941, recall 0.981, F1 0.961
- Accuracy: 0.930

### AUROC and related metrics
Using `P(class=1)`:
- AUROC: **0.9070**
- Average Precision: **0.9752**

The script also prints precision/recall/F1 computed from the default class prediction threshold (0.5) for the **Approved class**.

---

## Feature importance (XGBoost `feature_importances_`)

Logged importances:

1. Government Orientation — 0.481707
2. Has More Than 10 Authors — 0.206831
3. Party Popularity — 0.097529
4. Number of Authors (Truncated) — 0.094197
5. Historical Approval Rate — 0.071735
6. Author Popularity — 0.048001

The script also reports “new features contribution” as the share of total importance from:
- `party_popularity` + `historical_approval_rate`  
which equals **16.9%** in the logged run.

---

## Baselines and comparisons (what is actually being compared)

### 1) AUROC baseline
The baseline AUROC is **hard-coded** in the script:

- `governo_auroc = 0.8599`

The script reports:
- VOTE-RAP AUROC: 0.9070
- Baseline AUROC: 0.8599
- Difference: +0.0471 (+5.48%)

### 2) “Paired t-test” block
The script does **not** compute a paired test over repeated real measurements. Instead, it:
- simulates two AUROC distributions using a normal approximation and the test-set size (1,783),
- subtracts them, and
- runs a one-sample t-test over those simulated differences.

So interpret this as a *heuristic significance check* based on the simulation assumptions, not as a formal paired test across multiple folds/datasets.

### 3) Government-orientation baseline for rejected-class F1
A second comparison uses a simple rule on the test set:
- if `gov_orientation == 1` → predict Approved (1)
- if `gov_orientation == -1` → predict Rejected (0)
- otherwise → predict the rounded test-set mean label

It yields:
- VOTE-RAP F1 (Rejected): 0.706
- Vote-orientation F1 (Rejected): 0.637
- Improvement: +0.069 (+10.8%)

---

## Runtime (from the log)

The logged execution runs from:
- Start: 2025-11-28 15:37:49
- End:   2025-11-28 15:38:00

So this run completes in roughly **11 seconds**, with the hyperparameter search phase reported as **0.1 minutes**.

---

## Practical interpretation

This “FULL Enhanced” model combines:
- a strong political baseline signal (`gov_orientation`),
- coalition-size signals (`num_authors_trunc`, `has_more_than_10_authors`),
- and temporal history signals (`party_popularity`, `historical_approval_rate`, `author popularity`).

In the logged run, it achieves strong overall discrimination (AUROC 0.907) and significantly improves the minority-class F1 for rejections compared to a government-orientation-only rule.

---

*Last updated: 2025-11-28 (aligned to `global_votes_prediction_FULL_enhanced_output.txt`)*
