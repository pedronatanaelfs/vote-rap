# VOTE-RAP: Year-by-Year Prediction with Moving Window (Enhanced)

This script evaluates an **Enhanced VOTE-RAP** model in a **temporal moving-window** setup. Instead of a single train/test split, it trains a new model for each test year using the **previous 3 years** of data, and then tests on the **next year**.

- **Train window:** 3 consecutive years  
- **Test year:** the following year  
- **Test years covered:** 2007–2024 (18 yearly evaluations)

The “enhanced” part refers to adding two extra historical features:
- `party_popularity`
- `historical_approval_rate`

It also uses `popularity` (author popularity) and the baseline-oriented features (`gov_orientation`, author-count features).

---

## Files produced

The script generates:

1. `yearly_performance_with_presidents.png`  
   Year-by-year AUROC and F1 (Rejected) with presidential periods shaded.

2. `enhanced_vs_original_comparison.png`  
   Bar chart comparing the mean performance of this Enhanced yearly model vs a hard-coded “Original yearly” baseline.

And writes all console output to:

- `global_votes_prediction_yearly_enhanced_output.txt`

---

## Data inputs and merges

The script reads:

1) `data/vote_sessions_full.csv` (base)  
2) `scripts/01-feature-engineering/Author's Popularity/author_popularity.csv` → `popularity`  
3) `scripts/01-feature-engineering/Party Popularity/party_popularity_best_window_last_5_sessions.csv` → `party_popularity`  
4) `scripts/01-feature-engineering/Historical Approval Rate/proposition_history_predictions_historical_probability_rule.csv` → `historical_approval_rate`

Merge logic:
- Start from `vote_sessions_full.csv`
- Merge author popularity on `id == idVotacao`
- Merge party popularity on `id`
- Merge historical approval rate on `id`
- Drop `propositionID` and `idVotacao`
- Drop duplicate session ids (`id`, keep first)

**Logged sizes (2025-11-28 run):**
- Loaded vote sessions: 41,461 rows
- After merging/dedup: 9,260 rows
- After dropping missing target (`aprovacao`): 8,914 rows

---

## Feature engineering (exact behavior)

### 1) Government orientation (`gov_orientation`)
The script resolves differences between `GOV.` and `Governo`:

- If `GOV.` equals `Governo` → use that value  
- Else if `GOV.` is not zero → use `GOV.`  
- Else → use `Governo`

### 2) Author-count features
- `num_authors_trunc`: cap `num_authors` at **10**
- `has_more_than_10_authors`: boolean (`num_authors > 10`)

### 3) Missing values
- `popularity` → fill `NaN` with **0**
- `party_popularity` → fill `NaN` with **0**
- `historical_approval_rate` → fill `NaN` with **0.5**

### 4) Target cleaning
- `aprovacao` coerced to numeric Int64
- rows with missing `aprovacao` dropped

---

## Model inputs

### Feature list (used in every year)
The model uses exactly these 6 columns:

- `popularity`
- `gov_orientation`
- `num_authors_trunc`
- `has_more_than_10_authors`
- `party_popularity`
- `historical_approval_rate`

Target:
- `aprovacao` (0 = Rejected, 1 = Approved)

### Scaling
Only these are scaled with `StandardScaler` in each yearly loop (fit on train only):

- `popularity`
- `party_popularity`
- `historical_approval_rate`

Other features are left unscaled.

---

## Year-by-year training procedure

For each `test_year` in 2007–2024:

1. Define `train_years = [test_year-3, test_year-2, test_year-1]`
2. Train on rows where `year ∈ train_years`
3. Test on rows where `year == test_year`
4. Skip years only if `train_size < 50` or `test_size < 10` (this run did not skip any test year)

---

## XGBoost configuration (fixed, no hyperparameter search)

The model is trained with a fixed parameter set:

- `n_estimators=200`
- `max_depth=6`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `gamma=0.5`
- `min_child_weight=5`
- `reg_alpha=0.1`
- `reg_lambda=1.0`
- `tree_method='hist'`
- `eval_metric='logloss'`
- `random_state=42`

Class imbalance handling:
- `scale_pos_weight = (#rejected) / (#approved)` computed from the training window (else 1).

---

## Thresholding: optimized for the *Rejected* class each year

The script optimizes a **year-specific threshold** focusing on the Rejected class (label 0):

- It uses `y_proba_0 = P(class=0)` from `predict_proba`.
- It builds a precision–recall curve with `pos_label=0`.
- It chooses the threshold that maximizes F1 for label 0.

Prediction rule (equivalent form):
- predict Rejected (0) if `P(class=0) >= threshold`
- otherwise predict Approved (1)

The yearly results table includes the chosen threshold as `best_threshold`, although the printed “main table” in the log only shows accuracy/F1/AUROC.

---

## Reported results (from the execution log)

### Per-year metrics (2007–2024)

The script prints accuracy, F1 for Approved, F1 for Rejected, and AUROC for each test year.
Example lines:

- 2007: Acc=0.706, F1_rejected=0.594, AUROC=0.771
- 2019: Acc=0.830, F1_rejected=0.748, AUROC=0.910
- 2024: Acc=0.938, F1_rejected=0.702, AUROC=0.906

### Overall statistics (mean ± std across 18 test years)

- Mean Accuracy: **0.811 ± 0.083**
- Mean F1 Approved: **0.868 ± 0.067**
- Mean F1 Rejected: **0.624 ± 0.110**
- Mean AUROC: **0.823 ± 0.071**

### Performance range

- Best Accuracy: **0.938 (2024)**
- Worst Accuracy: **0.694 (2015)**
- Best F1 Rejected: **0.784 (2021)**
- Worst F1 Rejected: **0.425 (2015)**
- Best AUROC: **0.910 (2019)**
- Worst AUROC: **0.697 (2015)**

### Trend correlations (year vs metric)

- Year vs Accuracy correlation: **+0.457**
- Year vs F1 Rejected correlation: **+0.555**
- Year vs AUROC correlation: **+0.720**

### Presidential-period grouping (as printed by the script)

- Lula II (2007–2010): Avg AUROC 0.780, Avg F1 Rejected 0.609
- Dilma Rousseff (2011–2016): Avg AUROC 0.773, Avg F1 Rejected 0.518
- Michel Temer (2016–2018): Avg AUROC 0.849, Avg F1 Rejected 0.659
- Jair Bolsonaro (2019–2022): Avg AUROC 0.893, Avg F1 Rejected 0.744
- Lula III (2023–2024): Avg AUROC 0.896, Avg F1 Rejected 0.686

---

## Plot details (what is actually drawn)

### 1) `yearly_performance_with_presidents.png`
- Line 1: AUROC over test years
- Line 2: F1 (Rejected) over test years
- Background shaded by presidential period (Lula II, Dilma, Temer, Bolsonaro, Lula III)
- Annotates four “major political events” exactly as coded:
  - 2008: “Dry Law & Card CPI”
  - 2013: “June Protests”
  - 2014: “Car Wash & Internet Law”
  - 2020: “COVID-19 Aid”

### 2) `enhanced_vs_original_comparison.png`
Compares mean metrics against a hard-coded “Original yearly stats” dictionary:

Original yearly stats (hard-coded in the script):
- Accuracy 0.779
- F1 Approved 0.840
- F1 Rejected 0.587
- AUROC 0.767

Enhanced yearly stats (computed from this run):
- Accuracy 0.811
- F1 Approved 0.868
- F1 Rejected 0.624
- AUROC 0.823

The script prints the improvement table:
- Accuracy: +0.032 (+4.1%)
- F1 Approved: +0.028 (+3.3%)
- F1 Rejected: +0.037 (+6.3%)
- AUROC: +0.056 (+7.4%)

---

## Interpretation (what these results mean)

This evaluation answers a very specific question:

> “If I’m only allowed to train on the last 3 completed years, how well can the model predict the next year?”

The main takeaways from the logged run:
- Performance is **not constant** year-to-year; 2015 is the weakest year across multiple metrics.
- Later years generally improve (positive correlations), consistent with larger training windows and/or more stable learned patterns in the available labels/features.

---

## Reproducibility notes (as implemented)

Deterministic controls set:
- `np.random.seed(42)` and `random.seed(42)`
- Chronological windowing is explicit by `year` selection
- Scaling is fit on train only in each window

Potential sources of variation across environments:
- XGBoost internals can still vary slightly across versions/hardware, though `tree_method='hist'` and fixed seeds reduce variance.

---

*Last updated: 2025-11-28 (aligned to `global_votes_prediction_yearly_enhanced_output.txt`)*
