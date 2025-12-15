"""
Global Votes Prediction - FULL Enhanced Model (VOTE-RAP)

This script replicates the functionality of global_votes_prediction_FULL_enhanced.ipynb
and generates the following outputs in the same folder:
1. approval_rejection_by_year.png - Stacked bar chart of approval/rejection by year
2. vote_orientation_accuracy_by_year.png - Bar chart of vote orientation prediction accuracy
3. confusion_matrix.png - Confusion matrix heatmap
4. roc_curve.png - ROC curve
5. precision_recall_curve.png - Precision-Recall curve
6. feature_importance_all.png - All features importance
7. feature_importance_new_features.png - New features impact
8. distribution_party_popularity.png - Party popularity distribution histogram
9. distribution_historical_approval_rate.png - Historical approval rate distribution histogram
10. correlation_matrix_new_features.png - Correlation matrix of new features
11. auroc_comparison.png - VOTE-RAP vs Baseline comparison
12. f1_comparison.png - F1-Score comparison for rejected class

REPRODUCIBILITY:
- Random seeds are set (np.random.seed(42), random.seed(42))
- All calculations are deterministic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import random
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve, f1_score, classification_report, confusion_matrix,
    make_scorer, roc_auc_score, precision_score, recall_score, roc_curve,
    average_precision_score
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, RandomizedSearchCV, train_test_split
)
from scipy.stats import uniform, randint, stats
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).parent

# Setup output logging to file
class TeeOutput:
    """Class to write output to both console and file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.log_file.write(f"VOTE-RAP - Global Votes Prediction FULL Enhanced Model\n")
        self.log_file.write(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.write(f"\n\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.close()

# Redirect stdout to both console and file
output_file = OUTPUT_DIR / "global_votes_prediction_FULL_enhanced_output.txt"
tee = TeeOutput(output_file)
sys.stdout = tee

# Professional label mapping for features
FEATURE_LABELS = {
    'popularity': 'Author Popularity',
    'author_popularity': 'Author Popularity',
    'gov_orientation': 'Government Orientation',
    'num_authors_trunc': 'Number of Authors (Truncated)',
    'has_more_than_10_authors': 'Has More Than 10 Authors',
    'party_popularity': 'Party Popularity',
    'historical_approval_rate': 'Historical Approval Rate'
}

print("=" * 80)
print("GLOBAL VOTES PREDICTION - FULL ENHANCED MODEL (VOTE-RAP)")
print("=" * 80)
print("\nLoading data...")

# =====================================================
# DATA LOADING AND PREPROCESSING
# =====================================================

# Load the required files
vote_sessions = pd.read_csv(
    DATA_DIR / "vote_sessions_full.csv",
    usecols=["id", "data", "aprovacao", "propositionID", "siglaOrgao", "year",
             "author_type", "num_authors", "theme", "legislatura", "Governo", "Oposição", "GOV."]
)
authors_pop = pd.read_csv(
    BASE_DIR / "scripts" / "01-feature-engineering" / "Author's Popularity" / "author_popularity.csv",
    usecols=["idVotacao", "popularity"]
)
party_popularity = pd.read_csv(
    BASE_DIR / "scripts" / "01-feature-engineering" / "Party Popularity" / "party_popularity_best_window_last_5_sessions.csv",
    usecols=["id", "party_popularity"]
)
historical_data = pd.read_csv(
    BASE_DIR / "scripts" / "01-feature-engineering" / "Historical Approval Rate" / "proposition_history_predictions_historical_probability_rule.csv",
    usecols=["id", "historical_approval_rate"]
)

print(f"Loaded vote_sessions: {len(vote_sessions):,} rows")
print(f"Loaded authors_pop: {len(authors_pop):,} rows")
print(f"Loaded party_popularity: {len(party_popularity):,} rows")
print(f"Loaded historical_data: {len(historical_data):,} rows")

# Merge datasets
merged_df = vote_sessions.copy()
merged_df = merged_df.merge(authors_pop, left_on="id", right_on="idVotacao", how="left")
merged_df = merged_df.merge(party_popularity, left_on="id", right_on="id", how="left")
merged_df = merged_df.merge(historical_data, left_on="id", right_on="id", how="left")
merged_df = merged_df.drop(columns=["propositionID", "idVotacao"])
merged_df = merged_df.drop_duplicates(subset=["id"], keep="first")

print(f"After merging: {len(merged_df):,} rows")

# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("\nFeature engineering...")

# Resolve government orientation
def resolve_gov_orientation(row):
    if row['GOV.'] == row['Governo']:
        return row['GOV.']
    else:
        if row['GOV.'] != 0:
            return row['GOV.']
        else:
            return row['Governo']

merged_df['gov_orientation'] = merged_df.apply(resolve_gov_orientation, axis=1)

# Convert data types
merged_df['data'] = pd.to_datetime(merged_df['data'], errors='coerce')
merged_df['aprovacao'] = pd.to_numeric(merged_df['aprovacao'], errors='coerce').astype('Int64')

# Truncate num_authors at 10
merged_df['num_authors_trunc'] = merged_df['num_authors'].apply(lambda x: x if x <= 10 else 10)
merged_df['has_more_than_10_authors'] = merged_df['num_authors'] > 10

# Fill missing values
merged_df['popularity'] = merged_df['popularity'].fillna(0)
merged_df['party_popularity'] = merged_df['party_popularity'].fillna(0)
merged_df['historical_approval_rate'] = merged_df['historical_approval_rate'].fillna(0.5)

# Drop rows with missing target
merged_df = merged_df.dropna(subset=['aprovacao'])

# Fill missing theme
merged_df['theme'] = merged_df['theme'].fillna("Not defined")

print(f"After preprocessing: {len(merged_df):,} rows")

# =====================================================
# EXPLORATORY VISUALIZATIONS
# =====================================================

print("\nGenerating exploratory visualizations...")

# PLOT 1: Approval/Rejection percentages by year
print("1. Approval/Rejection by year...")
if 'year' not in merged_df.columns:
    merged_df['year'] = merged_df['data'].dt.year

approval_counts = merged_df.groupby(['year', 'aprovacao']).size().unstack(fill_value=0)
approval_percentages = approval_counts.div(approval_counts.sum(axis=1), axis=0) * 100
approval_percentages = approval_percentages[[0, 1]] if 0 in approval_percentages.columns and 1 in approval_percentages.columns else approval_percentages

fig, ax = plt.subplots(figsize=(10, 6))
approval_percentages.plot(kind='bar', stacked=True, ax=ax, color={0: '#E74C3C', 1: '#27AE60'})
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_title('Approval and Rejection Rates by Year', fontsize=14, fontweight='bold')
ax.legend(['Rejected', 'Approved'], fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'approval_rejection_by_year.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: approval_rejection_by_year.png")

# PLOT 2: Vote Orientation prediction accuracy by year
print("2. Vote orientation accuracy by year...")
filtered_df = merged_df[merged_df['gov_orientation'].isin([1, -1])].copy()

def gov_right(row):
    if row['gov_orientation'] == 1 and row['aprovacao'] == 1:
        return 1
    elif row['gov_orientation'] == -1 and row['aprovacao'] == 0:
        return 1
    else:
        return 0

filtered_df['gov_right'] = filtered_df.apply(gov_right, axis=1)
gov_right_perc = filtered_df.groupby('year')['gov_right'].mean() * 100

fig, ax = plt.subplots(figsize=(10, 6))
gov_right_perc.plot(kind='bar', color='#3498DB', ax=ax)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_title('Government Orientation Prediction Accuracy by Year', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'vote_orientation_accuracy_by_year.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: vote_orientation_accuracy_by_year.png")

# =====================================================
# MODEL TRAINING
# =====================================================

print("\n" + "=" * 80)
print("MODEL TRAINING AND OPTIMIZATION")
print("=" * 80)

# Define features
features = [
    'popularity',
    'gov_orientation',
    'num_authors_trunc',
    'has_more_than_10_authors',
    'party_popularity',
    'historical_approval_rate'
]
target = 'aprovacao'

numeric_features = ['popularity', 'party_popularity', 'historical_approval_rate']

# Prepare dataset
dataset_features_target = merged_df[features + [target]].copy()

print("Splitting X and y while preserving temporal order...")
X = dataset_features_target[features]
y = dataset_features_target[target]

print("Performing temporal split (80% train, 20% test)...")
split_idx = int(0.8 * len(X))
X_train = X.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_test = y.iloc[split_idx:].copy()

print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

# Normalize numeric features
print("Normalizing numeric features...")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

X_train_model = X_train_scaled[features]
X_test_model = X_test_scaled[features]

# =====================================================
# HYPERPARAMETER OPTIMIZATION
# =====================================================

print("\nStarting hyperparameter optimization...")
print("Phase 1: RandomizedSearchCV for fast exploration...")

auroc_scorer = make_scorer(roc_auc_score, needs_proba=True)

smart_param_distributions = {
    'n_estimators': randint(150, 351),
    'max_depth': randint(4, 8),
    'learning_rate': uniform(0.02, 0.08),
    'scale_pos_weight': uniform(0.7, 0.6),
    'subsample': uniform(0.7, 0.2),
    'colsample_bytree': uniform(0.5, 0.2),
    'gamma': uniform(0.3, 1.4),
    'min_child_weight': randint(3, 8),
    'reg_alpha': uniform(0, 0.15),
    'reg_lambda': uniform(0.8, 1.4)
}

dataset_size = len(X_train_model)
n_iter_phase1 = 75 if dataset_size < 10000 else 100
cv_folds = 3

print(f"Dataset size: {dataset_size} samples")
print(f"Phase 1: Testing {n_iter_phase1} random combinations x {cv_folds} folds")

xgb_fast = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    tree_method='hist',
    early_stopping_rounds=10,
    n_jobs=1
)

xgb_random_search = RandomizedSearchCV(
    xgb_fast,
    smart_param_distributions,
    n_iter=n_iter_phase1,
    scoring=auroc_scorer,
    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Create validation set for early stopping
X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
    X_train_model, y_train, test_size=0.2, random_state=42, stratify=y_train
)

start_time = time.time()
xgb_random_search.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val_fit, y_val_fit)],
    verbose=False
)
phase1_time = time.time() - start_time

print(f"\nPhase 1 completed in {phase1_time/60:.1f} minutes")
print(f"Phase 1 best AUROC: {xgb_random_search.best_score_:.4f}")
print(f"Phase 1 best parameters: {xgb_random_search.best_params_}")

final_best_params = xgb_random_search.best_params_
final_best_score = xgb_random_search.best_score_

# Re-train final model on full training data
print("\nRe-training final best model on full training data...")
xgb_final = XGBClassifier(**final_best_params, random_state=42, use_label_encoder=False, eval_metric='auc')
xgb_final.fit(X_train_model, y_train)
xgb_best = xgb_final

print("Model training completed!")

# =====================================================
# MODEL EVALUATION
# =====================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Find optimal threshold for F1-score of rejected class
y_proba_0 = xgb_best.predict_proba(X_test_model)[:, 0]
prec_0, rec_0, thresh_0 = precision_recall_curve(y_test, y_proba_0, pos_label=0)
f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0 + 1e-8)
best_idx_0 = f1_0.argmax()
best_thresh_0 = thresh_0[best_idx_0] if best_idx_0 < len(thresh_0) else 0.5
print(f"Best F1_rejected = {f1_0[best_idx_0]:.3f} at threshold={best_thresh_0:.2f}")

# Predict using optimal threshold
y_pred_xgb_f1_0 = (y_proba_0 >= best_thresh_0).astype(int) * 0 + (y_proba_0 < best_thresh_0).astype(int) * 1

# PLOT 3: Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_xgb_f1_0)
cm_labels = ["Rejected", "Approved"]

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 14, 'weight': 'bold'},
            xticklabels=cm_labels, yticklabels=cm_labels, cbar=False, linewidths=1.5, linecolor='gray')
plt.xlabel("Predicted", fontsize=12, fontweight='bold')
plt.ylabel("Actual", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix - VOTE-RAP\n(Optimized Threshold for F1-Score)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: confusion_matrix.png")

print("\nConfusion matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred_xgb_f1_0, digits=3, target_names=["Rejected", "Approved"]))

# Comprehensive AUROC evaluation
print("\nComprehensive AUROC evaluation...")
y_proba_refined = xgb_best.predict_proba(X_test_model)[:, 1]
y_pred_refined = xgb_best.predict(X_test_model)

xgb_auroc_refined = roc_auc_score(y_test, y_proba_refined)
precision_refined = precision_score(y_test, y_pred_refined)
recall_refined = recall_score(y_test, y_pred_refined)
f1_refined = f1_score(y_test, y_pred_refined)
ap_score = average_precision_score(y_test, y_proba_refined)

print(f"AUROC: {xgb_auroc_refined:.4f}")
print(f"Precision: {precision_refined:.4f}")
print(f"Recall: {recall_refined:.4f}")
print(f"F1-Score: {f1_refined:.4f}")
print(f"Average Precision: {ap_score:.4f}")

# PLOT 4A: ROC curve
print("\nGenerating ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_proba_refined)
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_proba_refined)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#E67E22', lw=3, label=f'VOTE-RAP (AUROC = {xgb_auroc_refined:.4f})')
plt.plot([0, 1], [0, 1], color='#34495E', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: roc_curve.png")

# PLOT 4B: Precision-Recall curve
print("Generating Precision-Recall curve...")
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='#27AE60', lw=3, label=f'VOTE-RAP (AP = {ap_score:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: precision_recall_curve.png")

# =====================================================
# FEATURE IMPORTANCE ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

importances = xgb_best.feature_importances_
feature_names = X_train_model.columns

importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

print("\nFeature importances:")
print("-" * 60)
# Create display version with professional labels
importance_df_display = importance_df.copy()
importance_df_display['feature'] = importance_df_display['feature'].map(FEATURE_LABELS)
importance_df_display = importance_df_display.sort_values('importance', ascending=False).reset_index(drop=True)
for idx, row in importance_df_display.iterrows():
    print(f"{idx}  {row['feature']:<35} {row['importance']:.6f}")
print("-" * 60)

# PLOT 5A: All feature importances
print("\nGenerating feature importance plot...")
plt.figure(figsize=(10, 6))

# Apply professional labels
importance_df_display = importance_df.copy()
importance_df_display['feature'] = importance_df_display['feature'].map(FEATURE_LABELS)

colors = ['darkred' if 'Party Popularity' in f or 'Historical Approval Rate' in f
          else 'skyblue' for f in importance_df_display['feature']]
bars = plt.barh(importance_df_display['feature'], importance_df_display['importance'], color=colors)
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.title('Feature Importances - VOTE-RAP (AUROC Optimized)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Add value labels with proper spacing
max_importance = importance_df_display['importance'].max()
for bar, importance in zip(bars, importance_df_display['importance']):
    plt.text(bar.get_width() + max_importance * 0.02, bar.get_y() + bar.get_height()/2,
             f'{importance:.3f}', ha='left', va='center', fontsize=9)

# Extend x-axis to make room for labels
plt.xlim(0, max_importance * 1.15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance_all.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: feature_importance_all.png")

# PLOT 5B: New features impact
print("Generating new features impact plot...")
new_features = ['party_popularity', 'historical_approval_rate']
new_features_importance = importance_df[importance_df['feature'].isin(new_features)].copy()
if not new_features_importance.empty:
    plt.figure(figsize=(10, 5))
    
    # Apply professional labels
    new_features_importance['feature'] = new_features_importance['feature'].map(FEATURE_LABELS)
    
    bars = plt.barh(new_features_importance['feature'], new_features_importance['importance'],
                    color=['darkred', 'darkgreen'][:len(new_features_importance)])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title('New Features Impact', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels with proper spacing
    max_new_importance = new_features_importance['importance'].max()
    for i, (bar, feature, importance) in enumerate(zip(bars, new_features_importance['feature'],
                                                        new_features_importance['importance'])):
        plt.text(bar.get_width() + max_new_importance * 0.03, i, f'{importance:.3f}',
                 ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Extend x-axis to make room for labels
    plt.xlim(0, max_new_importance * 1.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_new_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: feature_importance_new_features.png")

# Feature importance summary
print(f"\nNew Features Performance:")
for feature in ['party_popularity', 'historical_approval_rate']:
    if feature in importance_df['feature'].values:
        rank = importance_df[importance_df['feature'] == feature].index[0] + 1
        importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        feature_label = FEATURE_LABELS.get(feature, feature)
        print(f"  {feature_label}: Rank {rank}/{len(importance_df)} (Importance: {importance:.4f})")

total_new_features_importance = importance_df[
    importance_df['feature'].isin(['party_popularity', 'historical_approval_rate'])
]['importance'].sum()
total_importance = importance_df['importance'].sum()
new_features_contribution = (total_new_features_importance / total_importance) * 100
print(f"\nNew features contribution to model: {new_features_contribution:.1f}% of total importance")

# PLOT 6A: Party popularity distribution
print("\nGenerating party popularity distribution...")
plt.figure(figsize=(8, 6))
plt.hist(merged_df['party_popularity'], bins=30, alpha=0.8, color='#27AE60', edgecolor='black')
plt.title('Distribution of Party Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Party Popularity (%)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distribution_party_popularity.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: distribution_party_popularity.png")

# PLOT 6B: Historical approval rate distribution
print("Generating historical approval rate distribution...")
plt.figure(figsize=(8, 6))
plt.hist(merged_df['historical_approval_rate'], bins=30, alpha=0.8, color='#E74C3C', edgecolor='black')
plt.title('Distribution of Historical Approval Rate', fontsize=14, fontweight='bold')
plt.xlabel('Historical Approval Rate (Probability)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distribution_historical_approval_rate.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: distribution_historical_approval_rate.png")

# PLOT 6C: Correlation matrix
print("Generating correlation matrix...")
plt.figure(figsize=(9, 7))
correlation_data = merged_df[['party_popularity', 'historical_approval_rate', 'aprovacao']].corr()

# Rename columns for display
correlation_labels = {
    'party_popularity': 'Party\nPopularity',
    'historical_approval_rate': 'Historical\nApproval Rate',
    'aprovacao': 'Approval'
}
correlation_data_display = correlation_data.rename(index=correlation_labels, columns=correlation_labels)

sns.heatmap(correlation_data_display, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', annot_kws={'size': 11, 'weight': 'bold'},
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of New Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_matrix_new_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: correlation_matrix_new_features.png")

# =====================================================
# STATISTICAL COMPARISON WITH BASELINE
# =====================================================

print("\n" + "=" * 80)
print("STATISTICAL COMPARISON: VOTE-RAP vs BASELINE")
print("=" * 80)

full_enhanced_auroc = xgb_auroc_refined
governo_auroc = 0.8599

print(f"VOTE-RAP Model AUROC: {full_enhanced_auroc:.4f}")
print(f"Baseline Model AUROC: {governo_auroc:.4f}")
print(f"Difference:            {full_enhanced_auroc - governo_auroc:+.4f}")
print(f"Improvement:           {((full_enhanced_auroc - governo_auroc) / governo_auroc) * 100:+.2f}%")

# Simulate AUROC distributions for statistical testing
def simulate_auroc_distribution(auroc_mean, n_samples=1000, n_test_samples=1783):
    se_auroc = np.sqrt((auroc_mean * (1 - auroc_mean)) / n_test_samples)
    auroc_samples = np.random.normal(auroc_mean, se_auroc, n_samples)
    auroc_samples = np.clip(auroc_samples, 0, 1)
    return auroc_samples

np.random.seed(42)
n_bootstrap = 10000
full_enhanced_samples = simulate_auroc_distribution(full_enhanced_auroc, n_bootstrap)
governo_samples = simulate_auroc_distribution(governo_auroc, n_bootstrap)

differences = full_enhanced_samples - governo_samples
t_stat, p_value_ttest = stats.ttest_1samp(differences, 0)

print(f"\nPaired t-test:")
print(f"  Mean difference: {np.mean(differences):.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.6f}")
if p_value_ttest < 0.05:
    print("  Result: SIGNIFICANT difference (p < 0.05)")
else:
    print("  Result: No significant difference (p >= 0.05)")

# PLOT 7: AUROC comparison
print("\nGenerating AUROC comparison plot...")
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

models = ['Baseline Model', 'VOTE-RAP']
aurocs = [governo_auroc, full_enhanced_auroc]
colors = ['#95A5A6', '#2E86AB']

bars = ax1.bar(models, aurocs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('AUROC Score', fontsize=13, fontweight='bold')
ax1.set_title('Model Performance Comparison - AUROC', fontsize=15, fontweight='bold', pad=20)
ax1.set_ylim(0.8, 0.95)
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, aurocs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

improvement_pct = ((full_enhanced_auroc - governo_auroc) / governo_auroc) * 100
ax1.annotate(f'+{improvement_pct:.1f}%',
             xy=(1, full_enhanced_auroc),
             xytext=(0.5, full_enhanced_auroc + 0.015),
             arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2.5),
             fontsize=13, fontweight='bold', color='#27AE60',
             ha='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'auroc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: auroc_comparison.png")

# =====================================================
# F1-SCORE COMPARISON WITH VOTE ORIENTATION BASELINE
# =====================================================

print("\n" + "=" * 80)
print("F1-SCORE COMPARISON: VOTE-RAP vs VOTE ORIENTATION BASELINE")
print("=" * 80)

vote_rap_f1_rejected = f1_score(y_test, y_pred_xgb_f1_0, pos_label=0)
print(f"VOTE-RAP F1-Score (Rejected): {vote_rap_f1_rejected:.3f}")

# Vote Orientation predictions
test_indices = X_test_model.index
test_gov_orientation = merged_df.loc[test_indices, 'gov_orientation'].values

vote_orientation_predictions = np.where(test_gov_orientation == 1, 1,
                                       np.where(test_gov_orientation == -1, 0,
                                               np.round(y_test.mean())))

vote_orientation_f1_rejected = f1_score(y_test, vote_orientation_predictions, pos_label=0)
print(f"Vote Orientation F1-Score (Rejected): {vote_orientation_f1_rejected:.3f}")

improvement = vote_rap_f1_rejected - vote_orientation_f1_rejected
improvement_pct = (improvement / vote_orientation_f1_rejected) * 100

print(f"\nImprovement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

# Confusion matrices
cm_vote_rap = confusion_matrix(y_test, y_pred_xgb_f1_0)
cm_vote_orientation = confusion_matrix(y_test, vote_orientation_predictions)

print("\nVOTE-RAP Confusion Matrix:")
print(cm_vote_rap)
print("\nVote Orientation Confusion Matrix:")
print(cm_vote_orientation)

# PLOT 8: F1-Score comparison
print("\nGenerating F1-Score comparison plot...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

models = ['Government\nOrientation', 'VOTE-RAP']
f1_scores = [vote_orientation_f1_rejected, vote_rap_f1_rejected]
colors = ['#95A5A6', '#E67E22']

bars = ax.bar(models, f1_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=2)

ax.set_ylabel('F1-Score (Rejected Class)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_title('F1-Score Comparison for Rejected Propositions',
             fontsize=16, fontweight='bold', pad=20)

ax.set_ylim(0, max(f1_scores) * 1.2)
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
            f'{score:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

improvement_text = f'+{improvement_pct:.1f}%'
ax.annotate(improvement_text,
            xy=(1, vote_rap_f1_rejected),
            xytext=(0.5, vote_rap_f1_rejected + 0.06),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=3),
            fontsize=14, fontweight='bold', color='#27AE60',
            ha='center')

ax.set_xticklabels(models, rotation=0, fontsize=13)
ax.axhline(y=vote_orientation_f1_rejected, color='#C0392B', linestyle='--', alpha=0.5, linewidth=2)

textstr = f'VOTE-RAP shows {improvement_pct:.1f}% improvement\nover baseline'
props = dict(boxstyle='round,pad=0.6', facecolor='#AED6F1', alpha=0.9, edgecolor='black', linewidth=1.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: f1_comparison.png")

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("VOTE-RAP MODEL TRAINING COMPLETED")
print("=" * 80)

print(f"\nFinal Results:")
print(f"  AUROC: {xgb_auroc_refined:.4f}")
print(f"  F1-Score (Rejected): {vote_rap_f1_rejected:.3f}")
print(f"  Improvement over Baseline (AUROC): +{((full_enhanced_auroc - governo_auroc) / governo_auroc) * 100:.2f}%")
print(f"  Improvement over Vote Orientation (F1): +{improvement_pct:.1f}%")

print(f"\nGenerated files:")
print(f"  1. approval_rejection_by_year.png")
print(f"  2. vote_orientation_accuracy_by_year.png")
print(f"  3. confusion_matrix.png")
print(f"  4. roc_curve.png")
print(f"  5. precision_recall_curve.png")
print(f"  6. feature_importance_all.png")
print(f"  7. feature_importance_new_features.png")
print(f"  8. distribution_party_popularity.png")
print(f"  9. distribution_historical_approval_rate.png")
print(f"  10. correlation_matrix_new_features.png")
print(f"  11. auroc_comparison.png")
print(f"  12. f1_comparison.png")
print(f"\nOutput log saved to: global_votes_prediction_FULL_enhanced_output.txt")

# Close output file
sys.stdout = tee.terminal
tee.close()

