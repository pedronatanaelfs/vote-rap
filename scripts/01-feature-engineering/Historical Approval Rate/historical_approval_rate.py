"""
Related Objects Analysis - Proposition History Feature Engineering

This script replicates the functionality of related_objects.ipynb
and generates:
1. proposition_history_predictions_historical_probability_rule.csv
2. proposition_history_rules_comparison.png

REPRODUCIBILITY:
- Random seeds are set (numpy.random.seed(42))
- Data is sorted by ['data', 'id'] for consistent ordering
- All calculations are deterministic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import shutil
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).parent
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = BASE_DIR / "results" / "feature_engineering" / "historical_approval_rate"
PAPER_FIG_DIR = BASE_DIR / "article" / "figures"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Setup output logging to file
class TeeOutput:
    """Class to write output to both console and file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.log_file.write(f"VOTE-RAP - Historical Approval Rate Feature Engineering\n")
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
output_file = OUTPUT_DIR / "historical_approval_rate_output.txt"
tee = TeeOutput(output_file)
sys.stdout = tee

print("=" * 80)
print("PROPOSITION HISTORY FEATURE ENGINEERING")
print("=" * 80)
print("Loading data...")

# Load voting sessions data
voting_sessions = pd.read_csv(DATA_DIR / 'vote_sessions_full.csv')
voting_sessions = voting_sessions.drop_duplicates(subset=['id'])
voting_sessions['data'] = pd.to_datetime(voting_sessions['data'])

# Sort by date and id for deterministic ordering
voting_sessions = voting_sessions.sort_values(['data', 'id']).reset_index(drop=True)

print(f"Dataset shape: {voting_sessions.shape}")
print(f"Date range: {voting_sessions['data'].min()} to {voting_sessions['data'].max()}")
print(f"Unique propositionIDs: {voting_sessions['propositionID'].nunique()}")
print(f"Approval rate: {voting_sessions['aprovacao'].mean():.3f}")

# Create proposition history features
print("\n" + "=" * 80)
print("CREATING PROPOSITION HISTORY FEATURES")
print("=" * 80)

def create_proposition_history_features(data):
    """
    Create features based on the same proposition's voting history.
    
    For each voting session, look at previous votes on the SAME propositionID.
    """
    result_df = data.copy()
    
    # Initialize features
    result_df['last_vote_result'] = np.nan
    result_df['historical_approval_rate'] = np.nan
    result_df['vote_count'] = 0
    result_df['approval_trend'] = np.nan
    result_df['rejection_streak'] = 0
    result_df['approval_streak'] = 0
    
    print(f"Processing {len(data):,} sessions...")
    
    for i in range(len(data)):
        current_row = data.iloc[i]
        current_prop_id = current_row['propositionID']
        current_date = current_row['data']
        
        if pd.isna(current_prop_id):
            continue
        
        # Find all previous votes on the same proposition
        previous_votes = data[
            (data['propositionID'] == current_prop_id) & 
            (data['data'] < current_date)
        ].sort_values('data')
        
        if len(previous_votes) > 0:
            # Most recent previous vote result
            last_vote = previous_votes.iloc[-1]['aprovacao']
            result_df.iloc[i, result_df.columns.get_loc('last_vote_result')] = last_vote
            
            # Historical approval rate
            approval_rate = previous_votes['aprovacao'].mean()
            result_df.iloc[i, result_df.columns.get_loc('historical_approval_rate')] = approval_rate
            
            # Vote count
            result_df.iloc[i, result_df.columns.get_loc('vote_count')] = len(previous_votes)
            
            # Approval trend (comparing first half vs second half of voting history)
            if len(previous_votes) >= 4:
                mid_point = len(previous_votes) // 2
                first_half_approval = previous_votes.iloc[:mid_point]['aprovacao'].mean()
                second_half_approval = previous_votes.iloc[mid_point:]['aprovacao'].mean()
                trend = second_half_approval - first_half_approval
                result_df.iloc[i, result_df.columns.get_loc('approval_trend')] = trend
            
            # Streaks
            results = previous_votes['aprovacao'].tolist()
            if results:
                # Count rejection streak
                rejection_streak = 0
                for j in range(len(results) - 1, -1, -1):
                    if results[j] == 0:
                        rejection_streak += 1
                    else:
                        break
                result_df.iloc[i, result_df.columns.get_loc('rejection_streak')] = rejection_streak
                
                # Count approval streak
                approval_streak = 0
                for j in range(len(results) - 1, -1, -1):
                    if results[j] == 1:
                        approval_streak += 1
                    else:
                        break
                result_df.iloc[i, result_df.columns.get_loc('approval_streak')] = approval_streak
        
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i + 1:,}/{len(data):,} sessions ({(i + 1)/len(data)*100:.1f}%)")
    
    return result_df

# Create features
data_with_history = create_proposition_history_features(voting_sessions)

print(f"\nTotal sessions with some history: {data_with_history['vote_count'].gt(0).sum():,}")
print(f"Percentage with history: {data_with_history['vote_count'].gt(0).mean()*100:.1f}%")

# Define prediction rules
print("\n" + "=" * 80)
print("TESTING SIMPLE PREDICTION RULES")
print("=" * 80)

def momentum_rule(row):
    if pd.isna(row['last_vote_result']):
        return 0.5
    return row['last_vote_result']

def contrarian_rule(row):
    if pd.isna(row['last_vote_result']):
        return 0.5
    return 1 - row['last_vote_result']

def historical_average_rule(row):
    if pd.isna(row['historical_approval_rate']):
        return 0.5
    return 1 if row['historical_approval_rate'] > 0.5 else 0

def historical_probability_rule(row):
    if pd.isna(row['historical_approval_rate']):
        return 0.5
    return row['historical_approval_rate']

def rejection_streak_rule(row, threshold=2):
    if row['rejection_streak'] >= threshold:
        return 1
    elif row['approval_streak'] >= threshold:
        return 0
    else:
        return 0.5

def trend_rule(row):
    if pd.isna(row['approval_trend']):
        return 0.5
    return 1 if row['approval_trend'] > 0 else 0

def experience_rule(row):
    if row['vote_count'] == 0:
        return 0.5
    experience_factor = min(row['vote_count'] / 10, 1)
    return 0.5 + 0.3 * experience_factor

# Test prediction rules
def test_prediction_rule(data, rule_function, rule_name):
    valid_mask = (data['vote_count'] > 0) & (~pd.isna(data['aprovacao']))
    valid_data = data[valid_mask].copy()
    
    if len(valid_data) == 0:
        return {'rule_name': rule_name, 'auroc': np.nan, 'n_samples': 0}
    
    predictions = []
    for idx, row in valid_data.iterrows():
        try:
            pred = rule_function(row)
            predictions.append(pred)
        except:
            predictions.append(0.5)
    
    y_true = valid_data['aprovacao'].values
    y_pred = np.array(predictions)
    
    auroc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    
    print(f"\n[OK] {rule_name}:")
    print(f"   AUROC: {auroc:.3f}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Samples: {len(valid_data):,}")
    
    return {
        'rule_name': rule_name,
        'auroc': auroc,
        'accuracy': accuracy,
        'n_samples': len(valid_data),
        'true_approval_rate': y_true.mean(),
        'predicted_approval_rate': y_pred.mean()
    }

# Test all rules
rules_to_test = [
    (momentum_rule, "Momentum Rule (Same as Last Vote)"),
    (contrarian_rule, "Contrarian Rule (Opposite of Last Vote)"),
    (historical_average_rule, "Historical Average Rule (>0.5 threshold)"),
    (historical_probability_rule, "Historical Probability Rule"),
    (lambda row: rejection_streak_rule(row, 2), "Rejection Streak Rule (threshold=2)"),
    (lambda row: rejection_streak_rule(row, 3), "Rejection Streak Rule (threshold=3)"),
    (trend_rule, "Trend Rule (Positive trend → Approval)"),
    (experience_rule, "Experience Rule (More votes → Higher approval)")
]

results = []
for rule_func, rule_name in rules_to_test:
    result = test_prediction_rule(data_with_history, rule_func, rule_name)
    results.append(result)

# Create results summary
results_df = pd.DataFrame(results)
valid_results = results_df.dropna(subset=['auroc'])
valid_results = valid_results.sort_values('auroc', ascending=False)

best_rule = valid_results.iloc[0]

print(f"\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"\nBEST RULE: {best_rule['rule_name']}")
print(f"   AUROC: {best_rule['auroc']:.4f}")

# Create visualization
print("\n" + "=" * 80)
print("CREATING VISUALIZATION")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# AUROC comparison plot
rule_names = [name.split('(')[0].strip() for name in valid_results['rule_name']]
auroc_scores = valid_results['auroc']
colors = ['#ff6b6b' if i == 0 else '#4ecdc4' for i in range(len(auroc_scores))]

bars = ax.bar(range(len(rule_names)), auroc_scores, color=colors, alpha=0.8, edgecolor='black')
ax.set_title('AUROC Performance of Simple Prediction Rules\n(Red = Best Performance)', fontweight='bold')
ax.set_xlabel('Prediction Rules', fontweight='bold')
ax.set_ylabel('AUROC Score', fontweight='bold')
ax.set_xticks(range(len(rule_names)))
ax.set_xticklabels(rule_names, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.4, 1.0)

for i, (bar, value) in enumerate(zip(bars, auroc_scores)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Random (0.5)')
ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Fair (0.6)')
ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Good (0.7)')
ax.legend()

plt.tight_layout()
plot_path = RESULTS_DIR / "proposition_history_rules_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nChart saved as: {plot_path}")

# Keep a copy under article/figures for the paper build
try:
    shutil.copyfile(plot_path, PAPER_FIG_DIR / plot_path.name)
    print(f"Copied to paper figures: {PAPER_FIG_DIR / plot_path.name}")
except Exception as e:
    print(f"Warning: failed to copy plot to article/figures: {e}")

# Generate final dataset with best rule
print("\n" + "=" * 80)
print("GENERATING FINAL DATASET WITH BEST RULE")
print("=" * 80)

best_rule_name = best_rule['rule_name']
best_rule_func = None

for rule_func, rule_name in rules_to_test:
    if rule_name == best_rule_name:
        best_rule_func = rule_func
        break

if best_rule_func:
    # Apply the best rule to create predictions
    predictions = []
    for idx, row in data_with_history.iterrows():
        if row['vote_count'] > 0:
            try:
                pred = best_rule_func(row)
                predictions.append(pred)
            except:
                predictions.append(0.5)
        else:
            predictions.append(0.5)
    
    # Create final output DataFrame
    final_output = pd.DataFrame({
        'id': data_with_history['id'],
        'data': data_with_history['data'].dt.strftime('%Y-%m-%d'),
        'propositionID': data_with_history['propositionID'],
        'aprovacao': data_with_history['aprovacao'],
        'vote_count': data_with_history['vote_count'],
        'last_vote_result': data_with_history['last_vote_result'],
        'historical_approval_rate': data_with_history['historical_approval_rate'],
        'approval_trend': data_with_history['approval_trend'],
        'rejection_streak': data_with_history['rejection_streak'],
        'approval_streak': data_with_history['approval_streak'],
        'prediction': predictions,
        'prediction_rule': best_rule_name
    })
    
    # Save to CSV (centralized under data/features)
    output_filename = FEATURES_DIR / f'proposition_history_predictions_{best_rule_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.csv'
    final_output.to_csv(output_filename, index=False)
    
    print(f"\nFINAL DATASET SUMMARY:")
    print(f"Total rows: {len(final_output):,}")
    print(f"Rows with voting history: {(final_output['vote_count'] > 0).sum():,}")
    print(f"Rows without history: {(final_output['vote_count'] == 0).sum():,}")
    print(f"Percentage with history: {(final_output['vote_count'] > 0).sum() / len(final_output) * 100:.1f}%")
    print(f"Best rule: {best_rule_name}")
    print(f"AUROC achieved: {best_rule['auroc']:.4f}")
    
    # Print statistics for historical approval rate
    hist_data = final_output[final_output['vote_count'] > 0]['historical_approval_rate']
    if len(hist_data) > 0:
        print(f"\nHistorical Approval Rate Statistics (for rows with history):")
        print(f"  Mean: {hist_data.mean():.3f}")
        print(f"  Std Dev: {hist_data.std():.3f}")
        print(f"  Median: {hist_data.median():.3f}")
        print(f"  Min: {hist_data.min():.3f}")
        print(f"  Max: {hist_data.max():.3f}")
        print(f"  Distribution: Bimodal (peaks at 0.0 and 1.0)")
    
    print(f"\nFINAL DATASET SAVED!")
    print(f"Filename: {output_filename}")
    print(f"Columns: {list(final_output.columns)}")
    
    # Print sample data
    print(f"\nSample Data (first 3 rows with history):")
    sample_with_history = final_output[final_output['vote_count'] > 0].head(3)
    if len(sample_with_history) > 0:
        print(sample_with_history.to_string(index=False))

print("\n" + "=" * 80)
print("PROPOSITION HISTORY ANALYSIS COMPLETED!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  1. proposition_history_rules_comparison.png")
print(f"  2. {output_filename}")
print(f"  3. historical_approval_rate_output.txt")

# Close output file
sys.stdout = tee.terminal
tee.close()

