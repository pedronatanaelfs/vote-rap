"""
Party Popularity Feature Engineering Script

This script replicates the functionality of party_popularity.ipynb
and generates:
1. party_popularity_best_window_last_5_sessions.csv
2. party_popularity_auroc_comparison.png

REPRODUCIBILITY:
- Random seeds are set (np.random.seed(42), random.seed(42))
- Data is sorted by ['data', 'id'] to handle ties consistently
- Random Forest uses n_jobs=1 (parallel processing can introduce non-determinism)
- API calls are made in sorted order
- All random_state parameters are set to 42

Note: API data from Câmara API may change over time, which could affect results
if deputy party affiliations change. For complete reproducibility, use cached
party data instead of API calls.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as patheffects
import requests
import json
import time
from time import sleep
from pathlib import Path
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Set paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EXTRA_DATA_DIR = DATA_DIR / "extra"
OUTPUT_DIR = Path(__file__).parent
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = BASE_DIR / "results" / "feature_engineering" / "party_popularity"
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
        self.log_file.write(f"VOTE-RAP - Party Popularity Feature Engineering\n")
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
output_file = OUTPUT_DIR / "party_popularity_output.txt"
tee = TeeOutput(output_file)
sys.stdout = tee

print("=== PARTY POPULARITY FEATURE ENGINEERING ===")
print("Loading data...")

# Load voting sessions data
df_sessions = pd.read_csv(DATA_DIR / "vote_sessions_full.csv")
print(f"Loaded {len(df_sessions):,} rows")

# Remove duplicates
df_sessions = df_sessions.drop_duplicates(subset=['id'], keep='first')
print(f"After removing duplicates: {len(df_sessions):,} rows")

# Load deputy-party mapping data
print("Loading deputy-party mapping data...")
orgaos_deputados_frames = []
for legislature in range(51, 58):
    file_path = EXTRA_DATA_DIR / f"orgaosDeputados-L{legislature}.csv"
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, sep=';')
            if 'uriDeputado' in df.columns:
                df['idDeputado'] = df['uriDeputado'].astype(str).str.extract(r'/(\d+)$')
            df['legislature'] = legislature
            orgaos_deputados_frames.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

if orgaos_deputados_frames:
    orgaos_deputados_df = pd.concat(orgaos_deputados_frames, ignore_index=True)
    print(f"Loaded deputy-party data for {len(orgaos_deputados_df):,} records")
else:
    print("Warning: No deputy-party data loaded")
    orgaos_deputados_df = pd.DataFrame()

# Process party information
print("Processing party information...")
df_sessions['idDeputadoAutor_str'] = df_sessions['idDeputadoAutor'].fillna(-1).astype(int).astype(str)
df_sessions.loc[df_sessions['idDeputadoAutor'].isna(), 'idDeputadoAutor_str'] = 'nan'

if not orgaos_deputados_df.empty:
    orgaos_deputados_df['idDeputado_str'] = orgaos_deputados_df['idDeputado'].astype(str)
    # Sort for consistent iteration order
    orgaos_deputados_df = orgaos_deputados_df.sort_values(['idDeputado_str', 'legislature']).reset_index(drop=True)
    
    # Create lookup dictionaries
    deputy_party_lookup = {}
    for _, row in orgaos_deputados_df.iterrows():
        deputy_id = row['idDeputado_str']
        legislature = row['legislature']
        party = row['siglaPartido']
        if pd.notna(party) and deputy_id != 'nan':
            key = f"{deputy_id}_{legislature}"
            deputy_party_lookup[key] = party
    
    deputy_fallback_lookup = {}
    # Sort unique deputy IDs for consistent iteration
    for deputy_id in sorted(orgaos_deputados_df['idDeputado_str'].unique()):
        if deputy_id != 'nan':
            deputy_records = orgaos_deputados_df[orgaos_deputados_df['idDeputado_str'] == deputy_id]
            recent_with_party = deputy_records.dropna(subset=['siglaPartido']).sort_values('legislature', ascending=False)
            if not recent_with_party.empty:
                deputy_fallback_lookup[deputy_id] = recent_with_party.iloc[0]['siglaPartido']
    
    def get_deputy_party_fast(deputy_id_str, legislatura):
        if deputy_id_str == 'nan' or deputy_id_str == '-1':
            return None
        key = f"{deputy_id_str}_{legislatura}"
        if key in deputy_party_lookup:
            return deputy_party_lookup[key]
        if deputy_id_str in deputy_fallback_lookup:
            return deputy_fallback_lookup[deputy_id_str]
        return None
    
    # Assign parties to deputies
    df_sessions['party_or_author_type'] = df_sessions['author_type'].copy()
    deputy_mask = df_sessions['author_type'] == 'Deputado(a)'
    deputy_sessions = df_sessions[deputy_mask]
    
    # Sort indices for consistent iteration
    for idx in sorted(deputy_sessions.index):
        row = df_sessions.loc[idx]
        party = get_deputy_party_fast(row['idDeputadoAutor_str'], row['legislatura'])
        if party is not None:
            df_sessions.loc[idx, 'party_or_author_type'] = party

# Fetch missing party info from API
print("Fetching missing party information from API...")
print("WARNING: API data may change over time, affecting reproducibility")
deputies_without_party = df_sessions[
    (df_sessions['author_type'] == 'Deputado(a)') & 
    (df_sessions['party_or_author_type'] == 'Deputado(a)')
]
unique_missing_deputies = deputies_without_party['idDeputadoAutor_str'].unique()
unique_missing_deputies = [dep_id for dep_id in unique_missing_deputies if dep_id not in ['nan', '-1']]
# Sort for consistent API call order
unique_missing_deputies = sorted(unique_missing_deputies)

def fetch_deputy_party(deputy_id):
    url = f"https://dadosabertos.camara.leg.br/api/v2/deputados/{deputy_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'dados' in data and 'ultimoStatus' in data['dados']:
            return data['dados']['ultimoStatus'].get('siglaPartido')
    except:
        pass
    return None

api_party_lookup = {}
for i, deputy_id in enumerate(unique_missing_deputies):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(unique_missing_deputies)}")
    party = fetch_deputy_party(deputy_id)
    if party:
        api_party_lookup[deputy_id] = party
    sleep(0.1)  # Rate limiting

# Update dataframe with API data (iterate in sorted order for consistency)
for idx in sorted(df_sessions.index):
    row = df_sessions.loc[idx]
    if (row['author_type'] == 'Deputado(a)' and 
        row['party_or_author_type'] == 'Deputado(a)' and
        row['idDeputadoAutor_str'] in api_party_lookup):
        df_sessions.loc[idx, 'party_or_author_type'] = api_party_lookup[row['idDeputadoAutor_str']]

print("Party information processing complete!")

# Prepare base voting data
print("Preparing base voting data...")
base_voting_data = df_sessions.copy()
base_voting_data['data'] = pd.to_datetime(base_voting_data['data'])
# Sort by date and id for deterministic ordering (handles ties in date)
base_voting_data = base_voting_data.sort_values(['data', 'id']).reset_index(drop=True)
base_voting_data = base_voting_data[base_voting_data['aprovacao'].notna()].copy()
base_voting_data = base_voting_data.reset_index(drop=True)  # Reset index after filtering

print(f"Base voting data: {len(base_voting_data):,} rows")

# Party popularity calculation functions
def calculate_party_popularity_full_window(voting_data, current_idx, party):
    """Calculate party popularity using all previous sessions."""
    previous_sessions = voting_data.iloc[:current_idx]
    party_sessions = previous_sessions[previous_sessions['party_or_author_type'] == party]
    
    if len(party_sessions) == 0:
        return {'party_popularity': 0.0, 'party_total_sessions': 0, 'party_approved_sessions': 0}
    
    approved = party_sessions['aprovacao'].sum()
    total = len(party_sessions)
    popularity = (approved / total) * 100 if total > 0 else 0.0
    
    return {
        'party_popularity': popularity,
        'party_total_sessions': total,
        'party_approved_sessions': int(approved)
    }

def calculate_party_popularity_time_window(voting_data, current_idx, party, years):
    """Calculate party popularity using time window."""
    current_date = voting_data.iloc[current_idx]['data']
    cutoff_date = current_date - pd.DateOffset(years=years)
    
    previous_sessions = voting_data.iloc[:current_idx]
    previous_sessions = previous_sessions[previous_sessions['data'] >= cutoff_date]
    party_sessions = previous_sessions[previous_sessions['party_or_author_type'] == party]
    
    if len(party_sessions) == 0:
        return {'party_popularity': 0.0, 'party_total_sessions': 0, 'party_approved_sessions': 0}
    
    approved = party_sessions['aprovacao'].sum()
    total = len(party_sessions)
    popularity = (approved / total) * 100 if total > 0 else 0.0
    
    return {
        'party_popularity': popularity,
        'party_total_sessions': total,
        'party_approved_sessions': int(approved)
    }

def calculate_party_popularity_session_window(voting_data, current_idx, party, n_sessions):
    """Calculate party popularity using last N sessions by the party."""
    previous_sessions = voting_data.iloc[:current_idx]
    party_sessions = previous_sessions[previous_sessions['party_or_author_type'] == party]
    
    if len(party_sessions) == 0:
        return {'party_popularity': 0.0, 'party_total_sessions': 0, 'party_approved_sessions': 0}
    
    # Get last N sessions
    last_n_sessions = party_sessions.tail(n_sessions)
    
    approved = last_n_sessions['aprovacao'].sum()
    total = len(last_n_sessions)
    popularity = (approved / total) * 100 if total > 0 else 0.0
    
    return {
        'party_popularity': popularity,
        'party_total_sessions': total,
        'party_approved_sessions': int(approved)
    }

def create_party_popularity_features(voting_data, window_config):
    """Create party popularity features for all sessions."""
    features = {
        'party_popularity': [],
        'party_total_sessions': [],
        'party_approved_sessions': []
    }
    
    total_rows = len(voting_data)
    
    for idx in range(total_rows):
        if idx % 1000 == 0:
            print(f"    Processing row {idx:,}/{total_rows:,} ({idx/total_rows*100:.1f}%)")
        
        current_row = voting_data.iloc[idx]
        party = current_row['party_or_author_type']
        
        if window_config['type'] == 'full':
            party_stats = calculate_party_popularity_full_window(voting_data, idx, party)
        elif window_config['type'] == 'time':
            party_stats = calculate_party_popularity_time_window(
                voting_data, idx, party, window_config['years']
            )
        elif window_config['type'] == 'sessions':
            party_stats = calculate_party_popularity_session_window(
                voting_data, idx, party, window_config['n_sessions']
            )
        else:
            raise ValueError(f"Unknown window type: {window_config['type']}")
        
        features['party_popularity'].append(party_stats['party_popularity'])
        features['party_total_sessions'].append(party_stats['party_total_sessions'])
        features['party_approved_sessions'].append(party_stats['party_approved_sessions'])
    
    return features

# Window configurations
window_configurations = [
    {'name': 'Full Window', 'type': 'full'},
    {'name': '5-Year Window', 'type': 'time', 'years': 5},
    {'name': '1-Year Window', 'type': 'time', 'years': 1},
    {'name': 'Last 10 Sessions', 'type': 'sessions', 'n_sessions': 10},
    {'name': 'Last 5 Sessions', 'type': 'sessions', 'n_sessions': 5},
    {'name': 'Last 3 Sessions', 'type': 'sessions', 'n_sessions': 3},
    {'name': 'Last 1 Session', 'type': 'sessions', 'n_sessions': 1}
]

# Evaluation function
def evaluate_window_performance(df_with_features, window_name):
    """Evaluate window performance using ML models."""
    # Prepare data
    X = df_with_features[['party_popularity']].fillna(0)
    y = df_with_features['aprovacao'].fillna(0)
    
    # Chronological split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    results = {'window_name': window_name}
    
    # Random Forest (n_jobs=1 for reproducibility)
    try:
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        results['Random Forest_auroc'] = roc_auc_score(y_test, y_pred_proba)
    except:
        results['Random Forest_auroc'] = 0.0
    
    # Logistic Regression
    try:
        lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        results['Logistic Regression_auroc'] = roc_auc_score(y_test, y_pred_proba)
    except:
        results['Logistic Regression_auroc'] = 0.0
    
    return results

# Run window evaluation
print("\n=== EVALUATING WINDOW CONFIGURATIONS ===")
all_results = []

for i, window_config in enumerate(window_configurations):
    print(f"\nTesting window {i+1}/{len(window_configurations)}: {window_config['name']}")
    
    features = create_party_popularity_features(base_voting_data, window_config)
    
    df_with_features = base_voting_data.copy()
    for feature_name, feature_values in features.items():
        df_with_features[feature_name] = feature_values
    
    window_results = evaluate_window_performance(df_with_features, window_config['name'])
    all_results.append(window_results)

# Create results dataframe
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Random Forest_auroc', ascending=False).reset_index(drop=True)

# Print all window results
print("\n" + "=" * 60)
print("WINDOW EVALUATION RESULTS")
print("=" * 60)
print("\nAll Window Results:")
print("-" * 60)
print(f"{'Window Configuration':<25} {'RF AUROC':<15} {'LR AUROC':<15}")
print("-" * 60)
for _, row in results_df.iterrows():
    rf_auroc = row.get('Random Forest_auroc', 0.0)
    lr_auroc = row.get('Logistic Regression_auroc', 0.0)
    marker = " ⭐ BEST" if row['window_name'] == results_df.iloc[0]['window_name'] else ""
    print(f"{row['window_name']:<25} {rf_auroc:<15.4f} {lr_auroc:<15.4f}{marker}")
print("-" * 60)

# Find best window
best_window = results_df.iloc[0]

print(f"\nBest window: {best_window['window_name']} (AUROC: {best_window['Random Forest_auroc']:.4f})")
print(f"  Random Forest AUROC: {best_window.get('Random Forest_auroc', 0.0):.4f}")
print(f"  Logistic Regression AUROC: {best_window.get('Logistic Regression_auroc', 0.0):.4f}")

# Generate AUROC comparison image
print("\n=== GENERATING AUROC COMPARISON IMAGE ===")
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

windows = results_df['window_name'].tolist()
rf_auroc = results_df['Random Forest_auroc'].tolist()
lr_auroc = results_df['Logistic Regression_auroc'].tolist()

# Find best performer index
best_idx = rf_auroc.index(max(rf_auroc))

# Colors
colors_rf = ['#ff7f0e' if i == best_idx else '#1f77b4' for i in range(len(windows))]
colors_lr = ['#d62728' if i == best_idx else '#2ca02c' for i in range(len(windows))]

# Random Forest plot
bars1 = ax1.barh(windows, rf_auroc, color=colors_rf, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('AUROC Score', fontsize=12, fontweight='bold')
ax1.set_title('Random Forest AUROC by Party Popularity Window', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.50)')
ax1.legend()

for i, (bar, value) in enumerate(zip(bars1, rf_auroc)):
    label_color = 'white' if i == best_idx else 'black'
    text = ax1.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                   va='center', ha='left', fontweight='bold', fontsize=11, color=label_color)
    if i == best_idx:
        text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])

# Logistic Regression plot
bars2 = ax2.barh(windows, lr_auroc, color=colors_lr, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('AUROC Score', fontsize=12, fontweight='bold')
ax2.set_title('Logistic Regression AUROC by Party Popularity Window', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)
ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.50)')
ax2.legend()

for i, (bar, value) in enumerate(zip(bars2, lr_auroc)):
    label_color = 'white' if i == best_idx else 'black'
    text = ax2.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                   va='center', ha='left', fontweight='bold', fontsize=11, color=label_color)
    if i == best_idx:
        text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])

plt.tight_layout()
plt.suptitle('Party Popularity Window Size Comparison - AUROC Performance', 
             fontsize=16, fontweight='bold', y=1.02)

plot_filename = RESULTS_DIR / "party_popularity_auroc_comparison.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {plot_filename}")

# Keep a copy under article/figures for the paper build
try:
    shutil.copyfile(plot_filename, PAPER_FIG_DIR / plot_filename.name)
    print(f"Copied to paper figures: {PAPER_FIG_DIR / plot_filename.name}")
except Exception as e:
    print(f"Warning: failed to copy plot to article/figures: {e}")

# Generate final CSV with best window
print("\n=== GENERATING FINAL DATASET ===")
best_window_name = best_window['window_name']
best_config = None
for config in window_configurations:
    if config['name'] == best_window_name:
        best_config = config
        break

if best_config:
    print(f"Using best window: {best_window_name}")
    final_features = create_party_popularity_features(base_voting_data, best_config)
    
    final_output = base_voting_data.copy()
    for feature_name, feature_values in final_features.items():
        final_output[feature_name] = feature_values
    
    final_output['data'] = final_output['data'].dt.strftime('%Y-%m-%d')
    
    output_columns = ['id', 'data', 'party_or_author_type', 'party_popularity', 'aprovacao']
    final_dataset = final_output[output_columns].copy()
    
    best_window_clean_name = best_window_name.lower().replace(' ', '_').replace('-', '_')
    final_filename = FEATURES_DIR / f"party_popularity_best_window_{best_window_clean_name}.csv"
    final_dataset.to_csv(final_filename, index=False)
    print(f"Saved: {final_filename}")
    print(f"Rows: {len(final_dataset):,}")
    
    # Print dataset statistics
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Rows: {len(final_dataset):,}")
    print(f"Date Range: {final_dataset['data'].min()} to {final_dataset['data'].max()}")
    print(f"Unique Parties/Authors: {final_dataset['party_or_author_type'].nunique()}")
    print(f"\nParty Popularity Statistics:")
    print(f"  Mean: {final_dataset['party_popularity'].mean():.1f}%")
    print(f"  Std Dev: {final_dataset['party_popularity'].std():.1f}%")
    print(f"  Median: {final_dataset['party_popularity'].median():.1f}%")
    print(f"  Min: {final_dataset['party_popularity'].min():.1f}%")
    print(f"  Max: {final_dataset['party_popularity'].max():.1f}%")
    print(f"  25th Percentile: {final_dataset['party_popularity'].quantile(0.25):.1f}%")
    print(f"  75th Percentile: {final_dataset['party_popularity'].quantile(0.75):.1f}%")
    
    # Print sample data
    print(f"\nSample Data (first 3 rows):")
    print(final_dataset.head(3).to_string(index=False))

print("\n=== COMPLETE ===")
print(f"Generated files:")
print(f"  1. {plot_filename}")
print(f"  2. {final_filename}")
print(f"  3. party_popularity_output.txt")

# Close output file
sys.stdout = tee.terminal
tee.close()

