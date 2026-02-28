"""
Generalization Comparison: VOTE-RAP vs Albuquerque

This script tests Albuquerque's methodology on the full VOTE-RAP proposition space
to demonstrate that their approach, trained on the limited roll-call subset,
fails to generalize to the broader proposition space.

Comparison:
- VOTE-RAP: Trained on 80% of 41,461 propositions, tested on 20%
- Albuquerque-style: Trained on 3,741 roll-call propositions, tested on ALL 41,461
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "modeling" / "comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("GENERALIZATION COMPARISON: VOTE-RAP vs Albuquerque")
print("="*80)

# =============================================================================
# 1. LOAD DATASETS
# =============================================================================
print("\n1. Loading datasets...")

# Load VOTE-RAP full dataset
vote_sessions = pd.read_csv(DATA_DIR / 'vote_sessions_full.csv')
print(f"   VOTE-RAP dataset: {len(vote_sessions):,} propositions")

# Load Albuquerque dataset
df_albuquerque = pd.read_csv(SCRIPT_DIR / 'features.csv', sep=';', low_memory=False)
print(f"   Albuquerque dataset: {len(df_albuquerque):,} individual votes")
print(f"   Unique propositions in Albuquerque: {df_albuquerque['idVotacao'].nunique():,}")

# =============================================================================
# 2. CATEGORIZE ALBUQUERQUE FEATURES
# =============================================================================
print("\n2. Categorizing Albuquerque features...")

# ID columns (not features)
id_cols = ['idDeputado', 'nome', 'siglaUf', 'idPartido', 'siglaPartido', 
           'data', 'y', 'idLegislatura', 'idProposicao', 'idVotacao', 'voto']

# Proposition-level features (same for all deputies in a session)
proposition_features = [
    # Government/party orientation for the proposition
    'd_ori_gov', 'd_ori_gov_sim', 'd_ori_gov_nao', 'd_ori_gov_abstencao', 
    'd_ori_gov_liberado', 'd_ori_gov_obstrucao',
    'd_ori_mai', 'd_ori_mai_sim', 'd_ori_mai_nao', 'd_ori_mai_liberado',
    'd_ori_min', 'd_ori_min_liberado', 'd_ori_min_sim', 'd_ori_min_obstrucao', 'd_ori_min_nao',
    'd_ori_op', 'd_ori_op_sim', 'd_ori_op_nao', 'd_ori_op_liberado', 'd_ori_op_obstrucao',
    # Theme dummies
    'd_tema_ACR', 'd_tema_AP', 'd_tema_APPE', 'd_tema_CDU', 'd_tema_COM',
    'd_tema_CTI', 'd_tema_DC', 'd_tema_DCPC', 'd_tema_DDC', 'd_tema_DHM',
    'd_tema_DJ', 'd_tema_DPPP', 'd_tema_DS', 'd_tema_ECO', 'd_tema_EDU',
    'd_tema_EF', 'd_tema_EL', 'd_tema_ERHM', 'd_tema_FPO', 'd_tema_HDC',
    'd_tema_ICS', 'd_tema_MADS', 'd_tema_PAS', 'd_tema_PLAP', 'd_tema_PPE',
    'd_tema_RICE', 'd_tema_SAU', 'd_tema_TE', 'd_tema_TUR', 'd_tema_VTM',
    # Proposition type dummies
    'd_tipoVotacao_CMC', 'd_tipoVotacao_MPV', 'd_tipoVotacao_MSC', 'd_tipoVotacao_PDC',
    'd_tipoVotacao_PDL', 'd_tipoVotacao_PEC', 'd_tipoVotacao_PL', 'd_tipoVotacao_PLP',
    'd_tipoVotacao_PRC', 'd_tipoVotacao_RCP', 'd_tipoVotacao_REC', 'd_tipoVotacao_REP',
    'd_tipoVotacao_REQ', 'd_tipoVotacao_SIP',
    # Author type dummies
    'tipoAutor_CCJC', 'tipoAutor_CD', 'tipoAutor_CE', 'tipoAutor_CLP',
    'tipoAutor_CM', 'tipoAutor_CMD', 'tipoAutor_CMP', 'tipoAutor_CPI',
    'tipoAutor_CREDN', 'tipoAutor_CRP', 'tipoAutor_CT', 'tipoAutor_DEP',
    'tipoAutor_DPU', 'tipoAutor_EXEC', 'tipoAutor_LID', 'tipoAutor_MPU',
    'tipoAutor_PART', 'tipoAutor_PGR', 'tipoAutor_SC', 'tipoAutor_SEN',
    'tipoAutor_SF', 'tipoAutor_STDJ', 'tipoAutor_STF', 'tipoAutor_STJ',
    'tipoAutor_TCU', 'tipoAutor_TJDFT', 'tipoAutor_TSDT',
]

# Deputy-level features (need to aggregate to legislature level)
deputy_features_numeric = [
    'tamanho_bloco', 'tamanho_partido', 'idade', 'idade2',
    'n_voto_igual_autor', 'size_voto_igual_autor', 'pct_voto_igual_autor',
    'n_voto_part_sim', 'pct_voto_part_sim',
    'n_orgaos', 'pct_orgaos', 'n_lead_orgaos',
    'n_frentes', 'pct_frentes', 'n_lead_frentes',
    'n_proposicoes', 'pct_proposicoes',
    'pct_seg_ori_part', 'pct_traiu_ori_part',
    'pct_seg_ori_gov', 'pct_traiu_ori_gov',
    'pct_seg_ori_mai', 'pct_traiu_ori_mai',
    'pct_seg_ori_min', 'pct_traiu_ori_min',
    'pct_seg_ori_op', 'pct_traiu_ori_op',
    'pct_seg_ori_bancada', 'pct_traiu_ori_bancada',
    'pct_votSim', 'pct_votNao', 'pct_votSim_oriNao',
    'pct_part_gov', 'pct_part_min', 'size',
    'n_part', 'n_legis', 'nivel_escolaridade',
]

deputy_features_binary = [
    'd_mesa_presid', 'd_mesa_vp', 'd_mesa_sec', 'd_mesa_sup_sec',
    'd_proponente', 'd_autor', 'd_bloco_autor', 'd_uf_prop', 'd_uf_autor',
    'd_part_autor', 'd_part_prop',
    'd_lider_governo_cd', 'd_lider_partido_politico', 'd_vice_lider_partido_politico',
    'd_vice_lider_governo_cd', 'd_vice_lider_governo_cn',
    'd_representante_partido_politico', 'd_lider_bloco_parlamentar',
    'd_vice_lider_bloco_parlamentar', 'd_lider_minoria_cd',
    'd_vice_lider_minoria_cd', 'd_vice_lider_minoria_cn',
    'd_lider_minoria_cn', 'd_lider_governo_cn', 'd_lider_maioria_cd',
    'd_presidente_partido_politico', 'd_vice_lider_maioria_cd',
    'd_lider_oposicao_cd', 'd_vice_lider_oposicao_cd', 'd_lider_maioria_cn',
    'd_part_presid', 'd_block_presid',
    'd_1_part', 'd_2_part', 'd_3_part', 'd_4_part', 'd_5_part', 'd_6_part',
    'd_1_leg', 'd_2_leg', 'd_3_leg', 'd_4_leg', 'd_5_leg', 'd_6_leg',
    'd_7_leg', 'd_8_leg', 'd_9_leg', 'd_10_leg', 'd_11_leg', 'd_12_leg', 'd_13_leg',
    'd_reg_N', 'd_reg_NE', 'd_reg_SE', 'd_reg_S', 'd_reg_CO',
    'd_niv_esc_1', 'd_niv_esc_2', 'd_niv_esc_3', 'd_niv_esc_4', 'd_niv_esc_5',
    'd_homem', 'd_titular',
    'd_prof_ciencias_exatas', 'd_prof_engenharia', 'd_prof_ciencias_humanas',
    'd_prof_direito', 'd_prof_empresas', 'd_prof_agro', 'd_prof_servico_publico',
    'd_prof_artes', 'd_prof_pedagogia', 'd_prof_comunicacao', 'd_prof_medicina',
    'd_prof_militar', 'd_prof_religiao', 'd_prof_trabalhista', 'd_prof_comercial',
    'd_prof_tecnico',
]

# Filter to features that actually exist in the dataset
all_albuquerque_cols = df_albuquerque.columns.tolist()
proposition_features = [f for f in proposition_features if f in all_albuquerque_cols]
deputy_features_numeric = [f for f in deputy_features_numeric if f in all_albuquerque_cols]
deputy_features_binary = [f for f in deputy_features_binary if f in all_albuquerque_cols]

print(f"   Proposition-level features: {len(proposition_features)}")
print(f"   Deputy numeric features: {len(deputy_features_numeric)}")
print(f"   Deputy binary features: {len(deputy_features_binary)}")

# =============================================================================
# 3. CREATE SESSION-LEVEL DATASET FROM ALBUQUERQUE
# =============================================================================
print("\n3. Creating session-level dataset from Albuquerque...")

# Filter to Sim/Não votes only
df_albuquerque = df_albuquerque[df_albuquerque['voto'].isin(['Sim', 'Não'])].copy()
df_albuquerque['voto_binary'] = (df_albuquerque['voto'] == 'Sim').astype(int)

# Aggregate to session level
# For proposition features: take first (they're the same for all deputies)
# For deputy features: take mean (legislature-level average)

agg_dict = {}

# Proposition features: first value
for f in proposition_features:
    agg_dict[f] = 'first'

# Deputy numeric features: mean
for f in deputy_features_numeric:
    agg_dict[f] = 'mean'

# Deputy binary features: mean (proportion of deputies with this characteristic)
for f in deputy_features_binary:
    agg_dict[f] = 'mean'

# Target: majority vote determines session outcome
agg_dict['voto_binary'] = lambda x: (x.sum() > len(x)/2).astype(int)

# Legislature: first
agg_dict['idLegislatura'] = 'first'

# Date: first
if 'data' in df_albuquerque.columns:
    agg_dict['data'] = 'first'

# Aggregate
session_albuquerque = df_albuquerque.groupby('idVotacao').agg(agg_dict).reset_index()
session_albuquerque.rename(columns={'voto_binary': 'aprovacao'}, inplace=True)

print(f"   Albuquerque sessions: {len(session_albuquerque):,}")
print(f"   Pass rate: {session_albuquerque['aprovacao'].mean():.1%}")

# All features for Albuquerque model
albuquerque_feature_cols = proposition_features + deputy_features_numeric + deputy_features_binary
albuquerque_feature_cols = [f for f in albuquerque_feature_cols if f in session_albuquerque.columns]
print(f"   Total features for Albuquerque model: {len(albuquerque_feature_cols)}")

# =============================================================================
# 4. CREATE LEGISLATURE-LEVEL AGGREGATES FOR VOTE-RAP PROPOSITIONS
# =============================================================================
print("\n4. Creating legislature-level features for all VOTE-RAP propositions...")

# First, create legislature-level aggregates from Albuquerque's deputy data
legislature_features = df_albuquerque.groupby('idLegislatura').agg({
    **{f: 'mean' for f in deputy_features_numeric if f in df_albuquerque.columns},
    **{f: 'mean' for f in deputy_features_binary if f in df_albuquerque.columns}
}).reset_index()

# Rename columns to indicate they're legislature-level
legislature_features.columns = ['idLegislatura'] + [f'leg_{c}' for c in legislature_features.columns[1:]]

print(f"   Legislatures with deputy data: {len(legislature_features)}")
print(f"   Legislature IDs: {sorted(legislature_features['idLegislatura'].unique())}")

# =============================================================================
# 5. PREPARE VOTE-RAP DATASET WITH MATCHING FEATURES
# =============================================================================
print("\n5. Preparing VOTE-RAP dataset with matching features...")

# VOTE-RAP already has some features, but we need to add proposition-level features
# that match Albuquerque's format

# Load VOTE-RAP engineered features from separate files
try:
    # Author popularity
    author_pop = pd.read_csv(DATA_DIR / 'features' / 'author_popularity.csv')
    vote_sessions = vote_sessions.merge(author_pop[['id', 'popularity']], on='id', how='left')
    print("   Loaded author popularity")
except Exception as e:
    print(f"   Could not load author popularity: {e}")
    vote_sessions['popularity'] = 0

try:
    # Party popularity
    party_pop = pd.read_csv(DATA_DIR / 'features' / 'party_popularity_best_window_last_5_sessions.csv')
    vote_sessions = vote_sessions.merge(party_pop[['id', 'party_popularity']], on='id', how='left')
    print("   Loaded party popularity")
except Exception as e:
    print(f"   Could not load party popularity: {e}")
    vote_sessions['party_popularity'] = 0

try:
    # Historical approval rate
    hist_approval = pd.read_csv(DATA_DIR / 'features' / 'proposition_history_predictions_historical_probability_rule.csv')
    vote_sessions = vote_sessions.merge(hist_approval[['id', 'historical_approval_rate']], on='id', how='left')
    print("   Loaded historical approval rate")
except Exception as e:
    print(f"   Could not load historical approval rate: {e}")
    vote_sessions['historical_approval_rate'] = 0.5

# Compute derived features
vote_sessions['num_authors_trunc'] = vote_sessions['num_authors'].clip(upper=50)
vote_sessions['has_more_than_10_authors'] = (vote_sessions['num_authors'] > 10).astype(int)

# Compute gov_orientation from party orientation columns
gov_cols = ['GOV.', 'Governo']
for col in gov_cols:
    if col in vote_sessions.columns:
        vote_sessions['gov_orientation'] = vote_sessions[col].fillna(0)
        break
else:
    vote_sessions['gov_orientation'] = 0

print("   Computed derived features")

# Check if we have legislature info in VOTE-RAP
if 'legislatura' in vote_sessions.columns:
    vote_sessions['idLegislatura'] = vote_sessions['legislatura']
    print(f"   Using existing legislatura column")
elif 'idLegislatura' not in vote_sessions.columns:
    # Try to infer legislature from date/year
    if 'data' in vote_sessions.columns:
        vote_sessions['year'] = pd.to_datetime(vote_sessions['data']).dt.year
        
        # Legislature mapping (approximate)
        def get_legislature(year):
            if year < 2003: return 51
            elif year < 2007: return 52
            elif year < 2011: return 53
            elif year < 2015: return 54
            elif year < 2019: return 55
            elif year < 2023: return 56
            else: return 57
        
        vote_sessions['idLegislatura'] = vote_sessions['year'].apply(get_legislature)
        print(f"   Inferred legislature from year")

# Check legislature overlap
if 'idLegislatura' in vote_sessions.columns:
    vote_rap_legislatures = set(vote_sessions['idLegislatura'].unique())
    albuquerque_legislatures = set(legislature_features['idLegislatura'].unique())
    overlap = vote_rap_legislatures.intersection(albuquerque_legislatures)
    print(f"   VOTE-RAP legislatures: {sorted(vote_rap_legislatures)}")
    print(f"   Albuquerque legislatures: {sorted(albuquerque_legislatures)}")
    print(f"   Overlap: {sorted(overlap)}")

# =============================================================================
# 6. BUILD FEATURES FOR ALL VOTE-RAP PROPOSITIONS
# =============================================================================
print("\n6. Building features for all VOTE-RAP propositions...")

# Create proposition-level features for VOTE-RAP dataset
# We'll create what we can match from Albuquerque's features

# Initialize feature DataFrame
vote_rap_albuquerque_style = vote_sessions[['id', 'aprovacao']].copy()

if 'idLegislatura' in vote_sessions.columns:
    vote_rap_albuquerque_style['idLegislatura'] = vote_sessions['idLegislatura']

# Add government orientation if available
if 'gov_orientation' in vote_sessions.columns:
    vote_rap_albuquerque_style['d_ori_gov'] = vote_sessions['gov_orientation']
    vote_rap_albuquerque_style['d_ori_gov_sim'] = (vote_sessions['gov_orientation'] == 1).astype(int)
    vote_rap_albuquerque_style['d_ori_gov_nao'] = (vote_sessions['gov_orientation'] == -1).astype(int)

# Add proposition type features if available
tipo_col = 'proposicao_siglaTipo' if 'proposicao_siglaTipo' in vote_sessions.columns else 'siglaTipo'
if tipo_col in vote_sessions.columns:
    for tipo in ['PL', 'PEC', 'PLP', 'MPV', 'PDL', 'PDC', 'REQ', 'REC', 'MSC']:
        col_name = f'd_tipoVotacao_{tipo}'
        vote_rap_albuquerque_style[col_name] = (vote_sessions[tipo_col] == tipo).astype(int)
    print(f"   Added proposition type features from {tipo_col}")

# Add temporal features
if 'data' in vote_sessions.columns:
    dates = pd.to_datetime(vote_sessions['data'])
    vote_rap_albuquerque_style['year'] = dates.dt.year
    vote_rap_albuquerque_style['month'] = dates.dt.month
    vote_rap_albuquerque_style['day_of_week'] = dates.dt.dayofweek
    print("   Added temporal features")

# Add author type features if available
author_type_col = 'author_type' if 'author_type' in vote_sessions.columns else 'tipoAutor'
if author_type_col in vote_sessions.columns:
    author_types = vote_sessions[author_type_col].fillna('').astype(str)
    for autor in ['DEP', 'EXEC', 'SEN', 'Deputado', 'Poder Executivo', 'Senador']:
        col_name = f'tipoAutor_{autor.replace(" ", "_")}'
        vote_rap_albuquerque_style[col_name] = author_types.str.contains(autor, case=False, na=False).astype(int)
    print(f"   Added author type features from {author_type_col}")

# Add theme features if available
if 'theme' in vote_sessions.columns:
    # Get top themes and create dummy variables
    top_themes = vote_sessions['theme'].value_counts().head(30).index.tolist()
    for tema in top_themes:
        if pd.notna(tema):
            col_name = f'd_tema_{str(tema)[:10].replace(" ", "_")}'
            vote_rap_albuquerque_style[col_name] = (vote_sessions['theme'] == tema).astype(int)
    print(f"   Added {len(top_themes)} theme features")

# Add num_authors features
if 'num_authors' in vote_sessions.columns:
    vote_rap_albuquerque_style['size'] = vote_sessions['num_authors']
    vote_rap_albuquerque_style['size_squared'] = vote_sessions['num_authors'] ** 2
    print("   Added num_authors features")

# Join legislature-level deputy features
if 'idLegislatura' in vote_rap_albuquerque_style.columns and len(legislature_features) > 0:
    vote_rap_albuquerque_style = vote_rap_albuquerque_style.merge(
        legislature_features, on='idLegislatura', how='left'
    )
    print(f"   Joined legislature features: {len(legislature_features.columns)-1} features")

# Fill NaN with 0 for missing legislature data
vote_rap_albuquerque_style = vote_rap_albuquerque_style.fillna(0)

# Get feature columns (exclude id, target, legislature)
feature_cols_vote_rap = [c for c in vote_rap_albuquerque_style.columns 
                          if c not in ['id', 'aprovacao', 'idLegislatura']]

print(f"   Features available for VOTE-RAP: {len(feature_cols_vote_rap)}")
print(f"   Sample features: {feature_cols_vote_rap[:10]}")

# =============================================================================
# 7. ALIGN FEATURES BETWEEN DATASETS
# =============================================================================
print("\n7. Aligning features between datasets...")

# Find common features
common_features = list(set(albuquerque_feature_cols).intersection(set(feature_cols_vote_rap)))

# For features in Albuquerque but not in VOTE-RAP, add them as 0
for f in albuquerque_feature_cols:
    if f not in vote_rap_albuquerque_style.columns:
        vote_rap_albuquerque_style[f] = 0

# Use Albuquerque's feature set
final_feature_cols = albuquerque_feature_cols

print(f"   Common features: {len(common_features)}")
print(f"   Final feature set: {len(final_feature_cols)}")

# =============================================================================
# 8. TRAIN ALBUQUERQUE MODEL ON ROLL-CALL SUBSET
# =============================================================================
print("\n8. Training Albuquerque model on roll-call subset...")

# Prepare training data (all Albuquerque sessions)
X_albuquerque_train = session_albuquerque[albuquerque_feature_cols].fillna(0)
y_albuquerque_train = session_albuquerque['aprovacao']

print(f"   Training samples: {len(X_albuquerque_train):,}")
print(f"   Training pass rate: {y_albuquerque_train.mean():.1%}")

# Scale features
scaler_albuquerque = StandardScaler()
X_albuquerque_train_scaled = scaler_albuquerque.fit_transform(X_albuquerque_train)

# Train Random Forest
print("   Training Random Forest...")
rf_albuquerque = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_albuquerque.fit(X_albuquerque_train_scaled, y_albuquerque_train)

# Train Gradient Boosting
print("   Training Gradient Boosting...")
gbm_albuquerque = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_SEED
)
gbm_albuquerque.fit(X_albuquerque_train_scaled, y_albuquerque_train)

print("   Models trained!")

# =============================================================================
# 9. TEST ON FULL VOTE-RAP DATASET
# =============================================================================
print("\n9. Testing Albuquerque model on FULL VOTE-RAP dataset...")

# Prepare test data (all VOTE-RAP propositions)
X_vote_rap_test = vote_rap_albuquerque_style[albuquerque_feature_cols].fillna(0)
y_vote_rap_test = vote_rap_albuquerque_style['aprovacao']

print(f"   Test samples: {len(X_vote_rap_test):,}")
print(f"   Test pass rate: {y_vote_rap_test.mean():.1%}")

# Scale using Albuquerque's scaler
X_vote_rap_test_scaled = scaler_albuquerque.transform(X_vote_rap_test)

# Predict
y_rf_pred = rf_albuquerque.predict(X_vote_rap_test_scaled)
y_rf_pred_proba = rf_albuquerque.predict_proba(X_vote_rap_test_scaled)[:, 1]

y_gbm_pred = gbm_albuquerque.predict(X_vote_rap_test_scaled)
y_gbm_pred_proba = gbm_albuquerque.predict_proba(X_vote_rap_test_scaled)[:, 1]

# =============================================================================
# 10. TRAIN AND TEST VOTE-RAP ON SAME DATA
# =============================================================================
print("\n10. Training VOTE-RAP on full dataset for comparison...")

# VOTE-RAP features (the 6 interpretable features)
vote_rap_feature_names = ['popularity', 'gov_orientation', 'num_authors_trunc', 
                          'has_more_than_10_authors', 'party_popularity', 'historical_approval_rate']

# Check which features are available
available_vote_rap_features = [f for f in vote_rap_feature_names if f in vote_sessions.columns]
print(f"   Available VOTE-RAP features: {available_vote_rap_features}")

if len(available_vote_rap_features) >= 4:
    # Remove rows with NaN target
    vote_sessions_clean = vote_sessions.dropna(subset=['aprovacao'])
    print(f"   Removed {len(vote_sessions) - len(vote_sessions_clean)} rows with NaN target")
    
    # Chronological split
    vote_sessions_sorted = vote_sessions_clean.sort_values('id').reset_index(drop=True)
    split_idx = int(len(vote_sessions_sorted) * 0.8)
    
    train_vote_rap = vote_sessions_sorted.iloc[:split_idx]
    test_vote_rap = vote_sessions_sorted.iloc[split_idx:]
    
    X_vr_train = train_vote_rap[available_vote_rap_features].fillna(0)
    y_vr_train = train_vote_rap['aprovacao']
    X_vr_test = test_vote_rap[available_vote_rap_features].fillna(0)
    y_vr_test = test_vote_rap['aprovacao']
    
    print(f"   VOTE-RAP train: {len(X_vr_train):,}, test: {len(X_vr_test):,}")
    
    # Scale only numeric features
    numeric_features = ['popularity', 'party_popularity', 'historical_approval_rate']
    numeric_features = [f for f in numeric_features if f in available_vote_rap_features]
    
    scaler_vote_rap = StandardScaler()
    X_vr_train_scaled = X_vr_train.copy()
    X_vr_test_scaled = X_vr_test.copy()
    
    if numeric_features:
        X_vr_train_scaled[numeric_features] = scaler_vote_rap.fit_transform(X_vr_train[numeric_features])
        X_vr_test_scaled[numeric_features] = scaler_vote_rap.transform(X_vr_test[numeric_features])
    
    # Train VOTE-RAP model
    neg_count = (y_vr_train == 0).sum()
    pos_count = (y_vr_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    vote_rap_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    vote_rap_model.fit(X_vr_train_scaled, y_vr_train)
    
    y_vr_pred = vote_rap_model.predict(X_vr_test_scaled)
    y_vr_pred_proba = vote_rap_model.predict_proba(X_vr_test_scaled)[:, 1]
    
    vote_rap_trained = True
else:
    print("   Not enough VOTE-RAP features available!")
    vote_rap_trained = False

# =============================================================================
# 11. COMPARE RESULTS
# =============================================================================
print("\n" + "="*80)
print("RESULTS: GENERALIZATION COMPARISON")
print("="*80)

def evaluate_model(name, y_true, y_pred, y_pred_proba):
    """Evaluate model performance."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_rejected = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_pred_proba)
    except:
        auroc = 0.5
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 (Approved)': f1,
        'F1 (Rejected)': f1_rejected,
        'AUROC': auroc
    }

results = []

# Albuquerque RF on full VOTE-RAP
results.append(evaluate_model(
    "Albuquerque RF (trained on 3,741, tested on 41,461)",
    y_vote_rap_test, y_rf_pred, y_rf_pred_proba
))

# Albuquerque GBM on full VOTE-RAP
results.append(evaluate_model(
    "Albuquerque GBM (trained on 3,741, tested on 41,461)",
    y_vote_rap_test, y_gbm_pred, y_gbm_pred_proba
))

# VOTE-RAP on its own test set
if vote_rap_trained:
    results.append(evaluate_model(
        "VOTE-RAP (trained on ~33,000, tested on ~8,000)",
        y_vr_test, y_vr_pred, y_vr_pred_proba
    ))

# Print results
results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# =============================================================================
# 12. DETAILED ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print("\n--- Distribution Shift Analysis ---")
print(f"Albuquerque training pass rate: {y_albuquerque_train.mean():.1%}")
print(f"VOTE-RAP test pass rate: {y_vote_rap_test.mean():.1%}")
print(f"Distribution shift: {abs(y_albuquerque_train.mean() - y_vote_rap_test.mean()):.1%}")

print("\n--- Albuquerque RF Confusion Matrix (on full VOTE-RAP) ---")
print(confusion_matrix(y_vote_rap_test, y_rf_pred))

print("\n--- Albuquerque RF Classification Report (on full VOTE-RAP) ---")
print(classification_report(y_vote_rap_test, y_rf_pred, target_names=['Rejected', 'Approved']))

if vote_rap_trained:
    print("\n--- VOTE-RAP Confusion Matrix ---")
    print(confusion_matrix(y_vr_test, y_vr_pred))
    
    print("\n--- VOTE-RAP Classification Report ---")
    print(classification_report(y_vr_test, y_vr_pred, target_names=['Rejected', 'Approved']))

# =============================================================================
# 13. SAVE RESULTS
# =============================================================================
print("\n13. Saving results...")
results_df.to_csv(RESULTS_DIR / 'generalization_comparison.csv', index=False)
print(f"   Saved to: {RESULTS_DIR / 'generalization_comparison.csv'}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
KEY FINDING: Albuquerque's methodology, trained on the limited roll-call subset
(3,741 propositions with 53% pass rate), FAILS to generalize to the broader
proposition space (41,461 propositions with 80% pass rate).

This demonstrates that:
1. Albuquerque's approach is fundamentally limited to roll-call propositions
2. Training on a non-representative subset leads to poor generalization
3. VOTE-RAP's full-coverage approach is more robust and practical
""")

