"""
Get the key numbers for the comparison methodology:
1. Total propositions with outcome labels (VOTE-RAP)
2. Propositions with roll-call votes (Albuquerque intersection)
3. Pass/fail rate in full set
4. Pass/fail rate in roll-call subset
5. Close votes analysis
"""

import pandas as pd
import os
from pathlib import Path

# Get repository root
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
print(f"Repository root: {REPO_ROOT}")
print(f"Data directory: {DATA_DIR}")

# Load VOTE-RAP full dataset (outcome-level)
vote_sessions = pd.read_csv(DATA_DIR / 'vote_sessions_full.csv')
print('='*70)
print('VOTE-RAP FULL DATASET (Outcome-level)')
print('='*70)
print(f'Total propositions with outcome labels: {len(vote_sessions):,}')
print(f'Approval rate (pass): {vote_sessions["aprovacao"].mean():.2%}')
print(f'Rejection rate (fail): {1 - vote_sessions["aprovacao"].mean():.2%}')
print(f'Pass count: {vote_sessions["aprovacao"].sum():,}')
print(f'Fail count: {int((1-vote_sessions["aprovacao"]).sum()):,}')

# Load Albuquerque dataset (roll-call votes)
df_albuquerque = pd.read_csv(SCRIPT_DIR / 'features.csv', sep=';', low_memory=False)
print()
print('='*70)
print('ALBUQUERQUE ROLL-CALL DATASET (Vote-level)')
print('='*70)
print(f'Total individual votes: {len(df_albuquerque):,}')
print(f'Unique sessions (idVotacao): {df_albuquerque["idVotacao"].nunique():,}')

# Filter to Sim/Nao only
df_albuquerque_filtered = df_albuquerque[df_albuquerque['voto'].isin(['Sim', 'Não'])].copy()
print(f'After filtering to Sim/Não: {len(df_albuquerque_filtered):,} individual votes')
print(f'Unique sessions after filter: {df_albuquerque_filtered["idVotacao"].nunique():,}')

# Calculate session-level outcomes in Albuquerque data
df_albuquerque_filtered['voto_binary'] = (df_albuquerque_filtered['voto'] == 'Sim').astype(int)
session_outcomes = df_albuquerque_filtered.groupby('idVotacao').agg(
    votos_sim=('voto_binary', 'sum'),
    total=('voto_binary', 'count')
).reset_index()
session_outcomes['passed'] = (session_outcomes['votos_sim'] > session_outcomes['total']/2).astype(int)
session_outcomes['pct_sim'] = session_outcomes['votos_sim'] / session_outcomes['total']
session_outcomes['margin'] = abs(session_outcomes['pct_sim'] - 0.5)

print()
print(f'Sessions with majority pass: {session_outcomes["passed"].sum():,}')
print(f'Sessions with majority reject: {int((1-session_outcomes["passed"]).sum()):,}')
print(f'Pass rate in roll-call subset: {session_outcomes["passed"].mean():.2%}')

# Coverage calculation
coverage = df_albuquerque_filtered["idVotacao"].nunique() / len(vote_sessions) * 100
print(f'\nCoverage: {df_albuquerque_filtered["idVotacao"].nunique():,} / {len(vote_sessions):,} = {coverage:.1f}%')

# Close votes analysis
print()
print('='*70)
print('CLOSE VOTES ANALYSIS (for aggregation deflation)')
print('='*70)
for threshold in [0.05, 0.10, 0.15, 0.20]:
    close = session_outcomes[session_outcomes['margin'] <= threshold]
    print(f'Margin <= {threshold:.0%}: {len(close):,} sessions ({len(close)/len(session_outcomes)*100:.1f}% of roll-call subset)')
    if len(close) > 0:
        print(f'  Pass rate in close votes: {close["passed"].mean():.2%}')

# Summary for paper
print()
print('='*70)
print('SUMMARY FOR PAPER')
print('='*70)
print(f"""
Key Numbers:
1. Total propositions with outcome labels: {len(vote_sessions):,}
2. Propositions with roll-call votes: {df_albuquerque_filtered["idVotacao"].nunique():,}
3. Pass rate (full set): {vote_sessions["aprovacao"].mean():.1%}
4. Pass rate (roll-call subset): {session_outcomes["passed"].mean():.1%}
5. Coverage: {coverage:.1f}%

Close-Vote Subsets (for fair Albuquerque comparison):
- Margin <= 5%:  {len(session_outcomes[session_outcomes['margin'] <= 0.05]):,} sessions
- Margin <= 10%: {len(session_outcomes[session_outcomes['margin'] <= 0.10]):,} sessions
""")

# Votes per session stats (for understanding aggregation effect)
print('='*70)
print('VOTES PER SESSION (explains aggregation inflation)')
print('='*70)
votes_per_session = df_albuquerque_filtered.groupby('idVotacao').size()
print(f'Mean votes per session: {votes_per_session.mean():.1f}')
print(f'Median votes per session: {votes_per_session.median():.1f}')
print(f'Min votes per session: {votes_per_session.min()}')
print(f'Max votes per session: {votes_per_session.max()}')

