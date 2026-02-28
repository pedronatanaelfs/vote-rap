"""
Analyze what percentage of VOTE-RAP propositions have individual vote data available.

This script checks:
1. How many sessions are in VOTE-RAP's dataset (vote_sessions_full.csv)
2. How many of those sessions have individual deputy vote data available
3. The percentage of sessions without individual vote data
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directories to path
# Script is in: scripts/03-comparisons/albuquerque/
# Need to go up 3 levels to get to repo root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ALBUQUERQUE_DIR = BASE_DIR / "scripts" / "03-comparisons" / "albuquerque"

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"Checking if DATA_DIR exists: {DATA_DIR.exists()}")

print("="*80)
print("ANALYZING INDIVIDUAL VOTE DATA COVERAGE FOR VOTE-RAP")
print("="*80)

# 1. Load VOTE-RAP sessions
print("\n1. Loading VOTE-RAP sessions...")
vote_rap_sessions = pd.read_csv(DATA_DIR / "vote_sessions_full.csv")
print(f"   Total VOTE-RAP sessions: {len(vote_rap_sessions):,}")

# Get unique session IDs
vote_rap_session_ids = set(vote_rap_sessions['id'].astype(str).unique())
print(f"   Unique session IDs: {len(vote_rap_session_ids):,}")

# 2. Check if Albuquerque's dataset is available
print("\n2. Checking Albuquerque's dataset for individual votes...")
try:
    albuquerque_features = pd.read_csv(
        ALBUQUERQUE_DIR / "features.csv",
        sep=';',
        nrows=1000  # Just check structure first
    )
    print(f"   Albuquerque dataset columns: {list(albuquerque_features.columns)[:10]}...")
    
    # Check if idVotacao exists
    if 'idVotacao' in albuquerque_features.columns:
        print("   [OK] Found 'idVotacao' column in Albuquerque dataset")
        
        # Load full dataset to get all session IDs
        print("\n3. Loading full Albuquerque dataset to get session IDs...")
        albuquerque_full = pd.read_csv(
            ALBUQUERQUE_DIR / "features.csv",
            sep=';',
            usecols=['idVotacao'],
            low_memory=False
        )
        
        albuquerque_session_ids = set(albuquerque_full['idVotacao'].astype(str).unique())
        print(f"   Unique session IDs in Albuquerque dataset: {len(albuquerque_session_ids):,}")
        
        # 4. Compare session IDs
        print("\n4. Comparing session IDs...")
        sessions_with_votes = vote_rap_session_ids.intersection(albuquerque_session_ids)
        sessions_without_votes = vote_rap_session_ids - albuquerque_session_ids
        
        print(f"\n   Sessions WITH individual vote data: {len(sessions_with_votes):,}")
        print(f"   Sessions WITHOUT individual vote data: {len(sessions_without_votes):,}")
        print(f"   Total VOTE-RAP sessions: {len(vote_rap_session_ids):,}")
        
        pct_with_votes = (len(sessions_with_votes) / len(vote_rap_session_ids)) * 100
        pct_without_votes = (len(sessions_without_votes) / len(vote_rap_session_ids)) * 100
        
        print(f"\n   Percentage WITH individual votes: {pct_with_votes:.2f}%")
        print(f"   Percentage WITHOUT individual votes: {pct_without_votes:.2f}%")
        
        # 5. Analyze sessions without votes
        if len(sessions_without_votes) > 0:
            print("\n5. Analyzing sessions without individual vote data...")
            sessions_no_votes_df = vote_rap_sessions[
                vote_rap_sessions['id'].astype(str).isin(sessions_without_votes)
            ]
            
            print(f"   Date range (no votes): {sessions_no_votes_df['data'].min()} to {sessions_no_votes_df['data'].max()}")
            print(f"   Approval rate (no votes): {sessions_no_votes_df['aprovacao'].mean():.2%}")
            
            # Check by year
            if 'year' in sessions_no_votes_df.columns:
                print("\n   Distribution by year (sessions without votes):")
                year_counts = sessions_no_votes_df['year'].value_counts().sort_index()
                for year, count in year_counts.head(10).items():
                    print(f"     {year}: {count:,} sessions")
        
        # 6. Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"VOTE-RAP uses {len(vote_rap_session_ids):,} voting sessions")
        print(f"Of these, {len(sessions_with_votes):,} ({pct_with_votes:.2f}%) have individual deputy vote data")
        print(f"And {len(sessions_without_votes):,} ({pct_without_votes:.2f}%) do NOT have individual deputy vote data")
        print("\nThis means Albuquerque's approach (predicting individual votes) can only be")
        print(f"applied to {pct_with_votes:.2f}% of VOTE-RAP's dataset.")
        
    else:
        print("   [X] 'idVotacao' column not found in Albuquerque dataset")
        print("   Available columns:", list(albuquerque_features.columns)[:20])
        
except FileNotFoundError:
    print("   [X] Albuquerque features.csv not found")
    print("   Cannot compare - need features.csv in scripts/03-comparisons/albuquerque/")
except Exception as e:
    print(f"   [ERROR] Error loading Albuquerque dataset: {e}")

print("\n" + "="*80)

