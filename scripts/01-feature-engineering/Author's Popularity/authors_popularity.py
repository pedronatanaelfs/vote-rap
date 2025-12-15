import pandas as pd
import numpy as np
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path

# Setup output logging to file
OUTPUT_DIR = Path(__file__).parent
output_file = OUTPUT_DIR / "authors_popularity_output.txt"
log_file_handle = open(output_file, 'w', encoding='utf-8')
log_file_handle.write(f"VOTE-RAP - Author Popularity Feature Engineering\n")
log_file_handle.write(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file_handle.write("=" * 80 + "\n\n")
log_file_handle.flush()

# Function to print and log progress with timestamp
def log_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    print(f"[{timestamp}] {message}")
    log_file_handle.write(log_message)
    log_file_handle.flush()
    sys.stdout.flush()  # Force immediate output to console

# Checks if the output directories exist
def ensure_output_directory(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        log_progress(f"Created directory: {directory}")

# Start of execution
start_time = time.time()
log_progress("Starting author popularity calculation...")

try:
    # 1) Load the vote sessions data
    log_progress("Loading vote session data...")
    df_sessions = pd.read_csv("../../../data/vote_sessions_full.csv")
    if df_sessions.empty:
        log_progress("Error: Failed to load vote session data")
        sys.exit(1)
    else:
        log_progress(f"Vote session data loaded successfully: {len(df_sessions)} rows")
        log_progress(f"Available columns: {', '.join(df_sessions.columns)}")

    # Add a column for session importance/relevance (based on number of votes)
    log_progress("Calculating session importance metrics...")
    session_votes_count = df_sessions.groupby('id').size().reset_index(name='total_votes')
    log_progress(f"Vote count per session calculated: {len(session_votes_count)} unique sessions")
    
    df_sessions = df_sessions.merge(session_votes_count, on='id', how='left')
    log_progress(f"Merge completed. Shape after merge: {df_sessions.shape}")

    # 2) Drop duplicate sessions
    log_progress("Removing duplicate sessions...")
    df_sessions_unique = df_sessions.drop_duplicates(subset=['id'])
    log_progress(f"Number of unique sessions: {len(df_sessions_unique)}")

    # 3) Define the path pattern and years
    log_progress("Preparing to load vote data...")
    data_path = "../../../data/voting/votes/votacoesVotos-{year}.csv"
    years = range(2003, 2025)

    # 4) Load and concatenate all datasets
    log_progress("Loading vote data year by year...")
    dfs = []
    total_votes = 0
    
    for year in years:
        try:
            year_path = data_path.format(year=year)
            if not os.path.exists(year_path):
                log_progress(f"File not found for year {year}: {year_path}")
                continue
                
            log_progress(f"Loading data for year {year}...")
            df = pd.read_csv(year_path, delimiter=';', quotechar='"')
            dfs.append(df)
            total_votes += len(df)
            log_progress(f"Loaded {len(df)} votes for {year}. Total so far: {total_votes}")
        except Exception as e:
            log_progress(f"Error loading data for {year}: {e}")
            traceback.print_exc()

    log_progress(f"Concatenating {len(dfs)} vote dataframes...")
    df_votes = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    log_progress(f"Total votes loaded: {len(df_votes)}")

    # 5) Filter df_votes based on df_sessions
    if 'df_sessions' in locals() and 'id' in df_sessions_unique.columns and not df_votes.empty:
        log_progress("Filtering votes and merging with session data...")
        log_progress(f"Before filtering: {len(df_votes)} rows")
        
        session_ids = set(df_sessions_unique['id'])
        log_progress(f"Total session IDs for filtering: {len(session_ids)}")
        
        # Show sample of IDs for debugging
        sample_ids = list(session_ids)[:5] if len(session_ids) >= 5 else list(session_ids)
        log_progress(f"Sample session IDs: {sample_ids}")
        
        # Check data types
        log_progress(f"'idVotacao' dtype in df_votes: {df_votes['idVotacao'].dtype}")
        log_progress(f"'id' dtype in df_sessions_unique: {df_sessions_unique['id'].dtype}")
        
        # Convert data types if necessary
        if df_votes['idVotacao'].dtype != df_sessions_unique['id'].dtype:
            log_progress("Converting data types for compatibility...")
            if df_sessions_unique['id'].dtype == 'int64':
                df_votes['idVotacao'] = pd.to_numeric(df_votes['idVotacao'], errors='coerce')
            else:
                df_sessions_unique['id'] = df_sessions_unique['id'].astype(str)
                df_votes['idVotacao'] = df_votes['idVotacao'].astype(str)
        
        # Filter using the most efficient method
        df_votes_selected = df_votes[df_votes['idVotacao'].isin(session_ids)].copy()
        log_progress(f"After filtering: {len(df_votes_selected)} rows")
        
        # Check if filtering worked
        if len(df_votes_selected) == 0:
            log_progress("WARNING: No rows after filtering. Investigating data...")
            common_ids = set(df_votes['idVotacao'].unique()) & session_ids
            log_progress(f"Common IDs: {len(common_ids)}")
            
            if len(common_ids) == 0:
                log_progress("No common IDs between datasets. Showing sample vote IDs...")
                sample_vote_ids = list(df_votes['idVotacao'].unique())[:5]
                log_progress(f"Sample voting IDs: {sample_vote_ids}")
                
                # Attempt with string data types
                log_progress("Attempting data type conversion to string...")
                df_votes['idVotacao_str'] = df_votes['idVotacao'].astype(str)
                df_sessions_unique['id_str'] = df_sessions_unique['id'].astype(str)
                common_str_ids = set(df_votes['idVotacao_str'].unique()) & set(df_sessions_unique['id_str'].unique())
                log_progress(f"Common IDs after string conversion: {len(common_str_ids)}")
                
                if len(common_str_ids) > 0:
                    log_progress("Using IDs converted to string...")
                    df_votes_selected = df_votes[df_votes['idVotacao_str'].isin(common_str_ids)].copy()
                    log_progress(f"After filtering with strings: {len(df_votes_selected)} rows")
                else:
                    log_progress("CRITICAL ERROR: Could not find matches between datasets")
                    sys.exit(1)
        
        # Merge with author and relevance information
        log_progress("Merging with 'author' column and relevance metrics...")
        if len(df_votes_selected) > 0:
            merge_columns = ['id', 'author']
            if 'total_votes' in df_sessions_unique.columns:
                merge_columns.append('total_votes')
                
            df_votes_selected = df_votes_selected.merge(
                df_sessions_unique[merge_columns], 
                left_on='idVotacao', right_on='id', how='left'
            )
            log_progress(f"After merge: {len(df_votes_selected)} rows")
            
            if 'id' in df_votes_selected.columns:
                df_votes_selected.drop(columns=['id'], inplace=True)

    # 6) Convert 'dataHoraVoto' to datetime
    log_progress("Converting 'dataHoraVoto' to datetime...")
    try:
        df_votes_selected['dataHoraVoto'] = pd.to_datetime(df_votes_selected['dataHoraVoto'], errors='coerce')
        # Remove rows with invalid dates
        invalid_dates = df_votes_selected['dataHoraVoto'].isna().sum()
        if invalid_dates > 0:
            log_progress(f"Removing {invalid_dates} rows with invalid dates")
            df_votes_selected = df_votes_selected.dropna(subset=['dataHoraVoto'])
    except Exception as e:
        log_progress(f"Error converting dates: {e}")
        traceback.print_exc()

    # 7) Sort votes by 'dataHoraVoto' for chronological processing
    log_progress("Sorting votes by 'dataHoraVoto' for chronological processing...")
    df_votes_selected = df_votes_selected.sort_values(by='dataHoraVoto')

    # 8) Initialize parameters for improved popularity calculation
    log_progress("Setting parameters for popularity calculation...")
    TIME_WINDOW_DAYS = 365  # Rolling window of 1 year
    HALF_LIFE_DAYS = 90     # Half-life of 3 months for exponential decay
    MIN_VOTES_THRESHOLD = 5 # Minimum votes needed for reliable popularity
    DEFAULT_POPULARITY = 0.5 # Default popularity for new authors
    
    # Adjust parameters for faster processing if dataset is too large
    if len(df_votes_selected) > 1000000:
        log_progress(f"Very large dataset ({len(df_votes_selected)} votes). Optimizing parameters...")
        PROCESS_CHUNK_SIZE = 10000  # Process in chunks to avoid memory issues
    else:
        PROCESS_CHUNK_SIZE = len(df_votes_selected)

    # 9) Initialize data structures for efficient calculation
    log_progress("Initializing data structures for efficient calculation...")
    popularity_records = []
    author_votes = defaultdict(list)

    # 10) Process votes in chronological order - First pass
    total_rows = len(df_votes_selected)
    log_progress(f"Processing {total_rows} votes to calculate author popularity...")

    # First step: collect vote data
    log_progress("First pass: collecting vote data...")
    
    # Normalize vote importance in advance
    max_votes = df_sessions_unique['total_votes'].max() if 'total_votes' in df_sessions_unique.columns else 1
    log_progress(f"Max votes value for normalization: {max_votes}")
    
    # Process in chunks to monitor progress
    chunk_size = 100000
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        log_progress(f"Processing chunk {chunk_start}-{chunk_end} of {total_rows}...")
        
        chunk = df_votes_selected.iloc[chunk_start:chunk_end]
        
        for _, row in chunk.iterrows():
            author = row['author']
            vote = row['voto']
            vote_time = row['dataHoraVoto']
            session_id = row['idVotacao']
            
            # Avoid error if total_votes is not present
            if 'total_votes' in row:
                vote_importance = row['total_votes'] / max_votes  # Normalized importance
            else:
                vote_importance = 1.0  # Default importance if missing
            
            is_yes = 1 if isinstance(vote, str) and vote.lower() == 'sim' else 0
            
            # Store the vote with its metadata
            author_votes[author].append({
                'time': vote_time,
                'is_yes': is_yes,
                'importance': vote_importance,
                'session_id': session_id
            })
            
        log_progress(f"Processed {chunk_end} / {total_rows} votes. Unique authors: {len(author_votes)}")
        
    log_progress("First pass complete.")

    # Second pass: calculate time/importance-weighted popularity
    log_progress("Second pass: calculating improved popularity metrics...")
    
    # Global metrics for progress logging in second pass
    second_pass_start = time.time()
    total_authors = len(author_votes)
    total_votes_all = sum(len(v) for v in author_votes.values()) if total_authors > 0 else 0
    processed_votes_all = 0
    progress_log_interval = max(10000, total_votes_all // 100) if total_votes_all > 0 else 10000  # ~100 logs
    log_progress(f"Total authors: {total_authors}. Total votes to process in 2nd pass: {total_votes_all}")
    
    # Process authors in batches to show progress
    authors_processed = 0
    
    for author, votes in author_votes.items():
        # Progress monitoring
        authors_processed += 1
        if authors_processed % 100 == 0:
            elapsed = time.time() - start_time
            log_progress(f"Processing author {authors_processed}/{total_authors} ({(authors_processed/total_authors)*100:.1f}%). Elapsed time: {elapsed:.1f}s")
        
        # Sort votes by time for each author
        votes.sort(key=lambda x: x['time'])
        
        # Constants and structures for amortized O(1) updates
        total_votes = len(votes)
        lambda_decay = np.log(2) / HALF_LIFE_DAYS
        A_weighted_sum = 0.0  # sum of w_j * exp(-lambda * (t_i - t_j))
        B_weighted_sum = 0.0  # sum of w_j * y_j * exp(-lambda * (t_i - t_j))
        window_deque = deque()  # holds previous votes within [TIME_WINDOW_DAYS]
        raw_total_votes = 0
        raw_yes_votes = 0
        recent_deque = deque()  # 180-day window for volatility
        recent_count = 0
        recent_sum = 0.0
        recent_sumsq = 0.0

        for i in range(total_votes):
            current_vote = votes[i]
            current_time = current_vote['time']
            session_id = current_vote['session_id']

            # Update exponential accumulators with the immediately previous vote
            if i > 0:
                prev_vote = votes[i - 1]
                dt_days = (current_time - prev_vote['time']).days
                if dt_days < 0:
                    dt_days = 0
                decay_factor = np.exp(-lambda_decay * dt_days)
                A_weighted_sum = decay_factor * (A_weighted_sum + prev_vote['importance'])
                B_weighted_sum = decay_factor * (B_weighted_sum + prev_vote['importance'] * prev_vote['is_yes'])

                # Add the previous vote to the 365d and 180d windows
                window_deque.append((prev_vote['time'], prev_vote['is_yes'], prev_vote['importance']))
                raw_total_votes += 1
                raw_yes_votes += prev_vote['is_yes']

                recent_deque.append((prev_vote['time'], prev_vote['is_yes']))
                recent_count += 1
                recent_sum += prev_vote['is_yes']
                recent_sumsq += prev_vote['is_yes'] ** 2

            # Remove votes outside the 365-day window (adjust sums and accumulators)
            while window_deque:
                age_days = (current_time - window_deque[0][0]).days
                if age_days <= TIME_WINDOW_DAYS:
                    break
                old_time, old_yes, old_importance = window_deque.popleft()
                raw_total_votes -= 1
                raw_yes_votes -= old_yes
                # Remove the current contribution of the expired item from the exponential accumulators
                contrib_decay = np.exp(-lambda_decay * age_days)
                A_weighted_sum -= old_importance * contrib_decay
                B_weighted_sum -= old_importance * old_yes * contrib_decay

            # Remove votes outside the 180-day window (for volatility)
            while recent_deque:
                age_days_recent = (current_time - recent_deque[0][0]).days
                if age_days_recent <= 180:
                    break
                _, old_yes_recent = recent_deque.popleft()
                recent_count -= 1
                recent_sum -= old_yes_recent
                recent_sumsq -= old_yes_recent ** 2

            # Calculate various popularity metrics
            weighted_total_votes = A_weighted_sum
            weighted_yes_votes = B_weighted_sum

            if weighted_total_votes > 0:
                weighted_popularity = weighted_yes_votes / weighted_total_votes
            else:
                weighted_popularity = DEFAULT_POPULARITY

            if raw_total_votes >= MIN_VOTES_THRESHOLD:
                raw_popularity = raw_yes_votes / raw_total_votes if raw_total_votes > 0 else DEFAULT_POPULARITY
            else:
                # Bayesian smoothing for authors with few votes
                prior_votes = MIN_VOTES_THRESHOLD - raw_total_votes
                raw_popularity = (raw_yes_votes + prior_votes * DEFAULT_POPULARITY) / (raw_total_votes + prior_votes) if (raw_total_votes + prior_votes) > 0 else DEFAULT_POPULARITY

            # Combine metrics (weights can be adjusted)
            combined_popularity = 0.7 * weighted_popularity + 0.3 * raw_popularity

            # Volatility using incremental statistics in the 180-day window
            if recent_count >= 3:
                mean_recent = recent_sum / recent_count
                var_recent = max(0.0, (recent_sumsq / recent_count) - (mean_recent ** 2))
                volatility = float(np.sqrt(var_recent))
            else:
                volatility = 0.0

            # Store metrics
            popularity_records.append({
                'author': author,
                'popularity': combined_popularity,
                'weighted_popularity': weighted_popularity,
                'raw_popularity': raw_popularity,
                'volatility': volatility,
                'vote_count': raw_total_votes,
                'date': current_time.date(),
                'idVotacao': session_id
            })

            # Global progress
            processed_votes_all += 1
            if total_votes_all > 0 and (processed_votes_all % progress_log_interval == 0 or processed_votes_all == total_votes_all):
                elapsed = time.time() - second_pass_start
                rate = processed_votes_all / elapsed if elapsed > 0 else 0
                remaining = total_votes_all - processed_votes_all
                eta_seconds = (remaining / rate) if rate > 0 else float('inf')
                pct = (processed_votes_all / total_votes_all) * 100
                log_progress(f"Second pass: {processed_votes_all}/{total_votes_all} votes ({pct:.1f}%), ETA ~ {eta_seconds/60:.1f} min")

    # 11) Create DataFrame with computed popularity
    log_progress("Creating DataFrame with improved popularity metrics...")
    df_author_popularity = pd.DataFrame(popularity_records)
    log_progress(f"DataFrame created with {len(df_author_popularity)} records")
    
    # 12) Drop duplicate voting session IDs
    log_progress("Removing duplicate voting session IDs...")
    df_author_popularity = df_author_popularity.drop_duplicates(subset=['idVotacao'])
    log_progress(f"Final popularity DataFrame: {df_author_popularity.shape}")
    
    # 13) Save the DataFrame to a CSV file
    output_path = OUTPUT_DIR / 'author_popularity.csv'
    log_progress(f"Saving popularity metrics to {output_path}...")
    df_author_popularity.to_csv(output_path, index=False, encoding='utf-8')
    log_progress(f"Popularity metrics saved successfully")
    
    elapsed_time = time.time() - start_time
    log_progress(f"Processing complete in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    log_progress(f"Output log saved to: authors_popularity_output.txt")
    
except Exception as e:
    log_progress(f"CRITICAL ERROR: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
finally:
    # Close output file
    log_file_handle.write(f"\n\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file_handle.close()