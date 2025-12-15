# Author Popularity (Author Support) Feature Engineering

This script builds an **author-level popularity/support feature** for the VOTE-RAP dataset by using *individual roll-call vote records* and the *author associated with each voting session*. In the current implementation, “popularity” is **not** based on the session outcome (`aprovacao`). Instead, it measures the **historical share of “Sim” votes** (yes-votes) cast by legislators in sessions attributed to a given author, with **recency** and **session-size importance** weighting.

---

## What the script does (as implemented)

Given:

- A sessions table (`vote_sessions_full.csv`) that contains the voting session id (`id`) and an `author` label for that session.
- Per-year roll-call vote files (`votacoesVotos-{year}.csv`) that contain one row per legislator vote with:
  - `idVotacao` (session id),
  - `voto` (e.g., “Sim”),
  - `dataHoraVoto` (timestamp).

The script:

1. **Computes a “session size” proxy** called `total_votes` as the number of rows for each `id` in `vote_sessions_full.csv`.
2. Loads all roll-call votes from **2003–2024** and **filters** them down to only sessions present in `vote_sessions_full.csv`.
3. **Merges** votes with session metadata to attach the **session author** and the **`total_votes`** proxy to every vote row.
4. Creates an author→votes structure and runs a **two-pass** algorithm:
   - **Pass 1:** collect each author's vote history as a list of events (time, yes/no, importance, session id).
   - **Pass 2:** compute rolling-window and exponentially-decayed popularity metrics for each author over time.

Finally, it writes `author_popularity.csv`.

---

## Key implementation details

### 1) Vote label used
A vote is treated as **yes** if and only if:

- `voto` is a string and equals `"sim"` ignoring case.

Everything else becomes **0** (including “Não”, abstentions, obstructions, missing/other labels).

### 2) Recency weighting (exponential decay)
The script uses an exponential decay with half-life:

- `HALF_LIFE_DAYS = 90`

Decay parameter:

- `lambda_decay = ln(2) / HALF_LIFE_DAYS`

Recency is applied in an incremental way using differences in whole **days** between consecutive vote timestamps.

### 3) Rolling windows
- **Main window:** `TIME_WINDOW_DAYS = 365` (used for `raw_popularity`, `vote_count`, and also to expire terms from the decayed sums)
- **Volatility window:** 180 days (used for `volatility`)

### 4) Importance weighting (session-size proxy)
Each vote event is assigned an importance weight:

- `importance = total_votes / max_total_votes`

Where `max_total_votes` is the maximum `total_votes` found in `vote_sessions_full.csv`.

**Important note:** because the input vote files have one row per legislator vote, sessions with more recorded votes already contribute more events. This additional `importance` factor further emphasizes high-`total_votes` sessions.

### 5) One row per session in the final output
During computation, the script generates one popularity record per **(author, vote-event)** with an attached `idVotacao` (session id).  
At the end it runs:

- `drop_duplicates(subset=['idVotacao'])`

So the final CSV keeps **only one record per session id** (the *first* occurrence of that session id in the computed records, which corresponds to the earliest vote timestamp for that session within the author’s chronological list).

---

## Parameters

- `TIME_WINDOW_DAYS = 365`
- `HALF_LIFE_DAYS = 90`
- `MIN_VOTES_THRESHOLD = 5`
- `DEFAULT_POPULARITY = 0.5`

---

## Output file: `author_popularity.csv`

### Columns

- `author`  
  Author label associated with the session (as provided by `vote_sessions_full.csv`).

- `popularity` (**main feature**)  
  Combined popularity score:

  `popularity = 0.7 * weighted_popularity + 0.3 * raw_popularity`

- `weighted_popularity`  
  Recency-decayed and importance-weighted share of “Sim” votes (based on the author’s history).

- `raw_popularity`  
  Share of “Sim” votes within the last 365 days.
  If fewer than 5 votes exist in-window, Bayesian smoothing is applied toward `DEFAULT_POPULARITY`.

- `volatility`  
  Standard deviation of the (0/1) yes-vote indicator within the last 180 days of history.
  (Computed only when there are at least 3 votes in the 180-day window; otherwise 0.)

- `vote_count`  
  Number of historical vote events in the 365-day rolling window (raw count, not importance-weighted).

- `date`  
  The date of the vote timestamp used for the retained record.

- `idVotacao`  
  Voting session id (the key to merge back into your main dataset).

---

## How to use it in VOTE-RAP

Example merge pattern:

```python
import pandas as pd

author_pop = pd.read_csv("author_popularity.csv", usecols=["idVotacao", "popularity"])

df = df.merge(author_pop, left_on="id", right_on="idVotacao", how="left")
df["popularity"] = df["popularity"].fillna(0.5)
df = df.drop(columns=["idVotacao"])
```

---

## Run statistics from the logged execution (2025-11-28)

From `authors_popularity_output.txt`:

- Vote sessions loaded: **41,461** rows → **9,260** unique sessions  
- Total roll-call votes loaded (2003–2024): **1,696,248**  
- Votes after filtering to relevant sessions: **772,220**  
- Unique authors observed: **385**  
- Output size after deduplication: **1,990** rows × **8** columns  
- Total runtime: **100.15 seconds** (≈ 1.67 minutes)

---

## Interpretation (what the feature means)

For a given session (identified by `idVotacao`), the output `popularity` value represents:

- How strongly, **in the author’s recent historical record**, legislators tended to vote **“Sim”** in sessions attributed to that author,
- With higher emphasis on more recent history (90-day half-life),
- And an additional emphasis on larger sessions through the normalized `importance` weight.

Values close to **1.0** indicate that, historically and recently, sessions attributed to that author tended to receive many “Sim” votes (high support). Values closer to **0.0** indicate the opposite.

---

## Notes / caveats (alignment with current code)

- The script **does not** use the session outcome (`aprovacao`) to measure “author approval rate”. If you want “author success rate” (approved vs rejected propositions), the popularity definition would need to be changed to use that outcome rather than roll-call “Sim” votes.
- Because the computation is done at the level of *individual legislator votes*, high-participation sessions naturally contribute more events, and the explicit `importance` weighting further amplifies this effect.
- The final output keeps only one record per `idVotacao` via `drop_duplicates`, which keeps the first computed record for each session id.

---

*Last updated: 2025-11-28*
