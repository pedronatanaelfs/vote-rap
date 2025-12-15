# Party Popularity Feature Engineering (VOTE-RAP)

This script creates a **party-level historical approval feature** for each voting session and evaluates several “how much history should we look at?” window choices. The final output is a per-session dataset containing `party_popularity` (in percent) plus the ground-truth outcome `aprovacao`.

The script also produces a comparison plot of window AUROC scores and logs execution details.

---

## What the script generates

1. `party_popularity_best_window_last_5_sessions.csv` (final feature file)
2. `party_popularity_auroc_comparison.png` (window comparison chart)
3. `party_popularity_output.txt` (execution log)

---

## Inputs

### 1) Voting sessions
`data/vote_sessions_full.csv`

Used columns (as implemented):
- `id`: session id
- `data`: session date
- `aprovacao`: target (0/1, may have missing values)
- `author_type`: author type label (e.g., Deputado(a), Órgão do Poder Executivo, etc.)
- `idDeputadoAutor`: deputy id when the author is a deputy
- `legislatura`: legislature number

The script deduplicates sessions by `id` (`keep='first'`).

### 2) Deputy → party mapping (preferred source)
`data/extra/orgaosDeputados-L{legislature}.csv` for legislatures 51–57

The script extracts deputy ids from `uriDeputado` and uses `siglaPartido` as the party label.

### 3) Câmara API (fallback)
If a deputy’s party cannot be resolved from the files above, the script calls:

`https://dadosabertos.camara.leg.br/api/v2/deputados/{deputy_id}`

and reads `dados.ultimoStatus.siglaPartido`.

**Important:** the script explicitly warns that API data can change over time, which can affect reproducibility.

---

## Party label used per session: `party_or_author_type`

The script resolves a single string label for each session:

- If `author_type != 'Deputado(a)'` → keep `author_type` as the label.
- If `author_type == 'Deputado(a)'` → try to replace the label with the deputy’s party (`siglaPartido`) using:
  1) exact match by (deputy_id, legislatura),
  2) fallback to most recent party found in the CSV mapping,
  3) API lookup for remaining missing deputies.

This means `party_or_author_type` contains a mix of:
- party acronyms (e.g., “PT”, “PL”, …) for deputy-authored sessions, and
- other author type labels for non-deputy authors.

---

## Base dataset used for window testing

After resolving parties, the script builds `base_voting_data` as:

- sessions sorted by `['data', 'id']` (deterministic tie-breaking),
- **filtered to rows where `aprovacao` is not null**.

In the logged run this resulted in **8,914 rows**.

---

## Party popularity definition

For each row index `i` and its `party = party_or_author_type[i]`, the script looks only at **previous sessions**:

- `previous_sessions = voting_data.iloc[:i]`
- then filters those to `party_or_author_type == party`
- then applies a window policy (below)

Popularity is always computed as:

`party_popularity = 100 * (approved_sessions / total_sessions_in_window)`

where `approved_sessions` is the sum of `aprovacao` (0/1).

### Cold-start behavior
If a party/author-type has **no previous sessions** in the chosen window, the script returns:

- `party_popularity = 0.0`
- `party_total_sessions = 0`
- `party_approved_sessions = 0`

So early occurrences of a party/author-type (e.g., the first time “Órgão do Poder Executivo” appears) may have `party_popularity = 0.0` even if the current session ends up approved.

---

## Window configurations tested

The script evaluates 7 window configurations:

1. Full Window (all previous sessions)
2. 5-Year Window (previous sessions within last 5 years)
3. 1-Year Window (previous sessions within last 1 year)
4. Last 10 Sessions (previous 10 sessions for that party)
5. **Last 5 Sessions** ⭐ best (previous 5 sessions for that party)
6. Last 3 Sessions
7. Last 1 Session

Time windows use:

- `cutoff_date = current_date - DateOffset(years=Y)`
- and keep only previous sessions with `data >= cutoff_date`.

Session-count windows select `.tail(N)` from the party’s previous sessions.

---

## How window quality is evaluated

For each window:
- The script creates the `party_popularity` feature for all 8,914 rows.
- It trains two models using **only** `party_popularity` as input:
  - RandomForestClassifier (100 trees, `class_weight='balanced'`, `random_state=42`, `n_jobs=1`)
  - LogisticRegression (`class_weight='balanced'`, `random_state=42`, `max_iter=1000`)
- It uses a chronological 80/20 split (no shuffling).
- Primary metric reported: **AUROC**.

---

## Results (from the execution log)

In the logged run, the window results were:

- **Best window: Last 5 Sessions**
  - Random Forest AUROC: **0.6923**
  - Logistic Regression AUROC: **0.6919**

Other windows (RF / LR AUROC):
- Last 10 Sessions: 0.6908 / 0.6928
- Last 3 Sessions: 0.6883 / 0.6883
- Last 1 Session: 0.6378 / 0.6378
- 1-Year Window: 0.5473 / 0.6789
- 5-Year Window: 0.5302 / 0.6272
- Full Window: 0.5110 / 0.6352

---

## Final output: `party_popularity_best_window_last_5_sessions.csv`

This file is produced using the best window (Last 5 Sessions) and contains:

- `id`: session id
- `data`: date (YYYY-MM-DD)
- `party_or_author_type`: party acronym for deputies, otherwise author type
- `party_popularity`: percent approval rate over the last 5 prior sessions for that label
- `aprovacao`: ground-truth outcome (0/1)

Logged dataset statistics:
- Total rows: **8,914**
- Date range: **2003-02-19** to **2024-12-19**
- Unique `party_or_author_type`: **54**
- `party_popularity` mean / std: **79.0% ± 26.4%**
- Median: **80.0%**
- Min/Max: **0.0% / 100.0%**

---

## Interpretation (what `party_popularity` means)

For a given session, `party_popularity` answers:

> “For this party (or non-deputy author type), what fraction of its **last 5 prior sessions** were approved?”

Values near:
- **100** mean the label has a strong recent success record (5/5 approvals).
- **0** mean no approvals in the last-5 history (or a cold start with no history).

---

## Reproducibility notes (as implemented)

Deterministic elements:
- seeds set (`np.random.seed(42)` and `random.seed(42)`)
- consistent ordering by sorting on `['data', 'id']`
- `RandomForestClassifier(n_jobs=1)` to avoid parallel nondeterminism
- API calls made in sorted deputy-id order

Non-deterministic element:
- **Câmara API lookups** for party labels can change over time if the API returns different “ultimoStatus” values.

For fully repeatable results across time, cache and reuse the API-resolved deputy→party mapping.

---

## Suggested merge into the main dataset

```python
import pandas as pd

party_feat = pd.read_csv(
    "party_popularity_best_window_last_5_sessions.csv",
    usecols=["id", "party_popularity"]
)

df = df.merge(party_feat, on="id", how="left")
df["party_popularity"] = df["party_popularity"].fillna(0.0)
```

---

*Last updated: 2025-11-28 (based on `party_popularity_output.txt`)*
