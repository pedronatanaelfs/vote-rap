"""
VOTE-RAP vs Baselines vs VIOLA-style Methodology (Comparable Evaluation)

Goal
----
Provide a directly comparable evaluation of:
- VOTE-RAP (as implemented in this repository)
- Existing simple baselines
- A VIOLA-inspired, data-centric methodology adapted to the same task/metrics/split

Notes on "VIOLA methodology" adaptation
--------------------------------------
The attached paper focuses on predicting *individual roll-call votes* in a multi-party
legislative system, leveraging diverse structured + unstructured signals and building
meta-features derived from feature-importance information.

This repository's primary task (and metrics) are *global outcome prediction*:
`aprovacao` (0 = rejected, 1 = approved) at the vote-session level.

To keep the comparison fair and directly comparable, this script:
- Uses the same temporal split protocol (chronological 80/20) as VOTE-RAP scripts
- Uses the same headline metrics already used in the repo (AUROC + F1 for rejected)
- Builds a VIOLA-style model from *pre-vote* information only:
  - structured metadata (theme, organ, proposition type, author type, legislature, etc.)
  - unstructured text (ementa/keywords) from proposition metadata files
  - simple meta-features computed from global feature importance on a base model

Outputs
-------
- Prints a single comparison table to stdout
- Writes `scripts/02-modeling/comparison_vote_rap_baselines_viola.csv`
- Writes 3 plots in the same folder:
  - `comparison_auroc.png`
  - `comparison_f1_rejected.png`
  - `comparison_metrics_heatmap.png`
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy.sparse import issparse
from scipy.stats import randint, uniform
from xgboost import XGBClassifier


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results" / "modeling" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = BASE_DIR / "article" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Silence noisy openpyxl warnings found in these yearly XLSX dumps.
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
)


@dataclass(frozen=True)
class ModelResult:
    model: str
    accuracy: float
    precision_approved: float
    precision_rejected: float
    recall_approved: float
    recall_rejected: float
    f1_approved: float
    f1_rejected: float
    auroc: float
    average_precision: float | float("nan")
    best_threshold_rejected: float | float("nan")


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def resolve_gov_orientation(row: pd.Series) -> float:
    """
    Same logic used in the repository scripts.
    """
    if row.get("GOV.") == row.get("Governo"):
        return row.get("GOV.")
    if row.get("GOV.") != 0:
        return row.get("GOV.")
    return row.get("Governo")


def optimize_threshold_for_f1_rejected(y_true: pd.Series, proba_rejected: np.ndarray) -> float:
    """
    Chooses a threshold on P(class=0) that maximizes F1 for the rejected class.
    Mirrors the approach in `global_votes_prediction_FULL_enhanced.py`.
    """
    prec_0, rec_0, thresh_0 = precision_recall_curve(y_true, proba_rejected, pos_label=0)
    f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0 + 1e-8)
    best_idx = int(np.nanargmax(f1_0))
    if best_idx >= len(thresh_0):
        return 0.5
    return float(thresh_0[best_idx])


def predict_from_rejected_threshold(proba_rejected: np.ndarray, threshold_rejected: float) -> np.ndarray:
    """
    Predict Rejected (0) if P(rejected) >= threshold, else Approved (1).
    """
    return np.where(proba_rejected >= threshold_rejected, 0, 1).astype(int)


def compute_metrics(
    model_name: str,
    y_true: pd.Series,
    proba_approved: np.ndarray,
    proba_rejected: np.ndarray | None = None,
) -> ModelResult:
    proba_rejected = proba_rejected if proba_rejected is not None else (1.0 - proba_approved)
    best_thr = optimize_threshold_for_f1_rejected(y_true, proba_rejected)
    y_pred = predict_from_rejected_threshold(proba_rejected, best_thr)

    return ModelResult(
        model=model_name,
        accuracy=_safe_float(accuracy_score(y_true, y_pred)),
        precision_approved=_safe_float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        precision_rejected=_safe_float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        recall_approved=_safe_float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        recall_rejected=_safe_float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        f1_approved=_safe_float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f1_rejected=_safe_float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        auroc=_safe_float(roc_auc_score(y_true, proba_approved)),
        average_precision=_safe_float(average_precision_score(y_true, proba_approved)),
        best_threshold_rejected=_safe_float(best_thr),
    )


def load_vote_sessions_base() -> pd.DataFrame:
    # Keep a minimal, "pre-vote safe" set of columns.
    # Avoid leakage-prone text such as `descricao` which often contains "Aprovada/Rejeitada".
    usecols = [
        "id",
        "data",
        "aprovacao",
        "propositionID",
        "siglaOrgao",
        "proposicao_siglaTipo",
        "year",
        "author_type",
        "author_type_code",
        "num_authors",
        "theme",
        "legislatura",
        "Governo",
        "GOV.",
    ]
    df = pd.read_csv(DATA_DIR / "vote_sessions_full.csv", usecols=usecols)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["aprovacao"] = pd.to_numeric(df["aprovacao"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["data", "aprovacao"])
    return df


def load_vote_rap_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    authors_pop = pd.read_csv(
        DATA_DIR / "features" / "author_popularity.csv",
        usecols=["idVotacao", "popularity"],
    )
    party_popularity = pd.read_csv(
        DATA_DIR / "features" / "party_popularity_best_window_last_5_sessions.csv",
        usecols=["id", "party_popularity"],
    )
    historical_data = pd.read_csv(
        DATA_DIR / "features" / "proposition_history_predictions_historical_probability_rule.csv",
        usecols=["id", "historical_approval_rate"],
    )
    return authors_pop, party_popularity, historical_data


def load_propositions_text() -> pd.DataFrame:
    """
    Loads (and caches) proposition text fields from `data/propositions/*.xlsx`.

    To avoid re-reading 25 XLSX files on every run, we cache a compact CSV at:
      `data/extra/propositions_text_cache.csv`
    """
    cache_path = DATA_DIR / "extra" / "propositions_text_cache.csv"
    if cache_path.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Using cached proposition text: {cache_path}", flush=True)
        props = pd.read_csv(cache_path, usecols=["id", "proposition_text"])
        props["id"] = pd.to_numeric(props["id"], errors="coerce").astype("Int64")
        props["proposition_text"] = props["proposition_text"].fillna("").astype(str)
        return props

    files = sorted((DATA_DIR / "propositions").glob("proposicoes-*.xlsx"))
    if not files:
        raise FileNotFoundError("No proposition XLSX files found under data/propositions/")

    cols = ["id", "ementa", "ementaDetalhada", "keywords", "siglaTipo", "ano"]
    parts = []
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loading proposition text from {len(files)} XLSX files (one-time cache build)...",
        flush=True,
    )
    for i, f in enumerate(files, start=1):
        if i == 1 or i % 5 == 0 or i == len(files):
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Reading {i}/{len(files)}: {f.name}", flush=True)
        try:
            tmp = pd.read_excel(f, usecols=cols)
        except ValueError:
            # Some yearly files may lack some columns; fall back to reading and reindexing.
            tmp = pd.read_excel(f)
            tmp = tmp.reindex(columns=cols)
        parts.append(tmp)

    props = pd.concat(parts, ignore_index=True)
    props = props.drop_duplicates(subset=["id"], keep="last")
    props["id"] = pd.to_numeric(props["id"], errors="coerce").astype("Int64")
    for c in ["ementa", "ementaDetalhada", "keywords"]:
        if c in props.columns:
            props[c] = props[c].fillna("")
    props["proposition_text"] = (
        props.get("ementa", "").astype(str)
        + " "
        + props.get("ementaDetalhada", "").astype(str)
        + " "
        + props.get("keywords", "").astype(str)
    ).str.strip()

    # Write cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    props_out = props[["id", "proposition_text"]].copy()
    props_out.to_csv(cache_path, index=False, encoding="utf-8")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved cache: {cache_path}", flush=True)
    return props_out


def build_merged_df() -> pd.DataFrame:
    base = load_vote_sessions_base()
    authors_pop, party_popularity, historical_data = load_vote_rap_features()

    merged = base.merge(authors_pop, left_on="id", right_on="idVotacao", how="left")
    merged = merged.merge(party_popularity, left_on="id", right_on="id", how="left")
    merged = merged.merge(historical_data, left_on="id", right_on="id", how="left")
    merged = merged.drop(columns=["idVotacao"], errors="ignore")

    merged["gov_orientation"] = merged.apply(resolve_gov_orientation, axis=1)
    merged["num_authors_trunc"] = merged["num_authors"].apply(lambda x: x if pd.notna(x) and x <= 10 else (10 if pd.notna(x) else np.nan))
    merged["has_more_than_10_authors"] = (merged["num_authors"].fillna(0) > 10).astype(int)

    merged["popularity"] = merged["popularity"].fillna(0)
    merged["party_popularity"] = merged["party_popularity"].fillna(0)
    merged["historical_approval_rate"] = merged["historical_approval_rate"].fillna(0.5)

    # Make sure categorical columns exist and are stringy
    for c in ["siglaOrgao", "proposicao_siglaTipo", "author_type", "theme"]:
        merged[c] = merged[c].fillna("Unknown").astype(str)

    merged["author_type_code"] = pd.to_numeric(merged["author_type_code"], errors="coerce").fillna(-1).astype(int)
    merged["legislatura"] = pd.to_numeric(merged["legislatura"], errors="coerce").fillna(-1).astype(int)

    # Attach proposition text (rename to avoid clobbering vote-session id)
    props = load_propositions_text().rename(columns={"id": "proposition_id_text"})
    merged["propositionID"] = pd.to_numeric(merged["propositionID"], errors="coerce").astype("Int64")
    merged = merged.merge(props, left_on="propositionID", right_on="proposition_id_text", how="left")
    merged["proposition_text"] = merged["proposition_text"].fillna("")

    # Ensure temporal order is explicit
    merged = merged.sort_values("data").reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["id"], keep="first")

    return merged


def train_vote_rap_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    # Match the repository's training style: randomized search on AUROC.
    auroc_scorer = "roc_auc"
    param_distributions = {
        "n_estimators": randint(150, 351),
        "max_depth": randint(4, 8),
        "learning_rate": uniform(0.02, 0.08),
        "scale_pos_weight": uniform(0.7, 0.6),
        "subsample": uniform(0.7, 0.2),
        "colsample_bytree": uniform(0.5, 0.2),
        "gamma": uniform(0.3, 1.4),
        "min_child_weight": randint(3, 8),
        "reg_alpha": uniform(0, 0.15),
        "reg_lambda": uniform(0.8, 1.4),
    }

    base = XGBClassifier(
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
    )

    # Early stopping set created once for consistency
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    search = RandomizedSearchCV(
        base,
        param_distributions=param_distributions,
        n_iter=75,
        scoring=auroc_scorer,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_SEED,
    )
    search.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best = search.best_estimator_

    # Refit on full training data (no early stopping for final fit)
    best.set_params(early_stopping_rounds=None)
    best.fit(X_train, y_train)
    return best


def run_vote_rap_model(df: pd.DataFrame, split_idx: int) -> ModelResult:
    features = [
        "popularity",
        "gov_orientation",
        "num_authors_trunc",
        "has_more_than_10_authors",
        "party_popularity",
        "historical_approval_rate",
    ]
    numeric_features = ["popularity", "party_popularity", "historical_approval_rate"]

    X = df[features].copy()
    y = df["aprovacao"].astype(int)

    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    model = train_vote_rap_xgb(X_train, y_train)
    proba = model.predict_proba(X_test)
    proba_approved = proba[:, 1]
    proba_rejected = proba[:, 0]

    return compute_metrics("VOTE-RAP (XGBoost)", y_test, proba_approved, proba_rejected)


def run_viola_model(df: pd.DataFrame, split_idx: int) -> ModelResult:
    """
    VIOLA-inspired pipeline:
    - Structured: categorical + numeric metadata
    - Linguistic: TF-IDF of proposition ementa/keywords
    - Meta-features: computed from the top-N important features from a base model
    """

    y = df["aprovacao"].astype(int)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    numeric = ["gov_orientation", "num_authors_trunc", "has_more_than_10_authors", "year", "legislatura", "author_type_code"]
    categorical = ["siglaOrgao", "proposicao_siglaTipo", "author_type", "theme"]
    text = "proposition_text"

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
            (
                "text",
                TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 2),
                    min_df=2,
                    lowercase=True,
                ),
                text,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    base_model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", base_model)])
    pipe.fit(train_df, y_train)

    # Base predictions
    base_proba_test = pipe.predict_proba(test_df)
    base_proba_train = pipe.predict_proba(train_df)

    # Meta-features from feature-importance (top-N most important transformed features)
    model_fitted: XGBClassifier = pipe.named_steps["model"]
    importances = model_fitted.feature_importances_

    X_train_t = pipe.named_steps["pre"].transform(train_df)
    X_test_t = pipe.named_steps["pre"].transform(test_df)
    if not issparse(X_train_t):
        # ColumnTransformer can output dense for small feature spaces; keep code generic.
        X_train_t = np.asarray(X_train_t)
        X_test_t = np.asarray(X_test_t)

    top_n = min(50, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_n]
    imp_top = importances[top_idx]

    def _meta_features(X_t, base_proba) -> np.ndarray:
        n_rows = base_proba.shape[0]

        def _ensure_len(arr):
            if hasattr(arr, "toarray"):
                arr = arr.toarray()
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if arr.shape[0] == 1 and n_rows > 1:
                arr = np.repeat(arr, n_rows)
            return arr

        if issparse(X_t):
            X_top = X_t[:, top_idx]
            weighted_sum = _ensure_len(X_top @ imp_top)
            X_abs = X_top.copy()
            X_abs.data = np.abs(X_abs.data)
            weighted_abs_sum = _ensure_len(X_abs @ imp_top)
            nnz = _ensure_len(X_top.getnnz(axis=1))
            max_val = _ensure_len(X_top.max(axis=1))
        else:
            X_top = X_t[:, top_idx]
            weighted_sum = _ensure_len(X_top @ imp_top)
            weighted_abs_sum = _ensure_len(np.abs(X_top) @ imp_top)
            nnz = _ensure_len((X_top != 0).sum(axis=1))
            max_val = _ensure_len(X_top.max(axis=1))

        # Compact meta-representation
        return np.column_stack(
            [
                base_proba[:, 1],  # P(approved) from base model
                base_proba[:, 0],  # P(rejected) from base model
                weighted_sum,
                weighted_abs_sum,
                nnz,
                max_val,
            ]
        )

    meta_train = _meta_features(X_train_t, base_proba_train)
    meta_test = _meta_features(X_test_t, base_proba_test)

    meta_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        ]
    )
    meta_clf.fit(meta_train, y_train)
    final_proba = meta_clf.predict_proba(meta_test)

    return compute_metrics(
        "VIOLA-style (Structured+Text+Meta)",
        y_test,
        final_proba[:, 1],
        final_proba[:, 0],
    )


def run_simple_baselines(df: pd.DataFrame, split_idx: int) -> list[ModelResult]:
    y = df["aprovacao"].astype(int)
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results: list[ModelResult] = []

    # Random guess
    rng = np.random.default_rng(RANDOM_SEED)
    random_preds = rng.integers(0, 2, size=len(y_test))
    random_probs = rng.random(size=len(y_test))
    results.append(
        ModelResult(
            model="Random Guess",
            accuracy=_safe_float(accuracy_score(y_test, random_preds)),
            precision_approved=_safe_float(precision_score(y_test, random_preds, pos_label=1, zero_division=0)),
            precision_rejected=_safe_float(precision_score(y_test, random_preds, pos_label=0, zero_division=0)),
            recall_approved=_safe_float(recall_score(y_test, random_preds, pos_label=1, zero_division=0)),
            recall_rejected=_safe_float(recall_score(y_test, random_preds, pos_label=0, zero_division=0)),
            f1_approved=_safe_float(f1_score(y_test, random_preds, pos_label=1, zero_division=0)),
            f1_rejected=_safe_float(f1_score(y_test, random_preds, pos_label=0, zero_division=0)),
            auroc=_safe_float(roc_auc_score(y_test, random_probs)),
            average_precision=_safe_float(average_precision_score(y_test, random_probs)),
            best_threshold_rejected=float("nan"),
        )
    )

    # Majority class
    majority_class = int(y_train.mean() >= 0.5)
    maj_preds = np.full(len(y_test), majority_class)
    maj_probs = np.full(len(y_test), 1.0 if majority_class == 1 else 0.0)
    results.append(
        ModelResult(
            model="Majority Class",
            accuracy=_safe_float(accuracy_score(y_test, maj_preds)),
            precision_approved=_safe_float(precision_score(y_test, maj_preds, pos_label=1, zero_division=0)),
            precision_rejected=_safe_float(precision_score(y_test, maj_preds, pos_label=0, zero_division=0)),
            recall_approved=_safe_float(recall_score(y_test, maj_preds, pos_label=1, zero_division=0)),
            recall_rejected=_safe_float(recall_score(y_test, maj_preds, pos_label=0, zero_division=0)),
            f1_approved=_safe_float(f1_score(y_test, maj_preds, pos_label=1, zero_division=0)),
            f1_rejected=_safe_float(f1_score(y_test, maj_preds, pos_label=0, zero_division=0)),
            auroc=_safe_float(roc_auc_score(y_test, maj_probs)),
            average_precision=_safe_float(average_precision_score(y_test, maj_probs)),
            best_threshold_rejected=float("nan"),
        )
    )

    # Stratified probability (matches train approval rate)
    p_approved = float(y_train.mean())
    strat_probs = rng.random(size=len(y_test))
    strat_preds = (strat_probs < p_approved).astype(int)
    results.append(
        ModelResult(
            model="Stratified Probability",
            accuracy=_safe_float(accuracy_score(y_test, strat_preds)),
            precision_approved=_safe_float(precision_score(y_test, strat_preds, pos_label=1, zero_division=0)),
            precision_rejected=_safe_float(precision_score(y_test, strat_preds, pos_label=0, zero_division=0)),
            recall_approved=_safe_float(recall_score(y_test, strat_preds, pos_label=1, zero_division=0)),
            recall_rejected=_safe_float(recall_score(y_test, strat_preds, pos_label=0, zero_division=0)),
            f1_approved=_safe_float(f1_score(y_test, strat_preds, pos_label=1, zero_division=0)),
            f1_rejected=_safe_float(f1_score(y_test, strat_preds, pos_label=0, zero_division=0)),
            auroc=_safe_float(roc_auc_score(y_test, strat_probs)),
            average_precision=_safe_float(average_precision_score(y_test, strat_probs)),
            best_threshold_rejected=float("nan"),
        )
    )

    # Government orientation heuristic (same as in FULL enhanced script idea)
    # Note: proba heuristic is only used for AUROC computation.
    gov = df["gov_orientation"].iloc[split_idx:].to_numpy()
    gov_preds = np.where(gov == 1, 1, np.where(gov == -1, 0, int(round(p_approved))))
    gov_probs = np.where(gov == 1, 0.8, np.where(gov == -1, 0.2, p_approved))
    results.append(
        ModelResult(
            model="Government Orientation",
            accuracy=_safe_float(accuracy_score(y_test, gov_preds)),
            precision_approved=_safe_float(precision_score(y_test, gov_preds, pos_label=1, zero_division=0)),
            precision_rejected=_safe_float(precision_score(y_test, gov_preds, pos_label=0, zero_division=0)),
            recall_approved=_safe_float(recall_score(y_test, gov_preds, pos_label=1, zero_division=0)),
            recall_rejected=_safe_float(recall_score(y_test, gov_preds, pos_label=0, zero_division=0)),
            f1_approved=_safe_float(f1_score(y_test, gov_preds, pos_label=1, zero_division=0)),
            f1_rejected=_safe_float(f1_score(y_test, gov_preds, pos_label=0, zero_division=0)),
            auroc=_safe_float(roc_auc_score(y_test, gov_probs)),
            average_precision=_safe_float(average_precision_score(y_test, gov_probs)),
            best_threshold_rejected=float("nan"),
        )
    )

    return results


def plot_comparison(df_results: pd.DataFrame) -> None:
    # AUROC plot (best at top)
    fig, ax = plt.subplots(figsize=(12, 7))
    d = df_results.sort_values("auroc", ascending=False)
    ax.barh(d["model"], d["auroc"], color="#2E86AB", alpha=0.85, edgecolor="black", linewidth=1.2)
    for i, v in enumerate(d["auroc"].tolist()):
        if np.isfinite(v):
            ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="bold")
    ax.set_xlabel("AUROC")
    ax.set_title("Comparison: Baselines vs VOTE-RAP vs VIOLA-style (AUROC)")
    ax.set_xlim(0.4, min(1.0, max([x for x in d["auroc"] if np.isfinite(x)] + [0.8]) * 1.15))
    ax.invert_yaxis()  # highest first at top visually
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_auroc.png", dpi=300, bbox_inches="tight")
    plt.close()
    # Copy to paper figures
    try:
        shutil.copyfile(OUTPUT_DIR / "comparison_auroc.png", PAPER_FIG_DIR / "comparison_auroc.png")
    except Exception:
        pass

    # F1 rejected plot (best at top)
    fig, ax = plt.subplots(figsize=(12, 7))
    d_f1 = df_results.sort_values("f1_rejected", ascending=False)
    ax.barh(d_f1["model"], d_f1["f1_rejected"], color="#A23B72", alpha=0.85, edgecolor="black", linewidth=1.2)
    for i, v in enumerate(d_f1["f1_rejected"].tolist()):
        if np.isfinite(v):
            ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="bold")
    ax.set_xlabel("F1 (Rejected)")
    ax.set_title("Comparison: Baselines vs VOTE-RAP vs VIOLA-style (F1 Rejected)")
    ax.set_xlim(0.0, min(1.0, max([x for x in d_f1["f1_rejected"] if np.isfinite(x)] + [0.5]) * 1.2))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_f1_rejected.png", dpi=300, bbox_inches="tight")
    plt.close()
    # Copy to paper figures
    try:
        shutil.copyfile(OUTPUT_DIR / "comparison_f1_rejected.png", PAPER_FIG_DIR / "comparison_f1_rejected.png")
    except Exception:
        pass

    # Heatmap (drop incomplete rows)
    heat_cols = [
        "accuracy",
        "precision_approved",
        "precision_rejected",
        "recall_approved",
        "recall_rejected",
        "f1_approved",
        "f1_rejected",
        "auroc",
    ]
    # For the heatmap, order rows by AUROC descending to match the "best up" idea.
    heat = (
        df_results.dropna(subset=["accuracy"])
        .sort_values("auroc", ascending=False)
        .set_index("model")[heat_cols]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "Score", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Comprehensive Metrics Comparison")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Models")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_metrics_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 90)
    print("COMPARISON: VOTE-RAP + BASELINES + VIOLA-STYLE METHODOLOGY")
    print("=" * 90)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting run...", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building merged dataset...", flush=True)
    df = build_merged_df()
    split_idx = int(0.8 * len(df))
    print(f"Dataset size: {len(df):,} rows | train: {split_idx:,} | test: {len(df) - split_idx:,}")
    print(f"Time span: {df['data'].min().date()} -> {df['data'].max().date()}")

    results: list[ModelResult] = []
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running baselines...", flush=True)
    results.extend(run_simple_baselines(df, split_idx))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training VOTE-RAP model (XGBoost)...", flush=True)
    results.append(run_vote_rap_model(df, split_idx))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training VIOLA-style model (Structured+Text+Meta)...", flush=True)
    results.append(run_viola_model(df, split_idx))

    results_df = pd.DataFrame([r.__dict__ for r in results])

    # Sort for readability: baselines first, then learned models
    preferred_order = [
        "Random Guess",
        "Majority Class",
        "Stratified Probability",
        "Government Orientation",
        "VOTE-RAP (XGBoost)",
        "VIOLA-style (Structured+Text+Meta)",
    ]
    order_map = {m: i for i, m in enumerate(preferred_order)}
    results_df["__order"] = results_df["model"].map(order_map).fillna(999).astype(int)
    results_df = results_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    out_csv = OUTPUT_DIR / "comparison_vote_rap_baselines_viola.csv"
    results_df.to_csv(out_csv, index=False, encoding="utf-8")

    # Print concise table
    display_cols = ["model", "accuracy", "f1_rejected", "f1_approved", "auroc", "average_precision", "best_threshold_rejected"]
    disp = results_df[display_cols].copy()
    for c in display_cols[1:]:
        disp[c] = disp[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) else "N/A")

    print("\nResults (same split/metrics):")
    print(disp.to_string(index=False))
    print(f"\nSaved: {out_csv}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Writing plots...", flush=True)
    plot_comparison(results_df)
    print(f"Saved plots:")
    print(f"  - {OUTPUT_DIR / 'comparison_auroc.png'}")
    print(f"  - {OUTPUT_DIR / 'comparison_f1_rejected.png'}")
    print(f"  - {OUTPUT_DIR / 'comparison_metrics_heatmap.png'}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done.", flush=True)


if __name__ == "__main__":
    main()


