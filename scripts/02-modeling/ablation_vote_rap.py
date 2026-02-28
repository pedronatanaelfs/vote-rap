"""
VOTE-RAP Ablation Study (Government Orientation Dependence)
---------------------------------------------------------

Generates an explicit ablation study requested in paper feedback:

- Gov only
- Gov + coalition-size indicators
- Gov + temporal features
- Full VOTE-RAP

All variants:
- Use the same leakage-safe, chronological 80/20 split (no shuffling)
- Use the same preprocessing rules as the main pipeline
- Train a single XGBoost model with fixed hyperparameters (taken from the
  best config in `results/modeling/full_enhanced/global_votes_prediction_FULL_enhanced_output.txt`)
- Evaluate with AUROC and Rejected-class F1, where the rejected threshold is
  optimized on the evaluated split (consistent with repository comparison scripts)

Outputs
-------
- Prints a compact table to stdout
- Writes CSV to: results/modeling/ablation/ablation_vote_rap_results.csv
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results" / "modeling" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = False

# Best params (from full_enhanced run log)
BEST_PARAMS = {
    "colsample_bytree": 0.5749080237694725,
    "gamma": 1.6310000289738826,
    "learning_rate": 0.07855951534491241,
    "max_depth": 4,
    "min_child_weight": 7,
    "n_estimators": 252,
    "reg_alpha": 0.06687491292803867,
    "reg_lambda": 0.939964882145204,
    "scale_pos_weight": 0.9755493351795202,
    "subsample": 0.7667417222278043,
}


@dataclass(frozen=True)
class AblationResult:
    model: str
    features: str
    accuracy: float
    f1_rejected: float
    f1_approved: float
    auroc: float
    precision_rejected: float
    recall_rejected: float
    best_threshold_rejected: float


def resolve_gov_orientation(row: pd.Series) -> float:
    # Mirrors the deterministic resolution rule used in the paper and scripts.
    if row.get("GOV.") == row.get("Governo"):
        return row.get("GOV.")
    if row.get("GOV.") != 0:
        return row.get("GOV.")
    return row.get("Governo")


def optimize_threshold_for_f1_rejected(y_true: np.ndarray, proba_rejected: np.ndarray) -> float:
    prec_0, rec_0, thresh_0 = precision_recall_curve(y_true, proba_rejected, pos_label=0)
    f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0 + 1e-8)
    best_idx = int(np.nanargmax(f1_0))
    if best_idx >= len(thresh_0):
        return 0.5
    return float(thresh_0[best_idx])


def build_merged_df() -> pd.DataFrame:
    usecols = [
        "id",
        "data",
        "aprovacao",
        "num_authors",
        "Governo",
        "GOV.",
    ]
    base = pd.read_csv(DATA_DIR / "vote_sessions_full.csv", usecols=usecols)
    base["data"] = pd.to_datetime(base["data"], errors="coerce")
    base["aprovacao"] = pd.to_numeric(base["aprovacao"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["data", "aprovacao"])

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

    merged = base.merge(authors_pop, left_on="id", right_on="idVotacao", how="left")
    merged = merged.merge(party_popularity, on="id", how="left")
    merged = merged.merge(historical_data, on="id", how="left")
    merged = merged.drop(columns=["idVotacao"], errors="ignore")

    merged["gov_orientation"] = merged.apply(resolve_gov_orientation, axis=1)
    merged["gov_orientation"] = pd.to_numeric(merged["gov_orientation"], errors="coerce").fillna(0.0)

    merged["num_authors_trunc"] = merged["num_authors"].apply(
        lambda x: x if pd.notna(x) and x <= 10 else (10 if pd.notna(x) else np.nan)
    )
    merged["has_more_than_10_authors"] = (merged["num_authors"].fillna(0) > 10).astype(int)

    merged["popularity"] = pd.to_numeric(merged["popularity"], errors="coerce").fillna(0.0)
    merged["party_popularity"] = pd.to_numeric(merged["party_popularity"], errors="coerce").fillna(0.0)
    merged["historical_approval_rate"] = pd.to_numeric(merged["historical_approval_rate"], errors="coerce").fillna(0.5)

    merged = merged.sort_values("data").reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["id"], keep="first")

    # Drop rows with any missing engineered numeric features we rely on
    merged["num_authors_trunc"] = pd.to_numeric(merged["num_authors_trunc"], errors="coerce").fillna(0.0)

    return merged


def fit_and_eval_variant(df: pd.DataFrame, model_name: str, feature_cols: list[str]) -> AblationResult:
    df = df.copy()
    y = df["aprovacao"].astype(int).to_numpy()

    split_idx = int(len(df) * 0.8)
    # Use positional slicing (iloc) to guarantee alignment with numpy y slices.
    X_train = df.iloc[:split_idx][feature_cols].copy()
    y_train = y[:split_idx]
    X_test = df.iloc[split_idx:][feature_cols].copy()
    y_test = y[split_idx:]

    # Standardize only the temporal rate features when present (to match paper's comparison interface).
    numeric_temporal = [c for c in ["popularity", "party_popularity", "historical_approval_rate"] if c in feature_cols]
    if numeric_temporal:
        scaler = StandardScaler()
        X_train[numeric_temporal] = scaler.fit_transform(X_train[numeric_temporal])
        X_test[numeric_temporal] = scaler.transform(X_test[numeric_temporal])

    # Convert to numpy arrays to avoid pandas/QuantileDMatrix edge-cases on some Windows builds.
    X_train_mat = X_train.to_numpy(dtype=np.float32, copy=True)
    X_test_mat = X_test.to_numpy(dtype=np.float32, copy=True)

    if DEBUG:
        print(
            f"\n[{model_name}] X_train={X_train_mat.shape} y_train={y_train.shape} "
            f"X_test={X_test_mat.shape} y_test={y_test.shape} "
            f"nan_train={int(np.isnan(X_train_mat).sum())} nan_test={int(np.isnan(X_test_mat).sum())}"
        )

    clf = XGBClassifier(
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
        **BEST_PARAMS,
    )
    clf.fit(X_train_mat, y_train)

    proba_approved = clf.predict_proba(X_test_mat)[:, 1]
    proba_rejected = 1.0 - proba_approved

    thr = optimize_threshold_for_f1_rejected(y_test, proba_rejected)
    y_pred = np.where(proba_rejected >= thr, 0, 1).astype(int)

    return AblationResult(
        model=model_name,
        features=" + ".join(feature_cols),
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_rejected=float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        f1_approved=float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        auroc=float(roc_auc_score(y_test, proba_approved)),
        precision_rejected=float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        recall_rejected=float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        best_threshold_rejected=float(thr),
    )


def main() -> None:
    df = build_merged_df()
    assert len(df) > 0, "Empty dataset after preprocessing."

    if DEBUG:
        print(f"Dataset rows: {len(df):,} | unique ids: {df['id'].nunique():,}")
        vc = df["aprovacao"].astype(int).value_counts().to_dict()
        print(f"Target counts: {vc}")

    variants: list[tuple[str, list[str]]] = [
        ("Gov only", ["gov_orientation"]),
        ("Gov + coalition-size", ["gov_orientation", "num_authors_trunc", "has_more_than_10_authors"]),
        ("Gov + temporal", ["gov_orientation", "popularity", "party_popularity", "historical_approval_rate"]),
        (
            "Full VOTE-RAP",
            [
                "popularity",
                "gov_orientation",
                "num_authors_trunc",
                "has_more_than_10_authors",
                "party_popularity",
                "historical_approval_rate",
            ],
        ),
    ]

    results: list[AblationResult] = []
    for name, cols in variants:
        res = fit_and_eval_variant(df, name, cols)
        results.append(res)

    out_df = pd.DataFrame([r.__dict__ for r in results])
    out_path = OUTPUT_DIR / "ablation_vote_rap_results.csv"
    out_df.to_csv(out_path, index=False)

    # Pretty print
    pretty = out_df.copy()
    for c in ["accuracy", "f1_rejected", "f1_approved", "auroc", "precision_rejected", "recall_rejected", "best_threshold_rejected"]:
        pretty[c] = pretty[c].map(lambda x: f"{x:.3f}")
    print("\nAblation results (chronological 80/20):")
    print(pretty[["model", "accuracy", "f1_rejected", "auroc", "precision_rejected", "recall_rejected", "best_threshold_rejected"]].to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


