"""
Generate the paper figure:
  article/figures/data_pipeline.png

Figure caption in the paper:
  "Dataset construction and leakage-safe feature computation workflow, from raw
   Chamber open-data sources to the final chronological train-test split."
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, title, lines, fc="#F7F9FC", ec="#2C3E50", title_fc="#2E86AB"):
    r = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(r)

    tb_h = h * 0.22
    tb = FancyBboxPatch(
        (x, y + h - tb_h),
        w,
        tb_h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0,
        facecolor=title_fc,
    )
    ax.add_patch(tb)

    ax.text(
        x + 0.015,
        y + h - tb_h / 2,
        title,
        va="center",
        ha="left",
        color="white",
        weight="bold",
        fontsize=12,
    )

    body = "\n".join(lines)
    ax.text(
        x + 0.018,
        y + h - tb_h - 0.015,
        body,
        va="top",
        ha="left",
        color="#1F2D3D",
        fontsize=10,
    )


def _arrow(ax, x1, y1, x2, y2, label=None):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color="#34495E",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.02,
            label,
            ha="center",
            va="bottom",
            color="#34495E",
            fontsize=9,
        )


def main():
    out_dir = Path("article") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_pipeline.png"

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(14, 8), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout
    col1_x, col2_x, col3_x, col4_x = 0.04, 0.285, 0.53, 0.775
    w = 0.205

    h1, h2, h3 = 0.22, 0.28, 0.22
    r1_y, r2_y, r3_y = 0.72, 0.40, 0.12

    _box(
        ax,
        col1_x,
        r1_y,
        w,
        h1,
        "Raw Câmara Open Data",
        [
            "Roll-call vote sessions",
            "Proposition metadata",
            "Authors / parties / legislature",
            "Government orientation fields",
        ],
    )

    _box(
        ax,
        col2_x,
        r1_y,
        w,
        h1,
        "Ingestion + Harmonization",
        [
            "Download / load sources",
            "Normalize identifiers",
            "Parse dates and outcomes",
            "Remove duplicates",
        ],
    )

    _box(
        ax,
        col3_x,
        r1_y,
        w,
        h1,
        "Join & Filtering",
        [
            "Join sessions ↔ proposition info",
            "Join engineered feature tables",
            "Keep clear outcomes (0/1)",
            "Sort chronologically by date",
        ],
    )

    _box(
        ax,
        col4_x,
        r1_y,
        w,
        h1,
        "Leakage-safe Protocol",
        [
            "Features use only past info",
            "No look-ahead / no test leakage",
            "Chronological evaluation",
        ],
    )

    _box(
        ax,
        col2_x,
        r2_y,
        w,
        h2,
        "Feature Engineering (per session i)",
        [
            "Gov. orientation: resolve GOV. vs Governo",
            "Coalition size: num_authors_trunc, >10 flag",
            "Author popularity: past success (merged)",
            "Party popularity: last K sessions (K=5)",
            "HAR: past outcomes for same proposition",
            "Missing: popularity/party_pop=0, HAR=0.5",
        ],
        title_fc="#27AE60",
    )

    _box(
        ax,
        col4_x,
        r2_y,
        w,
        h2,
        "Modeling + Evaluation",
        [
            "Train: first 80% (time-ordered)",
            "Test: last 20% (future)",
            "Models: VOTE-RAP, baselines, VIOLA-style",
            "Metrics: AUROC + F1 (Rejected)",
            "Rejected threshold: maximize F1 for class 0",
        ],
        title_fc="#A23B72",
    )

    _box(
        ax,
        col1_x,
        r3_y,
        w,
        h3,
        "Artifacts (Saved Files)",
        [
            "vote_sessions_full.csv",
            "author_popularity.csv",
            "party_popularity_*.csv",
            "proposition_history_*.csv",
            "comparison_*.png / .csv",
        ],
        title_fc="#8E44AD",
    )

    # Flow arrows
    _arrow(ax, col1_x + w, r1_y + h1 / 2, col2_x, r1_y + h1 / 2)
    _arrow(ax, col2_x + w, r1_y + h1 / 2, col3_x, r1_y + h1 / 2)
    _arrow(ax, col3_x + w, r1_y + h1 / 2, col4_x, r1_y + h1 / 2)

    _arrow(
        ax,
        col3_x + w / 2,
        r1_y,
        col2_x + w / 2,
        r2_y + h2,
        label="compute features after merge\n(using only past sessions)",
    )
    _arrow(
        ax,
        col4_x + w / 2,
        r1_y,
        col4_x + w / 2,
        r2_y + h2,
        label="apply temporal split\n& evaluation",
    )

    _arrow(ax, col2_x + w, r2_y + h2 / 2, col4_x, r2_y + h2 / 2)
    _arrow(ax, col2_x, r2_y + 0.05, col1_x + w, r3_y + h3 - 0.02, label="outputs / caches")

    ax.text(
        0.5,
        0.035,
        "Dataset construction and leakage-safe feature computation workflow (VOTE-RAP).",
        ha="center",
        va="center",
        color="#34495E",
        fontsize=11,
        style="italic",
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


