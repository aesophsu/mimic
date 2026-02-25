#!/usr/bin/env python3
"""
Generate standalone Figure 1 missingness heatmap (panel-B style only).

Output:
- docs/suppl/figs/fig1_missing_heatmap.pdf
- docs/suppl/figs/fig1_missing_heatmap.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "mimic_raw_scale.csv"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "artifacts" / "features" / "selected_features.json"
OUT_DIR = PROJECT_ROOT / "docs" / "suppl" / "figs"
OUT_BASE = OUT_DIR / "fig1_missing_heatmap"


def _load_selected_features() -> list[str]:
    if not SELECTED_FEATURES_PATH.exists():
        return []
    with SELECTED_FEATURES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    seen = set()
    for outcome in ("pof", "mortality", "composite"):
        for feat in data.get(outcome, {}).get("features", []):
            if feat not in seen:
                seen.add(feat)
                out.append(feat)
    return out


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input: {DATA_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    selected = _load_selected_features()
    key_cols = [c for c in selected if c in df.columns]
    if not key_cols:
        raise ValueError("No selected features available for heatmap.")

    missing_rates = df[key_cols].isna().mean()
    sorted_cols = missing_rates.sort_values(ascending=False).index.tolist()
    mat = df[sorted_cols].isna().astype(int)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor="white")
    sns.heatmap(
        mat,
        cmap=["#F5F5F5", "#2E5A88"],
        cbar=True,
        yticklabels=False,
        xticklabels=sorted_cols,
        ax=ax,
        cbar_kws=dict(ticks=[0.25, 0.75], shrink=0.65, pad=0.02),
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(["Observed", "Missing"], rotation=90, va="center", fontsize=8)

    ax.set_xlabel("Clinical Features (model-selected)", fontsize=9, labelpad=6)
    ax.set_ylabel(f"Participants (N={len(df):,})", fontsize=9, labelpad=6)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_BASE}.pdf")
    print(f"Saved: {OUT_BASE}.png")


if __name__ == "__main__":
    main()
