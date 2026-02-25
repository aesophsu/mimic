"""Generate Fig2 quad-panel feature reduction journey figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs" / "main" / "figs"
FEATURE_DICT_PATH = PROJECT_ROOT / "artifacts" / "features" / "feature_dictionary.json"


def _load_feature_dict() -> Dict[str, Dict[str, str]]:
    if not FEATURE_DICT_PATH.exists():
        return {}
    return json.loads(FEATURE_DICT_PATH.read_text(encoding="utf-8"))


def _display_name(feature: str, feature_dict: Dict[str, Dict[str, str]]) -> str:
    item = feature_dict.get(feature, {})
    if item.get("display_name_en"):
        return item["display_name_en"]
    return feature.replace("_", " ")


def _prepare_lasso_cv(target: str):
    lasso_curve_path = PROJECT_ROOT / "artifacts" / "models" / target / "lasso_cv_curve.csv"
    if not lasso_curve_path.exists():
        raise FileNotFoundError(f"LASSO curve data not found: {lasso_curve_path}")
    df = pd.read_csv(lasso_curve_path)
    df = df.sort_values("lambda").reset_index(drop=True)
    x = df["log10_lambda"].values
    loss_mean = df["mean_deviance"].values
    loss_se = df["se_deviance"].values
    idx_min = int(df.index[df["is_dev_lambda_min"] == True][0])  # noqa: E712
    idx_1se = int(df.index[df["is_dev_lambda_1se"] == True][0])  # noqa: E712
    n_total = int(df["n_nonzero_coef"].max())
    selected_n = int(df["selected_n_features_final"].iloc[0])
    return x, loss_mean, loss_se, idx_min, idx_1se, n_total, selected_n


def _load_shap_rankings(target: str):
    stab_path = PROJECT_ROOT / "artifacts" / "models" / target / "xgb_shap_bootstrap_stability.csv"
    stab = pd.read_csv(stab_path)
    stab = stab[stab["target"] == target].copy()
    stab = stab.sort_values("shap_mean_abs_full", ascending=False).reset_index(drop=True)
    stab["pct"] = 100 * stab["shap_mean_abs_full"] / stab["shap_mean_abs_full"].sum()
    return stab


def _load_auc_by_k(target: str):
    internal = PROJECT_ROOT / "artifacts" / "models" / target / "xgb_internal_pruning_curve.csv"
    shap_curve = PROJECT_ROOT / "artifacts" / "models" / target / "xgb_shap_pruning_curve.csv"
    if internal.exists():
        df = pd.read_csv(internal)
    elif shap_curve.exists():
        df = pd.read_csv(shap_curve)
    else:
        raise FileNotFoundError("Pruning curve not found for panel D.")
    df = df[df["target"] == target].copy()
    df = df.sort_values("k").reset_index(drop=True)
    return df


def _load_recommended_k(target: str) -> int:
    rec_path = PROJECT_ROOT / "artifacts" / "models" / target / "xgb_shap_pruning_recommendation.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if "k_recommended" in rec:
            return int(rec["k_recommended"])
    stab_path = PROJECT_ROOT / "artifacts" / "models" / target / "xgb_shap_bootstrap_stability.csv"
    return int(pd.read_csv(stab_path).shape[0])


def _panel_a(ax, target: str):
    x, loss_mean, loss_se, idx_min, idx_1se, n_total, selected_n = _prepare_lasso_cv(target)
    ax.errorbar(
        x,
        loss_mean,
        yerr=loss_se,
        fmt="o",
        ms=3.5,
        color="#2563eb",
        ecolor="#93c5fd",
        elinewidth=1.0,
        capsize=2,
    )
    ax.plot(x, loss_mean, lw=1.2, color="#1d4ed8", alpha=0.8)
    ax.axvline(x[idx_min], ls="--", lw=1.5, color="#ef4444", label=r"$\lambda_{min}$")
    ax.axvline(x[idx_1se], ls="--", lw=1.5, color="#111827", label=r"$\lambda_{1se}$")
    y_annot = float(loss_mean.max() - (loss_mean.max() - loss_mean.min()) * 0.15)
    ax.text(
        x[idx_1se] + 0.04,
        y_annot,
        f"Selected by 1-SE:\n{selected_n} features",
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9ca3af", alpha=0.95),
    )
    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_ylabel("CV Binomial Deviance")
    ax.set_title("a) LASSO Cross-Validation Error Plot", loc="left", fontweight="bold")
    ax.grid(alpha=0.2, ls="--")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    # Move detailed textual note to markdown sidecar for better readability.


def _panel_b(
    ax,
    shap_rank: pd.DataFrame,
    feature_dict: Dict[str, Dict[str, str]],
    k_selected: int,
    display_n: int,
):
    top = shap_rank.head(display_n).copy()
    features = top["feature"].tolist()
    labels = [_display_name(f, feature_dict) for f in features]
    vals = top["shap_mean_abs_full"].values
    pct = top["pct"].values
    y = np.arange(len(top))

    colors = ["#dc2626" if (i + 1) <= k_selected else "#94a3b8" for i in range(len(top))]
    edges = ["#991b1b" if (i + 1) <= k_selected else "white" for i in range(len(top))]
    ax.barh(y, vals, color=colors, edgecolor=edges, linewidth=1.0, height=0.72)
    ax.set_yticks(y)
    ytick_fs = 8.3 if display_n <= 8 else 7.8
    ax.set_yticklabels(labels, fontsize=ytick_fs)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"b) SHAP Feature Importance (Top {display_n})", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.2, ls="--")

    for i, (v, p) in enumerate(zip(vals, pct)):
        ax.text(v + vals.max() * 0.015, i, f"{p:.1f}%", va="center", fontsize=8)

    # Move detailed textual note to markdown sidecar for better readability.


def _panel_c(
    ax,
    shap_rank: pd.DataFrame,
    feature_dict: Dict[str, Dict[str, str]],
    target: str,
    k_selected: int,
    display_n: int,
):
    shap_export = (
        PROJECT_ROOT
        / "results"
        / "supplementary"
        / "figures"
        / "S5_interpretation"
        / "shap_values"
        / f"SHAP_Data_Export_{target}.csv"
    )
    shap_df = pd.read_csv(shap_export)
    top_features: List[str] = shap_rank["feature"].head(display_n).tolist()
    cmap = mpl.colormaps["coolwarm"]

    for row_idx, feat in enumerate(top_features):
        shap_col = feat
        raw_col = f"raw_{feat}"
        if shap_col not in shap_df.columns or raw_col not in shap_df.columns:
            continue
        x = shap_df[shap_col].values
        raw = shap_df[raw_col].values
        if np.std(raw) == 0:
            norm = np.zeros_like(raw)
        else:
            norm = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw) + 1e-12)
        jitter = 0.11 if display_n <= 6 else 0.10
        point_size = 11 if display_n <= 6 else 9
        y = np.full_like(x, row_idx, dtype=float) + np.random.default_rng(42 + row_idx).normal(
            0, jitter, size=x.shape[0]
        )
        ax.scatter(x, y, c=norm, cmap=cmap, s=point_size, alpha=0.75, linewidths=0)

    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_yticks(np.arange(len(top_features)))
    ytick_fs = 8.5 if display_n <= 8 else 7.9
    ax.set_yticklabels([_display_name(f, feature_dict) for f in top_features], fontsize=ytick_fs)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on POF risk)")
    ax.set_title(f"c) SHAP Summary Plot (Top {display_n})", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.18, ls="--")

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap),
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )
    cbar.set_label("Feature value (low to high)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _panel_d(ax, shap_rank: pd.DataFrame, target: str, k_selected: int, display_n: int):
    full_vals = shap_rank["shap_mean_abs_full"].values
    pct_full = 100 * full_vals / np.sum(full_vals)
    pct = pct_full[:display_n]
    k_all = np.arange(1, len(pct) + 1)
    k_cum = float(np.sum(pct_full[: min(k_selected, len(pct_full))]))

    # Left axis: independent feature contribution bars
    bar_colors = ["#dc2626" if k <= k_selected else "#93c5fd" for k in k_all]
    ax.bar(k_all, pct, color=bar_colors, width=0.65, alpha=0.95)
    ax.set_xlim(0.5, len(pct) + 0.5)
    ax.set_ylim(0, max(30, float(np.max(pct) * 1.22)))
    ax.set_xticks(k_all)
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Independent SHAP contribution (%)", color="#1f2937")
    ax.tick_params(axis="y", labelcolor="#1f2937")

    # Right axis: AUROC vs number of features
    auc_df = _load_auc_by_k(target)
    ax2 = ax.twinx()
    ax2.plot(
        auc_df["k"],
        auc_df["auc"],
        color="#0f766e",
        marker="o",
        lw=2.0,
        ms=4.8,
        label="AUROC",
        zorder=4,
    )
    if {"auc_low", "auc_high"}.issubset(set(auc_df.columns)):
        ax2.fill_between(
            auc_df["k"].values,
            auc_df["auc_low"].values,
            auc_df["auc_high"].values,
            color="#0f766e",
            alpha=0.12,
            lw=0,
            zorder=3,
        )
    ax2.set_ylim(max(0.78, float(auc_df["auc"].min() - 0.01)), float(auc_df["auc"].max() + 0.012))
    ax2.set_ylabel("AUROC", color="#0f766e")
    ax2.tick_params(axis="y", labelcolor="#0f766e")

    # Emphasize k=3 clinical elbow/plateau
    auc_at_k = (
        float(auc_df.loc[auc_df["k"] == k_selected, "auc"].iloc[0]) if (auc_df["k"] == k_selected).any() else np.nan
    )
    if k_selected <= len(pct):
        ax.axvline(k_selected, ls="--", lw=1.2, color="#b91c1c", alpha=0.95)
    if not np.isnan(auc_at_k):
        ax2.scatter([k_selected], [auc_at_k], color="#b91c1c", s=30, zorder=5)
    plateau_note = f"AUROC plateaus after k={k_selected}" if k_selected >= 3 else "AUROC reaches stable range early"
    ax.text(
        0.02,
        0.975,
        f"SHAP-stability selected k={k_selected}\nTop-{k_selected} SHAP = {k_cum:.1f}%\n{plateau_note}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#9ca3af", alpha=0.95),
    )

    # Light labels on selected bars
    for i in range(min(k_selected, len(k_all))):
        ax.text(k_all[i], pct[i] + 0.6, f"{pct[i]:.1f}%", ha="center", va="bottom", fontsize=7.8, color="#7f1d1d")

    ax.set_title(f"d) Independent SHAP vs AUROC (Top {display_n}, selected k={k_selected})", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22, ls="--")


def main():
    parser = argparse.ArgumentParser(description="Generate feature reduction journey quad-panel figure.")
    parser.add_argument("--target", default="pof", choices=["pof", "mortality", "composite"])
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--display-n", type=int, default=8)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dict = _load_feature_dict()
    shap_rank = _load_shap_rankings(args.target)
    k_selected = max(1, min(_load_recommended_k(args.target), len(shap_rank)))
    display_n = max(args.display_n, k_selected)
    display_n = min(display_n, len(shap_rank))
    out_name: Optional[str] = args.out_name or f"feature_reduction_journey_{args.target}"

    fig = plt.figure(figsize=(14.8, 10.2), dpi=300, facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a, args.target)
    _panel_b(ax_b, shap_rank, feature_dict, k_selected, display_n)
    _panel_c(ax_c, shap_rank, feature_dict, args.target, k_selected, display_n)
    _panel_d(ax_d, shap_rank, args.target, k_selected, display_n)

    fig.subplots_adjust(left=0.055, right=0.988, top=0.975, bottom=0.07, wspace=0.22, hspace=0.24)
    out_base = out_dir / out_name
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Export verbose notes to markdown sidecar instead of tiny in-figure text.
    _, _, _, _, _, n_total, selected_n = _prepare_lasso_cv(args.target)
    sel = shap_rank.head(k_selected)
    sel_text = ", ".join([f"{_display_name(r.feature, feature_dict)} ({r.pct:.1f}%)" for _, r in sel.iterrows()])
    notes_path = out_base.parent / f"{out_base.name}_notes.md"
    notes_md = (
        f"# Feature Reduction Journey Notes ({args.target})\n\n"
        f"- Panel a note: {n_total} candidate features reduced to {selected_n} by LASSO 1-SE rule.\n"
        f"- Panel b note: SHAP-stability selected k={k_selected}; selected-set contributors: {sel_text}.\n"
    )
    notes_path.write_text(notes_md, encoding="utf-8")

    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")
    print(f"Saved: {notes_path}")


if __name__ == "__main__":
    main()
