"""Generate Figure 3 as a 2x3 matrix from precomputed upstream data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "main" / "figs"


def _eavg(cal_df: pd.DataFrame) -> float:
    if cal_df.empty:
        return float("nan")
    return float(np.mean(np.abs(cal_df["obs"].values - cal_df["pred"].values)))


def _smooth_roc(fpr: np.ndarray, tpr: np.ndarray, n_grid: int = 200, window: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ROC to uniform grid and smooth minor jaggedness for presentation."""
    x = np.linspace(0.0, 1.0, n_grid)
    y = np.interp(x, fpr, tpr)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    ys = np.convolve(ypad, kernel, mode="valid")
    ys = np.maximum.accumulate(np.clip(ys, 0, 1))
    ys[0], ys[-1] = 0.0, 1.0
    return x, ys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 3 (2x3 matrix) from precomputed data.")
    parser.add_argument("--target", default="pof", choices=["pof", "mortality", "composite"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    target = args.target
    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else (PROJECT_ROOT / "results" / "main" / "figures" / f"fig3_data_{target}")
    )
    out_dir = Path(args.out_dir)
    out_name = args.out_name or f"Fig3_2x3_{target}"

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    required = [
        "internal_roc_summary.csv",
        "internal_roc_curves.csv",
        "external_roc_summary.csv",
        "external_roc_bands.csv",
        "internal_calibration_curve.csv",
        "external_calibration_curve.csv",
        "internal_spike_probs.csv",
        "external_spike_probs.csv",
        "calibration_meta.json",
        "robustness_roc_curves.csv",
        "robustness_auc_summary.csv",
        "performance_summary_forest.csv",
    ]
    missing = [x for x in required if not (data_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing upstream Fig3 data in {data_dir}: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    int_sum = pd.read_csv(data_dir / "internal_roc_summary.csv")
    int_curves = pd.read_csv(data_dir / "internal_roc_curves.csv")
    ext_sum = pd.read_csv(data_dir / "external_roc_summary.csv")
    ext_bands = pd.read_csv(data_dir / "external_roc_bands.csv")
    cal_i = pd.read_csv(data_dir / "internal_calibration_curve.csv")
    cal_e = pd.read_csv(data_dir / "external_calibration_curve.csv")
    spike_i = pd.read_csv(data_dir / "internal_spike_probs.csv")
    spike_e = pd.read_csv(data_dir / "external_spike_probs.csv")
    rob_curves = pd.read_csv(data_dir / "robustness_roc_curves.csv")
    rob_sum = pd.read_csv(data_dir / "robustness_auc_summary.csv")
    forest = pd.read_csv(data_dir / "performance_summary_forest.csv")
    meta = json.loads((data_dir / "calibration_meta.json").read_text(encoding="utf-8"))

    color_pars = "#D7191C"
    color_full = "#2B83BA"
    color_other = "#999999"
    color_ideal = "#000000"
    color_no_event = "#3b82f6"
    color_event = "#ef4444"

    def _lbl(name: str, auc: float, lo: float, hi: float) -> str:
        return f"{name}: {auc:.3f} [{lo:.3f}-{hi:.3f}]"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=300, facecolor="white")
    ax_a, ax_b, ax_e = axes[0]
    ax_c, ax_d, ax_f = axes[1]

    # a) ROC internal: all algorithms
    for model in ["SVM", "Random Forest", "Decision Tree", "Logistic Regression"]:
        cdf = int_curves[int_curves["model"] == model]
        if cdf.empty:
            continue
        sdf = int_sum[int_sum["model"] == model].iloc[0]
        ax_a.plot(cdf["fpr"], cdf["tpr"], color=color_other, lw=1.4, ls=":", alpha=0.65, label=_lbl(model, sdf["auc"], sdf["auc_low"], sdf["auc_high"]))
    cdf = int_curves[int_curves["model"] == "XGBoost"]
    sdf = int_sum[int_sum["model"] == "XGBoost"].iloc[0]
    ax_a.plot(cdf["fpr"], cdf["tpr"], color=color_full, lw=2.5, ls=(0, (10, 5)), label=_lbl("XGBoost", sdf["auc"], sdf["auc_low"], sdf["auc_high"]))
    cdf = int_curves[int_curves["model"] == "Parsimonious XGBoost"]
    sdf = int_sum[int_sum["model"] == "Parsimonious XGBoost"].iloc[0]
    ax_a.plot(cdf["fpr"], cdf["tpr"], color=color_pars, lw=3.0, ls="-", label=_lbl("Parsimonious XGBoost", sdf["auc"], sdf["auc_low"], sdf["auc_high"]))
    ax_a.plot([0, 1], [0, 1], color=color_ideal, lw=1, ls="--")
    ax_a.set_title("a) ROC: Internal", loc="left", fontweight="bold")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("1 - Specificity")
    ax_a.set_ylabel("Sensitivity")
    ax_a.legend(loc="lower right", fontsize=8.7, frameon=False)

    # b) ROC external: full vs parsimonious
    for model, c, ls, lw, alpha_fill in [
        ("Full-feature XGBoost", color_full, (0, (10, 5)), 2.5, 0.08),
        ("Parsimonious XGBoost", color_pars, "-", 3.0, 0.20),
    ]:
        bdf = ext_bands[ext_bands["model"] == model]
        sdf = ext_sum[ext_sum["model"] == model].iloc[0]
        ax_b.fill_between(bdf["fpr"], bdf["tpr_low"], bdf["tpr_high"], color=c, alpha=alpha_fill, linewidth=0)
        name_show = "Full XGBoost" if model.startswith("Full") else "Parsimonious XGBoost"
        ax_b.plot(bdf["fpr"], bdf["tpr"], color=c, lw=lw, ls=ls, label=_lbl(name_show, sdf["auc"], sdf["auc_low"], sdf["auc_high"]))
    ax_b.plot([0, 1], [0, 1], color=color_ideal, lw=1, ls="--")
    ax_b.set_title("b) ROC: External", loc="left", fontweight="bold")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("1 - Specificity")
    ax_b.set_ylabel("Sensitivity")
    ax_b.legend(loc="lower right", fontsize=9.0, frameon=False)

    # e) ROC robustness
    for grp, c in [("Overall", "#4b5563"), ("Non-renal failure", color_pars)]:
        cdf = rob_curves[rob_curves["group"] == grp]
        if cdf.empty:
            continue
        sdf = rob_sum[rob_sum["group"] == grp].iloc[0]
        sfpr, stpr = _smooth_roc(cdf["fpr"].values, cdf["tpr"].values, n_grid=220, window=13)
        ax_e.plot(sfpr, stpr, color=c, lw=2.8, ls="-", label=_lbl(grp, sdf["auc"], sdf["auc_low"], sdf["auc_high"]))
    ax_e.plot([0, 1], [0, 1], color=color_ideal, lw=1, ls="--")
    ax_e.set_title("e) ROC: Robustness", loc="left", fontweight="bold")
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.set_xlabel("1 - Specificity")
    ax_e.set_ylabel("Sensitivity")
    ax_e.legend(loc="lower right", fontsize=8.8, frameon=False)

    # c) Calibration internal
    eavg_i = _eavg(cal_i)
    ax_c.plot([0, 1], [0, 1], ls="--", lw=1.2, color=color_ideal, label="Ideal")
    ax_c.plot(cal_i["pred"], cal_i["obs"], marker="o", lw=2.8, color=color_pars, label="Parsimonious XGBoost")
    inset_c = ax_c.inset_axes([0.12, -0.02, 0.84, 0.18])
    inset_c.patch.set_alpha(0.0)
    bins = np.linspace(0, 1, 26)
    cnt_no, e1 = np.histogram(spike_i.loc[spike_i["label"] == 0, "prob"], bins=bins)
    cnt_yes, _ = np.histogram(spike_i.loc[spike_i["label"] == 1, "prob"], bins=bins)
    ctr = (e1[:-1] + e1[1:]) / 2
    w = (e1[1] - e1[0]) * 0.9
    inset_c.bar(ctr, cnt_yes, width=w, color=color_event, alpha=0.60)
    inset_c.bar(ctr, -cnt_no, width=w, color=color_no_event, alpha=0.55)
    inset_c.axhline(0, color="#111827", lw=0.8, alpha=0.6)
    inset_c.set_xlim(0, 1)
    inset_c.set_xticks([])
    inset_c.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        inset_c.spines[sp].set_visible(False)
    ax_c.set_title("c) Calibration: Internal", loc="left", fontweight="bold")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("Predicted probability")
    ax_c.set_ylabel("Observed probability")
    handles_c = [
        Line2D([0], [0], color=color_ideal, lw=1.2, ls="--", label="Ideal"),
        Line2D([0], [0], color=color_pars, lw=2.8, ls="-", label="Parsimonious XGBoost"),
        Patch(facecolor=color_event, alpha=0.60, label="Event"),
        Patch(facecolor=color_no_event, alpha=0.55, label="No Event"),
    ]
    ax_c.legend(handles=handles_c, loc="lower right", fontsize=8.1, frameon=True, framealpha=0.9)
    ax_c.text(0.02, 0.98, f"Brier Score = {meta['internal']['brier']:.3f}\nEavg = {eavg_i:.3f}",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=8.8,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9ca3af", alpha=0.95))

    # d) Calibration external
    eavg_e = _eavg(cal_e)
    ax_d.plot([0, 1], [0, 1], ls="--", lw=1.2, color=color_ideal, label="Ideal")
    ax_d.plot(cal_e["pred"], cal_e["obs"], marker="o", lw=2.8, color=color_pars, label="Parsimonious XGBoost")
    inset_d = ax_d.inset_axes([0.12, -0.02, 0.84, 0.18])
    inset_d.patch.set_alpha(0.0)
    cnt_no_e, e2 = np.histogram(spike_e.loc[spike_e["label"] == 0, "prob"], bins=bins)
    cnt_yes_e, _ = np.histogram(spike_e.loc[spike_e["label"] == 1, "prob"], bins=bins)
    ctr_e = (e2[:-1] + e2[1:]) / 2
    w_e = (e2[1] - e2[0]) * 0.9
    inset_d.bar(ctr_e, cnt_yes_e, width=w_e, color=color_event, alpha=0.60)
    inset_d.bar(ctr_e, -cnt_no_e, width=w_e, color=color_no_event, alpha=0.55)
    inset_d.axhline(0, color="#111827", lw=0.8, alpha=0.6)
    inset_d.set_xlim(0, 1)
    inset_d.set_xticks([])
    inset_d.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        inset_d.spines[sp].set_visible(False)
    ax_d.set_title("d) Calibration: External", loc="left", fontweight="bold")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_xlabel("Predicted probability")
    ax_d.set_ylabel("Observed probability")
    handles_d = [
        Line2D([0], [0], color=color_ideal, lw=1.2, ls="--", label="Ideal"),
        Line2D([0], [0], color=color_pars, lw=2.8, ls="-", label="Parsimonious XGBoost"),
        Patch(facecolor=color_event, alpha=0.60, label="Event"),
        Patch(facecolor=color_no_event, alpha=0.55, label="No Event"),
    ]
    ax_d.legend(handles=handles_d, loc="lower right", fontsize=8.1, frameon=True, framealpha=0.9)
    ax_d.text(0.02, 0.98, f"Brier Score = {meta['external']['brier']:.3f}\nEavg = {eavg_e:.3f}",
              transform=ax_d.transAxes, ha="left", va="top", fontsize=8.8,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9ca3af", alpha=0.95))

    # f) Performance summary forest
    forest = forest.copy()
    forest["y"] = np.arange(len(forest))[::-1]
    ax_f.hlines(forest["y"], forest["auc_low"], forest["auc_high"], color="#6b7280", lw=2.0)
    ax_f.scatter(forest["auc"], forest["y"], color=color_pars, s=42, zorder=3)
    ax_f.set_yticks(forest["y"])
    ax_f.set_yticklabels(forest["label"])
    ax_f.set_xlim(0.70, 0.95)
    ax_f.set_xlabel("AUC")
    ax_f.set_title("f) Performance Summary", loc="left", fontweight="bold")
    ax_f.grid(axis="x", alpha=0.25, ls="--")
    for _, r in forest.iterrows():
        ax_f.text(min(r["auc_high"] + 0.005, 0.949), r["y"], f"{r['auc']:.3f} [{r['auc_low']:.3f}-{r['auc_high']:.3f}]",
                  va="center", fontsize=8.5)

    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08, wspace=0.20, hspace=0.28)
    out_base = out_dir / out_name
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save caption in markdown
    caption_md = (
        "# Figure 3 Caption\n\n"
        "Figure 3. Performance evaluation, calibration, and robustness of the predictive models.\n"
        "a) Receiver operating characteristic (ROC) curves of the development algorithms in the internal cohort.\n"
        "b) Comparison of ROC curves between the full-feature and parsimonious models in the external cohort.\n"
        "c) and d) Calibration plots for the internal and external cohorts, respectively; the histograms represent the distribution of predicted risks.\n"
        "e) Sensitivity analysis showing the ROC curves for the non-renal failure subgroup compared with the overall population, demonstrating model robustness.\n"
        "f) Summary of AUC values with 95% confidence intervals across different cohorts and clinical subgroups.\n"
    )
    (out_dir / f"{out_name}_caption.md").write_text(caption_md, encoding="utf-8")

    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")
    print(f"Saved: {out_dir / f'{out_name}_caption.md'}")


if __name__ == "__main__":
    main()
