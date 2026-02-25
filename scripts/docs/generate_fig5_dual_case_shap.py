"""Generate Fig5 dual-case SHAP waterfall contrast for POF."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "pof"
SHAP_DIR = PROJECT_ROOT / "results" / "supplementary" / "figures" / "S5_interpretation" / "shap_values"
OUT_DIR = PROJECT_ROOT / "docs" / "main" / "figs"
FEAT_DICT = PROJECT_ROOT / "artifacts" / "features" / "feature_dictionary.json"

# Harmonize figure units with Table1 conventions where applicable.
UNIT_HARMONIZE = {
    "creatinine_max": ("Creatinine (max)", "umol/L", lambda v: v * 88.4),   # mg/dL -> umol/L
    "bun_min": ("Urea (min)", "mmol/L", lambda v: v * 0.357),               # mg/dL -> mmol/L
    "bun_max": ("Urea (max)", "mmol/L", lambda v: v * 0.357),               # mg/dL -> mmol/L
    "pao2fio2ratio_min": ("P/F ratio (min)", "", lambda v: v),              # ratio: no unit shown
    "ph_min": ("pH (min)", "", lambda v: v),
    "wbc_max": ("WBC (max)", "10^9/L", lambda v: v),
    "lactate_max": ("Lactate (max)", "mmol/L", lambda v: v),
}


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_feature_dict() -> dict:
    if FEAT_DICT.exists():
        return json.loads(FEAT_DICT.read_text(encoding="utf-8"))
    return {}


def _fmt_feature_value(feature: str, raw_val: float, fd: dict) -> str:
    if feature in UNIT_HARMONIZE:
        name, unit, conv = UNIT_HARMONIZE[feature]
        v = conv(raw_val)
        if unit:
            return f"{name} = {v:.2f} {unit}"
        return f"{name} = {v:.2f}"
    cfg = fd.get(feature, {})
    name = cfg.get("display_name_en", feature)
    unit = cfg.get("unit", "")
    if unit:
        return f"{name} = {raw_val:.2f} {unit}"
    return f"{name} = {raw_val:.2f}"


def _prepare_case_data():
    shap_df = pd.read_csv(SHAP_DIR / "SHAP_Data_Export_pof.csv")
    base_logit = float((SHAP_DIR / "SHAP_BaseValue_pof.txt").read_text(encoding="utf-8").strip())
    eval_data = joblib.load(MODEL_DIR / "eval_data.pkl")
    models = joblib.load(MODEL_DIR / "all_models_dict.pkl")
    thr = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))["XGBoost"]

    y = np.asarray(eval_data["y_test"]).astype(int)
    p = models["XGBoost"].predict_proba(eval_data["X_test_pre"])[:, 1]
    pred = (p >= thr).astype(int)

    # Recover unscaled raw test values using persisted split indices
    bundle = joblib.load(MODEL_DIR / "deploy_bundle.pkl")
    test_idx = bundle.get("train_assets_bundle", {}).get("test_idx", None)
    raw_all = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "mimic_raw_scale.csv")
    if test_idx is None:
        raise RuntimeError("test_idx not found in deploy bundle; cannot map raw values for cases.")
    raw_test = raw_all.iloc[list(test_idx)].reset_index(drop=True)

    # Prefer complete cases to avoid NaN in visualized feature values.
    feature_cols = [c for c in shap_df.columns if not c.startswith("raw_")]
    complete_mask = raw_test[feature_cols].notna().all(axis=1).values

    # Case A: confident true positive; Case B: confident true negative
    tp_idx = np.where((y == 1) & (pred == 1) & complete_mask)[0]
    tn_idx = np.where((y == 0) & (pred == 0) & complete_mask)[0]
    # Fallback if strict complete filter is too restrictive
    if len(tp_idx) == 0:
        tp_idx = np.where((y == 1) & (pred == 1))[0]
    if len(tn_idx) == 0:
        tn_idx = np.where((y == 0) & (pred == 0))[0]
    case_a = int(tp_idx[np.argmax(p[tp_idx])])
    case_b = int(tn_idx[np.argmin(p[tn_idx])])
    return shap_df, raw_test, base_logit, p, y, case_a, case_b


def _build_steps(shap_row: pd.Series, top_n: int = 8):
    shap_vals = shap_row.copy()
    shap_vals = shap_vals[~shap_vals.index.str.startswith("raw_")]
    shap_vals = shap_vals.astype(float)
    ordered = shap_vals.reindex(shap_vals.abs().sort_values(ascending=False).index)
    selected = ordered.iloc[:top_n].copy()
    other_sum = float(ordered.iloc[top_n:].sum()) if len(ordered) > top_n else 0.0
    if abs(other_sum) > 1e-8:
        selected.loc["other_features"] = other_sum
    return selected


def _waterfall(
    ax,
    shap_row: pd.Series,
    raw_row: pd.Series,
    base_logit: float,
    pred_prob: float,
    y_true: int,
    title: str,
    fd: dict,
    labels_right: bool = False,
):
    steps = _build_steps(shap_row, top_n=8)
    # Draw bars from top to bottom by absolute contribution
    y_pos = np.arange(len(steps))[::-1]
    curr_logit = base_logit
    base_prob = float(_sigmoid(base_logit))
    for i, (feat, val) in enumerate(steps.items()):
        prev_prob = float(_sigmoid(curr_logit))
        curr_logit += float(val)
        next_prob = float(_sigmoid(curr_logit))
        left = min(prev_prob, next_prob)
        width = abs(next_prob - prev_prob)
        color = "#D7191C" if next_prob >= prev_prob else "#2B83BA"
        ax.barh(y_pos[i], width, left=left, height=0.72, color=color, alpha=0.92, edgecolor="white", linewidth=0.8)

        if feat == "other_features":
            label = "Other features (sum)"
        else:
            raw_val = float(raw_row[feat]) if feat in raw_row.index else float("nan")
            label = _fmt_feature_value(feat, raw_val, fd)
        if labels_right:
            ax.text(
                0.60,
                y_pos[i],
                label,
                va="center",
                ha="left",
                fontsize=8.4,
                color="#111827",
                clip_on=True,
            )
        else:
            ax.text(
                0.005,
                y_pos[i],
                label,
                va="center",
                ha="left",
                fontsize=8.4,
                color="#111827",
                clip_on=True,
            )

    # Base and final markers
    ax.axvline(base_prob, color="#111827", lw=1.2, ls="--")
    ax.axvline(pred_prob, color="#7c2d12", lw=1.2, ls="-")
    ax.text(base_prob, len(steps) + 0.15, f"Base = {base_prob:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(pred_prob, len(steps) + 0.15, f"f(x) = {pred_prob:.3f}", ha="center", va="bottom", fontsize=8.5, color="#7c2d12")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, len(steps) + 0.8)
    ax.set_yticks([])
    ax.set_xlabel("Predicted POF Probability")
    ax.set_title(f"{title}\n(True label={y_true}, Pred={pred_prob:.3f})", loc="left", fontweight="bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Return key drivers for external markdown notes
    pos = steps[steps > 0].sort_values(ascending=False)
    neg = steps[steps < 0].sort_values()
    pos_feat = pos.index[0] if len(pos) else None
    neg_feat = neg.index[0] if len(neg) else None
    return pos_feat, neg_feat


def _driver_sentence(feat: str | None, direction: str, fd: dict) -> str:
    if feat is None:
        return "Primary driver not identified."
    if feat == "pao2fio2ratio_min":
        if direction == "risk":
            return "Primary risk driver: Decreased P/F ratio."
        return "Risk mitigated by preserved oxygenation status."
    if feat == "creatinine_max":
        if direction == "risk":
            return "Primary risk driver: Elevated creatinine indicating renal dysfunction."
        return "Risk mitigated by preserved renal function (lower creatinine)."
    if feat == "ph_min":
        if direction == "risk":
            return "Primary risk driver: Lower pH suggesting acid-base imbalance."
        return "Risk mitigated by near-normal acid-base status."
    if feat == "other_features":
        if direction == "risk":
            return "Primary risk driver: Combined adverse contribution from other physiological factors."
        return "Risk mitigated by other physiological factors."
    label = fd.get(feat, {}).get("display_name_en", feat)
    if direction == "risk":
        return f"Primary risk driver: {label}."
    return f"Risk mitigated by {label}."


def main() -> None:
    warnings.filterwarnings("ignore")
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fd = _load_feature_dict()
    shap_df, raw_test, base_logit, p, y, idx_a, idx_b = _prepare_case_data()

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.8), dpi=300, facecolor="white")
    a_pos, a_neg = _waterfall(
        axes[0],
        shap_df.iloc[idx_a],
        raw_test.iloc[idx_a],
        base_logit=base_logit,
        pred_prob=float(p[idx_a]),
        y_true=int(y[idx_a]),
        title="a) Case A: High-Risk Patient (True Positive)",
        fd=fd,
        labels_right=False,
    )
    b_pos, b_neg = _waterfall(
        axes[1],
        shap_df.iloc[idx_b],
        raw_test.iloc[idx_b],
        base_logit=base_logit,
        pred_prob=float(p[idx_b]),
        y_true=int(y[idx_b]),
        title="b) Case B: Low-Risk Patient (True Negative)",
        fd=fd,
        labels_right=True,
    )

    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.15, wspace=0.12)
    out_base = OUT_DIR / "Fig5_dual_case_shap_waterfall"
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Missingness note (for transparency)
    miss_a = int(raw_test.iloc[idx_a][_build_steps(shap_df.iloc[idx_a], top_n=8).index.intersection(raw_test.columns)].isna().sum())
    miss_b = int(raw_test.iloc[idx_b][_build_steps(shap_df.iloc[idx_b], top_n=8).index.intersection(raw_test.columns)].isna().sum())
    notes = (
        "# Fig5 Notes\n\n"
        "- Case A: True Positive with high predicted POF risk.\n"
        "- Case B: True Negative with low predicted POF risk.\n"
        "- Red bars increase risk; blue bars decrease risk.\n"
        "- Units were harmonized to Table1 conventions where applicable "
        "(e.g., Creatinine in umol/L, Urea in mmol/L, P/F ratio shown without unit).\n"
        f"- Missing values among displayed top features: Case A = {miss_a}, Case B = {miss_b}. "
        "If non-zero, missingness was handled by model-imputation pipeline (median/MICE as configured upstream).\n"
        f"- Case A interpretation: {_driver_sentence(a_pos, 'risk', fd)} {_driver_sentence(a_neg, 'protect', fd)}\n"
        f"- Case B interpretation: {_driver_sentence(b_pos, 'risk', fd)} {_driver_sentence(b_neg, 'protect', fd)}\n"
    )
    (OUT_DIR / "Fig5_dual_case_shap_waterfall_notes.md").write_text(notes, encoding="utf-8")

    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")
    print(f"Saved: {OUT_DIR / 'Fig5_dual_case_shap_waterfall_notes.md'}")


if __name__ == "__main__":
    main()
