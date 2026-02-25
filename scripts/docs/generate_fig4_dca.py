"""Generate Fig4 Decision Curve Analysis (DCA) for POF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs" / "main" / "figs"


def _net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y_true)
    y_true = y_true.astype(int)
    out = []
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        out.append(nb)
    return np.array(out)


def _smooth_curve(y: np.ndarray, window: int = 9) -> np.ndarray:
    """Simple local smoothing to reduce DCA jaggedness from sparse tails."""
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    ys = np.convolve(ypad, kernel, mode="valid")
    return ys


def _fit_slim_model(target: str, model_dir: Path) -> tuple[CalibratedClassifierCV, list[str], StandardScaler]:
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "mimic_train_processed.csv")
    bundle = joblib.load(model_dir / "deploy_bundle.pkl")
    rec = json.loads((model_dir / "xgb_shap_pruning_recommendation.json").read_text(encoding="utf-8"))
    k = int(rec["k_recommended"])
    curve = pd.read_csv(model_dir / "xgb_shap_pruning_curve.csv")
    feats = curve.loc[curve["k"] == k, "features"].iloc[0].split(";")

    y_train = train_df[target].astype(int).values
    x_train = train_df[feats].copy()
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)

    best_model = bundle["best_model"]
    if hasattr(best_model, "calibrated_classifiers_") and best_model.calibrated_classifiers_:
        base = best_model.calibrated_classifiers_[0].estimator
    else:
        base = best_model
    params = base.get_params()
    params["n_jobs"] = 1
    params["eval_metric"] = "logloss"
    xgb = XGBClassifier(**params)
    slim = CalibratedClassifierCV(xgb, cv=3, method="isotonic", n_jobs=1)
    slim.fit(x_train_s, y_train)
    return slim, feats, scaler


def _plot_panel(ax, thresholds, nb_pars, nb_full, prevalence, title: str):
    nb_pars_s = _smooth_curve(nb_pars, window=9)
    nb_full_s = _smooth_curve(nb_full, window=9)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_all_s = _smooth_curve(nb_all, window=9)
    nb_none = np.zeros_like(thresholds)

    # Clinical relevant range: 0.1-0.5
    ax.axvspan(0.10, 0.50, color="#fde68a", alpha=0.10, zorder=0)
    ax.text(
        0.30, 0.93, "Clinical Relevant Range",
        transform=ax.transAxes, fontsize=12, color="#92400e", fontweight="bold", ha="center"
    )

    # Core curves (foreground)
    ax.plot(
        thresholds, nb_full_s, color="#2B83BA", lw=3.6, ls=(0, (12, 6)),
        alpha=0.60, label="Full XGBoost", zorder=3
    )
    ax.plot(
        thresholds, nb_pars_s, color="#D7191C", lw=4.8, ls="-",
        alpha=1.00, label="Parsimonious XGBoost", zorder=5
    )
    # Reference strategies (background)
    ax.plot(thresholds, nb_all_s, color="#4b5563", lw=1.2, ls="-", label="Intervene All", zorder=2)
    ax.plot(thresholds, nb_none, color="#6b7280", lw=1.2, ls="--", label="Intervene None", zorder=3)
    # Explicit zero-net-benefit baseline
    ax.axhline(0, color="#6b7280", lw=2.0, ls="-", alpha=0.95, zorder=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.00, 0.40)
    ax.set_xlabel("Threshold Probability", fontsize=13)
    ax.set_ylabel("Net Benefit", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="center right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig4-style DCA plot for target outcome.")
    parser.add_argument("--target", default="pof", choices=["pof", "mortality", "composite"])
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    target = args.target
    out_dir = Path(args.out_dir)
    out_name = args.out_name or f"Fig4_decision_curve_{target}"
    model_dir = PROJECT_ROOT / "artifacts" / "models" / target

    warnings.filterwarnings("ignore")
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_data = joblib.load(model_dir / "eval_data.pkl")
    models = joblib.load(model_dir / "all_models_dict.pkl")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "mimic_test_processed.csv")
    eicu = pd.read_csv(PROJECT_ROOT / "data" / "external" / f"eicu_processed_{target}.csv")

    x_test_pre = eval_data["X_test_pre"]
    y_test = np.asarray(eval_data["y_test"]).astype(int)
    full_feats = eval_data["features"]
    x_eicu_full = eicu[full_feats].values
    y_eicu = eicu[target].astype(int).values

    slim_model, slim_feats, slim_scaler = _fit_slim_model(target, model_dir)
    x_test_slim = slim_scaler.transform(test_df[slim_feats].values)
    x_eicu_slim = slim_scaler.transform(eicu[slim_feats].values)

    p_test_full = models["XGBoost"].predict_proba(x_test_pre)[:, 1]
    p_test_slim = slim_model.predict_proba(x_test_slim)[:, 1]
    p_eicu_full = models["XGBoost"].predict_proba(x_eicu_full)[:, 1]
    p_eicu_slim = slim_model.predict_proba(x_eicu_slim)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    nb_test_full = _net_benefit(y_test, p_test_full, thresholds)
    nb_test_slim = _net_benefit(y_test, p_test_slim, thresholds)
    nb_eicu_full = _net_benefit(y_eicu, p_eicu_full, thresholds)
    nb_eicu_slim = _net_benefit(y_eicu, p_eicu_slim, thresholds)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300, facecolor="white")
    _plot_panel(
        axes[0],
        thresholds,
        nb_test_slim,
        nb_test_full,
        prevalence=float(np.mean(y_test)),
        title="a) Decision Curve: Internal Validation (MIMIC-IV)",
    )
    _plot_panel(
        axes[1],
        thresholds,
        nb_eicu_slim,
        nb_eicu_full,
        prevalence=float(np.mean(y_eicu)),
        title="b) Decision Curve: External Validation (eICU-CRD)",
    )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.16, wspace=0.18)
    out_base = out_dir / out_name
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")


if __name__ == "__main__":
    main()
