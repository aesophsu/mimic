"""Prepare precomputed data artifacts for Fig3 plotting.

This script is upstream: it computes ROC/calibration related data once and
exports machine-readable files. Downstream plotting script should only read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import get_cleaned_path, get_external_dir, get_main_figure_dir, get_model_dir, ensure_dirs
from utils.logger import log as _log, log_header


TRAIN_PATH = get_cleaned_path("mimic_train_processed.csv")
TEST_PATH = get_cleaned_path("mimic_test_processed.csv")


def _auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        vals.append(float(roc_auc_score(yb, p[idx])))
    arr = np.sort(np.asarray(vals))
    if len(arr) == 0:
        return auc, np.nan, np.nan
    return auc, float(arr[int(0.025 * len(arr))]), float(arr[int(0.975 * len(arr))])


def _roc_band(y: np.ndarray, p: np.ndarray, n_boot: int = 400, seed: int = 42) -> pd.DataFrame:
    fpr_grid = np.linspace(0, 1, 201)
    rng = np.random.default_rng(seed)
    tprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        fpr, tpr, _ = roc_curve(yb, p[idx])
        tprs.append(np.interp(fpr_grid, fpr, tpr))
    arr = np.asarray(tprs)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    fpr, tpr, _ = roc_curve(y, p)
    mean_tpr = np.interp(fpr_grid, fpr, tpr)
    return pd.DataFrame({"fpr": fpr_grid, "tpr": mean_tpr, "tpr_low": lo, "tpr_high": hi})


def _hosmer_lemeshow(y: np.ndarray, p: np.ndarray, g: int = 10) -> tuple[float, float]:
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p")
    df["bin"] = pd.qcut(df["p"], q=g, duplicates="drop")
    grp = df.groupby("bin", observed=False)
    obs = grp["y"].sum().values
    exp = grp["p"].sum().values
    n = grp.size().values
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.nansum((obs - exp) ** 2 / (exp * (1 - exp / n) + 1e-12))
    dof = max(len(obs) - 2, 1)
    pval = float(1 - chi2.cdf(stat, dof))
    return float(stat), pval


def _calib_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    p2 = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p2 / (1 - p2))
    lr = LogisticRegression(fit_intercept=True, solver="lbfgs")
    lr.fit(logit_p.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _fit_slim_model(target: str, model_dir: str) -> tuple[CalibratedClassifierCV, list[str], StandardScaler]:
    train_df = pd.read_csv(TRAIN_PATH)
    bundle = joblib.load(os.path.join(model_dir, "deploy_bundle.pkl"))
    rec = json.load(open(os.path.join(model_dir, "xgb_shap_pruning_recommendation.json"), "r", encoding="utf-8"))
    k = int(rec["k_recommended"])
    curve = pd.read_csv(os.path.join(model_dir, "xgb_shap_pruning_curve.csv"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare upstream Fig3 plot data for a specific outcome.")
    parser.add_argument("--target", default="pof", choices=["pof", "mortality", "composite"])
    args = parser.parse_args()
    target = args.target
    model_dir = get_model_dir(target)
    out_dir = os.path.join(get_main_figure_dir(), f"fig3_data_{target}")
    eicu_path = os.path.join(get_external_dir(), f"eicu_processed_{target}.csv")

    log_header(f"🚀 10c_prepare_fig3_plot_data: export plot-ready data for Fig3 ({target})")
    ensure_dirs(out_dir)
    _log(f"Output: {os.path.abspath(out_dir)}", "INFO")

    eval_data = joblib.load(os.path.join(model_dir, "eval_data.pkl"))
    models = joblib.load(os.path.join(model_dir, "all_models_dict.pkl"))
    test_df = pd.read_csv(TEST_PATH)
    eicu = pd.read_csv(eicu_path)

    x_test_pre = eval_data["X_test_pre"]
    y_test = np.asarray(eval_data["y_test"]).astype(int)
    sub_mask = np.asarray(eval_data.get("sub_mask", np.ones_like(y_test))).astype(bool)
    full_feats = eval_data["features"]
    x_eicu_full = eicu[full_feats].values
    y_eicu = eicu[target].astype(int).values

    slim_model, slim_feats, slim_scaler = _fit_slim_model(target, model_dir)
    x_test_slim = slim_scaler.transform(test_df[slim_feats].values)
    x_eicu_slim = slim_scaler.transform(eicu[slim_feats].values)

    # Internal ROC: 5 full models + slim
    internal_pred = {name: clf.predict_proba(x_test_pre)[:, 1] for name, clf in models.items()}
    internal_pred["Parsimonious XGBoost"] = slim_model.predict_proba(x_test_slim)[:, 1]

    rows_summary = []
    rows_curve = []
    for name, p in internal_pred.items():
        auc, lo, hi = _auc_ci(y_test, p, n_boot=1000, seed=41)
        rows_summary.append({"model": name, "auc": auc, "auc_low": lo, "auc_high": hi})
        fpr, tpr, _ = roc_curve(y_test, p)
        rows_curve.extend([{"model": name, "fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)])
    pd.DataFrame(rows_summary).to_csv(os.path.join(out_dir, "internal_roc_summary.csv"), index=False)
    pd.DataFrame(rows_curve).to_csv(os.path.join(out_dir, "internal_roc_curves.csv"), index=False)

    # External ROC: full xgb + slim + CI bands
    ext_models = {
        "Full-feature XGBoost": models["XGBoost"].predict_proba(x_eicu_full)[:, 1],
        "Parsimonious XGBoost": slim_model.predict_proba(x_eicu_slim)[:, 1],
    }
    ext_sum = []
    ext_curve = []
    ext_band = []
    for name, p in ext_models.items():
        auc, lo, hi = _auc_ci(y_eicu, p, n_boot=1200, seed=84)
        ext_sum.append({"model": name, "auc": auc, "auc_low": lo, "auc_high": hi})
        fpr, tpr, _ = roc_curve(y_eicu, p)
        ext_curve.extend([{"model": name, "fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)])
        band = _roc_band(y_eicu, p, n_boot=400, seed=84)
        band["model"] = name
        ext_band.append(band)
    pd.DataFrame(ext_sum).to_csv(os.path.join(out_dir, "external_roc_summary.csv"), index=False)
    pd.DataFrame(ext_curve).to_csv(os.path.join(out_dir, "external_roc_curves.csv"), index=False)
    pd.concat(ext_band, ignore_index=True).to_csv(os.path.join(out_dir, "external_roc_bands.csv"), index=False)

    # Calibration data for slim model (internal + external)
    p_int_slim = slim_model.predict_proba(x_test_slim)[:, 1]
    p_ext_slim = ext_models["Parsimonious XGBoost"]
    pt_i, pp_i = calibration_curve(y_test, p_int_slim, n_bins=10, strategy="quantile")
    pt_e, pp_e = calibration_curve(y_eicu, p_ext_slim, n_bins=10, strategy="quantile")
    pd.DataFrame({"pred": pp_i, "obs": pt_i}).to_csv(os.path.join(out_dir, "internal_calibration_curve.csv"), index=False)
    pd.DataFrame({"pred": pp_e, "obs": pt_e}).to_csv(os.path.join(out_dir, "external_calibration_curve.csv"), index=False)
    pd.DataFrame({"prob": p_int_slim, "label": y_test}).to_csv(
        os.path.join(out_dir, "internal_spike_probs.csv"), index=False
    )
    pd.DataFrame({"prob": p_ext_slim, "label": y_eicu}).to_csv(
        os.path.join(out_dir, "external_spike_probs.csv"), index=False
    )

    hl_i, hlp_i = _hosmer_lemeshow(y_test, p_int_slim, g=10)
    hl_e, hlp_e = _hosmer_lemeshow(y_eicu, p_ext_slim, g=10)
    slope_e, intc_e = _calib_slope_intercept(y_eicu, p_ext_slim)
    meta = {
        "target": target,
        "internal": {"brier": float(brier_score_loss(y_test, p_int_slim)), "hl_stat": hl_i, "hl_p": hlp_i},
        "external": {
            "brier": float(brier_score_loss(y_eicu, p_ext_slim)),
            "hl_stat": hl_e,
            "hl_p": hlp_e,
            "slope": slope_e,
            "intercept": intc_e,
        },
        "slim_features": slim_feats,
    }
    with open(os.path.join(out_dir, "calibration_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Robustness ROC: overall vs non-renal subgroup (internal parsimonious model)
    rob_rows = []
    rob_sum = []
    # overall
    auc_o, lo_o, hi_o = _auc_ci(y_test, p_int_slim, n_boot=1000, seed=97)
    fpr_o, tpr_o, _ = roc_curve(y_test, p_int_slim)
    rob_rows.extend([{"group": "Overall", "fpr": float(f), "tpr": float(t)} for f, t in zip(fpr_o, tpr_o)])
    rob_sum.append({"group": "Overall", "auc": auc_o, "auc_low": lo_o, "auc_high": hi_o, "n": int(len(y_test))})
    # non-renal
    if sub_mask.sum() > 10 and np.unique(y_test[sub_mask]).size >= 2:
        auc_nr, lo_nr, hi_nr = _auc_ci(y_test[sub_mask], p_int_slim[sub_mask], n_boot=1000, seed=98)
        fpr_nr, tpr_nr, _ = roc_curve(y_test[sub_mask], p_int_slim[sub_mask])
        rob_rows.extend([{"group": "Non-renal failure", "fpr": float(f), "tpr": float(t)} for f, t in zip(fpr_nr, tpr_nr)])
        rob_sum.append(
            {
                "group": "Non-renal failure",
                "auc": auc_nr,
                "auc_low": lo_nr,
                "auc_high": hi_nr,
                "n": int(sub_mask.sum()),
            }
        )
    pd.DataFrame(rob_rows).to_csv(os.path.join(out_dir, "robustness_roc_curves.csv"), index=False)
    pd.DataFrame(rob_sum).to_csv(os.path.join(out_dir, "robustness_auc_summary.csv"), index=False)

    # Forest-style summary points for panel f
    forest = [
        {"label": "Internal", "auc": auc_o, "auc_low": lo_o, "auc_high": hi_o},
        {"label": "External", "auc": ext_sum[1]["auc"] if len(ext_sum) > 1 else ext_sum[0]["auc"],
         "auc_low": ext_sum[1]["auc_low"] if len(ext_sum) > 1 else ext_sum[0]["auc_low"],
         "auc_high": ext_sum[1]["auc_high"] if len(ext_sum) > 1 else ext_sum[0]["auc_high"]},
    ]
    if any(x["group"] == "Non-renal failure" for x in rob_sum):
        nr = [x for x in rob_sum if x["group"] == "Non-renal failure"][0]
        forest.append({"label": "Non-renal failure", "auc": nr["auc"], "auc_low": nr["auc_low"], "auc_high": nr["auc_high"]})
    pd.DataFrame(forest).to_csv(os.path.join(out_dir, "performance_summary_forest.csv"), index=False)

    _log("Fig3 plot data exported.", "OK")


if __name__ == "__main__":
    main()
