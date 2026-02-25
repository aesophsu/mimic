"""Generate Supplementary Table S6 in compact Internal vs External layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models"
RESULT_TABLE_DIR = PROJECT_ROOT / "results" / "main" / "tables"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
OUT_DIR = PROJECT_ROOT / "docs" / "suppl" / "tables"
OUT_CSV = OUT_DIR / "TableS6_internal_external_compact.csv"
OUT_MD = OUT_DIR / "TableS6_internal_external_compact.md"


ENDPOINTS = [
    ("Primary Endpoint (POF)", "pof"),
    ("Secondary Endpoint (28-day Mortality)", "mortality"),
    ("Composite Endpoint", "composite"),
]
FULL_MODELS = ["XGBoost", "Random Forest", "SVM", "Logistic Regression", "Decision Tree"]


def _fmt(x: float) -> str:
    return f"{float(x):.3f}"


def _fmt_auc_ci(auc: float, low: float, high: float) -> str:
    return f"{float(auc):.3f} ({float(low):.3f}-{float(high):.3f})"


def _extract_base_estimator(model: Any) -> Any:
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        cal_clf = model.calibrated_classifiers_[0]
        return getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", model))
    return model


def _auc_ci_bootstrap(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, y_prob[idx])))
    if not aucs:
        return np.nan, np.nan
    arr = np.sort(np.array(aucs))
    return float(arr[int(0.025 * len(arr))]), float(arr[int(0.975 * len(arr))])


def _sens_spec_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[float, float, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(sens), float(spec), float(f1)


def _load_full_internal_tables(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = pd.read_csv(MODEL_ROOT / "performance_report.csv")
    perf["Outcome"] = perf["Outcome"].str.lower()
    perf_t = perf[perf["Outcome"] == target].copy()
    # parse main CI text
    def parse_ci(s: str) -> tuple[float, float]:
        core = str(s).split("(", 1)[1].split(")", 1)[0]
        lo, hi = core.split("-")
        return float(lo), float(hi)
    perf_t[["AUC_Low", "AUC_High"]] = perf_t["Main CI"].apply(lambda s: pd.Series(parse_ci(s)))

    diag = pd.read_csv(MODEL_ROOT / target / "internal_diagnostic_perf.csv")
    diag = diag[diag["Group"] == "Full Population"].copy()
    return perf_t, diag


def _load_full_external_table(target: str) -> pd.DataFrame:
    ext = pd.read_csv(RESULT_TABLE_DIR / "Table4_external_validation.csv")
    ext["Target"] = ext["Target"].str.lower()
    return ext[ext["Target"] == target].copy()


def _load_slim_external_row(target: str) -> pd.Series:
    slim = pd.read_csv(RESULT_TABLE_DIR / "Table4_external_validation_slim.csv")
    slim["Target"] = slim["Target"].str.lower()
    return slim[slim["Target"] == target].iloc[0]


def _compute_slim_internal_metrics(target: str) -> dict[str, Any]:
    cfg = json.loads((MODEL_ROOT / target / "external_validation_slim_config.json").read_text(encoding="utf-8"))
    feats = [f for f in cfg.get("features", []) if isinstance(f, str)]
    threshold = float(cfg.get("threshold", 0.5))
    k = int(cfg.get("k", len(feats)))

    train_df = pd.read_csv(DATA_CLEANED / "mimic_train_processed.csv")
    test_df = pd.read_csv(DATA_CLEANED / "mimic_test_processed.csv")

    y_train = train_df[target].dropna().astype(int)
    y_test = test_df[target].dropna().astype(int)
    feats = [f for f in feats if f in train_df.columns and f in test_df.columns][:k]

    x_train = train_df.loc[y_train.index, feats]
    x_test = test_df.loc[y_test.index, feats]

    bundle = joblib.load(MODEL_ROOT / target / "deploy_bundle.pkl")
    base = _extract_base_estimator(bundle["best_model"])
    params = dict(base.get_params()) if hasattr(base, "get_params") else {}
    params["n_jobs"] = 1

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)
    clf = CalibratedClassifierCV(XGBClassifier(**params), cv=3, method="isotonic", n_jobs=1)
    clf.fit(x_train_sc, y_train.values)
    p_test = clf.predict_proba(x_test_sc)[:, 1]

    auc = float(roc_auc_score(y_test.values, p_test))
    auc_lo, auc_hi = _auc_ci_bootstrap(y_test.values, p_test, n_bootstraps=1000, seed=42)
    auprc = float(average_precision_score(y_test.values, p_test))
    sens, spec, f1 = _sens_spec_f1(y_test.values, p_test, threshold)
    brier = float(brier_score_loss(y_test.values, p_test))
    return {
        "k": k,
        "features": ";".join(feats),
        "auc": auc,
        "auc_low": auc_lo,
        "auc_high": auc_hi,
        "auprc": auprc,
        "sens": sens,
        "spec": spec,
        "f1": f1,
        "brier": brier,
    }


def build_table() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for endpoint_label, target in ENDPOINTS:
        perf_t, diag_t = _load_full_internal_tables(target)
        ext_t = _load_full_external_table(target)
        slim_ext = _load_slim_external_row(target)
        slim_int = _compute_slim_internal_metrics(target)

        # endpoint title row
        rows.append(
            {
                "Endpoints & Model": endpoint_label,
                "Internal AUROC (95% CI)": "",
                "Internal AUPRC": "",
                "Internal Sens/Spec": "",
                "Internal F1-score": "",
                "Internal Brier Score": "",
                "External AUROC (95% CI)": "",
                "External AUPRC": "",
                "External Sens/Spec": "",
                "External F1-score": "",
                "External Brier Score": "",
            }
        )

        # model order: slim second
        for model in FULL_MODELS:
            # full model row
            p = perf_t[perf_t["Algorithm"] == model].iloc[0]
            d = diag_t[diag_t["Algorithm"] == model].iloc[0]
            e = ext_t[ext_t["Algorithm"] == model].iloc[0]
            rows.append(
                {
                    "Endpoints & Model": f"{model} (n=12)",
                    "Internal AUROC (95% CI)": _fmt_auc_ci(p["Main AUC"], p["AUC_Low"], p["AUC_High"]),
                    "Internal AUPRC": _fmt(average_precision_score(joblib.load(MODEL_ROOT / target / "eval_data.pkl")["y_test"], joblib.load(MODEL_ROOT / target / "all_models_dict.pkl")[model].predict_proba(joblib.load(MODEL_ROOT / target / "eval_data.pkl")["X_test_pre"])[:, 1])),
                    "Internal Sens/Spec": f"{_fmt(d['Sensitivity'])}/{_fmt(d['Specificity'])}",
                    "Internal F1-score": _fmt(d["F1_Score"]),
                    "Internal Brier Score": _fmt(p["Brier"]),
                    "External AUROC (95% CI)": _fmt_auc_ci(e["AUC"], e["AUC_Low"], e["AUC_High"]),
                    "External AUPRC": _fmt(e["AUPRC"]),
                    "External Sens/Spec": f"{_fmt(e['Sensitivity'])}/{_fmt(e['Specificity'])}",
                    "External F1-score": _fmt(e["F1"]),
                    "External Brier Score": _fmt(e["Brier"]),
                }
            )

            # slim row right after XGBoost
            if model == "XGBoost":
                rows.append(
                    {
                        "Endpoints & Model": f"Parsimonious XGBoost (n={slim_int['k']})",
                        "Internal AUROC (95% CI)": _fmt_auc_ci(slim_int["auc"], slim_int["auc_low"], slim_int["auc_high"]),
                        "Internal AUPRC": _fmt(slim_int["auprc"]),
                        "Internal Sens/Spec": f"{_fmt(slim_int['sens'])}/{_fmt(slim_int['spec'])}",
                        "Internal F1-score": _fmt(slim_int["f1"]),
                        "Internal Brier Score": _fmt(slim_int["brier"]),
                        "External AUROC (95% CI)": _fmt_auc_ci(slim_ext["AUC"], slim_ext["AUC_Low"], slim_ext["AUC_High"]),
                        "External AUPRC": _fmt(slim_ext["AUPRC"]),
                        "External Sens/Spec": f"{_fmt(slim_ext['Sensitivity'])}/{_fmt(slim_ext['Specificity'])}",
                        "External F1-score": _fmt(slim_ext["F1"]),
                        "External Brier Score": _fmt(slim_ext["Brier"]),
                    }
                )

    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Two-row grouped header for CSV
    sub_cols = [
        "AUROC (95% CI)",
        "AUPRC",
        "Sens/Spec",
        "F1-score",
        "Brier Score",
    ]
    header_row1 = ["Endpoints & Model"] + ["Internal"] * len(sub_cols) + ["External"] * len(sub_cols)
    header_row2 = [""] + sub_cols + sub_cols
    data_cols = [
        "Endpoints & Model",
        "Internal AUROC (95% CI)",
        "Internal AUPRC",
        "Internal Sens/Spec",
        "Internal F1-score",
        "Internal Brier Score",
        "External AUROC (95% CI)",
        "External AUPRC",
        "External Sens/Spec",
        "External F1-score",
        "External Brier Score",
    ]
    data_rows = df[data_cols].values.tolist()
    csv_lines = [",".join(header_row1), ",".join(header_row2)]
    for row in data_rows:
        vals = [str(v) if v is not None else "" for v in row]
        csv_lines.append(",".join(vals))
    OUT_CSV.write_text("\n".join(csv_lines), encoding="utf-8")

    # Two-row grouped header for Markdown
    md_lines = []
    md_lines.append("| Endpoints & Model | Internal |  |  |  |  | External |  |  |  |  |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    md_lines.append("|  | AUROC (95% CI) | AUPRC | Sens/Spec | F1-score | Brier Score | AUROC (95% CI) | AUPRC | Sens/Spec | F1-score | Brier Score |")
    for _, r in df.iterrows():
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(r["Endpoints & Model"]),
                    str(r["Internal AUROC (95% CI)"]),
                    str(r["Internal AUPRC"]),
                    str(r["Internal Sens/Spec"]),
                    str(r["Internal F1-score"]),
                    str(r["Internal Brier Score"]),
                    str(r["External AUROC (95% CI)"]),
                    str(r["External AUPRC"]),
                    str(r["External Sens/Spec"]),
                    str(r["External F1-score"]),
                    str(r["External Brier Score"]),
                ]
            )
            + " |"
        )

    lines = [
        "# Supplementary Table S6. Internal vs External Validation (Compact Metrics)",
        "",
        *md_lines,
        "",
        "Notes:",
        "- Internal: MIMIC-IV validation. External: eICU-CRD validation.",
        "- Metrics shown: AUROC (95% CI), AUPRC, Sens/Spec, F1-score, Brier Score.",
        "- Parsimonious model is placed as the second row in each endpoint block.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")


def main() -> None:
    df = build_table()
    write_outputs(df)


if __name__ == "__main__":
    main()
