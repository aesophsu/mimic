import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log as _log, log_header
from utils.paths import get_main_table_dir, get_model_dir, ensure_dirs
from utils.study_config import OUTCOMES


TABLE_DIR = get_main_table_dir()
FULL_EXTERNAL_PATH = os.path.join(TABLE_DIR, "Table4_external_validation.csv")
SLIM_EXTERNAL_PATH = os.path.join(TABLE_DIR, "Table4_external_validation_slim.csv")
SHAP_SUMMARY_PATH = os.path.join(get_model_dir(), "xgb_shap_pruning_summary.csv")


def _safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


def _read_full_internal_auc(target: str) -> float:
    path = os.path.join(get_model_dir(target), "internal_diagnostic_perf.csv")
    if not os.path.exists(path):
        return np.nan
    try:
        df = pd.read_csv(path)
        sub = df[
            (df["Algorithm"].astype(str).str.lower() == "xgboost")
            & (df["Group"].astype(str).str.contains("Full", case=False, na=False))
            & (df["Outcome"].astype(str).str.lower() == target.lower())
        ]
        if sub.empty:
            return np.nan
        return _safe_float(sub.iloc[0]["AUC"])
    except Exception:
        return np.nan


def _read_full_external_auc(target: str) -> float:
    if not os.path.exists(FULL_EXTERNAL_PATH):
        return np.nan
    try:
        df = pd.read_csv(FULL_EXTERNAL_PATH)
        sub = df[
            (df["Target"].astype(str).str.lower() == target.lower())
            & (df["Algorithm"].astype(str).str.lower() == "xgboost")
        ]
        if sub.empty:
            return np.nan
        return _safe_float(sub.iloc[0]["AUC"])
    except Exception:
        return np.nan


def _read_slim_internal_row(target: str) -> tuple[float, float]:
    if not os.path.exists(SHAP_SUMMARY_PATH):
        return np.nan, np.nan
    try:
        df = pd.read_csv(SHAP_SUMMARY_PATH)
        sub = df[df["target"].astype(str).str.lower() == target.lower()]
        if sub.empty:
            return np.nan, np.nan
        row = sub.iloc[0]
        return _safe_float(row.get("auc_recommended", np.nan)), _safe_float(row.get("k_recommended", np.nan))
    except Exception:
        return np.nan, np.nan


def _read_slim_external_row(target: str) -> tuple[float, float]:
    if not os.path.exists(SLIM_EXTERNAL_PATH):
        return np.nan, np.nan
    try:
        df = pd.read_csv(SLIM_EXTERNAL_PATH)
        sub = df[df["Target"].astype(str).str.lower() == target.lower()]
        if sub.empty:
            return np.nan, np.nan
        row = sub.iloc[0]
        return _safe_float(row.get("AUC", np.nan)), _safe_float(row.get("K", np.nan))
    except Exception:
        return np.nan, np.nan


def _retention(slim: float, full: float) -> float:
    if np.isnan(slim) or np.isnan(full) or full <= 0:
        return np.nan
    return slim / full


def _delta(slim: float, full: float) -> float:
    if np.isnan(slim) or np.isnan(full):
        return np.nan
    return slim - full


def main() -> None:
    log_header("📊 10c_full_vs_slim_dual_domain: Full vs Slim 双域性能折损评估")
    ensure_dirs(TABLE_DIR)
    rows = []

    for target in OUTCOMES:
        auc_full_internal = _read_full_internal_auc(target)
        auc_full_external = _read_full_external_auc(target)
        auc_slim_internal, k_internal = _read_slim_internal_row(target)
        auc_slim_external, k_external = _read_slim_external_row(target)
        k_used = int(k_external) if not np.isnan(k_external) else (int(k_internal) if not np.isnan(k_internal) else np.nan)

        rows.append(
            {
                "target": target,
                "k_slim": k_used,
                "internal_auc_full": auc_full_internal,
                "internal_auc_slim": auc_slim_internal,
                "internal_auc_delta": _delta(auc_slim_internal, auc_full_internal),
                "internal_auc_retention": _retention(auc_slim_internal, auc_full_internal),
                "external_auc_full": auc_full_external,
                "external_auc_slim": auc_slim_external,
                "external_auc_delta": _delta(auc_slim_external, auc_full_external),
                "external_auc_retention": _retention(auc_slim_external, auc_full_external),
            }
        )

    out = pd.DataFrame(rows)
    out_path = os.path.join(TABLE_DIR, "Table4c_full_vs_slim_dual_domain.csv")
    out.to_csv(out_path, index=False)
    _log(f"双域对比表已导出: {os.path.abspath(out_path)}", "OK")

    for _, r in out.iterrows():
        _log(
            f"[{r['target']}] k={int(r['k_slim']) if not np.isnan(r['k_slim']) else 'NA'} | "
            f"internal retention={r['internal_auc_retention']:.3f} | "
            f"external retention={r['external_auc_retention']:.3f}",
            "INFO",
        )


if __name__ == "__main__":
    main()
