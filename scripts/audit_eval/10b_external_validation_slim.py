import argparse
import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log as _log, log_header
from utils.paths import (
    ensure_dirs,
    get_cleaned_path,
    get_external_dir,
    get_main_table_dir,
    get_model_dir,
)
from utils.study_config import OUTCOME_TYPE, OUTCOMES


TRAIN_PATH = get_cleaned_path("mimic_train_processed.csv")
TEST_PATH = get_cleaned_path("mimic_test_processed.csv")
EXTERNAL_DIR = get_external_dir()
TABLE_DIR = get_main_table_dir()

DEFAULT_K = {
    "pof": 3,
    "mortality": 4,
    "composite": 4,
}


def _load_bundle(target: str) -> dict[str, Any]:
    bundle_path = os.path.join(get_model_dir(target), "deploy_bundle.pkl")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"deploy_bundle 不存在: {bundle_path}")
    return joblib.load(bundle_path)


def _extract_base_estimator(model: Any) -> Any:
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        cal_clf = model.calibrated_classifiers_[0]
        return getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", model))
    return model


def _get_ranked_features(bundle: dict[str, Any]) -> list[str]:
    features = list(bundle.get("features") or bundle.get("feature_names") or [])
    if not features:
        raise ValueError("deploy_bundle 缺少 features")
    model = bundle.get("best_model")
    if model is None:
        return features
    try:
        base = _extract_base_estimator(model)
        importances = getattr(base, "feature_importances_", None)
        if importances is not None and len(importances) == len(features):
            pairs = sorted(zip(features, importances), key=lambda x: float(x[1]), reverse=True)
            return [name for name, _ in pairs]
    except Exception as e:
        _log(f"[{bundle.get('target_outcome', 'unknown')}] 读取特征重要性失败，回退原顺序: {e}", "WARN")
    return features


def _load_shap_recommendation(target: str) -> dict[str, Any] | None:
    rec_path = os.path.join(get_model_dir(target), "xgb_shap_pruning_recommendation.json")
    if not os.path.exists(rec_path):
        return None
    try:
        with open(rec_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        if isinstance(rec, dict):
            return rec
    except Exception as e:
        _log(f"[{target}] 读取 SHAP 推荐失败: {e}", "WARN")
    return None


def _parse_feature_str(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [x.strip() for x in value.split(";") if x.strip()]


def _get_shap_features_by_k(target: str, k: int) -> list[str] | None:
    curve_path = os.path.join(get_model_dir(target), "xgb_shap_pruning_curve.csv")
    if not os.path.exists(curve_path):
        return None
    try:
        df = pd.read_csv(curve_path)
        if "k" not in df.columns or "features" not in df.columns or df.empty:
            return None
        df["k"] = df["k"].astype(int)
        row = df[df["k"] == int(k)]
        if row.empty:
            row = df.sort_values("k").tail(1)
        return _parse_feature_str(row.iloc[0]["features"])
    except Exception as e:
        _log(f"[{target}] 读取 SHAP 曲线失败: {e}", "WARN")
        return None


def _resolve_k(target: str, k_override: int | None) -> int:
    if k_override is not None:
        return int(k_override)
    rec = _load_shap_recommendation(target)
    if rec and "k_recommended" in rec:
        try:
            return int(rec["k_recommended"])
        except Exception:
            pass
    return int(DEFAULT_K.get(target, 4))


def _resolve_features(
    target: str,
    bundle: dict[str, Any],
    feature_source: str,
    k_use: int,
) -> tuple[list[str], str]:
    if feature_source == "shap":
        shap_feats = _get_shap_features_by_k(target, k_use)
        if shap_feats:
            return shap_feats, "shap_curve"
        _log(f"[{target}] 未找到 SHAP 曲线特征，回退到 XGBoost importance", "WARN")
    ranked = _get_ranked_features(bundle)
    return ranked[:k_use], "xgb_importance"


def _get_xgb_params(bundle: dict[str, Any]) -> dict[str, Any]:
    default = {"random_state": 42, "eval_metric": "logloss", "n_jobs": 1}
    model = bundle.get("best_model")
    if model is None:
        return default
    try:
        base = _extract_base_estimator(model)
        params = dict(base.get_params())
        params["n_jobs"] = 1
        return params
    except Exception:
        return default


def _auc_ci_bootstrap(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, y_prob[idx]))
    if not aucs:
        return np.nan, np.nan
    aucs = np.sort(np.array(aucs))
    lo = float(aucs[int(0.025 * len(aucs))])
    hi = float(aucs[int(0.975 * len(aucs))])
    return lo, hi


def _select_threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    return float(thr[idx])


def _fit_slim_model(
    x_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    xgb_params: dict[str, Any],
) -> tuple[StandardScaler, CalibratedClassifierCV]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    xgb = XGBClassifier(**xgb_params)
    clf = CalibratedClassifierCV(xgb, cv=3, method="isotonic", n_jobs=1)
    clf.fit(x_train, y_train)
    return scaler, clf


def _run_one_target(
    target: str,
    k: int | None,
    feature_source: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_eicu: pd.DataFrame,
) -> dict[str, Any]:
    bundle = _load_bundle(target)
    k_use = _resolve_k(target, k)
    chosen_feats, source_used = _resolve_features(target, bundle, feature_source=feature_source, k_use=k_use)
    usable = [f for f in chosen_feats if f in df_train.columns and f in df_test.columns and f in df_eicu.columns]
    if not usable and feature_source == "shap":
        _log(f"[{target}] SHAP 特征在数据列中不可用，回退到 XGBoost importance", "WARN")
        fallback_ranked = _get_ranked_features(bundle)
        usable = [f for f in fallback_ranked if f in df_train.columns and f in df_test.columns and f in df_eicu.columns]
        source_used = "xgb_importance_fallback"
    if not usable:
        raise ValueError(f"{target}: 无可用特征可用于精简外部验证")
    feats = usable[: min(k_use, len(usable))]
    k_use = len(feats)

    y_train = df_train[target].dropna().astype(int)
    y_test = df_test[target].dropna().astype(int)
    y_eicu = df_eicu[target].dropna().astype(int)

    x_train_raw = df_train.loc[y_train.index, feats]
    x_test_raw = df_test.loc[y_test.index, feats]
    x_eicu_raw = df_eicu.loc[y_eicu.index, feats]

    scaler, clf = _fit_slim_model(x_train_raw, y_train.values, _get_xgb_params(bundle))
    p_test = clf.predict_proba(scaler.transform(x_test_raw))[:, 1]
    threshold = _select_threshold_by_youden(y_test.values, p_test)

    p_eicu = clf.predict_proba(scaler.transform(x_eicu_raw))[:, 1]
    y_pred_eicu = (p_eicu >= threshold).astype(int)
    auc = float(roc_auc_score(y_eicu.values, p_eicu))
    auc_lo, auc_hi = _auc_ci_bootstrap(y_eicu.values, p_eicu)
    auprc = float(average_precision_score(y_eicu.values, p_eicu))
    brier = float(brier_score_loss(y_eicu.values, p_eicu))
    tn, fp, fn, tp = confusion_matrix(y_eicu.values, y_pred_eicu).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    row = {
        "Endpoint": OUTCOME_TYPE.get(target, target),
        "Target": target,
        "Algorithm": f"XGBoost-Slim(k={k_use})",
        "K": int(k_use),
        "Features": ";".join(feats),
        "AUC": auc,
        "AUC_Low": auc_lo,
        "AUC_High": auc_hi,
        "AUPRC": auprc,
        "Brier": brier,
        "Sensitivity": sens,
        "Specificity": spec,
        "Threshold": threshold,
        "N_External": int(len(y_eicu)),
        "Feature_Source": source_used,
    }

    target_dir = get_model_dir(target)
    ensure_dirs(target_dir)
    pd.DataFrame([row]).to_csv(os.path.join(target_dir, "external_validation_slim.csv"), index=False)
    with open(os.path.join(target_dir, "external_validation_slim_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"target": target, "k": k_use, "features": feats, "threshold": threshold, "feature_source": source_used},
            f,
            ensure_ascii=False,
            indent=2,
        )

    _log(
        f"[{target}] k={k_use} | AUC={auc:.4f} ({auc_lo:.4f}-{auc_hi:.4f}) | Sens={sens:.4f} | Spec={spec:.4f}",
        "OK",
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="精简版（k）XGBoost eICU 外部验证（默认使用 SHAP 推荐特征）")
    parser.add_argument("--targets", nargs="+", default=OUTCOMES, help=f"结局列表，默认: {' '.join(OUTCOMES)}")
    parser.add_argument("--k-pof", type=int, default=None, help="POF 的精简特征数（默认读取 SHAP 推荐）")
    parser.add_argument("--k-mortality", type=int, default=None, help="Mortality 的精简特征数（默认读取 SHAP 推荐）")
    parser.add_argument("--k-composite", type=int, default=None, help="Composite 的精简特征数（默认读取 SHAP 推荐）")
    parser.add_argument(
        "--feature-source",
        choices=["shap", "xgb"],
        default="shap",
        help="特征来源：shap=读取 06c 结果；xgb=按 feature_importances_ 排序",
    )
    args = parser.parse_args()

    log_header("🚀 10b_external_validation_slim: 精简版 XGBoost 外部验证（eICU）")
    _log(f"TRAIN: {os.path.abspath(TRAIN_PATH)}", "INFO")
    _log(f"TEST:  {os.path.abspath(TEST_PATH)}", "INFO")
    _log(f"eICU:  {os.path.abspath(EXTERNAL_DIR)}", "INFO")

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("缺少 mimic_train_processed.csv 或 mimic_test_processed.csv，请先运行 02 与 06")

    ensure_dirs(TABLE_DIR)
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    k_map = {
        "pof": None if args.k_pof is None else int(args.k_pof),
        "mortality": None if args.k_mortality is None else int(args.k_mortality),
        "composite": None if args.k_composite is None else int(args.k_composite),
    }

    rows = []
    for target in [t.lower() for t in args.targets]:
        eicu_path = os.path.join(EXTERNAL_DIR, f"eicu_processed_{target}.csv")
        if target not in k_map:
            _log(f"跳过未知结局: {target}", "WARN")
            continue
        if not os.path.exists(eicu_path):
            _log(f"跳过 {target}: 缺少外部验证文件 {eicu_path}", "WARN")
            continue
        if target not in df_train.columns or target not in df_test.columns:
            _log(f"跳过 {target}: MIMIC train/test 中缺少结局列", "WARN")
            continue
        df_eicu = pd.read_csv(eicu_path)
        if target not in df_eicu.columns:
            _log(f"跳过 {target}: eICU 文件中缺少结局列", "WARN")
            continue
        try:
            rows.append(_run_one_target(target, k_map[target], args.feature_source, df_train, df_test, df_eicu))
        except Exception as e:
            _log(f"{target} 失败: {e}", "ERR")

    if not rows:
        _log("未生成任何精简外部验证结果。", "WARN")
        return

    out = pd.DataFrame(rows)
    order_map = {"pof": 0, "mortality": 1, "composite": 2}
    out["_order"] = out["Target"].map(lambda x: order_map.get(str(x).lower(), 9))
    out = out.sort_values(["_order"]).drop(columns=["_order"]).reset_index(drop=True)
    out_path = os.path.join(TABLE_DIR, "Table4_external_validation_slim.csv")
    out.to_csv(out_path, index=False)
    _log(f"表格已导出: {os.path.abspath(out_path)}", "OK")


if __name__ == "__main__":
    main()
