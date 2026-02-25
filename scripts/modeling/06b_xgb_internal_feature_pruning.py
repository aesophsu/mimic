import argparse
import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log as _log, log_header
from utils.paths import get_cleaned_path, get_model_dir, ensure_dirs
from utils.study_config import OUTCOMES


TRAIN_PATH = get_cleaned_path("mimic_train_processed.csv")
TEST_PATH = get_cleaned_path("mimic_test_processed.csv")


def _auc_ci_bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, y_prob[idx])))
    if not aucs:
        return np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def _load_target_bundle(target: str) -> dict[str, Any]:
    bundle_path = os.path.join(get_model_dir(target), "deploy_bundle.pkl")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"缺少 deploy_bundle: {bundle_path}")
    return joblib.load(bundle_path)


def _get_ranked_features(bundle: dict[str, Any]) -> list[str]:
    features = list(bundle.get("features") or bundle.get("feature_names") or [])
    if not features:
        raise ValueError("deploy_bundle 缺少 features")

    model = bundle.get("best_model")
    if model is None:
        return features

    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cal_clf = model.calibrated_classifiers_[0]
            base = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
        else:
            base = model
        importances = getattr(base, "feature_importances_", None)
        if importances is not None and len(importances) == len(features):
            pairs = sorted(zip(features, importances), key=lambda x: float(x[1]), reverse=True)
            return [p[0] for p in pairs]
    except Exception as e:
        _log(f"特征重要性读取失败，按原顺序回退: {e}", "WARN")
    return features


def _get_xgb_params(bundle: dict[str, Any]) -> dict[str, Any]:
    model = bundle.get("best_model")
    if model is None:
        return {"random_state": 42, "eval_metric": "logloss", "n_jobs": 1}

    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cal_clf = model.calibrated_classifiers_[0]
            base = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
        else:
            base = model
        params = dict(base.get_params())
        params["n_jobs"] = 1  # 沙箱/本地稳定运行
        return params
    except Exception:
        return {"random_state": 42, "eval_metric": "logloss", "n_jobs": 1}


def _fit_eval_topk(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    xgb_params: dict[str, Any],
    auc_ci_bootstraps: int = 1000,
    auc_ci_seed: int = 42,
) -> dict[str, float]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = XGBClassifier(**xgb_params)
    clf = CalibratedClassifierCV(model, cv=3, method="isotonic", n_jobs=1)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, y_prob))
    auc_lo, auc_hi = _auc_ci_bootstrap(y_test, y_prob, n_bootstraps=auc_ci_bootstraps, seed=auc_ci_seed)

    return {
        "auc": auc,
        "auc_low": auc_lo,
        "auc_high": auc_hi,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "brier": float(brier_score_loss(y_test, y_prob)),
    }


def _run_target(
    target: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    min_k: int,
    max_k: int | None,
    auc_tolerance: float,
    auc_ci_bootstraps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = _load_target_bundle(target)
    ranked_features = _get_ranked_features(bundle)
    xgb_params = _get_xgb_params(bundle)

    usable = [f for f in ranked_features if f in df_train.columns and f in df_test.columns]
    if not usable:
        raise ValueError(f"{target}: 无可用特征")

    y_train = df_train[target].dropna().astype(int)
    y_test = df_test[target].dropna().astype(int)
    X_train_base = df_train.loc[y_train.index, usable]
    X_test_base = df_test.loc[y_test.index, usable]

    k_min = max(1, min_k)
    k_max = len(usable) if max_k is None else min(max_k, len(usable))
    if k_min > k_max:
        raise ValueError(f"{target}: min_k({k_min}) > max_k({k_max})")

    rows = []
    for k in range(k_min, k_max + 1):
        feats = usable[:k]
        metrics = _fit_eval_topk(
            X_train_base[feats],
            y_train.values,
            X_test_base[feats],
            y_test.values,
            xgb_params,
            auc_ci_bootstraps=auc_ci_bootstraps,
            auc_ci_seed=42 + k,
        )
        rows.append(
            {
                "target": target,
                "k": k,
                "features": ";".join(feats),
                "auc": metrics["auc"],
                "auc_low": metrics["auc_low"],
                "auc_high": metrics["auc_high"],
                "accuracy": metrics["accuracy"],
                "brier": metrics["brier"],
                "n_test": int(len(y_test)),
            }
        )
        _log(
            f"[{target}] k={k:>2d} | AUC={metrics['auc']:.4f} ({metrics['auc_low']:.4f}-{metrics['auc_high']:.4f}) | "
            f"ACC={metrics['accuracy']:.4f} | Brier={metrics['brier']:.4f}",
            "INFO",
        )

    curve = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_auc = float(curve["auc"].max())
    threshold = best_auc - auc_tolerance
    candidates = curve[curve["auc"] >= threshold].sort_values("k")
    picked = candidates.iloc[0]
    best_row = curve.sort_values(["auc", "k"], ascending=[False, True]).iloc[0]

    recommendation = {
        "target": target,
        "k_recommended": int(picked["k"]),
        "auc_recommended": float(picked["auc"]),
        "auc_recommended_low": float(picked["auc_low"]),
        "auc_recommended_high": float(picked["auc_high"]),
        "accuracy_recommended": float(picked["accuracy"]),
        "k_best_auc": int(best_row["k"]),
        "auc_best": float(best_row["auc"]),
        "auc_best_low": float(best_row["auc_low"]),
        "auc_best_high": float(best_row["auc_high"]),
        "auc_tolerance": float(auc_tolerance),
        "auc_ci_bootstraps": int(auc_ci_bootstraps),
        "selection_rule": "smallest_k_within_auc_tolerance_on_internal_test",
    }
    return curve, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="仅开发集内 XGBoost 递减特征筛选（附加分析）")
    parser.add_argument("--targets", nargs="+", default=OUTCOMES, help=f"结局列表，默认: {' '.join(OUTCOMES)}")
    parser.add_argument("--min-k", type=int, default=3, help="最小特征数")
    parser.add_argument("--max-k", type=int, default=None, help="最大特征数（默认使用全部已选特征）")
    parser.add_argument("--auc-tolerance", type=float, default=0.01, help="AUC 容忍差值（推荐最小k阈值）")
    parser.add_argument("--auc-ci-bootstraps", type=int, default=1000, help="AUC 95%CI bootstrap 次数")
    args = parser.parse_args()

    log_header("🧪 06b_xgb_internal_feature_pruning: 开发集内附加变量筛选")
    _log("说明：本脚本仅使用 MIMIC 开发集（训练+内部测试），不读取 eICU 外部集。", "INFO")
    _log(f"训练集: {os.path.abspath(TRAIN_PATH)}", "INFO")
    _log(f"测试集: {os.path.abspath(TEST_PATH)}", "INFO")

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("缺少 mimic_train_processed.csv 或 mimic_test_processed.csv，请先运行 Step 02")

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    targets = [t.lower() for t in args.targets]
    summary_rows = []

    for target in targets:
        if target not in df_train.columns or target not in df_test.columns:
            _log(f"跳过 {target}: 结局列不存在", "WARN")
            continue

        target_dir = get_model_dir(target)
        ensure_dirs(target_dir)
        curve, rec = _run_target(
            target,
            df_train,
            df_test,
            args.min_k,
            args.max_k,
            args.auc_tolerance,
            args.auc_ci_bootstraps,
        )

        curve_path = os.path.join(target_dir, "xgb_internal_pruning_curve.csv")
        rec_path = os.path.join(target_dir, "xgb_internal_pruning_recommendation.json")
        curve.to_csv(curve_path, index=False)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        _log(f"[{target}] 曲线保存: {os.path.abspath(curve_path)}", "OK")
        _log(f"[{target}] 推荐保存: {os.path.abspath(rec_path)}", "OK")
        summary_rows.append(rec)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(get_model_dir(), "xgb_internal_pruning_summary.csv")
        summary.to_csv(summary_path, index=False)
        _log(f"汇总保存: {os.path.abspath(summary_path)}", "OK")
    else:
        _log("未生成任何结果（请检查 targets 参数）", "WARN")


if __name__ == "__main__":
    main()
