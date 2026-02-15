import argparse
import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
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


def _patch_shap_for_xgboost31() -> None:
    import shap.explainers._tree as _tree_mod

    if getattr(_patch_shap_for_xgboost31, "_patched", False):
        return
    orig_decode = _tree_mod.decode_ubjson_buffer

    def _patched_decode(fd):
        result = orig_decode(fd)
        learner = result.get("learner", {})
        params = learner.get("learner_model_param", {})
        base_score = params.get("base_score")
        if isinstance(base_score, str) and base_score.strip().startswith("[") and base_score.strip().endswith("]"):
            params["base_score"] = base_score.strip()[1:-1]
        return result

    _tree_mod.decode_ubjson_buffer = _patched_decode
    _patch_shap_for_xgboost31._patched = True


def _load_target_bundle(target: str) -> dict[str, Any]:
    bundle_path = os.path.join(get_model_dir(target), "deploy_bundle.pkl")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"缺少 deploy_bundle: {bundle_path}")
    return joblib.load(bundle_path)


def _extract_xgb_params(bundle: dict[str, Any]) -> dict[str, Any]:
    default = {"random_state": 42, "eval_metric": "logloss", "n_jobs": 1}
    model = bundle.get("best_model")
    if model is None:
        return default
    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cal_clf = model.calibrated_classifiers_[0]
            base = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", model))
        else:
            base = model
        params = dict(base.get_params())
        params["n_jobs"] = 1
        return params
    except Exception:
        return default


def _fit_base_xgb(
    x_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    xgb_params: dict[str, Any],
) -> tuple[StandardScaler, XGBClassifier]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    model = XGBClassifier(**xgb_params)
    model.fit(x_train, y_train)
    return scaler, model


def _mean_abs_shap_importance(
    model: XGBClassifier,
    scaler: StandardScaler,
    x_ref_raw: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    _patch_shap_for_xgboost31()
    x_ref = scaler.transform(x_ref_raw[features])
    x_ref_df = pd.DataFrame(x_ref, columns=features)
    explainer = shap.TreeExplainer(model)
    sv = explainer(x_ref_df)
    if len(sv.shape) == 3:
        sv = sv[:, :, 1]
    return np.abs(sv.values).mean(axis=0)


def _fit_eval_topk(
    x_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    x_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    xgb_params: dict[str, Any],
) -> dict[str, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    xgb = XGBClassifier(**xgb_params)
    clf = CalibratedClassifierCV(xgb, cv=3, method="isotonic", n_jobs=1)
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, y_prob)),
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
    n_bootstrap: int,
    stability_threshold: float,
    shap_ref_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bundle = _load_target_bundle(target)
    features = list(bundle.get("features") or bundle.get("feature_names") or [])
    if not features:
        raise ValueError(f"{target}: deploy_bundle 缺少 features")

    usable = [f for f in features if f in df_train.columns and f in df_test.columns]
    if len(usable) < 2:
        raise ValueError(f"{target}: 可用特征不足")

    y_train = df_train[target].dropna().astype(int)
    y_test = df_test[target].dropna().astype(int)
    x_train_all = df_train.loc[y_train.index, usable]
    x_test_all = df_test.loc[y_test.index, usable]

    k_min = max(2, min_k)
    k_max = len(usable) if max_k is None else min(max_k, len(usable))
    if k_min > k_max:
        raise ValueError(f"{target}: min_k({k_min}) > max_k({k_max})")

    xgb_params = _extract_xgb_params(bundle)
    rng = np.random.default_rng(42)
    ref_n = min(shap_ref_size, len(x_train_all))
    ref_idx = rng.choice(len(x_train_all), size=ref_n, replace=False)
    x_ref = x_train_all.iloc[ref_idx].reset_index(drop=True)

    _log(f"[{target}] SHAP bootstrap 开始: B={n_bootstrap}, refs={len(x_ref)}, features={len(usable)}", "INFO")
    rank_mat = np.zeros((n_bootstrap, len(usable)), dtype=int)
    topk_counts = {k: {f: 0 for f in usable} for k in range(k_min, k_max + 1)}

    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, len(x_train_all), size=len(x_train_all))
        x_boot = x_train_all.iloc[boot_idx].reset_index(drop=True)
        y_boot = y_train.values[boot_idx]
        scaler_b, model_b = _fit_base_xgb(x_boot, y_boot, xgb_params)
        imp_b = _mean_abs_shap_importance(model_b, scaler_b, x_ref, usable)
        order = np.argsort(imp_b)[::-1]
        inv_rank = np.empty_like(order)
        inv_rank[order] = np.arange(1, len(usable) + 1)
        rank_mat[b, :] = inv_rank
        ordered_feats = [usable[i] for i in order]
        for k in range(k_min, k_max + 1):
            for feat in ordered_feats[:k]:
                topk_counts[k][feat] += 1
        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            _log(f"[{target}] bootstrap {b + 1}/{n_bootstrap}", "INFO")

    rank_mean = rank_mat.mean(axis=0)
    rank_std = rank_mat.std(axis=0)

    scaler_full, model_full = _fit_base_xgb(x_train_all, y_train.values, xgb_params)
    imp_full = _mean_abs_shap_importance(model_full, scaler_full, x_ref, usable)
    order_full = np.argsort(imp_full)[::-1]
    ranked_features = [usable[i] for i in order_full]

    stability_rows = []
    for i, feat in enumerate(usable):
        row = {
            "target": target,
            "feature": feat,
            "shap_mean_abs_full": float(imp_full[i]),
            "rank_mean": float(rank_mean[i]),
            "rank_std": float(rank_std[i]),
        }
        for k in range(k_min, k_max + 1):
            row[f"selection_freq_top{k}"] = float(topk_counts[k][feat] / n_bootstrap)
        stability_rows.append(row)
    stability_df = pd.DataFrame(stability_rows).sort_values(["rank_mean", "rank_std"]).reset_index(drop=True)

    curve_rows = []
    for k in range(k_min, k_max + 1):
        feats_k = ranked_features[:k]
        metrics = _fit_eval_topk(
            x_train_all[feats_k],
            y_train.values,
            x_test_all[feats_k],
            y_test.values,
            xgb_params,
        )
        stable_freq = float(np.mean([topk_counts[k][f] / n_bootstrap for f in feats_k]))
        curve_rows.append(
            {
                "target": target,
                "k": k,
                "features": ";".join(feats_k),
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "brier": metrics["brier"],
                "mean_selection_freq": stable_freq,
            }
        )
        _log(
            f"[{target}] k={k:>2d} | AUC={metrics['auc']:.4f} | ACC={metrics['accuracy']:.4f} | "
            f"Brier={metrics['brier']:.4f} | Stability={stable_freq:.3f}",
            "INFO",
        )

    curve_df = pd.DataFrame(curve_rows).sort_values("k").reset_index(drop=True)
    best_auc = float(curve_df["auc"].max())
    threshold_auc = best_auc - auc_tolerance
    candidates = curve_df[curve_df["auc"] >= threshold_auc]
    stable_candidates = candidates[candidates["mean_selection_freq"] >= stability_threshold]
    if not stable_candidates.empty:
        picked = stable_candidates.sort_values("k").iloc[0]
        selection_rule = "smallest_k_within_auc_tolerance_and_stability_threshold_on_internal_test"
    else:
        picked = candidates.sort_values("k").iloc[0]
        selection_rule = "smallest_k_within_auc_tolerance_on_internal_test"
    best_row = curve_df.sort_values(["auc", "k"], ascending=[False, True]).iloc[0]

    rec = {
        "target": target,
        "k_recommended": int(picked["k"]),
        "auc_recommended": float(picked["auc"]),
        "accuracy_recommended": float(picked["accuracy"]),
        "mean_selection_freq_recommended": float(picked["mean_selection_freq"]),
        "k_best_auc": int(best_row["k"]),
        "auc_best": float(best_row["auc"]),
        "auc_tolerance": float(auc_tolerance),
        "stability_threshold": float(stability_threshold),
        "bootstrap_n": int(n_bootstrap),
        "selection_rule": selection_rule,
    }
    return stability_df, curve_df, rec


def main() -> None:
    parser = argparse.ArgumentParser(description="仅开发集内 SHAP+Bootstrap 稳定性变量筛选（附加分析）")
    parser.add_argument("--targets", nargs="+", default=OUTCOMES, help=f"结局列表，默认: {' '.join(OUTCOMES)}")
    parser.add_argument("--min-k", type=int, default=3, help="最小特征数")
    parser.add_argument("--max-k", type=int, default=None, help="最大特征数（默认使用 deploy 特征全集）")
    parser.add_argument("--auc-tolerance", type=float, default=0.01, help="AUC 容忍差值")
    parser.add_argument("--bootstrap-n", type=int, default=200, help="Bootstrap 次数")
    parser.add_argument("--stability-threshold", type=float, default=0.70, help="Top-k 平均入选频率阈值")
    parser.add_argument("--shap-ref-size", type=int, default=300, help="SHAP 参考样本数量（训练集内）")
    args = parser.parse_args()

    log_header("🧪 06c_shap_bootstrap_feature_pruning: 开发集内 SHAP 稳定性筛选")
    _log("说明：仅使用 MIMIC 开发集（训练+内部测试），不读取 eICU 外部集。", "INFO")
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
        try:
            stability_df, curve_df, rec = _run_target(
                target=target,
                df_train=df_train,
                df_test=df_test,
                min_k=args.min_k,
                max_k=args.max_k,
                auc_tolerance=args.auc_tolerance,
                n_bootstrap=args.bootstrap_n,
                stability_threshold=args.stability_threshold,
                shap_ref_size=args.shap_ref_size,
            )
        except Exception as e:
            _log(f"[{target}] 失败: {e}", "ERR")
            continue

        stability_path = os.path.join(target_dir, "xgb_shap_bootstrap_stability.csv")
        curve_path = os.path.join(target_dir, "xgb_shap_pruning_curve.csv")
        rec_path = os.path.join(target_dir, "xgb_shap_pruning_recommendation.json")
        stability_df.to_csv(stability_path, index=False)
        curve_df.to_csv(curve_path, index=False)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        _log(f"[{target}] 稳定性导出: {os.path.abspath(stability_path)}", "OK")
        _log(f"[{target}] 曲线导出: {os.path.abspath(curve_path)}", "OK")
        _log(f"[{target}] 推荐导出: {os.path.abspath(rec_path)}", "OK")
        summary_rows.append(rec)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(get_model_dir(), "xgb_shap_pruning_summary.csv")
        summary.to_csv(summary_path, index=False)
        _log(f"汇总保存: {os.path.abspath(summary_path)}", "OK")
    else:
        _log("未生成任何结果（请检查 targets 参数）", "WARN")


if __name__ == "__main__":
    main()
