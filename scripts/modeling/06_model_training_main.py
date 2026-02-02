import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.feature_formatter import FeatureFormatter
from utils.study_config import OUTCOMES
from utils.paths import get_project_root, get_cleaned_path, get_artifact_path, get_model_dir, get_supplementary_figure_dir, get_main_table_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.plot_config import (
    apply_medical_style, SAVE_DPI, PALETTE_MAIN, COLOR_REF_LINE,
    LABEL_FONT, TITLE_FONT, TICK_FONT, FIG_WIDTH_DOUBLE, save_fig_medical
)
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.utils import resample
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = get_project_root()
TRAIN_PATH = get_cleaned_path("mimic_train_processed.csv")
TEST_PATH = get_cleaned_path("mimic_test_processed.csv")
JSON_FEAT_PATH = get_artifact_path("features", "selected_features.json")
MODEL_ROOT = get_model_dir()
RESULT_ROOT = get_supplementary_figure_dir("S2_internal_ROC")
ensure_dirs(get_artifact_path("scalers"), get_main_table_dir())

def get_auc_ci(model, X_test, y_test, n_bootstraps=1000):
    scores = []
    X_arr = np.array(X_test)
    y_arr = np.array(y_test)
    indices = np.arange(len(y_arr))
    for i in range(n_bootstraps):
        resample_idx = resample(indices, random_state=i)
        y_res = y_arr[resample_idx]
        if len(np.unique(y_res)) < 2: continue
        prob = model.predict_proba(X_arr[resample_idx])[:, 1]
        scores.append(roc_auc_score(y_res, prob))
    sorted_scores = np.sort(scores)
    if len(sorted_scores) == 0: return 0, 0
    return sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]

def plot_performance(models, X_test, y_test, target, save_path, ci_stats=None):
    """ROC 与校准图：医学期刊格式（双栏 7.2in, DPI 600），含 95% CI"""
    apply_medical_style()
    ci_stats = ci_stats or {}
    colors = PALETTE_MAIN[:len(models)] if len(models) <= len(PALETTE_MAIN) else sns.color_palette("Set1", n_colors=len(models))
    plt.figure(figsize=(FIG_WIDTH_DOUBLE, 6), dpi=300, facecolor='white')
    ax = plt.gca()
    for i, (name, clf) in enumerate(models.items()):
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        if name in ci_stats and 'main' in ci_stats[name]:
            lo, hi = ci_stats[name]['main']
            label = f"{name} (AUC = {auc_val:.3f}, 95% CI: {lo:.3f}-{hi:.3f})"
        else:
            label = f"{name} (AUC = {auc_val:.3f})"
        plt.plot(fpr, tpr, lw=2.5, color=colors[i], label=label)
    plt.plot([0, 1], [0, 1], color=COLOR_REF_LINE, linestyle='--', lw=1.2, alpha=0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=LABEL_FONT, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=LABEL_FONT, labelpad=10)
    plt.title(f"ROC Analysis: {target.upper()}", fontsize=TITLE_FONT, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=9, frameon=False)
    plt.grid(color='whitesmoke', linestyle='-', linewidth=1)
    plt.tight_layout()
    roc_base = os.path.join(save_path, f"{target}_ROC")
    save_fig_medical(roc_base)
    plt.close()
    plt.figure(figsize=(FIG_WIDTH_DOUBLE, 6), dpi=300, facecolor='white')
    ax = plt.gca()
    for i, (name, clf) in enumerate(models.items()):
        y_prob = clf.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='s', ms=5, lw=2, 
                 color=colors[i], label=name, alpha=0.9)
    plt.plot([0, 1], [0, 1], color=COLOR_REF_LINE, linestyle=':', lw=1.5, label='Perfectly Calibrated')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Predicted Probability", fontsize=LABEL_FONT, labelpad=10)
    plt.ylabel("Actual Observed Probability", fontsize=LABEL_FONT, labelpad=10)
    plt.title(f"Calibration Analysis: {target.upper()}", fontsize=TITLE_FONT, fontweight='bold', pad=20)
    plt.legend(loc="upper left", fontsize=10, frameon=False)
    plt.grid(color='whitesmoke', linestyle='-', linewidth=1)
    plt.tight_layout()
    calib_base = os.path.join(save_path, f"{target}_Calibration")
    save_fig_medical(calib_base)
    plt.close()
    _log(f"图片已保存: {os.path.abspath(save_path)}", "OK")

def optimize_all_models(X_train, y_train):
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}
    # XGBoost
    _log("正在优化 XGBoost (n_trials=100)...", "INFO")
    def xgb_obj(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42, 'eval_metric': 'logloss'
        }
        model = XGBClassifier(**param)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_obj, n_trials=100)
    best_models["XGBoost"] = XGBClassifier(**study_xgb.best_params, random_state=42)
    # Random Forest
    _log("正在优化 Random Forest (n_trials=100)...", "INFO")
    def rf_obj(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestClassifier(**param, class_weight='balanced', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(rf_obj, n_trials=100)
    best_models["Random Forest"] = RandomForestClassifier(**study_rf.best_params, class_weight='balanced', random_state=42)
    # SVM
    _log("正在优化 SVM (n_trials=50)...", "INFO")
    def svm_obj(trial):
        param = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': 'rbf'
        }
        model = SVC(**param, probability=True, class_weight='balanced', max_iter=2000)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    
    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(svm_obj, n_trials=50) # SVM 耗时较长， trials 可略少
    best_models["SVM"] = SVC(**study_svm.best_params, probability=True, class_weight='balanced', max_iter=5000)
    # Decision Tree
    _log("正在优化 Decision Tree (n_trials=50)...", "INFO")
    def dt_obj(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
        }
        model = DecisionTreeClassifier(**param, class_weight='balanced', random_state=42)
        return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
    
    study_dt = optuna.create_study(direction='maximize')
    study_dt.optimize(dt_obj, n_trials=50)
    best_models["Decision Tree"] = DecisionTreeClassifier(**study_dt.best_params, class_weight='balanced', random_state=42)

    return best_models

def train_and_calibrate_all(X_train, y_train, best_instances):
    models = {
        **best_instances,  # 解包包含 XGB, RF, SVM, DT 的字典
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=2000)
    }
    calibrated_results = {}
    for name, m in models.items():
        _log(f"正在校准模型: {name}...", "INFO")
        clf = CalibratedClassifierCV(m, cv=3, method='isotonic', n_jobs=-1)
        try:
            clf.fit(X_train, y_train)
            calibrated_results[name] = clf
        except Exception as e:
            _log(f"模型 {name} 校准失败: {e}", "ERR")
    return calibrated_results

def evaluate_performance(models_dict, X_test, y_test, sub_mask):
    summary = []
    ci_stats = {}
    _log(f"{'Algorithm':<20} | {'Main AUC (95% CI)':<25} | {'Brier':<10}", "INFO")
    _log("-" * 65, "INFO")
    for name, clf in models_dict.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc_m = roc_auc_score(y_test, y_prob)
        low_m, high_m = get_auc_ci(clf, X_test, y_test)
        brier = brier_score_loss(y_test, y_prob)
        if len(np.unique(y_test[sub_mask])) > 1:
            auc_s = roc_auc_score(y_test[sub_mask], y_prob[sub_mask])
            low_s, high_s = get_auc_ci(clf, X_test[sub_mask], y_test[sub_mask])
        else:
            auc_s, low_s, high_s = 0, 0, 0
        main_ci_str = f"{auc_m:.3f} ({low_m:.3f}-{high_m:.3f})"
        sub_ci_str = f"{auc_s:.3f} ({low_s:.3f}-{high_s:.3f})"
        _log(f"{name:<20} | {main_ci_str:<25} | {brier:.4f}", "INFO")
        summary.append({
            "Algorithm": name,
            "Main AUC": round(auc_m, 4),
            "Main CI": main_ci_str,
            "Sub CI": sub_ci_str,
            "Brier": round(brier, 4)
        })
        ci_stats[name] = {
            "main": [float(low_m), float(high_m)], 
            "sub": [float(low_s), float(high_s)],
            "auc_ci": f"{auc_m:.3f} ({low_m:.3f}-{high_m:.3f})"  # 供 Step 07 绘图使用
        }
    return summary, ci_stats

def save_model_assets(target, target_dir, models_dict, scaler, ci_stats, features, X_train_cols):
    joblib.dump(models_dict, os.path.join(target_dir, "all_models_dict.pkl"))
    joblib.dump(scaler, os.path.join(target_dir, "scaler.pkl"))
    joblib.dump(ci_stats, os.path.join(target_dir, "bootstrap_ci_stats.pkl"))
    artifact_dir = get_artifact_path("scalers")
    deploy_bundle = {
        'feature_names': list(features),
        'scaler': scaler,
        'target_outcome': target
    }
    for key, fname in [
        ('imputer', 'mimic_mice_imputer.joblib'),
        ('mimic_scaler', 'mimic_scaler.joblib'),
        ('skewed_cols', 'skewed_cols_config.pkl'),
        ('train_assets_bundle', 'train_assets_bundle.pkl')
    ]:
        path = os.path.join(artifact_dir, fname)
        if os.path.exists(path):
            deploy_bundle[key] = joblib.load(path)
    if 'imputer' in deploy_bundle:
        imp = deploy_bundle['imputer']
        order = getattr(imp, 'feature_names_in_', None)
        if order is None:
            order = deploy_bundle.get('train_assets_bundle', {}).get('feature_order', [])
        deploy_bundle['imputer_feature_order'] = list(order) if order is not None else []
    joblib.dump(deploy_bundle, os.path.join(target_dir, "deploy_bundle.pkl"))
    imp_list = []
    for name in ["XGBoost", "Random Forest", "Logistic Regression"]:
        if name in models_dict:
            try:
                cal_clf = models_dict[name].calibrated_classifiers_[0]
                base = getattr(cal_clf, 'estimator', getattr(cal_clf, 'base_estimator', None))
                if name == "Logistic Regression":
                    weights = base.coef_.flatten()
                else:
                    weights = base.feature_importances_
                if len(weights) == len(features):
                    formatter = FeatureFormatter()
                    display_names = formatter.format_features(features)
                    imp_df = pd.DataFrame({
                        'feature': features,
                        'display_name': display_names,
                        'importance': weights, 
                        'algorithm': name, 
                        'outcome': target
                    })
                    imp_list.append(imp_df)
                else:
                    _log(f"{name} 权重长度({len(weights)})与特征数({len(features)})不匹配，已跳过。", "WARN")
            except Exception as e:
                _log(f"提取 {name} 重要性时发生错误: {e}", "WARN")
                continue
    if imp_list:
        final_imp_df = pd.concat(imp_list, ignore_index=True)
        final_imp_df.to_csv(os.path.join(target_dir, "feature_importance.csv"), index=False)
        _log(f"特征重要性已保存至: {target_dir}", "OK")
        
def run_model_training_flow():
    log_header("🚀 06_model_training_main: 模型训练与校准")
    _log(f"训练集: {os.path.abspath(TRAIN_PATH)}", "INFO")
    _log(f"测试集: {os.path.abspath(TEST_PATH)}", "INFO")
    _log(f"特征配置: {os.path.abspath(JSON_FEAT_PATH)}", "INFO")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"❌ 找不到输入数据，请先运行 05_feature_selection_lasso.py")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    with open(JSON_FEAT_PATH, 'r') as f:
        feature_config = json.load(f)
    global_performance = []
    for target, config in feature_config.items():
        if target not in df_train.columns or target not in df_test.columns:
            _log(f"跳过结局 {target}: 不在数据列中", "WARN")
            continue
        _log(f"启动临床结局分析: {target.upper()}", "INFO")
        target_model_dir = get_model_dir(target)
        target_fig_dir = get_supplementary_figure_dir("S2_internal_ROC", target.lower())
        ensure_dirs(target_model_dir, target_fig_dir)
        selected_features = config['features']
        missing_feats = [f for f in selected_features if f not in df_train.columns]
        if missing_feats:
            raise ValueError(f"❌ 结局 {target} 缺失关键特征: {missing_feats}")
        X_train = df_train[selected_features].copy()
        y_train = df_train[target]
        X_test = df_test[selected_features].copy()
        y_test = df_test[target]
        sub_test = df_test['subgroup_no_renal']
        train_valid = y_train.notna()
        test_valid = y_test.notna()
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_test, y_test = X_test[test_valid], y_test[test_valid]
        sub_test = sub_test[test_valid]
        scaler_pre = StandardScaler()
        X_train_pre = scaler_pre.fit_transform(X_train)
        X_test_pre = scaler_pre.transform(X_test)
        joblib.dump(scaler_pre, os.path.join(target_model_dir, "scaler.pkl"))
        sub_mask = (sub_test == 1).values
        best_tuned_instances = optimize_all_models(X_train_pre, y_train)
        calibrated_models = train_and_calibrate_all(X_train_pre, y_train, best_tuned_instances)
        summary_list, ci_stats = evaluate_performance(calibrated_models, X_test_pre, y_test, sub_mask)
        for s in summary_list: 
            s['Outcome'] = target 
        save_model_assets(target, target_model_dir, calibrated_models, scaler_pre, ci_stats, selected_features, X_train.columns)
        X_test_df = pd.DataFrame(X_test_pre, columns=selected_features)
        eval_bundle = {
            'X_test_pre': X_test_df,
            'X_test_raw': X_test,
            'y_test': y_test.values, 
            'sub_mask': sub_mask, 
            'features': selected_features
        }
        joblib.dump(eval_bundle, os.path.join(target_model_dir, "eval_data.pkl"))
        plot_performance(calibrated_models, X_test_pre, y_test, target, target_fig_dir, ci_stats)
        global_performance.extend(summary_list)

    if global_performance:
        perf_df = pd.DataFrame(global_performance)
        report_path = os.path.join(MODEL_ROOT, "performance_report.csv")
        perf_df.to_csv(report_path, index=False)
        _log("所有结局分析完成！", "OK")
        _log(f"汇总报告: {os.path.abspath(report_path)}", "OK")
        _log("下一步: 07_optimal_cutoff_analysis.py", "INFO")

if __name__ == "__main__":
    run_model_training_flow()
