import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.feature_formatter import FeatureFormatter
from utils.study_config import OUTCOMES
from utils.paths import get_project_root, get_artifact_path, get_cleaned_path, get_model_dir, get_supplementary_figure_dir, ensure_dirs
from utils.logger import log as _log
from utils.plot_config import (
    apply_medical_style, SAVE_DPI, FIG_DPI, FIG_WIDTH_DOUBLE,
    LABEL_FONT, TICK_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH_THIN,
    COLOR_POSITIVE, COLOR_NEGATIVE
)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = get_project_root()
INPUT_PATH = get_cleaned_path("mimic_train_processed.csv")
ARTIFACTS_DIR = get_artifact_path("features")
MODELS_ARTIFACTS_DIR = get_model_dir()
FIG_DIR = get_supplementary_figure_dir("S1_lasso")
ensure_dirs(ARTIFACTS_DIR, FIG_DIR)

def plot_academic_lasso(cv_model, X_columns, target):
    """LASSO 诊断图：符合医学期刊格式（双栏 176mm、DPI 600、Arial 8–12pt）"""
    apply_medical_style()
    Cs = cv_model.Cs_
    log_Cs = np.log10(Cs)
    pos_class_idx = cv_model.classes_[1]
    scores_mean = cv_model.scores_[pos_class_idx].mean(axis=0)
    scores_se = cv_model.scores_[pos_class_idx].std(axis=0) / np.sqrt(cv_model.scores_[pos_class_idx].shape[0])
    idx_max = np.argmax(scores_mean)
    target_score = scores_mean[idx_max] - scores_se[idx_max]
    eligible_indices = np.where(scores_mean >= target_score)[0]
    idx_1se = eligible_indices[np.argmin(Cs[eligible_indices])]
    # 双栏图：176mm ≈ 7.2 in，高度 ≤ 8.75 in（PLOS/Nature 上限）
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 5.0), dpi=FIG_DPI, facecolor='white')
    ax1.errorbar(log_Cs, scores_mean, yerr=scores_se, fmt='o', color=COLOR_POSITIVE,
                 ecolor='#BDC3C7', elinewidth=1.2, capsize=3, mfc=COLOR_POSITIVE, ms=5, label='CV ROC AUC')
    ax1.axvline(log_Cs[idx_max], color=COLOR_NEGATIVE, linestyle='--', lw=1.5,
                label=f'Max AUC (logC={log_Cs[idx_max]:.2f})')
    ax1.axvline(log_Cs[idx_1se], color='#2C3E50', linestyle='--', lw=1.5,
                label=f'1-SE Rule (logC={log_Cs[idx_1se]:.2f})')
    ax1.set_xlabel(r'$\log_{10}(C)$', fontsize=LABEL_FONT)
    ax1.set_ylabel('Mean ROC AUC', fontsize=LABEL_FONT)
    ax1.set_title(f'LASSO Selection (Logistic): {target.upper()}', fontsize=TITLE_FONT, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', frameon=False, fontsize=LEGEND_FONT)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Regularization Strength', fontsize=TICK_FONT)
    ax2.set_xticks([log_Cs.min(), log_Cs.max()])
    ax2.set_xticklabels(['Strong (Sparse)', 'Weak (Dense)'])
    plt.tight_layout()
    base = os.path.join(FIG_DIR, f"lasso_diag_{target}")
    save_kw = dict(bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.savefig(f"{base}.pdf", **save_kw)
    plt.savefig(f"{base}.png", dpi=SAVE_DPI, **save_kw)
    plt.close()
    
def plot_feature_importance(features, weights, target):
    """特征重要性条形图：符合医学期刊格式（双栏、DPI 600、显示名）"""
    if not features:
        return

    features = np.array(features)
    weights = np.array(weights)
    sorted_idx = np.argsort(np.abs(weights))
    formatter = FeatureFormatter()
    display_labels = formatter.format_features(features[sorted_idx].tolist())
    
    apply_medical_style()
    # 高度：12 特征约 5.5 in，上限 8.75 in（PLOS 最大高度）
    n_feat = len(features)
    fig_h = min(max(5.0, 0.45 * n_feat), 8.75)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, fig_h), dpi=FIG_DPI, facecolor='white')

    colors = [COLOR_POSITIVE if w > 0 else COLOR_NEGATIVE for w in weights[sorted_idx]]
    
    bars = ax.barh(range(len(features)), weights[sorted_idx], color=colors, 
                   edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(display_labels, fontsize=TICK_FONT, fontweight='medium')
    ax.axvline(0, color='black', lw=LINE_WIDTH_THIN, zorder=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.3f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if width > 0 else -5, 0),
                    textcoords="offset points",
                    ha='left' if width > 0 else 'right', 
                    va='center', fontsize=LEGEND_FONT, fontweight='bold',
                    color=bar.get_facecolor())

    ax.set_xlabel('Regression Coefficient (Standardized)', fontsize=LABEL_FONT, fontweight='bold')
    ax.set_title(f'Predictors for {target.upper()}', loc='left', fontsize=TITLE_FONT, fontweight='bold', pad=20)
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    base = os.path.join(FIG_DIR, f"lasso_importance_{target}")
    save_kw = dict(bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.savefig(f"{base}.pdf", **save_kw)
    plt.savefig(f"{base}.png", dpi=SAVE_DPI, **save_kw)
    plt.close()


def _compute_cv_deviance_curve(X, y, Cs, cv):
    """Compute CV binomial deviance(log-loss) across C grid."""
    fold_losses = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        losses = []
        for c in Cs:
            clf = LogisticRegression(
                C=c,
                penalty='l1',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            )
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_va)[:, 1]
            losses.append(log_loss(y_va, prob, labels=[0, 1]))
        fold_losses.append(losses)
    loss_arr = np.asarray(fold_losses)
    loss_mean = loss_arr.mean(axis=0)
    loss_se = loss_arr.std(axis=0, ddof=1) / np.sqrt(loss_arr.shape[0])
    return loss_mean, loss_se


def export_lasso_cv_curve(lasso_cv, X_train, y_train, selected_features, target, target_artifacts):
    """Export tabular LASSO CV curve data for plotting and manuscript figures."""
    Cs = lasso_cv.Cs_
    lambdas = 1.0 / Cs
    log10_C = np.log10(Cs)
    log10_lambda = np.log10(lambdas)

    pos_class = 1
    auc_mean = lasso_cv.scores_[pos_class].mean(axis=0)
    auc_se = lasso_cv.scores_[pos_class].std(axis=0, ddof=1) / np.sqrt(lasso_cv.scores_[pos_class].shape[0])
    idx_auc_max = int(np.argmax(auc_mean))
    auc_threshold = auc_mean[idx_auc_max] - auc_se[idx_auc_max]
    auc_eligible = np.where(auc_mean >= auc_threshold)[0]
    idx_auc_1se = int(auc_eligible[np.argmin(Cs[auc_eligible])])

    dev_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    dev_mean, dev_se = _compute_cv_deviance_curve(X_train.values, y_train, Cs, dev_cv)
    idx_dev_min = int(np.argmin(dev_mean))
    dev_threshold = dev_mean[idx_dev_min] + dev_se[idx_dev_min]
    dev_eligible = np.where(dev_mean <= dev_threshold)[0]
    idx_dev_1se = int(dev_eligible[np.argmin(Cs[dev_eligible])])

    n_nonzero = []
    for c in Cs:
        clf = LogisticRegression(
            C=c,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        clf.fit(X_train, y_train)
        n_nonzero.append(int(np.count_nonzero(clf.coef_[0])))

    out_df = pd.DataFrame({
        "target": target,
        "C": Cs,
        "lambda": lambdas,
        "log10_C": log10_C,
        "log10_lambda": log10_lambda,
        "mean_auc": auc_mean,
        "se_auc": auc_se,
        "mean_deviance": dev_mean,
        "se_deviance": dev_se,
        "n_nonzero_coef": n_nonzero
    })
    out_df["is_auc_lambda_min"] = False
    out_df.loc[idx_auc_max, "is_auc_lambda_min"] = True
    out_df["is_auc_lambda_1se"] = False
    out_df.loc[idx_auc_1se, "is_auc_lambda_1se"] = True
    out_df["is_dev_lambda_min"] = False
    out_df.loc[idx_dev_min, "is_dev_lambda_min"] = True
    out_df["is_dev_lambda_1se"] = False
    out_df.loc[idx_dev_1se, "is_dev_lambda_1se"] = True
    out_df["selected_n_features_final"] = int(len(selected_features))

    out_path = os.path.join(target_artifacts, "lasso_cv_curve.csv")
    out_df.to_csv(out_path, index=False)
    print(f"💾 LASSO CV曲线已保存: {os.path.abspath(out_path)}")

    summary = {
        "target": target,
        "n_candidate_features": int(X_train.shape[1]),
        "selected_n_features_final": int(len(selected_features)),
        "auc_lambda_min": float(lambdas[idx_auc_max]),
        "auc_lambda_1se": float(lambdas[idx_auc_1se]),
        "deviance_lambda_min": float(lambdas[idx_dev_min]),
        "deviance_lambda_1se": float(lambdas[idx_dev_1se]),
        "deviance_at_lambda_min": float(dev_mean[idx_dev_min]),
        "deviance_at_lambda_1se": float(dev_mean[idx_dev_1se])
    }
    summary_path = os.path.join(target_artifacts, "lasso_cv_curve_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 LASSO CV摘要已保存: {os.path.abspath(summary_path)}")

    
def run_lasso_selection_flow():
    targets = OUTCOMES
    df = pd.read_csv(INPUT_PATH)
    protected = ['pof', 'mortality', 'composite', 'subgroup_no_renal',
                 'resp_pof', 'cv_pof', 'renal_pof',
                 'sofa_score', 'apsiii', 'sapsii', 'oasis', 'lods',
                 'subject_id', 'hadm_id', 'stay_id', 'los',
                 'mechanical_vent_flag', 'vaso_flag']
    X_cols = [c for c in df.columns if c not in protected]
    # 仅对候选预测变量做统计审计（不含结局与保护列）
    print(f"[候选预测变量审计] N={len(X_cols)} 列")
    print(f"{'Feature':<22} | {'Miss%':>7} | {'Median':>9} | {'Mean':>9} | {'Max':>9}")
    for col in X_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            missing = df[col].isnull().mean() * 100
            med = series.median() if not series.empty else 0
            mean = series.mean() if not series.empty else 0
            v_max = series.max() if not series.empty else 0
            print(f"{col:<22} | {missing:>6.2f}% | {med:>9.2f} | {mean:>9.2f} | {v_max:>9.2f}")
    binary_cols = [c for c in X_cols if df[c].nunique() <= 2]
    continuous_cols = [c for c in X_cols if c not in binary_cols]
    
    if continuous_cols:
        max_mean_cont = df[continuous_cols].mean().abs().max()
        print(f"🔍 [标准化审计] 连续特征最大绝对均值: {max_mean_cont:.6f}")
        if max_mean_cont > 0.1:
            print("⚠️ 警告: 连续特征均值偏离0，请检查标准化步骤")
        else:
            print("✅ 审计通过: 连续特征已对齐")
        max_std_diff = (df[continuous_cols].std() - 1).abs().max()
        if max_std_diff > 0.2:
            print(f"⚠️ 警告: 连续特征标准差偏离1 (Max Diff: {max_std_diff:.2f})")
        
    all_outcomes_features = {}

    for target in targets:
        print(f"\n>>> 正在精炼: {target.upper()}")
        TARGET_ARTIFACTS = os.path.join(MODELS_ARTIFACTS_DIR, target)
        os.makedirs(TARGET_ARTIFACTS, exist_ok=True)

        df_target = df.dropna(subset=[target])
        X_train = df_target[X_cols]
        y_train = df_target[target].values

        classes = np.unique(y_train)
        print(f"📊 [结局审计] 类别分布: {classes}, 阳性样本数: {sum(y_train==1)}")
        if len(classes) != 2:
            print(f"❌ 错误: {target} 不是二分类结局，跳过。")
            continue

        print(f"📐 使用 Step 03 训练集 N={len(y_train)}（测试集不参与特征选择）")

        lasso_cv = LogisticRegressionCV(
            Cs=100, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            penalty='l1', solver='liblinear', scoring='roc_auc',
            random_state=42, max_iter=1000, n_jobs=1
        )
        lasso_cv.fit(X_train, y_train)

        pos_class = 1
        scores_mean = lasso_cv.scores_[pos_class].mean(axis=0)
        scores_se = lasso_cv.scores_[pos_class].std(axis=0) / np.sqrt(lasso_cv.scores_[pos_class].shape[0])
        idx_max = np.argmax(scores_mean)
        target_score = scores_mean[idx_max] - scores_se[idx_max]
        eligible_indices = np.where(scores_mean >= target_score)[0]
        idx_1se = eligible_indices[np.argmin(lasso_cv.Cs_[eligible_indices])]
        best_C_1se = lasso_cv.Cs_[idx_1se]

        final_lasso = LogisticRegression(
            C=best_C_1se, penalty='l1', solver='liblinear',
            random_state=42, max_iter=1000
        )
        final_lasso.fit(X_train, y_train)
        coef = final_lasso.coef_[0]

        selected_idx = np.where(coef != 0)[0]
        selected_features = X_train.columns[selected_idx].tolist()
        if len(selected_features) == 0:
            top_k = min(5, len(X_train.columns))
            selected_idx = np.argsort(np.abs(coef))[-top_k:]
            selected_features = X_train.columns[selected_idx].tolist()
            print(f"⚠️ 1-SE 准则无非零系数，兜底选取系数绝对值最大的 {top_k} 个特征")

        if len(selected_features) > 12:
            coef_abs = np.abs(coef[selected_idx])
            top_idx = np.argsort(coef_abs)[-12:]
            selected_features = [selected_features[i] for i in top_idx]

        all_outcomes_features[target] = {
            "n_features": len(selected_features),
            "features": selected_features,
            "weights": {f: round(float(coef[X_train.columns.get_loc(f)]), 4) for f in selected_features},
            "best_C": float(best_C_1se),
            "best_lambda": float(1 / best_C_1se)
        }

        plot_academic_lasso(lasso_cv, X_train.columns, target)
        current_weights = [all_outcomes_features[target]["weights"][f] for f in selected_features]
        plot_feature_importance(selected_features, current_weights, target)
        export_lasso_cv_curve(
            lasso_cv=lasso_cv,
            X_train=X_train,
            y_train=y_train,
            selected_features=selected_features,
            target=target,
            target_artifacts=TARGET_ARTIFACTS
        )
        print(f"🎯 选定特征 ({len(selected_features)} 个): {', '.join(selected_features)}")

        with open(os.path.join(TARGET_ARTIFACTS, "selected_features.json"), "w", encoding='utf-8') as f:
            json.dump(all_outcomes_features[target], f, ensure_ascii=False, indent=4)

    selected_path = os.path.join(ARTIFACTS_DIR, "selected_features.json")
    with open(selected_path, "w", encoding='utf-8') as f:
        json.dump(all_outcomes_features, f, ensure_ascii=False, indent=4)

    abs_selected = os.path.abspath(selected_path)
    abs_fig_dir = os.path.abspath(FIG_DIR)
    print(f"\n📂 全局特征清单: {abs_selected}")
    print(f"📊 图片输出目录: {abs_fig_dir}")
    _log("下一步：进入 06_model_training_main.py", "INFO")

if __name__ == "__main__":
    run_lasso_selection_flow()
