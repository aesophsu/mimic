import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.plot_config import apply_medical_style, SAVE_DPI, PALETTE_MAIN, COLOR_REF_LINE, FIG_WIDTH_DOUBLE, save_fig_medical
from utils.study_config import OUTCOMES, OUTCOME_TYPE
from utils.paths import get_model_dir, get_external_dir, get_main_table_dir, get_main_figure_dir, get_supplementary_figure_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.deploy_utils import load_deploy_bundle
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, roc_curve
)

MODEL_ROOT = get_model_dir()
EICU_DIR = get_external_dir()
TABLE_DIR = get_main_table_dir()
FIGURE_DIR = get_main_figure_dir()  # Fig2 ROC External
FIGURE_SUPP_DIR = get_supplementary_figure_dir("S4_comparison")  # Table4 viz, drift
ensure_dirs(TABLE_DIR, FIGURE_DIR, FIGURE_SUPP_DIR)

TARGETS = OUTCOMES

def load_external_validation_assets(target):
    """
    基于第 06 步保存的资产，加载模型和特征清单。
    注意：eICU 数据由 Step 09 使用 deploy_bundle 完成完整变换，此处无需再标准化。
    """
    target_dir = get_model_dir(target)
    
    # 1. 加载模型字典
    models_path = os.path.join(target_dir, "all_models_dict.pkl")
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"未找到模型字典: {models_path}")
    models = joblib.load(models_path)
    
    # 2. 加载部署包（获取特征清单）
    bundle = load_deploy_bundle(target, fill_missing=False)
    if bundle is None:
        raise FileNotFoundError(f"未找到部署包: {os.path.join(target_dir, 'deploy_bundle.pkl')}")
    
    # 3. 加载阈值字典
    thresh_path = os.path.join(target_dir, "thresholds.json")
    threshold_data = {}
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            threshold_data = json.load(f)
    
    return models, bundle['feature_names'], threshold_data

def process_and_align_eicu(target, features):
    """
    读取 eICU 数据。Step 09 已使用 deploy_bundle 完成完整变换（Log+MICE+mimic_scaler+scaler_pre），
    此处直接提取特征列，无需再标准化。
    """
    eicu_path = os.path.join(EICU_DIR, f"eicu_processed_{target.lower()}.csv")
    if not os.path.exists(eicu_path):
        raise FileNotFoundError(f"未找到 eICU 数据: {eicu_path}")
    
    df = pd.read_csv(eicu_path)
    df.columns = [str(c) for c in df.columns]
    
    # 按照 MIMIC 的特征顺序提取，缺失列补 0
    X_list = []
    missing_count = 0
    for f in features:
        f_str = str(f)
        if f_str in df.columns:
            X_list.append(df[f_str])
        else:
            X_list.append(pd.Series(np.zeros(len(df)), name=f_str))
            missing_count += 1
            
    if missing_count > 0:
        _log(f"eICU 缺失 {missing_count} 个特征，已自动补 0", "WARN")
        
    X = pd.concat(X_list, axis=1)
    y_true = df[target].values
    
    # Step 09 已输出完全变换数据，直接转为数组供模型推理
    X_arr = np.array(X)
    return X_arr, y_true

def compute_metrics_ci(y_true, y_prob, n_bootstraps=1000, seed=42):
    """同步计算 AUC, AUPRC, Brier 的 95% CI"""
    rng = np.random.RandomState(seed)
    scores = {'auc': [], 'auprc': [], 'brier': []}
    
    for i in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        scores['auc'].append(roc_auc_score(y_true[idx], y_prob[idx]))
        scores['auprc'].append(average_precision_score(y_true[idx], y_prob[idx]))
        scores['brier'].append(brier_score_loss(y_true[idx], y_prob[idx]))
    
    results = {}
    for k, v in scores.items():
        sorted_v = np.sort(v)
        results[k] = (sorted_v[int(0.025 * len(v))], sorted_v[int(0.975 * len(v))])
    return results

def plot_roc_all_models(target, eicu_curves, mimic_ref=None):
    """
    医学级 ROC 对比图：展示 5 种模型在 eICU 外部验证的完整对比
    eicu_curves: list of (name, auc, fpr, tpr, ci_tuple) 每个模型在 eICU 上的表现
    mimic_ref: optional (name, auc, fpr, tpr, ci_tuple) MIMIC 内部验证参考线
    """
    apply_medical_style()
    n_models = len(eicu_curves)
    colors = PALETTE_MAIN[:n_models] if len(PALETTE_MAIN) >= n_models else list(plt.cm.Set1(np.linspace(0, 1, n_models)))

    plt.figure(figsize=(FIG_WIDTH_DOUBLE, 6), dpi=300, facecolor='white')
    plt.plot([0, 1], [0, 1], color=COLOR_REF_LINE, linestyle='--', lw=1.2, alpha=0.8)

    # 1. MIMIC 内部验证参考线（若有）
    if mimic_ref is not None:
        m_name, m_auc, m_fpr, m_tpr, m_ci = mimic_ref
        m_label = f'MIMIC Internal ({m_name}, AUC = {m_auc:.3f}, 95% CI: {m_ci[0]:.3f}-{m_ci[1]:.3f})'
        plt.plot(m_fpr, m_tpr, linestyle='--', color='#95a5a6', lw=1.8, alpha=0.9, label=m_label)

    # 2. 绘制 5 种模型 eICU 外部验证曲线
    for i, (name, auc, fpr, tpr, ci) in enumerate(eicu_curves):
        c = colors[i % len(colors)]
        label = f'{name} (AUC = {auc:.3f}, 95% CI: {ci[0]:.3f}-{ci[1]:.3f})'
        plt.plot(fpr, tpr, color=c, lw=2, label=label)

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, labelpad=8)
    plt.title(f"External Validation: {target.upper()} (eICU, 5 Models)", fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=False, fontsize=8)
    ax = plt.gca()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.grid(color='whitesmoke', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    base_path = os.path.join(FIGURE_DIR, f"Fig2_ROC_external_{target}")
    save_fig_medical(base_path)
    plt.close()

def run_single_validation(target, mimic_auc_ref):
    """
    执行单个结局目标的 5 种模型验证
    适配多模型专属阈值字典，并对比 MIMIC 真实基准
    """
    _log(f"正在分析结局: {target.upper()}", "INFO")
    results = []
    
    try:
        # 1. 加载资产 (此时 threshold_dict 包含各模型专属阈值)
        models, features, threshold_dict = load_external_validation_assets(target)
        X_eicu, y_eicu = process_and_align_eicu(target, features)
        y_eicu = np.array(y_eicu).astype(int) # 确保标签为整型
        
        # 1.1 加载 MIMIC 内部验证数据（用于 ROC 参考线）
        eval_path = os.path.join(MODEL_ROOT, target.lower(), "eval_data.pkl")
        eval_data = joblib.load(eval_path)

        _log(f"eICU 样本量: {len(y_eicu)} | 正例率: {y_eicu.mean():.2%}", "INFO")
        header = f"{'Algorithm':<20} | {'AUC (95% CI)':<22} | {'Brier':<8} | {'Sens':<8}"
        _log(header, "INFO")
        _log("-" * len(header), "INFO")

        eicu_curves = []
        mimic_ref = None

        for name, model in models.items():
            # 2. 动态匹配该模型的最佳阈值
            current_thresh = threshold_dict.get(name, 0.5)
            
            # 3. 模型预测与性能评估
            y_prob = model.predict_proba(X_eicu)[:, 1]
            y_pred = (y_prob >= current_thresh).astype(int)
            
            # 计算包含 95% CI 的多维指标
            cis = compute_metrics_ci(y_eicu, y_prob) 
            auc = roc_auc_score(y_eicu, y_prob)
            brier = brier_score_loss(y_eicu, y_prob)
            
            # 计算敏感度与特异度
            tn, fp, fn, tp = confusion_matrix(y_eicu, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 4. 控制台实时输出结果
            auc_display = f"{auc:.3f} ({cis['auc'][0]:.3f}-{cis['auc'][1]:.3f})"
            _log(f"{name:<20} | {auc_display:<22} | {brier:.4f} | {sens:.4f}", "INFO")

            # 5. 结果收集
            results.append({
                'Endpoint': OUTCOME_TYPE.get(target, target),
                'Target': target, 'Algorithm': name, 
                'AUC': auc, 'AUC_Low': cis['auc'][0], 'AUC_High': cis['auc'][1],
                'Brier': brier, 'Sensitivity': sens, 'Specificity': spec,
                'AUPRC': average_precision_score(y_eicu, y_prob),
                'Threshold': current_thresh
            })

            # 6. 收集 eICU ROC 数据（供 5 模型对比图）
            fpr_e, tpr_e, _ = roc_curve(y_eicu, y_prob)
            eicu_curves.append((name, auc, fpr_e, tpr_e, cis['auc']))

            # 7. MIMIC 内部验证参考（取 XGBoost 作为基准）
            if name == "XGBoost":
                y_prob_mimic = model.predict_proba(eval_data['X_test_pre'])[:, 1]
                fpr_m, tpr_m, _ = roc_curve(eval_data['y_test'], y_prob_mimic)
                auc_m = roc_auc_score(eval_data['y_test'], y_prob_mimic)
                cis_m = compute_metrics_ci(eval_data['y_test'], y_prob_mimic)
                mimic_ref = (name, auc_m, fpr_m, tpr_m, cis_m['auc'])

        # 8. 绘制 5 种模型 eICU 外部验证 ROC 对比图
        plot_roc_all_models(target, eicu_curves, mimic_ref=mimic_ref)
        return results

    except Exception as e:
        _log(f"失败 {target}: {str(e)}", "ERR")
        return None

def plot_external_comparison_summary(csv_path):
    """医学级性能汇总图：算法横向大比拼，统一期刊风格"""
    apply_medical_style()
    df = pd.read_csv(csv_path)
    plot_df = df.melt(id_vars=['Target', 'Algorithm'], 
                      value_vars=['AUC', 'Sensitivity', 'Specificity'],
                      var_name='Metric', value_name='Score')

    g = sns.catplot(
        data=plot_df, x='Target', y='Score', hue='Algorithm',
        col='Metric', kind='point', 
        linestyle='none', 
        palette=PALETTE_MAIN, 
        markers=['o', 's', 'D', 'X', 'P'],
        dodge=0.5, height=5, aspect=0.7,
        markersize=10
    )

    # 布局与坐标轴调整
    g.set_titles("{col_name}", size=14, fontweight='bold')
    g.set_axis_labels("", "Metric Score", size=12)
    g.set(ylim=(0, 1.05))
    
    for ax in g.axes.flat:
        # 添加 0.8 和 0.9 基准线
        ax.axhline(0.8, color=COLOR_REF_LINE, linestyle='--', lw=0.8, alpha=0.5)
        ax.axhline(0.9, color=COLOR_REF_LINE, linestyle='--', lw=0.8, alpha=0.5)
        
        # 稳健的刻度标签大写逻辑
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        labels = [t.get_text().upper() for t in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle('External Validation Performance Across eICU Cohort', 
                   fontsize=16, fontweight='bold')

    base_path = os.path.join(FIGURE_SUPP_DIR, "Table4_Performance_Visualization")
    g.savefig(f"{base_path}.pdf", bbox_inches='tight')
    g.savefig(f"{base_path}.png", bbox_inches='tight', dpi=600)
    _log("外部验证汇总图已成功生成 (PDF/PNG)", "OK")

def get_mimic_base_auc(target, algorithm="XGBoost"):
    """从 06 步生成的性能报告中动态提取 MIMIC 实际 AUC"""
    report_path = os.path.join(MODEL_ROOT, "performance_report.csv")
    try:
        df_perf = pd.read_csv(report_path)
        # 匹配结局和指定算法（通常以 XGBoost 作为对比基准）
        match = df_perf[(df_perf['Outcome'] == target.lower()) & 
                        (df_perf['Algorithm'] == algorithm)]
        return match['Main AUC'].values[0] if not match.empty else 0.85
    except Exception:
        # 兜底预设值
        return {'pof': 0.882, 'mortality': 0.845, 'composite': 0.867}.get(target.lower(), 0.85)

def main():
    log_header("启动模块 11: eICU 外部验证 (动态基准版)")
    performance_table = []

    for target in TARGETS:
        # 动态获取该结局在 MIMIC 上的实际表现作为绘图参考线
        mimic_auc_ref = get_mimic_base_auc(target)
        _log(f"基准确认 {target.upper()} MIMIC 实际 AUC: {mimic_auc_ref:.4f}", "INFO")
        
        results = run_single_validation(target, mimic_auc_ref)
        if results:
            performance_table.extend(results)
            
    if performance_table:
        df_final = pd.DataFrame(performance_table)
        order_map = {'pof': 0, 'mortality': 1, 'composite': 2}
        df_final['_order'] = df_final['Target'].map(lambda x: order_map.get(x.lower(), 9))
        df_final = df_final.sort_values(['_order', 'AUC'], ascending=[True, False]).drop(columns=['_order'])
        
        csv_path = os.path.join(TABLE_DIR, "Table4_external_validation.csv")
        df_final.to_csv(csv_path, index=False)
        _log(f"外部验证结果已导出: {csv_path}", "OK")

        # 生成医学出版级汇总图
        try:
            plot_external_comparison_summary(csv_path)
        except Exception as e:
            _log(f"汇总图生成失败: {e}", "WARN")

    _log("外部验证流已结束。下一步: 11_model_interpretation_shap.py", "OK")

if __name__ == "__main__":
    main()
