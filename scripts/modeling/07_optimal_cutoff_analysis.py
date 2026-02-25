import os
import sys
import json
import joblib
import re
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.feature_formatter import FeatureFormatter
from utils.study_config import OUTCOMES, OUTCOME_TYPE
from utils.paths import get_project_root, get_model_dir, get_main_table_dir, get_supplementary_table_dir, get_supplementary_figure_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.plot_config import (
    apply_medical_style, SAVE_DPI, PALETTE_MAIN, COLOR_POSITIVE, COLOR_NEGATIVE,
    COLOR_REF_LINE, OR_POINT_COLOR, FIG_WIDTH_DOUBLE, save_fig_medical
)

BASE_DIR = get_project_root()
MODEL_ROOT = get_model_dir()
FIG_CUTOFF_DIR = get_supplementary_figure_dir("S3_cutoff")
TABLE_ROOT = get_main_table_dir()
SUPP_TABLE_DIR = get_supplementary_table_dir()
ensure_dirs(FIG_CUTOFF_DIR, TABLE_ROOT, SUPP_TABLE_DIR)

def calculate_detailed_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        "Threshold": round(threshold, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "PPV": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
        "NPV": round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0,
        "F1_Score": round(f1_score(y_true, y_pred), 4),
        "Accuracy": round((tp + tn) / (tp + tn + fp + fn), 4)
        # "Sen_CI": "N/A"  # 如果不跑 Bootstrap，建议先注释掉或设为 N/A
    }
    return metrics

def plot_diagnostic_viz(y_true, y_prob, threshold, name, target, save_dir):
    """诊断图：医学期刊格式（双栏、DPI 600）"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    apply_medical_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE * 1.5, 5), dpi=300, facecolor='white')
    c_normal = COLOR_NEGATIVE
    c_event = COLOR_POSITIVE
    c_main = OR_POINT_COLOR
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    auc_label = f'{name} (AUC = {auc_val:.3f})'
    ci_lo, ci_hi = None, None
    ci_path = os.path.join(get_model_dir(target), "bootstrap_ci_stats.pkl")
    if os.path.exists(ci_path):
        try:
            boot_stats = joblib.load(ci_path)
            if name in boot_stats and 'main' in boot_stats[name]:
                ci_lo, ci_hi = boot_stats[name]['main']
                auc_label = f"{name} (AUC = {auc_val:.3f}, 95% CI: {ci_lo:.3f}-{ci_hi:.3f})"
        except Exception:
            pass

    ax1.plot(fpr, tpr, label=auc_label, color=c_main, lw=2.5)
    ax1.plot([0, 1], [0, 1], linestyle='--', color=COLOR_REF_LINE, alpha=0.5, lw=1)

    text_auc = f'AUC = {auc_val:.3f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f})' if (ci_lo is not None and ci_hi is not None) else f'AUC = {auc_val:.3f}'
    ax1.text(0.6, 0.1, text_auc, fontsize=12, 
             fontweight='bold', color=c_main, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    perf = calculate_detailed_metrics(y_true, y_prob, threshold)
    ax1.scatter(1-perf['Specificity'], perf['Sensitivity'], 
                color=c_event, s=120, edgecolors='white', zorder=5,
                label=f'Optimal Cutoff: {threshold:.3f}')
    ax1.annotate(f'Sensitivity: {perf["Sensitivity"]:.2f}\nSpecificity: {perf["Specificity"]:.2f}',
                 xy=(1-perf['Specificity'], perf['Sensitivity']), 
                 xytext=(1-perf['Specificity']+0.12, perf['Sensitivity']-0.15),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color='black', connectionstyle="arc3,rad=.2"))
    ax1.set_xlabel('1 - Specificity (False Positive Rate)', labelpad=10)
    ax1.set_ylabel('Sensitivity (True Positive Rate)', labelpad=10)
    ax1.set_title(f'Diagnostic Performance: {name}\n({target.upper()})', fontweight='bold', pad=15)
    ax1.legend(loc='lower right', frameon=False)
    ax1.set_aspect('equal')
    df_prob = pd.DataFrame({'prob': y_prob, 'target': y_true})
    sns.kdeplot(data=df_prob[df_prob['target'] == 0], x='prob', fill=True, 
                label='Normal/Survival', color=c_normal, ax=ax2, alpha=0.4, lw=2)
    sns.kdeplot(data=df_prob[df_prob['target'] == 1], x='prob', fill=True, 
                label='Outcome Event', color=c_event, ax=ax2, alpha=0.4, lw=2)
    ax2.axvline(threshold, color=c_main, linestyle='--', lw=2, alpha=0.8)
    ylim = ax2.get_ylim()[1]
    text_style = dict(fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    
    ax2.text(threshold-0.03, ylim*0.92, 'LOW RISK', ha='right', color=c_normal, **text_style)
    ax2.text(threshold+0.03, ylim*0.85, 'HIGH RISK', ha='left', color=c_event, **text_style)
    ax2.set_xlabel('Predicted Risk Probability', labelpad=10)
    ax2.set_ylabel('Population Density', labelpad=10)
    ax2.set_title('Clinical Risk Stratification', fontweight='bold', pad=15)
    ax2.legend(frameon=False)
    sns.despine() # 移除上方和右侧边框
    plt.tight_layout()
    save_filename = f"07_Diagnostic_{name}_{target}"
    save_base = os.path.join(save_dir, save_filename)
    save_fig_medical(save_base)
    plt.close()
    _log(f"诊断图: {os.path.abspath(save_base)}", "OK")

def export_formatted_table3():
    summary_path = os.path.join(MODEL_ROOT, "global_diagnostic_summary.csv")
    if not os.path.exists(summary_path):
        _log("未找到汇总表，无法导出 Table 3", "WARN"); return
    
    df = pd.read_csv(summary_path)
    formatted_rows = []
    ci_cache = {}
    
    for _, row in df.iterrows():
        target = str(row['Outcome']).lower()
        algo = row['Algorithm']
        group_raw = str(row['Group'])
        
        # 1. 加载第 06 步 CI 资产
        if target not in ci_cache:
            ci_path = os.path.join(get_model_dir(target), "bootstrap_ci_stats.pkl")
            ci_cache[target] = joblib.load(ci_path) if os.path.exists(ci_path) else None

        # 2. 人群分类与 CI 索引匹配
        if 'Full' in group_raw:
            display_group, ci_key, group_order = 'Full Population', 'main', 0
        else:
            display_group, ci_key, group_order = 'Subgroup (No Renal)', 'sub', 1

        # 3. 格式化 AUC (95% CI)
        auc_val = row['AUC']
        auc_str = f"{auc_val:.3f}"
        target_ci = ci_cache.get(target)
        if target_ci and algo in target_ci:
            try:
                low, high = target_ci[algo][ci_key]
                auc_str = f"{auc_val:.3f} ({low:.3f}–{high:.3f})"
            except (KeyError, TypeError):
                pass 

        outcome_order = {'pof': 0, 'mortality': 1, 'composite': 2}.get(target, 9)
        formatted_rows.append({
            'Endpoint': OUTCOME_TYPE.get(target, target),
            'Outcome': row['Outcome'].upper(),
            'Group': display_group,
            'Model': algo,
            'AUC (95% CI)': auc_str,
            'Sens.': f"{row['Sensitivity']:.3f}",
            'Spec.': f"{row['Specificity']:.3f}",
            'F1': f"{row['F1_Score']:.3f}",
            'Optimal Cut-off': f"{row['Threshold']:.3f}",
            'auc_numeric': auc_val,
            'group_priority': group_order,
            'outcome_order': outcome_order
        })

    # 5. 多级逻辑排序并导出
    table3 = pd.DataFrame(formatted_rows)
    table3 = table3.sort_values(
        ['outcome_order', 'group_priority', 'auc_numeric'],
        ascending=[True, True, False]
    )

    final_columns = ['Endpoint', 'Outcome', 'Group', 'Model', 'AUC (95% CI)', 'Sens.', 'Spec.', 'F1', 'Optimal Cut-off']
    output_path = os.path.join(TABLE_ROOT, "Table3_performance.csv")
    table3[final_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
    _log(f"Table 3: {os.path.abspath(output_path)}", "OK")
    
class ResultVisualizer:
    def __init__(self, base_dir=None):
        base_dir = base_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
        self.model_root = os.path.join(base_dir, "artifacts/models")
        self.result_root = get_supplementary_figure_dir("S3_cutoff")
        self.report_path = os.path.join(self.model_root, "performance_report.csv")
        os.makedirs(self.result_root, exist_ok=True)
        
        apply_medical_style()
        self.sci_palette = PALETTE_MAIN

    def summarize_feature_importance(self, outcomes=None, top_n=15):
        outcomes = outcomes or OUTCOMES
        _log("正在生成出版级特征重要性图...", "INFO")
        all_imps = []
        for target in outcomes:
            path = os.path.join(self.model_root, target, "feature_importance.csv")
            if os.path.exists(path):
                all_imps.append(pd.read_csv(path))
        
        if not all_imps: return

        full_df = pd.concat(all_imps, ignore_index=True)
        # 若无 display_name 列则补充（兼容旧版 feature_importance.csv）
        if 'display_name' not in full_df.columns:
            formatter = FeatureFormatter()
            full_df['display_name'] = full_df['feature'].map(lambda x: formatter.get_label(x))
        pivot_df = full_df.groupby(['feature', 'outcome'])['importance'].mean().unstack()
        pivot_df['Global_Avg'] = pivot_df.mean(axis=1)
        top_feats = pivot_df.sort_values('Global_Avg', ascending=False).head(top_n).index.tolist()
        feat_to_display = full_df.drop_duplicates('feature').set_index('feature')['display_name'].to_dict()
        top_display = [feat_to_display.get(f, f) for f in top_feats]

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 6), dpi=300, facecolor='white')
        plot_data = full_df[full_df['feature'].isin(top_feats)]
        
        sns.barplot(data=plot_data, y='display_name', x='importance', hue='outcome', 
                    order=top_display, palette="mako", alpha=0.9, edgecolor="white", linewidth=0.5)
        
        # 添加轻量级垂直网格线
        ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
        ax.set_axisbelow(True)

        plt.title("Primary Clinical Predictors of Outcomes", loc='left', pad=20)
        plt.xlabel("Mean Relative Feature Importance (Normalized)")
        plt.ylabel("")
        
        # 优化图例
        plt.legend(title="Clinical Outcome", frameon=False, loc='lower right', bbox_to_anchor=(1, 0.05))
        
        sns.despine()
        plt.tight_layout()
        save_base = os.path.join(self.result_root, "sci_feature_importance")
        save_fig_medical(save_base)
        plt.close()

    def plot_performance_forest(self):
        """医学出版级森林图（更清晰的分组与对齐）"""
        _log("正在生成出版级森林图...", "INFO")
        if not os.path.exists(self.report_path): return

        df = pd.read_csv(self.report_path)
        def parse_ci(s):
            vals = re.findall(r"([0-9.]+)", str(s))
            return [float(x) for x in vals[:3]] if len(vals) >= 3 else [np.nan]*3
        parsed = df['Main CI'].apply(parse_ci).tolist()
        df[['auc', 'low', 'high']] = pd.DataFrame(parsed, index=df.index)

        outcomes = df['Outcome'].unique()
        models = df['Algorithm'].unique()
        color_map = dict(zip(models, self.sci_palette))

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 6), dpi=300, facecolor='white')
        
        y_pos = 0
        y_ticks, y_labels = [], []

        for outcome in reversed(outcomes):
            sub_df = df[df['Outcome'] == outcome]
            
            # 结局分组横带 (医学期刊常用风格)
            y_pos += 1
            ax.axhspan(y_pos-0.5, y_pos+len(sub_df)+0.5, color='#F8F9FA', alpha=0.8, zorder=0)
            
            # 分组标题
            ax.text(0.51, y_pos + (len(sub_df)/2) + 0.5, f"Outcome: {outcome.upper()}", 
                    va='center', ha='left', fontweight='black', fontsize=12, 
                    color='#2C3E50', style='italic')
            
            for _, row in sub_df.iterrows():
                y_pos += 1
                y_ticks.append(y_pos)
                y_labels.append(row['Algorithm'])
                
                # 绘制置信区间线条
                ax.plot([row['low'], row['high']], [y_pos, y_pos], color=color_map[row['Algorithm']], 
                        linewidth=2.5, solid_capstyle='round', zorder=3)
                
                # 绘制中心点 (Marker)
                ax.scatter(row['auc'], y_pos, color=color_map[row['Algorithm']], 
                           s=80, edgecolors='white', linewidth=1, zorder=4)
                
                # 数值标注 (对齐对齐)
                label_text = f"{row['auc']:.3f} [{row['low']:.3f} - {row['high']:.3f}]"
                ax.text(1.01, y_pos, label_text, va='center', ha='left', fontsize=9.5, fontfamily='monospace')

            y_pos += 1.5 # 组间距

        # 图表修饰
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontweight='normal')
        ax.axvline(0.5, color='#34495E', linestyle='-', linewidth=1.5, alpha=0.8, label='Chance level')
        ax.axvline(0.8, color='#BDC3C7', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel('Area Under the ROC Curve (95% CI)', labelpad=15)
        ax.set_xlim(0.5, 1.0) # AUC 不可能超过 1.0
        
        # 移除坐标轴冗余
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        
        plt.title("Predictive Performance Across Clinical Outcomes", loc='left', pad=30, fontsize=16)
        
        # 图例美化
        patches = [mpatches.Patch(color=color_map[m], label=m) for m in models]
        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=len(models), frameon=False, fontsize=9)

        plt.tight_layout()
        save_base = os.path.join(self.result_root, "sci_forest_plot")
        save_fig_medical(save_base)
        plt.close()
        _log(f"森林图: {os.path.abspath(save_base)}.png", "OK")

# 2. 核心执行逻辑
def run_cutoff_optimization_flow():
    log_header("🚀 07_optimal_cutoff_analysis: 阈值寻优与临床效能审计")
    _log(f"模型目录: {os.path.abspath(MODEL_ROOT)}", "INFO")
    global_summary = []

    for target in OUTCOMES:
        target_dir = get_model_dir(target)
        fig_save_dir = os.path.join(FIG_CUTOFF_DIR, target)
        ci_stats = None
        
        if not os.path.exists(target_dir):
            _log(f"跳过 {target}: 路径缺失", "WARN"); continue

        _log(f"正在处理终点: [{target.upper()}]", "INFO")
        ci_path = os.path.join(target_dir, "bootstrap_ci_stats.pkl")
        if os.path.exists(ci_path):
            try:
                ci_stats = joblib.load(ci_path)
            except Exception:
                ci_stats = None

        try:
            # 1. 资产加载与特征对齐
            models_dict = joblib.load(os.path.join(target_dir, "all_models_dict.pkl"))
            eval_data = joblib.load(os.path.join(target_dir, "eval_data.pkl"))
            X_test_pre, y_test = eval_data['X_test_pre'], eval_data['y_test']
            feat_path = os.path.join(target_dir, "selected_features.json")
            if os.path.exists(feat_path):
                with open(feat_path, 'r') as f:
                    feat_data = json.load(f)
                
                # 关键修复：从字典中提取 "features" 列表
                if isinstance(feat_data, dict) and "features" in feat_data:
                    selected_features = feat_data["features"]
                else:
                    selected_features = list(feat_data) # 兜底逻辑
                
                # 仅选择测试集中存在的特征
                valid_cols = [c for c in selected_features if c in X_test_pre.columns]
                X_eval = X_test_pre[valid_cols].values
            else:
                X_eval = X_test_pre.values

            best_model_name = max(
                models_dict.keys(), 
                key=lambda n: roc_auc_score(y_test, models_dict[n].predict_proba(X_eval)[:, 1])
            )
            _log(f"选定最佳模型进行诊断可视化: {best_model_name}", "OK")

        except Exception as e:
            _log(f"资产解析失败 ({target}): {e}", "ERR")
            continue

        target_thresholds, target_perf_report = {}, []

        # 2. 遍历模型：计算阈值与多维效能
        for name, clf in models_dict.items():
            y_prob = clf.predict_proba(X_eval)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            
            # Youden Index 寻优 (全人群)
            if len(thresholds) <= 1:
                best_th = 0.5
            else:
                youden_index = tpr + (1 - fpr) - 1
                best_th = float(thresholds[np.argmax(youden_index)])
            
            best_th = min(1.0, best_th) # 修正非法阈值
            target_thresholds[name] = best_th

            # --- 全人群效能审计 ---
            perf_main = calculate_detailed_metrics(y_test, y_prob, best_th)
            auc_low_main, auc_high_main = np.nan, np.nan
            if isinstance(ci_stats, dict) and name in ci_stats and "main" in ci_stats[name]:
                try:
                    auc_low_main, auc_high_main = ci_stats[name]["main"]
                except Exception:
                    pass
            perf_main.update({
                'Algorithm': name, 
                'Group': 'Full Population', 
                'Outcome': target,
                'AUC': round(auc_val, 4), # 【新增】加入 AUC
                'AUC_Low': round(float(auc_low_main), 4) if not np.isnan(auc_low_main) else np.nan,
                'AUC_High': round(float(auc_high_main), 4) if not np.isnan(auc_high_main) else np.nan,
            })
            target_perf_report.append(perf_main)

            # --- 亚组效能审计 (肾病亚组保护逻辑) ---
            if 'sub_mask' in eval_data:
                mask = eval_data['sub_mask']
                # 【新增：样本量保护检查】
                if mask.sum() > 10 and len(np.unique(y_test[mask])) > 1:
                    y_prob_sub = y_prob[mask]
                    y_test_sub = y_test[mask]
                    
                    # 【优化：计算亚组独立最优截断值】
                    fpr_s, tpr_s, th_s = roc_curve(y_test_sub, y_prob_sub)
                    youden_s = tpr_s + (1 - fpr_s) - 1
                    best_th_sub = float(th_s[np.argmax(youden_s)])
                    
                    # 使用主人群阈值评估当前性能
                    perf_sub = calculate_detailed_metrics(y_test_sub, y_prob_sub, best_th)
                    auc_sub = roc_auc_score(y_test_sub, y_prob_sub)
                    auc_low_sub, auc_high_sub = np.nan, np.nan
                    if isinstance(ci_stats, dict) and name in ci_stats and "sub" in ci_stats[name]:
                        try:
                            auc_low_sub, auc_high_sub = ci_stats[name]["sub"]
                        except Exception:
                            pass
                    perf_sub.update({
                        'Algorithm': name, 
                        'Group': 'Subgroup (Non-Renal)', 
                        'Outcome': target,
                        'AUC': round(auc_sub, 4),
                        'AUC_Low': round(float(auc_low_sub), 4) if not np.isnan(auc_low_sub) else np.nan,
                        'AUC_High': round(float(auc_high_sub), 4) if not np.isnan(auc_high_sub) else np.nan,
                        'Subgroup_Specific_Th': round(best_th_sub, 4) # 存储独立的建议阈值
                    })
                    target_perf_report.append(perf_sub)

            # 仅为最佳模型生成诊断可视化图，避免图片冗余
            if name == best_model_name:
                plot_diagnostic_viz(y_test, y_prob, best_th, name, target, fig_save_dir)

        # 3. 资产持久化
        # 保存阈值 JSON (用于 eICU 外部验证一键映射)
        with open(os.path.join(target_dir, "thresholds.json"), 'w') as f:
            json.dump(target_thresholds, f, indent=4)
        
        # 存入结局子目录与 Table 汇总目录
        perf_df = pd.DataFrame(target_perf_report)
        perf_df.to_csv(os.path.join(target_dir, "internal_diagnostic_perf.csv"), index=False)
        perf_df.to_csv(os.path.join(SUPP_TABLE_DIR, f"Table3_Perf_{target}.csv"), index=False)
        
        global_summary.extend(target_perf_report)
        
        # 实时反馈
        best_perf = next(p for p in target_perf_report if p['Algorithm'] == best_model_name and p['Group'] == 'Full Population')
        _log(f"审计完成。最优模型 F1: {best_perf['F1_Score']} (AUC: {best_perf['AUC']})", "OK")

    # 4. 全局汇总并按学术逻辑排序
    if global_summary:
        summary_df = pd.DataFrame(global_summary)
        # 排序：结局升序 -> 分组升序 -> AUC 降序
        summary_df = summary_df.sort_values(['Outcome', 'Group', 'AUC'], ascending=[True, True, False])
        summary_df.to_csv(os.path.join(MODEL_ROOT, "global_diagnostic_summary.csv"), index=False)
        _log(f"任务圆满完成！全局报告见: {MODEL_ROOT}/global_diagnostic_summary.csv", "OK")

if __name__ == "__main__":
    # 1. 执行 07 步主流程：计算阈值与效能
    run_cutoff_optimization_flow()
    
    # 2. 自动导出 Table 3 (Excel 或 CSV)
    export_formatted_table3()
    
    # 3. 实例化可视化工具
    viz = ResultVisualizer(base_dir=BASE_DIR)
    
    # 4. 生成特征重要性汇总图 (基于第 7 步后的资产)
    viz.summarize_feature_importance(outcomes=OUTCOMES, top_n=15)
    
    # 5. 【关键修复】生成森林图：强制使用第 06 步生成的置信区间报告
    step7_report_path = os.path.join(MODEL_ROOT, "performance_report.csv")
    
    if os.path.exists(step7_report_path):
        _log(f"正在基于第 06 步的置信区间数据生成森林图: {step7_report_path}", "INFO")
        viz.report_path = step7_report_path
        viz.plot_performance_forest()
    else:
        _log("警告: 未找到第 06 步的 performance_report.csv。", "WARN")
        _log("请确保已运行第 06 步脚本，森林图需要其提供的 95% CI 数据。", "WARN")

    _log(f"07 步图片输出目录: {os.path.abspath(FIG_CUTOFF_DIR)}", "OK")
