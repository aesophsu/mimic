import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dcurves import dca
from sklearn.calibration import calibration_curve
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.paths import get_model_dir, get_external_dir, get_main_figure_dir, get_supplementary_table_dir, ensure_dirs
from utils.plot_config import apply_medical_style, SAVE_DPI, PALETTE_MAIN, FIG_DCA_FIGSIZE, save_fig_medical
from utils.study_config import OUTCOMES
from utils.logger import log as _log, log_header

MODEL_ROOT = get_model_dir()
EICU_DIR = get_external_dir()
FIGURE_DIR = get_main_figure_dir()  # Fig3 DCA
SUPP_TABLE_DIR = get_supplementary_table_dir()
ensure_dirs(FIGURE_DIR, SUPP_TABLE_DIR)

TARGETS = OUTCOMES

def load_model_and_data(target):
    """加载模型和 eICU 数据（含肾功能亚组标记）"""
    target_dir = get_model_dir(target)
    models_path = os.path.join(target_dir, "all_models_dict.pkl")
    # 假设 eICU 验证集已准备好
    eicu_path = os.path.join(EICU_DIR, f"eicu_processed_{target.lower()}.csv")
    
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"模型资产缺失: {target}")
    
    models = joblib.load(models_path)
    # 这里的 df_eicu 应该是你在 Module 11 中生成的外部验证 DataFrame
    df_eicu = pd.read_csv(eicu_path)
    
    y_true = df_eicu[target].values
    # 确保特征列与训练时一致
    eval_data = joblib.load(os.path.join(target_dir, "eval_data.pkl"))
    features = eval_data['features']
    X = df_eicu[features]

    # 肾功能亚组：若缺失则全部视作 1，避免后续逻辑报错
    if 'subgroup_no_renal' in df_eicu.columns:
        renal_subgroup = df_eicu['subgroup_no_renal'].values.astype(int)
    else:
        renal_subgroup = np.ones(len(df_eicu), dtype=int)
        _log("警告: eICU 数据中缺失 subgroup_no_renal 列，DCA 分层分析将退化为整体人群。", "WARN")
    
    return models, X, y_true, renal_subgroup

def plot_dca(dca_res, data_df, target, subgroup_label=None):
    """
    医学出版级 DCA 与 Calibration 组合绘图
    增强版：包含 Treat All/None 标注、最佳阈值辅助线及次轴校准曲线
    """
    # 1. 数据预处理
    plot_df = dca_res.copy()
    plot_df['model'] = plot_df['model'].str.replace('_prob', '', regex=False)
    
    # 获取绘图范围最大值以便放置文字
    max_nb = plot_df['net_benefit'].max()
    
    # 加载最佳阈值：07 输出 thresholds.json（各模型专属阈值），优先取 XGBoost
    thresh_path = os.path.join(MODEL_ROOT, target.lower(), "thresholds.json")
    optimal_th = None
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            th_dict = json.load(f)
            optimal_th = th_dict.get('XGBoost') or th_dict.get('best_threshold')
            if optimal_th is None and th_dict:
                optimal_th = next(iter(th_dict.values()), 0.5)

    apply_medical_style()
    # 与外部校准曲线图统一：双栏 7.2in × 6in
    fig, ax1 = plt.subplots(figsize=FIG_DCA_FIGSIZE, dpi=300, facecolor='white')
    
    # 2. 绘制 DCA 基准线 (All vs None)
    sns.lineplot(data=plot_df[plot_df['model'] == 'all'], x='threshold', y='net_benefit', 
                 color='black', linestyle='--', lw=1.2, ax=ax1, zorder=1)
    sns.lineplot(data=plot_df[plot_df['model'] == 'none'], x='threshold', y='net_benefit', 
                 color='black', lw=1.2, ax=ax1, zorder=1)
    
    # 3. 绘制预测模型 DCA 曲线
    pred_models = plot_df[~plot_df['model'].isin(['all', 'none'])]
    sns.lineplot(data=pred_models, x='threshold', y='net_benefit', 
                 hue='model', lw=2.5, palette=PALETTE_MAIN, ax=ax1, zorder=2)
    
    # 4. 显式文本标注 (Treat All / Treat None)
    # Treat All: 放在曲线左上角（threshold 低处），与虚线起始段视觉关联，避免低 prevalence 时落到底部
    all_df = plot_df[plot_df['model'] == 'all']
    th_all = all_df['threshold'].values
    nb_all = all_df['net_benefit'].values
    idx_left = np.argmin(np.abs(th_all - 0.08))
    ax1.text(0.10, nb_all[idx_left] + 0.015, 'Treat All', fontsize=9, fontweight='bold', color='black',
             ha='left', va='bottom')
    # Treat None: 放在 y≈0 基线附近，绘图区域内
    ax1.text(0.68, 0.015, 'Treat None', fontsize=9, fontweight='bold', color='black',
             ha='left', va='bottom')
    
    # 5. 添加最佳阈值标注线
    if optimal_th is not None:
        ax1.axvline(optimal_th, color='gray', linestyle=':', alpha=0.6, lw=1.5)
        ax1.text(optimal_th + 0.01, max_nb * 0.55, f'Optimal Th: {optimal_th:.3f}', 
                 fontsize=9, color='dimgray', fontstyle='italic', fontweight='bold')
    
    # 6. 叠加校准曲线 (使用右侧次坐标轴)
    main_model_col = [c for c in data_df.columns if '_prob' in c][0]
    prob_true, prob_pred = calibration_curve(data_df['outcome'], data_df[main_model_col], n_bins=10)
    
    ax2 = ax1.twinx()
    ax2.plot(prob_pred, prob_true, 'o--', color='purple', alpha=0.35, label='Reliability (Calibration)')
    ax2.set_ylabel('Observed Fraction (Actual)', color='purple', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(0, 1)

    # 7. 出版级样式美化
    title_suffix = f" (subgroup_no_renal = {subgroup_label})" if subgroup_label is not None else ""
    ax1.set_title(f"Clinical Benefit & Reliability Analysis: {target.upper()}{title_suffix}", 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Threshold Probability (Risk Threshold)", fontsize=11)
    ax1.set_ylabel("Net Benefit (Clinical Utility)", fontsize=11)
    ax1.set_xlim(0, 0.75)
    ax1.set_ylim(-0.015, max_nb * 1.15)
    
    # 合并图例
    ax1.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=9)
    ax1.grid(axis='y', color='whitesmoke', linestyle='--', zorder=0)
    sns.despine(ax=ax1, right=False)

    # 收紧布局，减少底部留白
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.12, top=0.92)

    # 8. 导出文件（最小边距以减少留白）
    if subgroup_label is not None:
        base_n = os.path.join(FIGURE_DIR, f"Fig3_DCA_calibration_{target}_subgroupNoRenal_{subgroup_label}")
    else:
        base_n = os.path.join(FIGURE_DIR, f"Fig3_DCA_calibration_{target}")
    save_fig_medical(base_n, pad_inches=0.01)
    plt.close()
    _log(f"DCA-Calibration: {os.path.abspath(base_n)}.png", "OK")

def compute_net_benefit(models, X_scaled, y_true, target):
    """
    执行 DCA 计算
    """
    _log(f"Calculating Net Benefit for {target.upper()}...", "INFO")
    
    data_df = pd.DataFrame({'outcome': y_true})
    predictor_cols = []
    X_input = X_scaled.values if hasattr(X_scaled, 'values') else X_scaled
    
    for name, model in models.items():
        prob_col = f"{name}_prob"
        # 使用 X_input (numpy array) 进行预测，消除警告
        data_df[prob_col] = model.predict_proba(X_input)[:, 1]
        predictor_cols.append(prob_col)

    thresholds = np.arange(0, 0.76, 0.01)

    # 位置参数调用 (保持之前成功的逻辑)
    try:
        dca_res = dca(data_df, 'outcome', predictor_cols, thresholds)
    except Exception:
        try:
            dca_res = dca(data=data_df, outcome='outcome', model_names=predictor_cols, thresholds=thresholds)
        except:
            dca_res = dca(data=data_df, outcome='outcome', predictors=predictor_cols, thresholds=thresholds)
    
    return dca_res, data_df


def main():
    log_header("Module 13: Clinical Utility (DCA) & External Validation")
    
    for target in TARGETS:
        _log(f"Analyzing Outcome: {target.upper()}", "INFO")
        try:
            models, X_scaled, y_true, renal_sub = load_model_and_data(target)

            # 整体人群 DCA + 校准（保持原有 Figure 3）
            dca_results, eval_df = compute_net_benefit(models, X_scaled, y_true, target)
            plot_dca(dca_results, eval_df, target)
            dca_results.to_csv(os.path.join(SUPP_TABLE_DIR, f"DCA_Data_{target}.csv"), index=False)

            # 按肾功能亚组 (subgroup_no_renal) 分层 DCA + 校准
            renal_arr = np.asarray(renal_sub).astype(int)
            for grp in [1, 0]:
                mask = renal_arr == grp
                if mask.sum() < 10:
                    _log(f"DCA: subgroup_no_renal={grp} 样本量过少 (n={mask.sum()})，跳过分层 DCA。", "WARN")
                    continue
                y_sub = np.asarray(y_true)[mask]
                if len(np.unique(y_sub)) < 2:
                    _log(f"DCA: subgroup_no_renal={grp} 仅有单一结局，跳过分层 DCA。", "WARN")
                    continue

                # X_scaled 可能是 DataFrame 或 ndarray，分别处理
                if hasattr(X_scaled, 'iloc'):
                    X_sub = X_scaled.iloc[mask]
                else:
                    X_sub = X_scaled[mask]

                dca_sub, eval_sub = compute_net_benefit(models, X_sub, y_sub, target)
                plot_dca(dca_sub, eval_sub, target, subgroup_label=str(grp))
                dca_sub.to_csv(
                    os.path.join(SUPP_TABLE_DIR, f"DCA_Data_{target}_subgroupNoRenal_{grp}.csv"),
                    index=False,
                )
            
        except Exception:
            _log(f"Critical Error Failed for {target}:", "ERR")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
