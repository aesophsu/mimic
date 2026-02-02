import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.feature_formatter import FeatureFormatter
from utils.study_config import OUTCOMES
from utils.paths import get_cleaned_path, get_external_dir, get_validation_dir, get_supplementary_figure_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.plot_config import apply_medical_style, SAVE_DPI, COLOR_POSITIVE, COLOR_NEGATIVE, FIG_WIDTH_DOUBLE, save_fig_medical

MIMIC_PROCESSED = get_cleaned_path("mimic_processed.csv")
EICU_PROCESSED_DIR = get_external_dir()
VALIDATION_DIR = get_validation_dir()
FIGURE_DIR = get_supplementary_figure_dir("S4_comparison")
ensure_dirs(VALIDATION_DIR, FIGURE_DIR)

TARGETS = OUTCOMES

def load_processed_data(target):
    mimic_path = MIMIC_PROCESSED
    eicu_path = os.path.join(EICU_PROCESSED_DIR, f"eicu_processed_{target}.csv")
    
    if not os.path.exists(mimic_path) or not os.path.exists(eicu_path):
        raise FileNotFoundError(f"缺失文件: {mimic_path} 或 {eicu_path}")
    
    df_mimic = pd.read_csv(mimic_path)
    df_eicu = pd.read_csv(eicu_path)
    
    # 排除非预测特征
    exclude = [
        'pof', 'mortality', 'composite', 'subgroup_no_renal', 
        'gender', 'malignant_tumor', 'mechanical_vent_flag', 
        'vaso_flag', 'dialysis_flag', 'uniquepid', 'patientunitstayid'
    ]
    
    # 共有特征
    common_features = [c for c in df_eicu.columns if c in df_mimic.columns and c not in exclude]
    
    # 数值型
    features = [c for c in common_features if pd.api.types.is_numeric_dtype(df_mimic[c])]
    
    _log(f"对齐成功: 结局 [{target.upper()}] 共有 {len(features)} 个特征参与漂移分析", "INFO")
    
    return df_mimic[features], df_eicu[features], features

def ks_drift_test(mimic_series, eicu_series):
    if len(mimic_series.dropna()) < 5 or len(eicu_series.dropna()) < 5:
        return {"statistic": np.nan, "pvalue": np.nan, "drift": "样本不足"}
    
    ks_stat, p_value = ks_2samp(mimic_series.dropna(), eicu_series.dropna())
    drift_level = "显著" if p_value < 0.05 else "不显著"
    return {
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 4),
        "drift_significant": drift_level,
        "max_diff": round(ks_stat, 4)  # KS 统计量本身即最大累积差异
    }

def plot_distribution_comparison(mimic_series, eicu_series, feature_name, target):
    apply_medical_style()
    formatter = FeatureFormatter()
    display_name = formatter.get_label(feature_name)
    plt.figure(figsize=(FIG_WIDTH_DOUBLE, 5), dpi=300, facecolor='white')
    sns.kdeplot(mimic_series.dropna(), label='MIMIC (Train)', color=COLOR_NEGATIVE, linewidth=2)
    sns.kdeplot(eicu_series.dropna(), label='eICU (Validation)', color=COLOR_POSITIVE, linewidth=2)
    
    plt.title(f'Distribution Comparison: {display_name}\n({target.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel(display_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    sns.despine()
    
    ks_result = ks_drift_test(mimic_series, eicu_series)
    plt.text(0.02, 0.95, f"KS Statistic: {ks_result['ks_statistic']:.4f}\np-value: {ks_result['p_value']:.4f}",
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    base_path = os.path.join(FIGURE_DIR, f"dist_drift_{feature_name}_{target}")
    save_fig_medical(base_path)
    plt.close()
    return base_path

def main():
    log_header("🚀 09_cross_cohort_audit: 跨队列漂移分析 (MIMIC vs eICU)")
    _log(f"MIMIC: {os.path.abspath(MIMIC_PROCESSED)}", "INFO")
    _log(f"eICU: {os.path.abspath(EICU_PROCESSED_DIR)}", "INFO")
    drift_summary = {}

    for target in TARGETS:
        _log(f"分析结局: {target.upper()}", "INFO")
        try:
            df_mimic, df_eicu, features = load_processed_data(target)
        except Exception as e:
            _log(f"加载失败: {e}", "ERR")
            continue

        drift_summary[target] = {}
        top_drift_features = []

        for feature in features:
            if feature not in df_mimic.columns or feature not in df_eicu.columns:
                continue

            mimic_vals = df_mimic[feature]
            eicu_vals = df_eicu[feature]

            ks_result = ks_drift_test(mimic_vals, eicu_vals)
            drift_summary[target][feature] = ks_result

            if ks_result['p_value'] < 0.05:
                top_drift_features.append((feature, ks_result['ks_statistic']))

            if ks_result['p_value'] < 0.05:
                plot_distribution_comparison(mimic_vals, eicu_vals, feature, target)

        drift_summary[target]['summary'] = {
            "total_features": len(features),
            "significant_drift": len([f for f in drift_summary[target] if drift_summary[target][f]['p_value'] < 0.05]),
            "top_drift": sorted(top_drift_features, key=lambda x: x[1], reverse=True)[:10]
        }

    drift_json_path = os.path.join(VALIDATION_DIR, "eicu_vs_mimic_drift.json")
    with open(drift_json_path, 'w', encoding='utf-8') as f:
        json.dump(drift_summary, f, ensure_ascii=False, indent=4)

    _log("漂移分析完成", "OK")
    _log(f"报告: {os.path.abspath(drift_json_path)}", "INFO")
    _log("下一步: 10_external_validation_perf.py", "INFO")

if __name__ == "__main__":
    main()
