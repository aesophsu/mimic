"""04: MIMIC 描述统计审计 - 缺失值热图"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.plot_config import apply_medical_style, SAVE_DPI, FIG_WIDTH_DOUBLE, save_fig_medical
from utils.feature_formatter import FeatureFormatter
from utils.paths import get_cleaned_path, get_main_table_dir, get_main_figure_dir, ensure_dirs
from utils.logger import log as _log, log_header

DATA_PATH = get_cleaned_path("mimic_raw_scale.csv")
TABLE_DIR = get_main_table_dir()
FIGURE_DIR = get_main_figure_dir()
ensure_dirs(FIGURE_DIR)

def plot_heatmap():
    """缺失值热图：医学期刊格式（双栏宽、DPI 600）"""
    df = pd.read_csv(DATA_PATH)
    missing_rates = df.isnull().mean()
    sorted_cols = missing_rates.sort_values(ascending=False).index
    df_sorted = df[sorted_cols]

    formatter = FeatureFormatter()
    display_labels = formatter.format_features(sorted_cols.tolist())

    apply_medical_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE * 1.5, 6), facecolor='white')

    sns.heatmap(
        df_sorted.isnull(), 
        cmap=['#F5F5F5', '#2E5A88'],
        cbar=True, 
        yticklabels=False,
        ax=ax
    )

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['Observed', 'Missing'])
    colorbar.outline.set_visible(True)

    plt.title("Pattern of Missing Clinical Observations", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Clinical Features (Sorted by Missing Rate)", fontsize=12, labelpad=10)
    plt.ylabel(f"Study Participants (N={len(df)})", fontsize=12, labelpad=10)

    ax.set_xticks(np.arange(len(display_labels)) + 0.5)
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=6)
    
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    save_base = os.path.join(FIGURE_DIR, "Fig1_missing_heatmap")
    save_fig_medical(save_base)
    plt.close()
    return os.path.abspath(f"{save_base}.png")

def main():
    log_header("🚀 04_mimic_stat_audit: 描述统计与缺失热图 (MIMIC-IV)")
    _log(f"数据: {os.path.abspath(DATA_PATH)}", "INFO")
    _log(f"输出: {os.path.abspath(FIGURE_DIR)}", "INFO")

    t1_path = os.path.join(TABLE_DIR, "Table1_baseline.csv")
    t2_path = os.path.join(TABLE_DIR, "Table2_renal_subgroup.csv")
    _log(f"Table1: {'存在' if os.path.exists(t1_path) else '缺失'} {os.path.abspath(t1_path)}", "OK" if os.path.exists(t1_path) else "WARN")
    _log(f"Table2: {'存在' if os.path.exists(t2_path) else '缺失'} {os.path.abspath(t2_path)}", "OK" if os.path.exists(t2_path) else "WARN")

    heatmap_path = plot_heatmap()
    _log(f"热图已保存: {heatmap_path}", "OK")
    _log("下一步: 05_feature_selection_lasso.py", "INFO")

if __name__ == "__main__":
    main()
