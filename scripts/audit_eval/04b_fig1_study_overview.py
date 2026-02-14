"""
04b: Figure 1 Study Overview — Panel A (CONSORT flowchart) + Panel B (missingness heatmap).
Panel B uses all model-selected features from selected_features.json (22 unique), sorted by missing rate.
横版：Fig1_study_overview.pdf, .png；竖版：Fig1_study_overview_portrait.pdf, .png。
两者均保存到 results/main/figures/ 与 docs/figures/main/。
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.plot_config import apply_medical_style, SAVE_DPI, FIG_WIDTH_DOUBLE, FIG_HEIGHT_MAX, save_fig_medical
from utils.feature_formatter import FeatureFormatter
from utils.paths import get_cleaned_path, get_main_figure_dir, get_artifact_path, get_project_root, ensure_dirs
from utils.logger import log as _log, log_header

# Flowchart counts (from MANUSCRIPT_METHODS / SQL flowchart)
MIMIC_COUNTS = [1686, 1405, 1100, 1100, 1095]
EICU_COUNTS = [1658, 1322, 1143, 1142, 1112]

DATA_PATH = get_cleaned_path("mimic_raw_scale.csv")
SELECTED_FEATURES_PATH = get_artifact_path("features", "selected_features.json")
FIGURE_DIR = get_main_figure_dir()
ensure_dirs(FIGURE_DIR)


def load_fig1_heatmap_features():
    """从 selected_features.json 加载三个结局的去重特征并保持顺序（按首次出现）。"""
    if not os.path.exists(SELECTED_FEATURES_PATH):
        return []
    with open(SELECTED_FEATURES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    seen = set()
    order = []
    for outcome in ("pof", "mortality", "composite"):
        if outcome not in data or "features" not in data[outcome]:
            continue
        for f in data[outcome]["features"]:
            if f not in seen:
                seen.add(f)
                order.append(f)
    return order


def draw_flowchart_panel(ax, box_w=4.4, title_fontsize=6.5, box_fontsize=5.5):
    """Panel A: Cohort Selection Flowchart — two parallel columns, 6 boxes + vertical arrows each.
    Column titles: MIMIC-IV (Development Cohort), eICU (External Validation Cohort).
    box_w, title_fontsize, box_fontsize 可传参以适配竖版等不同布局。
    """
    ax.set_xlim(0, 10)
    y_start = 38.0   # 流程图数据坐标起点
    gap_after_col_title = 1.2   # 队列标题与第一个框的间距（与 draw_column 内一致）
    content_height = gap_after_col_title + 5 * 7.4 + 5 * (2.0 + 1.0 + 0.35) + 6  # ~60.95
    y_bottom = y_start - content_height - 0.5
    # 上界略留白；流程图在整图中的下移靠 main() 里调整 ax 位置
    ax.set_ylim(y_bottom, y_start + 5.0)
    ax.axis('off')

    gap_before_arrow = 2.0   # 框底与箭头之间的空隙（保持不变）
    arrow_len = 1.0          # 箭头段长
    gap_after_arrow = 0.35   # 箭头与下文框间距；框间总距加大
    arrow_lw = 2.8

    # 根据全局样本量计算每一步排除人数
    # MIMIC_COUNTS / EICU_COUNTS: [initial, after step1, after step2, after step3, after step4]
    mimic_excluded = [
        MIMIC_COUNTS[i] - MIMIC_COUNTS[i + 1] for i in range(len(MIMIC_COUNTS) - 1)
    ]  # 例如 [281, 305, 0, 5]
    eicu_excluded = [
        EICU_COUNTS[i] - EICU_COUNTS[i + 1] for i in range(len(EICU_COUNTS) - 1)
    ]  # 例如 [336, 179, 1, 30]

    mimic_remaining = MIMIC_COUNTS[1:]
    eicu_remaining = EICU_COUNTS[1:]

    # Left: MIMIC-IV — 精简版排除条件与人数（Remaining 放在文本框最后一行）
    MIMIC_BOXES = [
        (f"AP admissions\n(ICD-9: 577.0; ICD-10: K85.x)\nn = {MIMIC_COUNTS[0]:,}", 7.4, False),
        (f"ICU LOS < 24 h\nn = {mimic_excluded[0]:,}\nRemaining: {mimic_remaining[0]:,}", 7.4, False),
        (f"Non-index admission\nn = {mimic_excluded[1]:,}\nRemaining: {mimic_remaining[1]:,}", 7.4, False),
        (f"Age < 18\nn = {mimic_excluded[2]:,}\nRemaining: {mimic_remaining[2]:,}", 7.4, False),
        (f"Missing predictors > 80%\nn = {mimic_excluded[3]:,}\nRemaining: {mimic_remaining[3]:,}", 7.4, False),
        (f"Final (n = {MIMIC_COUNTS[-1]:,})", 6, True),
    ]
    # Right: eICU — 精简版排除条件与人数（Remaining 放在文本框最后一行）
    EICU_BOXES = [
        (f"AP admissions\n(Keyword: pancreatit; excl. chronic)\nn = {EICU_COUNTS[0]:,}", 7.4, False),
        (f"ICU LOS < 24 h\nn = {eicu_excluded[0]:,}\nRemaining: {eicu_remaining[0]:,}", 7.4, False),
        (f"Non-index admission\nn = {eicu_excluded[1]:,}\nRemaining: {eicu_remaining[1]:,}", 7.4, False),
        (f"Age < 18\nn = {eicu_excluded[2]:,}\nRemaining: {eicu_remaining[2]:,}", 7.4, False),
        (f"Missing predictors > 80%\nn = {eicu_excluded[3]:,}\nRemaining: {eicu_remaining[3]:,}", 7.4, False),
        (f"Final (n = {EICU_COUNTS[-1]:,})", 6, True),
    ]

    title_y_offset = 1.2   # 队列标题相对 y_start 上移，靠近标题 A

    def draw_column(x_center, boxes, col_title):
        y = y_start
        ax.text(x_center, y_start + title_y_offset, col_title, ha='center', va='bottom', fontsize=title_fontsize, fontweight='bold')
        y -= gap_after_col_title
        for i, (text, box_h, is_final) in enumerate(boxes):
            rect = mpatches.FancyBboxPatch(
                (x_center - box_w/2, y - box_h), box_w, box_h,
                boxstyle="round,pad=0.04", facecolor='#2E5A88' if is_final else '#E8EEF4',
                edgecolor='#2E5A88', linewidth=1.2
            )
            ax.add_patch(rect)
            ax.text(x_center, y - box_h/2, text, ha='center', va='center', fontsize=box_fontsize,
                    color='white' if is_final else 'black', fontweight='bold' if is_final else 'normal')
            y -= box_h
            if i < len(boxes) - 1:
                y -= gap_before_arrow
                ax.annotate('', xy=(x_center, y - arrow_len), xytext=(x_center, y),
                            arrowprops=dict(arrowstyle='->', color='#2E5A88', lw=arrow_lw))
                y -= arrow_len + gap_after_arrow

    # remaining_counts: 每一步排除后剩余样本量（不含初始）
    draw_column(2.5, MIMIC_BOXES, "MIMIC-IV (Development)")
    draw_column(7.5, EICU_BOXES, "eICU (Validation)")


def plot_heatmap_panel(ax, df, key_cols, formatter, n_cols=None):
    """Panel B: Missingness heatmap for model-selected features (sorted by missing rate)."""
    n_cols = n_cols or len(key_cols)
    missing_rates = df[key_cols].isnull().mean()
    sorted_cols = missing_rates.sort_values(ascending=False).index.tolist()
    df_plot = df[sorted_cols]
    display_labels = formatter.format_features(sorted_cols)

    sns.heatmap(
        df_plot.isnull().astype(int),
        cmap=['#F5F5F5', '#2E5A88'],
        cbar=True,
        yticklabels=False,
        ax=ax,
        cbar_kws=dict(ticks=[0.25, 0.75], shrink=0.5, pad=0.02)
    )
    cbar = ax.collections[0].colorbar
    # 图例字体与坐标轴标签保持一致
    cbar.ax.set_yticklabels(['Observed', 'Missing'], rotation=90, va='center', fontsize=7)
    cbar.outline.set_visible(True)

    # 小画布同步缩小字号
    xtick_fontsize = 5 if n_cols >= 18 else 6
    ax.set_xticks(np.arange(len(display_labels)) + 0.5)
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=xtick_fontsize)
    # 轴标题精简
    ax.set_xlabel("Clinical features", fontsize=7, labelpad=6)
    ax.set_ylabel(f"Participants (N={len(df):,})", fontsize=7, labelpad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    log_header("04b_fig1_study_overview: Figure 1 Study Overview (Panel A + B)")
    _log(f"Data: {os.path.abspath(DATA_PATH)}", "INFO")
    _log(f"Output: {os.path.abspath(FIGURE_DIR)}", "INFO")

    if not os.path.exists(DATA_PATH):
        _log(f"Cleaned data not found: {DATA_PATH}. Run 01/02 first.", "ERR")
        return

    df = pd.read_csv(DATA_PATH)
    all_selected = load_fig1_heatmap_features()
    key_cols = [c for c in all_selected if c in df.columns]
    if not all_selected:
        _log("selected_features.json not found or empty; heatmap will be empty.", "WARN")
    elif len(key_cols) < len(all_selected):
        missing = set(all_selected) - set(key_cols)
        _log(f"Features in JSON but not in CSV (skipped): {missing}", "WARN")
    _log(f"Heatmap variables (model-selected): {len(key_cols)}", "INFO")

    apply_medical_style()
    # 横版画布：宽=双栏 7.2in，高按宽高比 2.33 推算，符合医学期刊
    fig_w = FIG_WIDTH_DOUBLE  # 7.2 in
    fig_h = fig_w / 2.33      # ~3.09 in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs = GridSpec(1, 2, width_ratios=[0.5, 0.5], wspace=0.12, figure=fig)  # 左右排版

    # Panel A（左）
    ax_a = fig.add_subplot(gs[0])
    draw_flowchart_panel(ax_a)
    # 面板标识 A（轴坐标系内，避免被裁剪）
    ax_a.text(-0.06, 1.02, "A", transform=ax_a.transAxes,
              fontsize=9, fontweight='bold', ha='left', va='bottom')

    # Panel B（右）
    ax_b = fig.add_subplot(gs[1])
    formatter = FeatureFormatter()
    plot_heatmap_panel(ax_b, df, key_cols, formatter, n_cols=len(key_cols))
    # 面板标识 B
    ax_b.text(-0.06, 1.02, "B", transform=ax_b.transAxes,
              fontsize=9, fontweight='bold', ha='left', va='bottom')

    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.96, wspace=0.12)
    # 热图与色条：先缩小间距，再向右扩展（保持 ax_b 左缘不动，图 B 标题与图 A 对齐）
    pos_b = ax_b.get_position()
    cbar = ax_b.collections[0].colorbar
    pos_c = cbar.ax.get_position()
    gap = pos_c.x0 - pos_b.x1
    ax_b.set_position([pos_b.x0, pos_b.y0, pos_b.width + gap * 0.5, pos_b.height])
    pos_b = ax_b.get_position()
    pos_c = cbar.ax.get_position()
    gap = pos_c.x0 - pos_b.x1
    ax_b.set_position([pos_b.x0, pos_b.y0, pos_b.width + gap * 0.5, pos_b.height])
    pos_b = ax_b.get_position()
    pos_c = cbar.ax.get_position()
    # 色条整体右移，热图右缘与色条左缘留间隙，避免重叠
    cbar_right = 0.995
    gap_heatmap_cbar = 0.02   # 热图与色条之间的空隙
    ax_b_right = (cbar_right - pos_c.width) - gap_heatmap_cbar
    ax_b.set_position([pos_b.x0, pos_b.y0, ax_b_right - pos_b.x0, pos_b.height])
    cbar.ax.set_position([cbar_right - pos_c.width, pos_c.y0, pos_c.width, pos_c.height])
    # 标题 A 与 B 对齐；Panel A 向下延伸（更高、底边更低）
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    extra_h = 0.06   # Panel A 向下多延伸的高度
    ax_a.set_position([pos_a.x0, pos_b.y0 - extra_h, pos_a.width, pos_b.height + extra_h])

    save_base = os.path.join(FIGURE_DIR, "Fig1_study_overview")
    # 在画布顶部画一不可见占位，使 bbox_inches='tight' 保留 A 标题上方留白（否则会被裁掉）
    fig.text(0.5, 0.992, " ", fontsize=7, alpha=0, transform=fig.transFigure)
    save_fig_medical(save_base)
    # 横版图同时保存到 docs/figures/main
    docs_fig_main = os.path.join(get_project_root(), "docs", "figures", "main")
    ensure_dirs(docs_fig_main)
    save_fig_medical(os.path.join(docs_fig_main, "Fig1_study_overview"))
    plt.close()
    _log(f"Saved: {save_base}.pdf, .png + docs/figures/main/Fig1_study_overview.pdf, .png (横版)", "OK")

    # 竖版：A 上 B 下，画布宽 170mm
    fig_w_single = 170 / 25.4   # 170 mm → in
    fig_h_portrait = 7.2                   # 竖版总高
    fig2 = plt.figure(figsize=(fig_w_single, fig_h_portrait), facecolor='white')
    gs2 = GridSpec(2, 1, height_ratios=[1.15, 1], hspace=0.22, figure=fig2)
    ax_a2 = fig2.add_subplot(gs2[0])
    draw_flowchart_panel(ax_a2, box_w=3.5, title_fontsize=7.5, box_fontsize=7.0)
    ax_a2.text(-0.08, 1.02, "A", transform=ax_a2.transAxes,
               fontsize=9, fontweight='bold', ha='left', va='bottom')
    ax_b2 = fig2.add_subplot(gs2[1])
    # 竖版下方 Panel B：不再放“B. Missingness heatmap”文字，仅保留 B 标识
    plot_heatmap_panel(ax_b2, df, key_cols, formatter, n_cols=len(key_cols))
    fig2.subplots_adjust(top=0.96, bottom=0.06, left=0.12, right=0.96)
    # 竖版图 B：热图 + 色条位置与标题分离（标题左对齐 A，热图再右移）
    pos_b2 = ax_b2.get_position()
    cbar2 = ax_b2.collections[0].colorbar
    pos_c2 = cbar2.ax.get_position()
    ax_b2.set_position([pos_b2.x0, pos_b2.y0, pos_b2.width + (pos_c2.x0 - pos_b2.x1) * 0.5, pos_b2.height])
    pos_b2 = ax_b2.get_position()
    pos_c2 = cbar2.ax.get_position()
    pos_a2 = ax_a2.get_position()
    cbar2_right = 0.999
    gap_heatmap_cbar2 = 0.02
    shift_right = 0.08
    heatmap_shift = 0.05   # 仅竖版：热图与色条相对标题的右移量，标题不动
    ax_b2_right = (cbar2_right - pos_c2.width) - gap_heatmap_cbar2
    heatmap_w = ax_b2_right - pos_b2.x0
    panel_left = pos_a2.x0 + shift_right
    ax_a2.set_position([panel_left, pos_a2.y0, pos_a2.width, pos_a2.height])
    ax_b2.set_position([panel_left, pos_b2.y0, heatmap_w, pos_b2.height])
    cbar2.ax.set_position([pos_c2.x0 + shift_right, pos_c2.y0, pos_c2.width, pos_c2.height])
    # 图 B 标题条删除，仅调整热图与色条位置
    title_strip_height = 0.04
    ax_b2.set_position([panel_left + heatmap_shift, pos_b2.y0, heatmap_w - heatmap_shift, pos_b2.height])
    ax_b2.set_zorder(1)
    ax_b2.text(-0.08, 1.02, "B", transform=ax_b2.transAxes,
               fontsize=9, fontweight='bold', ha='left', va='bottom')
    pos_c2 = cbar2.ax.get_position()
    cbar2.ax.set_position([pos_c2.x0 + heatmap_shift, pos_c2.y0, pos_c2.width, pos_c2.height])
    fig2.text(0.5, 0.992, " ", fontsize=7, alpha=0, transform=fig2.transFigure)
    save_base_portrait = os.path.join(FIGURE_DIR, "Fig1_study_overview_portrait")
    save_fig_medical(save_base_portrait)
    save_fig_medical(os.path.join(docs_fig_main, "Fig1_study_overview_portrait"))
    plt.close(fig2)
    _log(f"Saved: {save_base_portrait}.pdf, .png + docs/figures/main/Fig1_study_overview_portrait.pdf, .png (竖版)", "OK")


if __name__ == "__main__":
    main()
