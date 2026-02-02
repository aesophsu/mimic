"""医学期刊投稿级绘图配置 (DPI 300–600, Arial, 色盲友好)
参考: PLOS Medicine, BMC, Nature - 双栏 176mm, 高度≤8.75in, 字体 8–12pt"""
import os
import matplotlib.pyplot as plt

FIG_DPI = 300
SAVE_DPI = 600
FIG_WIDTH_SINGLE = 3.5   # 单栏 ~85mm
FIG_WIDTH_DOUBLE = 7.2   # 双栏 ~176mm (PLOS max 7.5in)
FIG_WIDTH_LARGE = 10
FIG_HEIGHT_MAX = 8.75   # PLOS/Nature 最大高度 (in)

# DCA/校准组合图：双栏宽 × 适度高，宽高比 ~1.5:1 符合医学期刊审美
FIG_DCA_FIGSIZE = (7.2, 4.8)

FONT_FAMILY = 'sans-serif'
FONT_SANS = ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans']
LABEL_FONT = 11
TICK_FONT = 10
TITLE_FONT = 12
LEGEND_FONT = 9

PALETTE_MAIN = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
COLOR_POSITIVE = '#E64B35'
COLOR_NEGATIVE = '#4DBBD5'
COLOR_REF_LINE = '#95A5A6'
COLOR_GRID = '#ECF0F1'
OR_POINT_COLOR = '#2C3E50'
OR_CI_COLOR = '#7F8C8D'
OR_REF_LINE_COLOR = '#BDC3C7'
LOG_OR_TICKS = [0.1, 0.25, 0.5, 1, 2, 4, 10]

LINE_WIDTH = 2.0
LINE_WIDTH_THIN = 1.2
MARKER_SIZE = 5
CAPSIZE = 3

FOREST_FIGSIZE = (7.2, 6)


class PlotConfig:
    FIG_DPI = FIG_DPI
    SAVE_DPI = SAVE_DPI
    FOREST_FIGSIZE = FOREST_FIGSIZE
    LABEL_FONT = LABEL_FONT
    TICK_FONT = TICK_FONT
    TITLE_FONT = TITLE_FONT
    OR_POINT_COLOR = OR_POINT_COLOR
    OR_CI_COLOR = OR_CI_COLOR
    OR_REF_LINE_COLOR = OR_REF_LINE_COLOR
    LOG_OR_TICKS = LOG_OR_TICKS


def apply_medical_style():
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.sans-serif': FONT_SANS,
        'font.size': TICK_FONT,
        'axes.titlesize': TITLE_FONT,
        'axes.labelsize': LABEL_FONT,
        'xtick.labelsize': TICK_FONT,
        'ytick.labelsize': TICK_FONT,
        'legend.fontsize': LEGEND_FONT,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.alpha': 0.4,
        'grid.color': COLOR_GRID,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'figure.autolayout': False,
        'savefig.bbox': 'tight',
        'savefig.dpi': SAVE_DPI,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
    })


def style_axis_clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_fig_medical(base_path, formats=('pdf', 'png'), dpi=SAVE_DPI, **kwargs):
    """统一医学期刊级图片保存：PDF+PNG, 白底, 2pt 边距；kwargs 可覆盖 pad_inches 等"""
    opts = dict(bbox_inches='tight', facecolor='white', pad_inches=0.02)
    opts.update(kwargs)
    for fmt in formats:
        path = f"{base_path}.{fmt}"
        if fmt == 'png':
            plt.savefig(path, dpi=dpi, **opts)
        else:
            plt.savefig(path, **opts)
    return os.path.abspath(f"{base_path}.png")
