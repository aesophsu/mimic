import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
import matplotlib.ticker as ticker
_sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from utils.feature_formatter import FeatureFormatter
from utils.study_config import OUTCOMES
from utils.paths import get_project_root, get_model_dir, get_main_figure_dir, get_main_table_dir, get_supplementary_table_dir, get_cleaned_path, get_supplementary_figure_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.plot_utils import PlotUtils
from utils.plot_config import PlotConfig, apply_medical_style, save_fig_medical

BASE_DIR = get_project_root()
MODEL_ROOT = get_model_dir()
FIGURE_DIR = get_main_figure_dir()  # Fig5 Nomogram, Forest
TABLE_DIR = get_main_table_dir()
SUPP_TABLE_DIR = get_supplementary_table_dir()  # OR_Statistics
SUPP_FIG_NOMOGRAM_CALIB = get_supplementary_figure_dir("S5_nomogram_calibration")
ensure_dirs(FIGURE_DIR, TABLE_DIR, SUPP_TABLE_DIR)

TARGETS = OUTCOMES

# 设置绘图风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'], # 添加备选字体
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix'  # 使用类 Times New Roman 的数学字体，支持极好
})

def get_or_with_bootstrap(model, X_test, y_test, features, n_iterations=500):
    """
    真实的 Bootstrap 估算 OR 的 95% CI
    通过对原始测试集进行有放回抽样，计算系数的分布
    """
    _log(f"Starting Real Bootstrap for 95% CI (n={n_iterations})...", "INFO")
    
    # 1. 提取底层逻辑回归模型
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model
    
    raw_coefs = base_model.coef_[0]
    raw_ors = np.exp(raw_coefs)
    
    # 2. 准备数据容器
    boot_coefs = []
    X_arr = X_test.values if hasattr(X_test, 'values') else X_test
    y_arr = np.array(y_test)
    
    # 3. 迭代重采样
    for i in range(n_iterations):
        try:
            X_res, y_res = resample(
                X_arr, y_arr,
                random_state=i,
                stratify=y_arr
            )
        except ValueError:
            continue
        
        # 使用与训练一致的参数拟合
        params = base_model.get_params()
        params.update({'max_iter': 2000})

        boot_lr = LogisticRegression(**params)
        
        try:
            boot_lr.fit(X_res, y_res)
            boot_coefs.append(boot_lr.coef_[0])
        except Exception:
            continue
            
        if (i + 1) % 100 == 0:
            _log(f"Progress: {i + 1}/{n_iterations} iterations completed", "INFO")

    # 4. 计算 2.5% 和 97.5% 分位数
    boot_coefs = np.array(boot_coefs)
    lower_coefs = np.percentile(boot_coefs, 2.5, axis=0)
    upper_coefs = np.percentile(boot_coefs, 97.5, axis=0)

    if boot_coefs.shape[0] < n_iterations * 0.8:
        _log("Warning: Too many bootstrap failures, CI may be unstable.", "WARN")

    return pd.DataFrame({
        'Feature': features,
        'OR': raw_ors,
        'OR_Lower': np.exp(lower_coefs),
        'OR_Upper': np.exp(upper_coefs),
        'Coef': raw_coefs
    })


def plot_lr_calibration_internal(model, X_test, y_test, target, lang="en"):
    """
    针对构建 Nomogram 的逻辑回归模型，绘制内部验证校准曲线。
    使用与 Step 06 一致的风格，仅包含 Logistic Regression 一条曲线。
    """
    apply_medical_style()
    plt.figure(figsize=(7.2, 6), dpi=300, facecolor="white")
    ax = plt.gca()

    # 支持 CalibratedClassifierCV 或裸 LogisticRegression
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    plt.plot(
        prob_pred,
        prob_true,
        marker="s",
        ms=6,
        lw=2,
        color="#2c3e50",
        label="Logistic Regression",
        alpha=0.95,
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="#7f8c8d",
        linestyle=":",
        lw=1.5,
        label="Perfectly Calibrated",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Predicted Probability", fontsize=PlotConfig.LABEL_FONT, labelpad=10)
    plt.ylabel("Observed Probability", fontsize=PlotConfig.LABEL_FONT, labelpad=10)

    title = (
        f"Calibration of Nomogram (Internal, {target.upper()})"
        if lang == "en"
        else f"{target.upper()} 列线图内部校准曲线"
    )
    plt.title(title, fontsize=PlotConfig.TITLE_FONT, fontweight="bold", pad=18)
    plt.legend(loc="upper left", fontsize=10, frameon=False)
    plt.grid(color="whitesmoke", linestyle="-", linewidth=1)
    plt.tight_layout()

    save_base = os.path.join(
        SUPP_FIG_NOMOGRAM_CALIB,
        f"Fig5_nomogram_LR_calibration_{target}_internal",
    )
    ensure_dirs(SUPP_FIG_NOMOGRAM_CALIB)
    save_fig_medical(save_base)
    plt.close()
    _log(f"Nomogram Calibration (internal, {target}): {os.path.abspath(save_base)}.png", "OK")

def plot_forest_or(or_df, target, formatter, lang='en', show_or_text=True):
    """
    Publication-grade forest plot with OR (95% CI) text annotation
    """
    # ===============================
    # 1. 排序 & 预处理
    # ===============================
    or_df = or_df.sort_values('OR', ascending=True).copy()

    plot_utils = PlotUtils(formatter, lang)
    labels = plot_utils.format_feature_labels(
        or_df['Feature'], with_unit=True
    )
    left_err, right_err = plot_utils.compute_or_error(or_df)
    x_min, x_max = plot_utils.compute_or_xlim(or_df)

    # ===============================
    # 2. 画布与基础 Forest Plot
    # ===============================
    apply_medical_style()
    plt.figure(figsize=PlotConfig.FOREST_FIGSIZE, dpi=PlotConfig.FIG_DPI)
    y_pos = np.arange(len(or_df))

    plt.errorbar(
        or_df['OR'],
        y_pos,
        xerr=[left_err, right_err],
        fmt='s',
        markersize=6,
        color=PlotConfig.OR_POINT_COLOR,
        ecolor=PlotConfig.OR_CI_COLOR,
        elinewidth=2,
        capsize=4
    )

    # Reference line (OR = 1)
    plt.axvline(
        1,
        color=PlotConfig.OR_REF_LINE_COLOR,
        linestyle='--',
        lw=1.2
    )

    # ===============================
    # 3. Y / X 轴设置
    # ===============================
    plt.yticks(y_pos, labels, fontsize=PlotConfig.TICK_FONT)
    plt.xscale('log')
    plt.xlabel('Odds Ratio (log scale)', fontsize=PlotConfig.LABEL_FONT)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(PlotConfig.LOG_OR_TICKS)
    plt.xlim(x_min, x_max)

    # ===============================
    # 4. OR (95% CI) 文本显示（核心新增）
    # ===============================
    if show_or_text:
        # 文本列放在最右侧
        log_max = np.log10(or_df['OR_Upper'].max())
        text_x = 10 ** (log_max + 0.25)  # 向右移动 0.25 log units

        for y, (_, row) in zip(y_pos, or_df.iterrows()):
            or_text = PlotUtils.format_or_ci(
                row['OR'], row['OR_Lower'], row['OR_Upper']
            )
            plt.text(
                text_x, y,
                or_text,
                va='center',
                fontsize=PlotConfig.TICK_FONT,
                color='#222222'
            )
        plt.xlim(x_min, 10 ** (log_max + 0.45))

    # ===============================
    # 5. 标题与美化
    # ===============================
    title = f'Risk Factors for {target.upper()}'
    if lang != 'en':
        title += '（中文版）'

    plt.title(
        title,
        fontsize=PlotConfig.TITLE_FONT,
        fontweight='bold',
        pad=18
    )

    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    if show_or_text:
        plt.text(
            text_x, y_pos[-1] + 1,
            'OR (95% CI)',
            ha='left',
            va='bottom',
            fontsize=PlotConfig.TICK_FONT,
            fontweight='bold'
        )

    # ===============================
    # 6. 保存
    # ===============================
    save_fn = os.path.join(
        FIGURE_DIR,
        f"Fig5_forest_{target}_{lang}"
    )

    save_fig_medical(save_fn)
    plt.close()
    _log(f"Forest Plot: {os.path.abspath(save_fn)}.png", "OK")

def plot_nomogram_standard(
    or_df,
    target,
    formatter,
    feature_ranges=None,
    intercept=0.0,
    max_coef=None,
    lang='en',
    top_k=10
):
    """
    Publication-grade clinical nomogram for logistic regression models

    Parameters
    ----------
    or_df : DataFrame
        Must contain: Feature, Coef
    feature_ranges : dict
        {feature: (min, max)} for physical scale ticks
    intercept : float
        Logistic regression intercept
    max_coef : float
        Maximum absolute coefficient for scaling points (if None, inferred)
    """

    # ===============================
    # 1. Feature selection & scaling
    # ===============================
    df = or_df.copy()
    df['abs_coef'] = df['Coef'].abs()

    top_df = (
        df.sort_values('abs_coef', ascending=False)
          .head(top_k)
          .copy()
    )

    if max_coef is None:
        max_coef = top_df['abs_coef'].max()

    # Linear point assignment (standard nomogram convention)
    top_df['Points_max'] = (top_df['abs_coef'] / max_coef) * 100
    top_df = top_df.sort_values('Points_max', ascending=True)

    n_feat = len(top_df)

    # ===============================
    # 2. Canvas
    # ===============================
    apply_medical_style()
    fig, ax = plt.subplots(figsize=(7.2, 8), dpi=300, facecolor='white')

    # ===============================
    # 3. Global Points ruler (0–100)
    # ===============================
    main_y = n_feat + 1.2
    ax.hlines(main_y, 0, 100, lw=2, color='black')

    for p in range(0, 101, 10):
        ax.vlines(p, main_y, main_y + 0.25, lw=1.4, color='black')
        if p % 20 == 0:
            ax.text(
                p, main_y + 0.55, f"{p}",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
            ax.vlines(
                p, -5, main_y,
                lw=0.6, ls='--',
                color='gray', alpha=0.3, zorder=0
            )

    ax.text(
        -12, main_y + 0.2,
        'Points' if lang == 'en' else '评分',
        ha='right', va='center',
        fontsize=15, fontweight='bold'
    )

    # ===============================
    # 4. Feature-specific axes
    # ===============================
    for i, (_, row) in enumerate(top_df.iterrows()):
        feat = row['Feature']
        coef = row['Coef']
        max_pt = row['Points_max']

        ax.hlines(i, 0, max_pt, lw=1.6, color='black')

        # ---- Physical value ticks ----
        if feature_ranges and feat in feature_ranges:
            v_min, v_max = feature_ranges[feat]
            v_ticks = np.linspace(v_min, v_max, 5)

            p_ticks = (
                np.linspace(max_pt, 0, 5)
                if coef < 0 else
                np.linspace(0, max_pt, 5)
            )

            for v, p in zip(v_ticks, p_ticks):
                ax.vlines(p, i, i + 0.18, lw=1, color='black')
                ax.text(
                    p, i - 0.35, f"{v:.1f}",
                    ha='center', va='top',
                    fontsize=9, color='#333333'
                )

        # ---- Feature label (formatter) ----
        feat_label = formatter.get_label(
            feat, lang=lang, with_unit=True
        )
        ax.text(
            -12, i,
            feat_label,
            ha='right', va='center',
            fontsize=12, fontweight='bold'
        )

    # ===============================
    # 5. Total Points axis
    # ===============================
    tp_y = -2.3
    ax.hlines(tp_y, 0, 100, lw=2.6, color='darkred')

    ax.text(
        -12, tp_y,
        'Total Points' if lang == 'en' else '总评分',
        ha='right', va='center',
        fontsize=14, fontweight='bold',
        color='darkred'
    )

    for p in np.linspace(0, 100, 11):
        ax.vlines(p, tp_y, tp_y - 0.28, lw=2, color='darkred')
        ax.text(
            p, tp_y - 0.9,
            f"{int(p)}",
            ha='center',
            fontsize=11,
            fontweight='bold',
            color='darkred'
        )

    # ===============================
    # 6. Risk probability axis
    # ===============================
    prob_y = -4.8
    ax.hlines(prob_y, 0, 100, lw=2.6, color='darkblue')

    risk_label = (
        f'Risk of {target.upper()}'
        if lang == 'en'
        else f'{target.upper()} 发生风险'
    )

    ax.text(
        -12, prob_y,
        risk_label,
        ha='right', va='center',
        fontsize=14, fontweight='bold',
        color='darkblue'
    )

    probs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for j, p in enumerate(probs):
        logit_p = np.log(p / (1 - p))
        total_pt = (logit_p - intercept) / max_coef * 100

        if 0 <= total_pt <= 100:
            ax.vlines(
                total_pt,
                prob_y,
                prob_y + 0.28,
                lw=2,
                color='darkblue'
            )
            ax.text(
                total_pt,
                prob_y - (0.9 if j % 2 == 0 else 1.4),
                f"{p:.2f}",
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='darkblue'
            )

    # ===============================
    # 7. Global styling
    # ===============================
    title = (
        f"Clinical Nomogram for Predicting {target.upper()}"
        if lang == 'en'
        else f"{target.upper()} 预测列线图"
    )

    ax.set_title(
        title,
        fontsize=20,
        fontweight='bold',
        pad=60
    )

    ax.set_xlim(-38, 112)
    ax.set_ylim(prob_y - 2.8, main_y + 2.8)
    ax.axis('off')

    # ===============================
    # 8. Save
    # ===============================
    plt.tight_layout()
    save_path = os.path.join(
        FIGURE_DIR,
        f"Fig5_nomogram_{target}_{lang}"
    )

    save_fig_medical(save_path)
    plt.close()
    _log(f"Nomogram ({lang}): {os.path.abspath(save_path)}.png", "OK")    

def load_raw_reference(file_path):
    """加载用于 Nomogram 刻度的原始物理数据"""
    if os.path.exists(file_path):
        raw_df = pd.read_csv(file_path)
        _log("Raw physical data loaded for Nomogram scales.", "INFO")
        return raw_df
    else:
        _log(f"Warning: {file_path} not found. Scales will be empty.", "WARN")
        return None

def calculate_feature_physical_ranges(or_results, raw_df):
    """计算 Top 10 特征的 1% - 99% 分位数物理范围"""
    feat_ranges = {}
    if raw_df is not None:
        # 提取系数绝对值最大的前 10 个特征
        top_feats = or_results.assign(abs_c=lambda x: x['Coef'].abs())\
                              .sort_values('abs_c', ascending=False)\
                              .head(10)['Feature']
        for f in top_feats:
            if f in raw_df.columns:
                # 使用分位数避免离群值导致坐标轴过长
                feat_ranges[f] = np.percentile(raw_df[f].dropna(), [1, 99])
    return feat_ranges


def verify_feature_units(or_results, feat_ranges, formatter):
    """
    校验列线图特征的单位与物理刻度是否一致。
    单位来自 feature_dictionary.json，物理刻度来自 mimic_raw_scale（02 清洗后）。
    """
    top_feats = or_results.assign(abs_c=lambda x: x['Coef'].abs())\
                          .sort_values('abs_c', ascending=False)\
                          .head(10)['Feature']
    issues = []
    for f in top_feats:
        label_with_unit = formatter.get_label(f, lang='en', with_unit=True)
        has_unit = '(' in label_with_unit and ')' in label_with_unit
        if f in feat_ranges:
            v1, v2 = feat_ranges[f]
            # 合理性检查：常见单位与数值范围
            if 'mg/dL' in label_with_unit and (v2 > 5000 or v1 < -1):
                issues.append(f"{f}: 数值 [{v1:.2f}, {v2:.2f}] 可能与 mg/dL 不符")
            elif ('×10' in label_with_unit or '10^' in label_with_unit) and 'L' in label_with_unit:
                if 'platelets' in f.lower() and (v2 > 5000 or v1 < 0):
                    issues.append(f"{f}: 血小板 [{v1:.1f}, {v2:.1f}] 请确认单位 ×10^9/L")
                elif 'wbc' in f.lower() and (v2 > 500 or v1 < 0):
                    issues.append(f"{f}: WBC [{v1:.2f}, {v2:.2f}] 请确认单位 ×10^9/L")
        elif has_unit:
            issues.append(f"{f}: 有单位但无物理刻度（raw_df 中可能缺失该列）")
    if issues:
        _log("Unit Check 潜在单位不一致:", "WARN")
        for msg in issues:
            _log(f"    {msg}", "WARN")
    return len(issues) == 0

def process_single_target(target, raw_df):
    """处理单个结局指标的完整流水线"""
    _log(f"Processing Outcome: {target.upper()}", "INFO")
    try:
        # 1. 资产加载
        target_dir = os.path.join(MODEL_ROOT, target.lower())
        models = joblib.load(os.path.join(target_dir, "all_models_dict.pkl"))
        eval_data = joblib.load(os.path.join(target_dir, "eval_data.pkl"))
        formatter = FeatureFormatter()
        model = models.get("Logistic Regression")
        if model is None:
            _log("Skipped: Logistic Regression not found.", "WARN")
            return

        X_test = eval_data['X_test_pre']
        y_test = eval_data['y_test']
        features = eval_data['features']
        
        # 2. 统计计算 (Bootstrap OR)
        or_results = get_or_with_bootstrap(model, X_test, y_test, features, n_iterations=500)
        
        # 保存 CSV 结果（增加 display_name 列统一特征展示名）
        formatter = FeatureFormatter()
        or_results['display_name'] = or_results['Feature'].map(lambda x: formatter.get_label(x))
        table_fn = os.path.join(SUPP_TABLE_DIR, f"OR_Statistics_{target}.csv")
        json_fn = os.path.join(SUPP_TABLE_DIR, f"OR_Json_{target}.json")
        or_results.to_csv(table_fn, index=False)
        or_results.to_json(json_fn, orient='records', indent=4)
        
        # 3. 计算物理刻度范围
        feat_ranges = calculate_feature_physical_ranges(or_results, raw_df)

        # 3.1 校验特征单位与物理刻度一致性
        verify_feature_units(or_results, feat_ranges, formatter)

        # 4. 绘图 (plot_nomogram_standard 内部已包含 PDF 保存逻辑)
        # 4.1 提取精确绘图所需的模型参数 (Intercept 和 Max Coef)
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator
        else:
            base_model = model
            
        model_intercept = base_model.intercept_[0]
        # 获取 OR 结果中绝对值最大的系数，用于标准化点数基准
        max_abs_coef = or_results['Coef'].abs().max()

        # 5. 绘制森林图与列线图
        plot_forest_or(or_results, target, formatter=formatter)
        plot_nomogram_standard(
            or_results, 
            target,
            formatter=formatter,
            feature_ranges=feat_ranges,
            intercept=model_intercept,
            max_coef=max_abs_coef
        )

        # 6. 逻辑回归 Nomogram 的内部校准曲线
        plot_lr_calibration_internal(model, X_test, y_test, target, lang='en')
        
        _log(f"Statistics and Figures generated for {target}.", "OK")
        
    except Exception as e:
        _log(f"Critical Error Failed for {target}: {str(e)}", "ERR")
        # import traceback; traceback.print_exc() # 调试时可开启

def main():
    log_header("🚀 13_nomogram_odds_ratio: 临床转化 - Bootstrap OR 与列线图")
    _log(f"模型目录: {os.path.abspath(MODEL_ROOT)}", "INFO")
    _log(f"输出: {os.path.abspath(FIGURE_DIR)}", "INFO")
    
    # 初始化：加载物理数据基准
    RAW_DATA_PATH = get_cleaned_path("mimic_raw_scale.csv")
    raw_df = load_raw_reference(RAW_DATA_PATH)
    
    # 循环处理每个目标
    for target in TARGETS:
        process_single_target(target, raw_df)

    _log("13 步完成！", "OK")
    _log(f"图表: {os.path.abspath(FIGURE_DIR)}", "INFO")
    _log(f"表格: {os.path.abspath(TABLE_DIR)}", "INFO")

if __name__ == "__main__":
    main()
