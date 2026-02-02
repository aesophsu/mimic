"""
生成 Table 1 基线特征表（论文规范版）

改进：
1. 结构分组：Demographics, Severity Scores, Organ Support, Laboratory, Outcomes
2. SMD 列：标准化均数差，SMD > 0.1 表示存在显著差异
3. 单位统一：实验室指标附单位
4. 分类变量合并：仅显示 Male n(%) 等阳性类别
5. 展示名规范：Mechanical ventilation, Creatinine 等
6. eICU 外部验证列：展示人群漂移

前置：01_mimic_cleaning（mimic_raw_scale.csv）；完整 Table 1 需 08_eicu_alignment_cleaning（eicu_raw_scale.csv）。
run_all 全流程时 03 在 08 之后运行；mimic-only 时 03 在 02 之后运行。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.feature_formatter import FeatureFormatter
from utils.study_config import OUTCOMES
from utils.paths import get_cleaned_path, get_external_path, get_artifact_path, get_main_table_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.outcome_utils import normalize_gender
from utils.table1_config import (
    TABLE1_GROUPS,
    TABLE1_DISPLAY_OVERRIDES,
    TABLE1_UNITS,
    BINARY_SHOW_POSITIVE_ONLY,
    TABLE1_FOOTNOTES,
    TABLE1_USE_INDENT,
)

MIMIC_PATH = get_cleaned_path("mimic_raw_scale.csv")
EICU_PATH = get_external_path("eicu_raw_scale.csv")
DICT_PATH = get_artifact_path("features", "feature_dictionary.json")
TABLE_DIR = get_main_table_dir()
GROUPBY_COL = "pof"  # Non-POF (0) vs POF (1)


def _get_display_label(col, with_unit=True):
    """获取 Table 1 展示名，优先使用覆盖配置"""
    override = TABLE1_DISPLAY_OVERRIDES.get(col)
    if override:
        label = override
    else:
        formatter = FeatureFormatter()
        label = formatter.get_label(col, with_unit=False)
    if with_unit and col in TABLE1_UNITS and TABLE1_UNITS[col]:
        unit = TABLE1_UNITS[col].replace("×", "x").replace("⁹", "9")  # CSV 兼容
        label = f"{label} ({unit})"
    return label


def _smd_continuous(x0, x1):
    """连续变量 SMD: (mean1 - mean0) / pooled_sd"""
    m0, m1 = x0.mean(), x1.mean()
    s0, s1 = x0.std(), x1.std()
    n0, n1 = len(x0.dropna()), len(x1.dropna())
    if n0 < 2 or n1 < 2:
        return np.nan
    pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    if pooled == 0:
        return 0.0
    return abs((m1 - m0) / pooled)


def _smd_binary(p0, p1, n0, n1):
    """二分类 SMD: Cohen's h 或 (p1-p0)/sqrt(p*(1-p))"""
    if n0 == 0 or n1 == 0:
        return np.nan
    p_pool = (n0 * p0 + n1 * p1) / (n0 + n1)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0
    denom = np.sqrt(p_pool * (1 - p_pool))
    if denom == 0:
        return 0.0
    return abs((p1 - p0) / denom)


def _format_smd(smd):
    """SMD 展示：>0.1 显示数值，否则 <0.1"""
    if np.isnan(smd):
        return "—"
    if smd < 0.1:
        return "<0.1"
    return f"{smd:.2f}"


def _median_iqr(series):
    """median [Q1, Q3]"""
    s = series.dropna()
    if s.empty:
        return "—"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def _n_pct(n, total):
    return f"{n} ({100*n/total:.1f}%)" if total > 0 else "—"


def _pvalue_continuous(x0, x1):
    try:
        _, p = stats.kruskal(x0.dropna(), x1.dropna())
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return "—"


def _pvalue_categorical(tab):
    try:
        _, p, _, _ = stats.chi2_contingency(tab)
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return "—"


def build_mimic_table1(df_mimic):
    """构建 MIMIC 分组基线表：Overall, Non-POF, POF, P-value, SMD (POF vs Non-POF)"""
    g0 = df_mimic[df_mimic[GROUPBY_COL] == 0]
    g1 = df_mimic[df_mimic[GROUPBY_COL] == 1]
    n0, n1 = len(g0), len(g1)
    n_total = n0 + n1

    rows = [{
        "Characteristic": "n",
        "Overall": str(n_total),
        "Non-POF": str(n0),
        "POF": str(n1),
        "P-value": "",
        "SMD (POF vs Non-POF)": "",
    }]
    for group_name, cols in TABLE1_GROUPS.items():
        rows.append({
            "Characteristic": group_name,
            "Overall": "",
            "Non-POF": "",
            "POF": "",
            "P-value": "",
            "SMD (POF vs Non-POF)": "",
        })
        for col in cols:
            if col not in df_mimic.columns:
                continue
            label = _get_display_label(col)
            # POF 为分组变量：Overall 显示发生率，Non-POF=0%, POF=100%
            if col == GROUPBY_COL:
                v_total = (df_mimic[col] == 1).sum()
                overall = _n_pct(v_total, n_total)
                non_pof = "0 (0.0%)"
                pof = _n_pct(n1, n1)  # 100%
                pval = "—"
                smd = np.nan
            elif col in BINARY_SHOW_POSITIVE_ONLY:
                v0 = (g0[col] == 1).sum()
                v1 = (g1[col] == 1).sum()
                v_total = (df_mimic[col] == 1).sum()
                overall = _n_pct(v_total, n_total)
                non_pof = _n_pct(v0, n0)
                pof = _n_pct(v1, n1)
                tab = np.array([[n0 - v0, v0], [n1 - v1, v1]])
                pval = _pvalue_categorical(tab)
                smd = _smd_binary(v0 / n0, v1 / n1, n0, n1)
            else:
                overall = _median_iqr(df_mimic[col])
                non_pof = _median_iqr(g0[col])
                pof = _median_iqr(g1[col])
                pval = _pvalue_continuous(g0[col], g1[col])
                smd = _smd_continuous(g0[col], g1[col])
            char_display = f"  {label}" if TABLE1_USE_INDENT else label
            rows.append({
                "Characteristic": char_display,
                "Overall": overall,
                "Non-POF": non_pof,
                "POF": pof,
                "P-value": pval,
                "SMD (POF vs Non-POF)": _format_smd(smd),
            })
    return pd.DataFrame(rows)


def _char_to_col(char, label_to_col):
    """Characteristic 可能带缩进，需 strip 后匹配"""
    return label_to_col.get(char) or label_to_col.get(char.strip())


# eICU 首日干预列映射（与 MIMIC intime 至 intime+24h 对齐，用于 Table 1 基线）
EICU_DAY1_COL_MAP = {
    "mechanical_vent_flag": "mechanical_vent_flag_day1",
    "vaso_flag": "vaso_flag_day1",
}


def _build_eicu_column(df_table1, df_eicu):
    """按 Table 1 行顺序构建 eICU 列；Organ Support 使用首日干预（与 MIMIC 对齐）"""
    eicu_n = len(df_eicu)
    label_to_col = {}
    for _, cols in TABLE1_GROUPS.items():
        for col in cols:
            label_to_col[_get_display_label(col)] = col

    eicu_vals = []
    for _, row in df_table1.iterrows():
        char = row["Characteristic"]
        if char == "n":
            eicu_vals.append(str(eicu_n))
        elif char in TABLE1_GROUPS:
            eicu_vals.append("")
        else:
            col_found = _char_to_col(char, label_to_col)
            if col_found is None:
                eicu_vals.append("—")
                continue
            # eICU 基线干预使用首日列（与 MIMIC intime+24h 对齐）
            eicu_col = EICU_DAY1_COL_MAP.get(col_found, col_found)
            if eicu_col not in df_eicu.columns:
                eicu_col = col_found
            if eicu_col not in df_eicu.columns:
                eicu_vals.append("—")
            elif col_found in BINARY_SHOW_POSITIVE_ONLY:
                v = (df_eicu[eicu_col] == 1).sum()
                eicu_vals.append(_n_pct(v, eicu_n))
            else:
                eicu_vals.append(_median_iqr(df_eicu[eicu_col]))
    return eicu_vals


def _build_smd_mimic_vs_eicu(df_table1, df_mimic, df_eicu):
    """构建 SMD (MIMIC vs eICU) 列：量化人群漂移；Organ Support 使用 eICU 首日列"""
    label_to_col = {}
    for _, cols in TABLE1_GROUPS.items():
        for col in cols:
            label_to_col[_get_display_label(col)] = col

    n_mimic, n_eicu = len(df_mimic), len(df_eicu)
    smd_vals = []
    for _, row in df_table1.iterrows():
        char = row["Characteristic"]
        if char in ["n", ""] or char in TABLE1_GROUPS:
            smd_vals.append("")
            continue
        col = _char_to_col(char, label_to_col)
        if col is None:
            smd_vals.append("—")
            continue
        eicu_col = EICU_DAY1_COL_MAP.get(col, col)
        if eicu_col not in df_eicu.columns:
            eicu_col = col
        if eicu_col not in df_eicu.columns:
            smd_vals.append("—")
            continue
        if col in BINARY_SHOW_POSITIVE_ONLY:
            p_mimic = (df_mimic[col] == 1).mean()
            p_eicu = (df_eicu[eicu_col] == 1).mean()
            smd = _smd_binary(p_mimic, p_eicu, n_mimic, n_eicu)
        else:
            x_mimic = df_mimic[col].dropna()
            x_eicu = df_eicu[eicu_col].dropna()
            if len(x_mimic) < 2 or len(x_eicu) < 2:
                smd_vals.append("—")
                continue
            smd = _smd_continuous(x_mimic, x_eicu)
        smd_vals.append(_format_smd(smd))
    return smd_vals


def _add_footnotes(df_table1):
    """在表格底部添加脚注行"""
    n_cols = len(df_table1.columns)
    footnote_rows = []
    for i, note in enumerate(TABLE1_FOOTNOTES):
        row = [""] * n_cols
        row[0] = f"Note {i+1}. {note}"
        footnote_rows.append(row)
    df_foot = pd.DataFrame(footnote_rows, columns=df_table1.columns)
    return pd.concat([df_table1, df_foot], ignore_index=True)


def build_table1_with_eicu(df_table1, df_mimic, df_eicu):
    """添加 eICU 列、SMD (MIMIC vs eICU) 列及脚注"""
    if df_eicu is None or len(df_eicu) == 0:
        return _add_footnotes(df_table1)
    df_table1 = df_table1.copy()
    eicu_vals = _build_eicu_column(df_table1, df_eicu)
    df_table1.insert(4, "eICU (External Validation)", eicu_vals)
    smd_drift = _build_smd_mimic_vs_eicu(df_table1, df_mimic, df_eicu)
    df_table1["SMD (MIMIC vs eICU)"] = smd_drift
    return _add_footnotes(df_table1)


def main():
    log_header("🚀 03_table1_baseline: 论文规范版 Table 1")

    if not os.path.exists(MIMIC_PATH):
        _log(f"MIMIC 数据不存在: {MIMIC_PATH}", "ERR")
        return

    df_mimic = pd.read_csv(MIMIC_PATH)

    # 预处理：与 03 一致
    df_mimic = normalize_gender(df_mimic)
    if "creatinine_max" in df_mimic.columns and "chronic_kidney_disease" in df_mimic.columns:
        df_mimic["subgroup_no_renal"] = (
            (df_mimic["creatinine_max"] < 1.5) & (df_mimic["chronic_kidney_disease"] == 0)
        ).astype(int)

    if GROUPBY_COL not in df_mimic.columns:
        _log(f"分组列 {GROUPBY_COL} 不存在", "ERR")
        return

    df_table1 = build_mimic_table1(df_mimic)

    # eICU 列、SMD (MIMIC vs eICU)、脚注（若存在 eICU）
    df_eicu = None
    if os.path.exists(EICU_PATH):
        df_eicu = pd.read_csv(EICU_PATH)
        _log(f"已加载 eICU 数据 N={len(df_eicu)}，添加外部验证列、SMD (MIMIC vs eICU)、脚注", "OK")
        df_table1 = build_table1_with_eicu(df_table1, df_mimic, df_eicu)
    else:
        _log("eICU 数据不存在，跳过外部验证列（运行 08 后可重新生成）", "WARN")
        df_table1 = _add_footnotes(df_table1)

    ensure_dirs(TABLE_DIR)
    out_path = os.path.join(TABLE_DIR, "Table1_baseline.csv")
    df_table1.to_csv(out_path, index=False)
    _log(f"Table 1 已保存: {os.path.abspath(out_path)}", "OK")
    _log("下一步: 04_mimic_stat_audit.py 或 09_cross_cohort_audit.py（全流程时 03 在 08 后运行）", "INFO")


if __name__ == "__main__":
    main()
