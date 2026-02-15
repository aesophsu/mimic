import json
import os
import io
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
SCALERS_DIR = os.path.join(ARTIFACTS_DIR, "scalers")
FEATURE_DICT_PATH = os.path.join(ARTIFACTS_DIR, "features", "feature_dictionary.json")
TABLE4_PATH = os.path.join(PROJECT_ROOT, "results", "main", "tables", "Table4_external_validation.csv")
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "mimic_train_processed.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "mimic_test_processed.csv")
DYNAMIC_DATA_AVAILABLE = os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)

TARGETS = {
    "POF（主要终点）": "pof",
    "28天死亡（次要终点）": "mortality",
    "Composite（混合终点）": "composite",
}

FAST_TOPN_BY_TARGET = {
    "pof": 3,
    "composite": 4,
    "mortality": 4,
}

# 统一换算到模型训练使用的标准单位（base_unit）
UNIT_CONVERSIONS = {
    "creatinine_max": {"base_unit": "mg/dL", "factors": {"mg/dL": 1.0, "umol/L": 1.0 / 88.4}},
    "albumin_min": {"base_unit": "g/dL", "factors": {"g/dL": 1.0, "g/L": 0.1}},
    "albumin_max": {"base_unit": "g/dL", "factors": {"g/dL": 1.0, "g/L": 0.1}},
    "bun_min": {"base_unit": "mg/dL", "factors": {"mg/dL": 1.0, "mmol/L": 2.801}},
    "bun_max": {"base_unit": "mg/dL", "factors": {"mg/dL": 1.0, "mmol/L": 2.801}},
    "lactate_max": {"base_unit": "mmol/L", "factors": {"mmol/L": 1.0, "mg/dL": 1.0 / 9.0}},
    "phosphate_min": {"base_unit": "mg/dL", "factors": {"mg/dL": 1.0, "mmol/L": 3.097}},
    "hemoglobin_min": {"base_unit": "g/dL", "factors": {"g/dL": 1.0, "g/L": 0.1}},
    "wbc_max": {"base_unit": "×10⁹/L", "factors": {"×10⁹/L": 1.0, "×10³/µL": 1.0}},
    "wbc_min": {"base_unit": "×10⁹/L", "factors": {"×10⁹/L": 1.0, "×10³/µL": 1.0}},
    "ptt_min": {"base_unit": "s", "factors": {"s": 1.0, "min": 60.0}},
    "pao2fio2ratio_min": {"base_unit": "mmHg", "factors": {"mmHg": 1.0, "kPa": 7.50062}},
    "aniongap_min": {"base_unit": "mmol/L", "factors": {"mmol/L": 1.0, "mEq/L": 1.0}},
    "spo2_max": {"base_unit": "%", "factors": {"%": 1.0, "fraction(0-1)": 100.0}},
    "spo2_min": {"base_unit": "%", "factors": {"%": 1.0, "fraction(0-1)": 100.0}},
    "spo2_slope": {"base_unit": "% per hour", "factors": {"% per hour": 1.0, "fraction per hour": 100.0}},
    "admission_age": {"base_unit": "years", "factors": {"years": 1.0, "months": 1.0 / 12.0}},
}

GROUP_LABELS = {
    "oxygenation": {"en": "Oxygenation & Acid-Base", "zh": "氧合与酸碱"},
    "renal_metabolic": {"en": "Renal & Metabolic", "zh": "肾功能与代谢"},
    "hematology_coag": {"en": "Hematology & Coagulation", "zh": "血液与凝血"},
    "inflammation": {"en": "Inflammation", "zh": "炎症相关"},
    "demographic_comorb": {"en": "Demographics & Comorbidity", "zh": "人口学与合并症"},
    "other": {"en": "Other", "zh": "其他"},
}

I18N = {
    "en": {
        "page_title": "AP Risk Calculator",
        "title": "Acute Pancreatitis Risk Calculator",
        "caption": "Single-patient risk estimation powered by fixed training-time preprocessing and calibrated XGBoost models.",
        "lang_switch": "Language",
        "endpoint": "Endpoint",
        "target_pof": "POF (Primary endpoint)",
        "target_mortality": "28-day Mortality (Secondary endpoint)",
        "target_composite": "Composite endpoint",
        "model_info": "Endpoint: `{target}` | Features in model: `{n}` | Engine: `Calibrated XGBoost`",
        "input_hint": "You may leave blank and let the model impute",
        "input_hint_range": "{range}; you may leave blank and let the model impute",
        "bin_missing": "Missing (impute by model)",
        "submit": "Predict Risk",
        "predict_ok": "Prediction completed",
        "prob": "Predicted Probability",
        "risk_level": "Risk Category",
        "risk_low": "Low",
        "risk_mid": "Medium",
        "risk_high": "High",
        "input_review": "Input Review",
        "missing": "missing",
        "load_failed": "Failed to load model: {err}",
        "predict_failed": "Prediction failed: {err}",
        "parse_failed": "Invalid input:\n- {errors}",
        "range_prefix": "Reference range: {lo} ~ {hi}",
        "footer": "本工具仅用于学术演示，不直接用于临床决策支持（For research purpose only, not for clinical diagnostic use）",
        "hero_kicker": "Research Use Interface",
        "hero_desc": "Designed for transparent workflow: structured input, model-based risk estimate, and traceable completeness reporting.",
        "section_inputs": "Clinical Input Form",
        "section_result": "Risk Output",
        "risk_bar": "Risk intensity",
        "number_err": "enter a number or leave blank",
        "input_mode": "Input mode",
        "mode_smart": "Smart (recommended)",
        "mode_full": "Full (12 variables)",
        "smart_hint": "Show core variables first; other variables are optional and can be imputed if left blank.",
        "smart_source": "Smart subset source: {source} (k={k})",
        "optional_vars": "Optional variables",
        "unit": "Unit",
        "model_unit": "Model unit: {unit}",
        "converted_value": "Converted value: {value} {unit}",
        "normal_range": "Normal range: {lo} ~ {hi} {unit}",
        "composite_fast_note": "Composite fast mode uses 4 core variables for simplicity; if you prioritize maximal discrimination, consider adding optional variables.",
        "group_title": "Field Groups",
        "explain_title": "Risk Explanation",
        "explain_none": "No obvious out-of-range signal from the provided values.",
        "explain_high": "{name} is above reference ({value} vs {lo}-{hi}).",
        "explain_low": "{name} is below reference ({value} vs {lo}-{hi}).",
        "shap_title": "Individual SHAP Explanation",
        "shap_run": "Generate SHAP plots",
        "shap_note": "Model-based feature attribution for this single case.",
        "shap_mode_label": "SHAP mode",
        "shap_mode_full": "Full-model SHAP",
        "shap_mode_smart": "Smart-subset SHAP",
        "shap_mode_full_note": "Explain prediction with the full deployed model.",
        "shap_mode_smart_note": "Explain prediction with a subset model retrained on development train set using smart features.",
        "shap_mode_smart_unavailable": "Smart-subset SHAP is unavailable because development train/test files are not bundled.",
        "shap_mode_source": "Subset source: {source} (k={k})",
        "shap_waterfall": "Waterfall plot",
        "shap_force": "Force plot",
        "shap_top": "Top feature contributions",
        "shap_failed": "SHAP generation failed: {err}",
        "shap_stale": "Inputs changed. Click to regenerate SHAP plots.",
        "missing_rate_all": "Missing rate (all model features): {pct}% ({miss}/{total})",
        "missing_rate_core": "Missing rate (core features): {pct}% ({miss}/{total})",
        "missing_hint": "Higher missing rate means stronger dependence on imputation.",
        "export_btn": "Export Result (CSV)",
        "export_file": "ap_risk_result_{target}.csv",
        "group_header": "{name} ({n})",
        "bench_title": "Reference Performance (XGBoost)",
        "bench_internal": "Internal",
        "bench_external": "External (eICU)",
        "bench_auc": "AUC",
        "bench_sens": "Sensitivity",
        "bench_spec": "Specificity",
        "bench_na": "N/A",
        "completeness_weighted": "Model completeness (impact-weighted): {pct}%",
        "completeness_hint": "Calculated from filled features weighted by XGBoost feature importance.",
        "dynamic_title": "Performance Retention vs Full-Feature Baseline",
        "dynamic_note": "Internal estimate from development data. This reflects model-level retention, not single-patient prediction certainty.",
        "dynamic_need_more": "Fill at least 2 model features to compute dynamic ratio.",
        "dynamic_unavailable": "Dynamic retention is disabled in this deployment (development train/test files are not bundled).",
        "dynamic_subset": "Subset size",
        "dynamic_ratio_auc": "AUC ratio",
        "dynamic_ratio_acc": "Accuracy ratio",
        "dynamic_delta_auc": "AUC delta",
        "dynamic_delta_acc": "Accuracy delta",
        "dynamic_list_title": "Key numbers",
        "dynamic_auc_subset": "Subset AUC",
        "dynamic_auc_full": "Full-feature AUC",
        "dynamic_auc_ratio": "AUC retention ratio",
        "dynamic_ci": "possible range (95% CI): {lo} - {hi}",
        "dynamic_feats": "Current filled features",
        "value_source": "Value source",
        "provided": "provided",
        "imputed": "imputed",
        "default_zero": "default(0)",
        "value_model_used": "Value used by model",
        "summary_title": "Prediction Summary",
        "summary_note": "Use this output as a research reference only.",
        "details_tab_1": "Interpretation",
        "details_tab_2": "Completeness",
        "details_tab_3": "Retention",
        "details_tab_4": "Data & Export",
        "bench_toggle": "Show reference performance",
    },
    "zh": {
        "page_title": "AP 风险计算器",
        "title": "急性胰腺炎风险计算器",
        "caption": "基于固定训练期预处理参数与校准 XGBoost 模型的单病例风险估计。",
        "lang_switch": "语言",
        "endpoint": "选择预测终点",
        "target_pof": "POF（主要终点）",
        "target_mortality": "28天死亡（次要终点）",
        "target_composite": "Composite（混合终点）",
        "model_info": "当前终点: `{target}` | 模型特征数: `{n}` | 引擎: `校准 XGBoost`",
        "input_hint": "可留空由模型插补",
        "input_hint_range": "{range}；可留空由模型插补",
        "bin_missing": "留空（由模型插补）",
        "submit": "计算风险",
        "predict_ok": "预测完成",
        "prob": "预测概率",
        "risk_level": "风险分层",
        "risk_low": "低风险",
        "risk_mid": "中风险",
        "risk_high": "高风险",
        "input_review": "本次输入（用于审阅）",
        "missing": "missing",
        "load_failed": "加载模型失败: {err}",
        "predict_failed": "预测失败: {err}",
        "parse_failed": "输入格式错误：\n- {errors}",
        "range_prefix": "参考范围: {lo} ~ {hi}",
        "footer": "本工具仅用于学术演示，不直接用于临床决策支持（For research purpose only, not for clinical diagnostic use）",
        "hero_kicker": "科研用途界面",
        "hero_desc": "强调可追溯流程：结构化输入、模型风险输出、完整度可解释展示。",
        "section_inputs": "临床输入表单",
        "section_result": "风险输出",
        "risk_bar": "风险强度",
        "number_err": "请输入数字或留空",
        "input_mode": "输入模式",
        "mode_smart": "精简模式（推荐）",
        "mode_full": "完整模式（12变量）",
        "smart_hint": "优先填写核心变量，其他变量可留空由模型插补。",
        "smart_source": "精简特征来源: {source}（k={k}）",
        "optional_vars": "可选变量",
        "unit": "单位",
        "model_unit": "模型单位: {unit}",
        "converted_value": "换算后数值: {value} {unit}",
        "normal_range": "正常范围: {lo} ~ {hi} {unit}",
        "composite_fast_note": "Composite 精简模式默认4项核心变量；若更重视最高区分能力，建议补充可选变量。",
        "group_title": "字段分组",
        "explain_title": "结果解释",
        "explain_none": "当前已填值未出现明显超出参考范围的信号。",
        "explain_high": "{name} 高于参考范围（{value}，参考 {lo}-{hi}）。",
        "explain_low": "{name} 低于参考范围（{value}，参考 {lo}-{hi}）。",
        "shap_title": "个体 SHAP 解释",
        "shap_run": "生成 SHAP 图",
        "shap_note": "针对当前病例的模型特征归因解释。",
        "shap_mode_label": "SHAP 模式",
        "shap_mode_full": "全模型 SHAP",
        "shap_mode_smart": "精简子集 SHAP",
        "shap_mode_full_note": "使用完整部署模型解释本次预测。",
        "shap_mode_smart_note": "使用精简特征在开发集重训的子模型解释本次预测。",
        "shap_mode_smart_unavailable": "当前部署未包含开发集训练/测试文件，精简子集 SHAP 不可用。",
        "shap_mode_source": "子集来源: {source}（k={k}）",
        "shap_waterfall": "瀑布图",
        "shap_force": "Force 图",
        "shap_top": "主要贡献特征",
        "shap_failed": "SHAP 生成失败: {err}",
        "shap_stale": "输入已变更，请重新点击生成 SHAP 图。",
        "missing_rate_all": "缺失率（全部模型特征）: {pct}%（{miss}/{total}）",
        "missing_rate_core": "缺失率（核心特征）: {pct}%（{miss}/{total}）",
        "missing_hint": "缺失率越高，结果越依赖插补。",
        "export_btn": "导出结果（CSV）",
        "export_file": "ap_risk_result_{target}.csv",
        "group_header": "{name}（{n}项）",
        "bench_title": "参考性能（XGBoost）",
        "bench_internal": "内部验证",
        "bench_external": "外部验证（eICU）",
        "bench_auc": "AUC",
        "bench_sens": "敏感度",
        "bench_spec": "特异度",
        "bench_na": "暂无",
        "completeness_weighted": "模型完整度（按影响加权）: {pct}%",
        "completeness_hint": "基于 XGBoost 特征重要性，对已填写变量进行加权统计。",
        "dynamic_title": "性能保留率（相对全特征基线）",
        "dynamic_note": "基于开发集的内部估计。该指标反映模型层面的保留率，不代表单个病例预测确定性。",
        "dynamic_need_more": "至少填写2个模型特征后才可计算动态比值。",
        "dynamic_unavailable": "当前部署未包含开发集训练/测试文件，动态保留率已禁用。",
        "dynamic_subset": "子集特征数",
        "dynamic_ratio_auc": "AUC 比值",
        "dynamic_ratio_acc": "准确率比值",
        "dynamic_delta_auc": "AUC 差值",
        "dynamic_delta_acc": "准确率差值",
        "dynamic_list_title": "关键数值",
        "dynamic_auc_subset": "子集 AUC",
        "dynamic_auc_full": "全特征 AUC",
        "dynamic_auc_ratio": "AUC 保留率",
        "dynamic_ci": "可能区间（95%CI）: {lo} - {hi}",
        "dynamic_feats": "当前已填特征",
        "value_source": "取值来源",
        "provided": "已填写",
        "imputed": "插补",
        "default_zero": "默认置0",
        "value_model_used": "模型使用值",
        "summary_title": "预测摘要",
        "summary_note": "该输出仅供科研参考。",
        "details_tab_1": "解释",
        "details_tab_2": "完整度",
        "details_tab_3": "保留率",
        "details_tab_4": "数据与导出",
        "bench_toggle": "显示参考性能",
    },
}


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

            :root {
                --bg-main: #eef3f8;
                --bg-card: #ffffff;
                --bg-hero: linear-gradient(90deg, #1f3f63 0%, #204f7d 55%, #2b5b7e 100%);
                --text-main: #1b2a3a;
                --text-sub: #5a6a7a;
                --border-soft: #cfd8e3;
                --accent: #1f4f7d;
                --accent-2: #3f88c5;
                --risk-low: #2f855a;
                --risk-mid: #c27c2c;
                --risk-high: #c53030;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 20% 0%, #dfe8f3 0%, rgba(223,232,243,0) 40%),
                    radial-gradient(circle at 95% 10%, #edf3f9 0%, rgba(237,243,249,0) 35%),
                    var(--bg-main);
            }

            html, body, [class*="css"], [data-testid="stMarkdownContainer"], .stTextInput label, .stSelectbox label {
                font-family: "IBM Plex Sans", sans-serif !important;
                color: var(--text-main);
            }

            /* 强制正文区域高对比度，避免白字在浅底不可读 */
            [data-testid="stAppViewContainer"] p,
            [data-testid="stAppViewContainer"] label,
            [data-testid="stAppViewContainer"] span,
            [data-testid="stAppViewContainer"] li,
            [data-testid="stAppViewContainer"] div {
                color: var(--text-main);
            }

            /* 保持 hero 区为浅色文字，不被全局覆盖 */
            .hero-wrap, .hero-wrap p, .hero-wrap span, .hero-wrap div, .hero-wrap h1 {
                color: #f4f8ff !important;
            }
            .hero-desc { color: #d6e6fa !important; }
            .hero-meta { color: #d8e8ff !important; }

            .main .block-container {
                max-width: 1120px;
                padding-top: 1.4rem;
                padding-bottom: 5rem;
            }

            .hero-wrap {
                background: var(--bg-hero);
                color: #f4f8ff;
                border-radius: 14px;
                padding: 16px 18px 14px 18px;
                box-shadow: 0 8px 24px rgba(18, 48, 78, 0.24);
                margin-bottom: 10px;
                border: 1px solid rgba(255,255,255,0.16);
            }
            .hero-kicker {
                display: inline-block;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.25);
                color: #e9f2ff;
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.04em;
                margin-bottom: 6px;
                text-transform: uppercase;
            }
            .hero-title {
                font-weight: 700;
                font-size: clamp(1.2rem, 2.0vw, 1.7rem);
                line-height: 1.25;
                margin: 0 0 6px 0;
            }
            .hero-desc {
                margin: 0;
                color: #d6e6fa;
                font-size: 0.9rem;
                line-height: 1.5;
            }
            .hero-meta {
                margin-top: 10px;
                padding-top: 9px;
                border-top: 1px dashed rgba(255,255,255,0.28);
                font-family: "IBM Plex Mono", monospace;
                font-size: 0.75rem;
                color: #d8e8ff;
                display: flex;
                gap: 14px;
                flex-wrap: wrap;
            }

            .panel-card {
                background: var(--bg-card);
                border: 1px solid var(--border-soft);
                border-radius: 12px;
                padding: 10px 12px 4px 12px;
                box-shadow: 0 6px 16px rgba(26, 45, 71, 0.08);
                margin-bottom: 10px;
            }
            .panel-title {
                font-size: 0.98rem;
                font-weight: 700;
                margin: 6px 0 8px 0;
                color: #1e3b5e;
                letter-spacing: 0.01em;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid #d4dfeb;
                background: linear-gradient(180deg, #f2f6fb 0%, #eaf1f8 100%);
            }

            .lang-top {
                margin-top: 6px;
            }

            .stButton > button, .stForm button[kind="primary"] {
                border-radius: 8px !important;
                border: 1px solid #245280 !important;
                background: linear-gradient(135deg, #2a5d91 0%, #214b75 100%) !important;
                color: #ffffff !important;
                font-weight: 600 !important;
                padding: 0.5rem 1rem !important;
            }

            /* 输入控件内部文字颜色 */
            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea {
                color: #14273d !important;
                -webkit-text-fill-color: #14273d !important;
                background-color: #ffffff !important;
            }

            .stSelectbox div[data-baseweb="select"] > div {
                color: #14273d !important;
                background-color: #ffffff !important;
            }
            .stSelectbox div[data-baseweb="select"] span {
                color: #14273d !important;
            }

            /* 单选框、提示、expander 标题 */
            [data-testid="stRadio"] label,
            [data-testid="stCaptionContainer"] *,
            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] summary * {
                color: #27405d !important;
            }

            /* DataFrame 文本 */
            [data-testid="stDataFrame"] * {
                color: #1f334d !important;
            }

            [data-testid="stMetricValue"] {
                font-family: "IBM Plex Mono", monospace;
                font-weight: 600;
                color: #193b5f;
            }

            .risk-chip {
                display: inline-block;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 600;
                color: #fff;
                margin-left: 4px;
                font-family: "IBM Plex Mono", monospace;
                letter-spacing: 0.02em;
            }
            .risk-low { background: var(--risk-low); }
            .risk-mid { background: var(--risk-mid); }
            .risk-high { background: var(--risk-high); }

            .result-card {
                border-radius: 12px;
                padding: 12px 14px 8px 14px;
                border: 1px solid var(--border-soft);
                background: #ffffff;
                box-shadow: 0 6px 16px rgba(26, 45, 71, 0.08);
                margin-top: 8px;
            }
            .result-title {
                font-size: 0.95rem;
                font-weight: 700;
                margin: 0 0 8px 0;
                color: #1e3b5e;
            }
            .summary-note {
                color: #5f7289;
                font-size: 0.82rem;
                margin-top: 2px;
            }
            .section-gap {
                margin-top: 6px;
                margin-bottom: 2px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_feature_dict() -> dict[str, Any]:
    if not os.path.exists(FEATURE_DICT_PATH):
        return {}
    with open(FEATURE_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _bundle_path(target: str) -> str:
    return os.path.join(MODELS_DIR, target, "deploy_bundle.pkl")


@st.cache_resource
def load_bundle(target: str) -> dict[str, Any]:
    path = _bundle_path(target)
    if not os.path.exists(path):
        raise FileNotFoundError(f"deploy_bundle 不存在: {path}")

    bundle = joblib.load(path)
    model_dir = os.path.join(MODELS_DIR, target)

    if "features" not in bundle and "feature_names" in bundle:
        bundle["features"] = list(bundle["feature_names"])
    if "feature_names" not in bundle and "features" in bundle:
        bundle["feature_names"] = list(bundle["features"])

    if "best_model" not in bundle:
        all_models_path = os.path.join(model_dir, "all_models_dict.pkl")
        if os.path.exists(all_models_path):
            all_models = joblib.load(all_models_path)
            bundle["best_model"] = all_models.get("XGBoost")

    if "imputer" not in bundle:
        bundle["imputer"] = joblib.load(os.path.join(SCALERS_DIR, "mimic_mice_imputer.joblib"))
    if "mimic_scaler" not in bundle:
        bundle["mimic_scaler"] = joblib.load(os.path.join(SCALERS_DIR, "mimic_scaler.joblib"))
    if "scaler" not in bundle:
        bundle["scaler"] = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    if "imputer_feature_order" not in bundle:
        imp = bundle["imputer"]
        order = getattr(imp, "feature_names_in_", None)
        if order is None:
            train_assets_path = os.path.join(SCALERS_DIR, "train_assets_bundle.pkl")
            order = []
            if os.path.exists(train_assets_path):
                train_assets = joblib.load(train_assets_path)
                order = train_assets.get("feature_order", [])
        bundle["imputer_feature_order"] = list(order) if order is not None else []

    if not bundle.get("features"):
        raise ValueError("deploy_bundle 缺少 features/feature_names")
    if bundle.get("best_model") is None:
        raise ValueError("deploy_bundle 缺少 best_model（XGBoost）")

    return bundle


def is_binary_feature(feat_name: str, feat_meta: dict[str, Any]) -> bool:
    rr = feat_meta.get("ref_range", {}) if feat_meta else {}
    lo, hi = rr.get("logical_min"), rr.get("logical_max")
    if lo == 0 and hi == 1:
        return True
    return feat_name in {
        "gender",
        "heart_failure",
        "chronic_kidney_disease",
        "malignant_tumor",
        "mechanical_vent_flag",
        "vaso_flag",
    }


def preprocess_single_row(bundle: dict[str, Any], input_values: dict[str, float | None]) -> np.ndarray:
    imputer_cols = bundle.get("imputer_feature_order") or list(getattr(bundle["imputer"], "feature_names_in_", []))
    if not imputer_cols:
        raise ValueError("无法确定 imputer 输入列顺序（imputer_feature_order 为空）")

    df = pd.DataFrame([{c: np.nan for c in imputer_cols}])
    for k, v in input_values.items():
        if k in df.columns:
            df.at[0, k] = v

    for col in bundle.get("skewed_cols", []):
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    imputed = bundle["imputer"].transform(df)
    df_imputed = pd.DataFrame(imputed, columns=imputer_cols, index=df.index)

    mimic_scaled = bundle["mimic_scaler"].transform(df_imputed)
    df_mimic_scaled = pd.DataFrame(mimic_scaled, columns=imputer_cols, index=df.index)

    features = list(bundle["features"])
    X_selected = df_mimic_scaled.reindex(columns=features).fillna(0.0)
    X_final = bundle["scaler"].transform(X_selected)
    return X_final


def predict_risk(bundle: dict[str, Any], input_values: dict[str, float | None]) -> float:
    X = preprocess_single_row(bundle, input_values)
    model = bundle["best_model"]
    proba = float(model.predict_proba(X)[0, 1])
    return proba


@st.cache_data
def load_shap_background(target: str, features_key: tuple[str, ...], max_rows: int = 40) -> pd.DataFrame:
    features = list(features_key)
    bundle = load_bundle(target)
    for path in [TEST_PATH, TRAIN_PATH]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if not set(features).issubset(set(df.columns)):
                continue
            x_raw = df[features].dropna().head(max_rows)
            if len(x_raw) < 5:
                continue
            x_bg = bundle["scaler"].transform(x_raw)
            return pd.DataFrame(x_bg, columns=features)
        except Exception:
            continue
    fallback_rows = max(5, min(20, max_rows))
    return pd.DataFrame(np.zeros((fallback_rows, len(features))), columns=features)


def build_single_case_shap_explanation(
    target: str,
    bundle: dict[str, Any],
    input_values: dict[str, float | None],
    feat_dict: dict[str, Any],
    lang_key: str,
    feature_subset: list[str] | None = None,
) -> shap.Explanation:
    if not feature_subset:
        features = list(bundle["features"])
        x_case = pd.DataFrame(preprocess_single_row(bundle, input_values), columns=features)
        x_bg = load_shap_background(target, tuple(features), max_rows=40)
        model = bundle["best_model"]
        explainer = shap.Explainer(model.predict_proba, x_bg)
        sv = explainer(x_case)
        exp = sv[0, :, 1]
        display_names = [feature_title(f, feat_dict.get(f, {}), lang_key) for f in features]
        base_value = float(np.array(exp.base_values).reshape(-1)[0])
        return shap.Explanation(
            values=np.array(exp.values),
            base_values=base_value,
            data=np.array(exp.data),
            feature_names=display_names,
        )

    if not DYNAMIC_DATA_AVAILABLE:
        raise FileNotFoundError("train/test files are unavailable for subset SHAP")
    feats = [f for f in feature_subset if f in bundle.get("features", [])]
    if len(feats) < 2:
        raise ValueError("subset SHAP requires at least 2 features")

    df_train = pd.read_csv(TRAIN_PATH)
    y_train = df_train[target].dropna().astype(int)
    x_train_raw = df_train.loc[y_train.index, feats]

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_raw)
    params = _extract_xgb_params(bundle)
    xgb = XGBClassifier(**params)
    model_sub = CalibratedClassifierCV(xgb, cv=3, method="isotonic", n_jobs=1)
    model_sub.fit(x_train_s, y_train.values)

    value_trace = compute_value_trace(bundle, input_values, list(bundle["features"]))
    x_case_raw = pd.DataFrame([[value_trace[f]["value"] for f in feats]], columns=feats)
    x_case = pd.DataFrame(scaler.transform(x_case_raw), columns=feats)

    x_bg_raw = None
    for path in [TEST_PATH, TRAIN_PATH]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if not set(feats).issubset(df.columns):
                continue
            cand = df[feats].dropna().head(40)
            if len(cand) >= 5:
                x_bg_raw = cand
                break
        except Exception:
            continue
    if x_bg_raw is None:
        x_bg_raw = pd.DataFrame(np.zeros((20, len(feats))), columns=feats)
    x_bg = pd.DataFrame(scaler.transform(x_bg_raw), columns=feats)

    explainer = shap.Explainer(model_sub.predict_proba, x_bg)
    sv = explainer(x_case)
    exp = sv[0, :, 1]
    display_names = [feature_title(f, feat_dict.get(f, {}), lang_key) for f in feats]
    base_value = float(np.array(exp.base_values).reshape(-1)[0])
    return shap.Explanation(
        values=np.array(exp.values),
        base_values=base_value,
        data=np.array(exp.data),
        feature_names=display_names,
    )


def render_shap_images(exp: shap.Explanation, max_display: int = 12) -> tuple[bytes, bytes]:
    buf_wf = io.BytesIO()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(buf_wf, format="png", dpi=220, bbox_inches="tight")
    plt.close()

    buf_force = io.BytesIO()
    safe_names = []
    for i, name in enumerate(list(exp.feature_names)):
        s = str(name).replace("×", "x").replace("⁹", "9").replace("µ", "u")
        s = s.encode("ascii", "ignore").decode("ascii").strip()
        safe_names.append(s if s else f"f{i+1}")
    try:
        plt.figure(figsize=(12, 2.8), dpi=180)
        shap.force_plot(
            exp.base_values,
            exp.values,
            exp.data,
            feature_names=safe_names,
            matplotlib=True,
            show=False,
            text_rotation=12,
            contribution_threshold=0.03,
        )
        plt.tight_layout()
        plt.savefig(buf_force, format="png", dpi=220, bbox_inches="tight")
        plt.close()
    except Exception:
        plt.close()
        plt.figure(figsize=(10, 2.4), dpi=180)
        plt.axis("off")
        plt.text(0.01, 0.5, "Force plot unavailable for current labels.", fontsize=11, va="center")
        plt.tight_layout()
        plt.savefig(buf_force, format="png", dpi=220, bbox_inches="tight")
        plt.close()
    return buf_wf.getvalue(), buf_force.getvalue()


def top_shap_contributions(exp: shap.Explanation, top_n: int = 5) -> list[str]:
    values = np.array(exp.values)
    names = list(exp.feature_names)
    idx = np.argsort(np.abs(values))[::-1][:top_n]
    lines = []
    for i in idx:
        sign = "+" if values[i] >= 0 else "-"
        lines.append(f"{names[i]}: {sign}{abs(values[i]):.3f}")
    return lines


def feature_title(feature: str, meta: dict[str, Any], lang: str) -> str:
    if lang == "zh":
        label = (meta or {}).get("display_name_cn") or (meta or {}).get("display_name") or feature
    else:
        label = (meta or {}).get("display_name_en") or (meta or {}).get("display_name") or feature
    unit = (meta or {}).get("unit")
    return f"{label} ({unit})" if unit else label


def range_text(meta: dict[str, Any], lang_pack: dict[str, str]) -> str:
    rr = (meta or {}).get("ref_range", {})
    lo, hi = rr.get("logical_min"), rr.get("logical_max")
    if lo is None and hi is None:
        return ""
    return lang_pack["range_prefix"].format(lo=lo, hi=hi)


def parse_float_or_none(raw: str) -> float | None:
    s = raw.strip()
    if s == "":
        return None
    return float(s)


def convert_to_model_unit(feature: str, value: float, selected_unit: str) -> float:
    cfg = UNIT_CONVERSIONS.get(feature)
    if not cfg:
        return value
    factor = cfg["factors"].get(selected_unit, 1.0)
    return float(value) * float(factor)


def convert_model_to_selected_unit(feature: str, value: float, selected_unit: str) -> float:
    cfg = UNIT_CONVERSIONS.get(feature)
    if not cfg:
        return value
    factor = cfg["factors"].get(selected_unit, 1.0)
    if factor == 0:
        return value
    return float(value) / float(factor)


def fmt_num(v: float | None) -> str:
    if v is None:
        return ""
    av = abs(float(v))
    if av >= 100:
        return f"{v:.1f}"
    if av >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def infer_group(feature: str, meta: dict[str, Any]) -> str:
    if feature in {"pao2fio2ratio_min", "ph_min", "spo2_min", "spo2_max", "spo2_slope"}:
        return "oxygenation"
    if feature in {"creatinine_max", "bun_min", "bun_max", "lactate_max", "phosphate_min", "albumin_min", "albumin_max", "aniongap_min"}:
        return "renal_metabolic"
    if feature in {"wbc_min", "wbc_max"}:
        return "inflammation"
    if feature in {"hemoglobin_min", "ptt_min"}:
        return "hematology_coag"
    if feature in {"admission_age", "malignant_tumor"}:
        return "demographic_comorb"

    cat = (meta or {}).get("category")
    if cat in {"demographic", "comorbidity"}:
        return "demographic_comorb"
    if cat in {"vital_sign", "trend"}:
        return "oxygenation"
    if cat == "lab_test":
        return "renal_metabolic"
    return "other"


def group_features(feature_list: list[str], feat_dict: dict[str, Any]) -> list[tuple[str, list[str]]]:
    order = ["oxygenation", "renal_metabolic", "hematology_coag", "inflammation", "demographic_comorb", "other"]
    grouped = {k: [] for k in order}
    for feat in feature_list:
        g = infer_group(feat, feat_dict.get(feat, {}))
        grouped.setdefault(g, []).append(feat)
    return [(g, grouped[g]) for g in order if grouped.get(g)]


def rank_features_for_display(bundle: dict[str, Any], top_n: int = 6) -> tuple[list[str], list[str]]:
    features = list(bundle.get("features", []))
    if len(features) <= top_n:
        return features, []

    model = bundle.get("best_model")
    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cal_clf = model.calibrated_classifiers_[0]
            base = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
        else:
            base = model
        importances = getattr(base, "feature_importances_", None)
        if importances is not None and len(importances) == len(features):
            pairs = sorted(zip(features, importances), key=lambda x: float(x[1]), reverse=True)
            ranked = [p[0] for p in pairs]
            core = ranked[:top_n]
            optional = [f for f in features if f not in core]
            return core, optional
    except Exception:
        pass

    core = features[:top_n]
    optional = features[top_n:]
    return core, optional


def _load_shap_recommendation(target: str) -> dict[str, Any] | None:
    rec_path = os.path.join(MODELS_DIR, target, "xgb_shap_pruning_recommendation.json")
    if not os.path.exists(rec_path):
        return None
    try:
        with open(rec_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        return rec if isinstance(rec, dict) else None
    except Exception:
        return None


def _get_shap_features_for_k(target: str, k: int) -> list[str]:
    curve_path = os.path.join(MODELS_DIR, target, "xgb_shap_pruning_curve.csv")
    if not os.path.exists(curve_path):
        return []
    try:
        df = pd.read_csv(curve_path)
        if "k" not in df.columns or "features" not in df.columns or df.empty:
            return []
        df["k"] = df["k"].astype(int)
        row = df[df["k"] == int(k)]
        if row.empty:
            return []
        feats = str(row.iloc[0]["features"]).split(";")
        return [x.strip() for x in feats if x.strip()]
    except Exception:
        return []


def get_smart_subset_features(target: str, bundle: dict[str, Any]) -> tuple[list[str], list[str], str, int]:
    features = list(bundle.get("features", []))
    fallback_k = int(FAST_TOPN_BY_TARGET.get(target, 6))

    rec = _load_shap_recommendation(target)
    if rec and "k_recommended" in rec:
        try:
            k_rec = int(rec["k_recommended"])
            shap_feats = _get_shap_features_for_k(target, k_rec)
            shap_feats = [f for f in shap_feats if f in features]
            if shap_feats:
                optional = [f for f in features if f not in shap_feats]
                return shap_feats, optional, "SHAP recommendation", len(shap_feats)
        except Exception:
            pass

    core, optional = rank_features_for_display(bundle, top_n=fallback_k)
    return core, optional, "XGBoost importance fallback", len(core)


def render_feature_inputs(
    feature_list: list[str],
    feat_dict: dict[str, Any],
    lang_key: str,
    L: dict[str, str],
    user_values: dict[str, float | None],
    parse_errors: list[str],
    key_prefix: str,
) -> None:
    cols = st.columns(2)
    for i, feat in enumerate(feature_list):
        meta = feat_dict.get(feat, {})
        title = feature_title(feat, meta, lang_key)
        help_txt = range_text(meta, L)
        col = cols[i % 2]
        if is_binary_feature(feat, meta):
            opt = col.selectbox(
                title,
                options=[L["bin_missing"], 0, 1],
                index=0,
                help=help_txt or None,
                key=f"{key_prefix}_bin_{feat}",
            )
            user_values[feat] = None if isinstance(opt, str) else float(opt)
        else:
            ucfg = UNIT_CONVERSIONS.get(feat)
            if ucfg:
                input_col, unit_col = col.columns([3, 2])
                txt = input_col.text_input(
                    title,
                    value="",
                    help=L["input_hint_range"].format(range=help_txt) if help_txt else L["input_hint"],
                    key=f"{key_prefix}_num_{feat}",
                )
                unit = unit_col.selectbox(
                    L["unit"],
                    options=list(ucfg["factors"].keys()),
                    index=0,
                    key=f"{key_prefix}_unit_{feat}",
                    help=L["model_unit"].format(unit=ucfg["base_unit"]),
                )
                rr = (meta or {}).get("ref_range", {})
                lo_m, hi_m = rr.get("logical_min"), rr.get("logical_max")
            else:
                txt = col.text_input(
                    title,
                    value="",
                    help=L["input_hint_range"].format(range=help_txt) if help_txt else L["input_hint"],
                    key=f"{key_prefix}_num_{feat}",
                )
                unit = None
                lo_m = hi_m = None
            try:
                parsed = parse_float_or_none(txt)
                if parsed is None:
                    user_values[feat] = None
                elif unit is None:
                    user_values[feat] = parsed
                else:
                    converted = convert_to_model_unit(feat, parsed, unit)
                    user_values[feat] = converted
                    col.markdown(
                        f"<div style='color:#6b7280;font-size:12px;margin-top:-2px;'>"
                        f"{L['converted_value'].format(value=fmt_num(converted), unit=ucfg['base_unit'])}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if lo_m is not None and hi_m is not None:
                        lo_u = convert_model_to_selected_unit(feat, float(lo_m), unit)
                        hi_u = convert_model_to_selected_unit(feat, float(hi_m), unit)
                        col.markdown(
                            f"<div style='color:#6b7280;font-size:12px;margin-top:-6px;'>"
                            f"{L['normal_range'].format(lo=fmt_num(lo_u), hi=fmt_num(hi_u), unit=unit)}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            except ValueError:
                parse_errors.append(f"{feat}: {L['number_err']}")


def render_grouped_inputs(
    feature_list: list[str],
    feat_dict: dict[str, Any],
    lang_key: str,
    L: dict[str, str],
    user_values: dict[str, float | None],
    parse_errors: list[str],
    key_prefix: str,
) -> None:
    groups = group_features(feature_list, feat_dict)
    for group_id, feats in groups:
        group_name = GROUP_LABELS.get(group_id, {}).get(lang_key, group_id)
        st.markdown(f"**{L['group_header'].format(name=group_name, n=len(feats))}**")
        render_feature_inputs(
            feats,
            feat_dict,
            lang_key,
            L,
            user_values,
            parse_errors,
            key_prefix=f"{key_prefix}_{group_id}",
        )


def build_risk_explanations(
    user_values: dict[str, float | None],
    feat_dict: dict[str, Any],
    lang_key: str,
    L: dict[str, str],
    top_n: int = 3,
) -> list[str]:
    signals = []
    for feat, val in user_values.items():
        if val is None:
            continue
        meta = feat_dict.get(feat, {})
        rr = (meta or {}).get("ref_range", {})
        lo, hi = rr.get("logical_min"), rr.get("logical_max")
        if lo is None or hi is None:
            continue
        span = float(hi) - float(lo)
        if span <= 0:
            continue
        if val < float(lo):
            dev = (float(lo) - float(val)) / span
            signals.append((dev, "low", feat, val, float(lo), float(hi), meta))
        elif val > float(hi):
            dev = (float(val) - float(hi)) / span
            signals.append((dev, "high", feat, val, float(lo), float(hi), meta))
    signals.sort(key=lambda x: x[0], reverse=True)

    lines = []
    for _, direction, feat, val, lo, hi, meta in signals[:top_n]:
        name = feature_title(feat, meta, lang_key)
        if direction == "high":
            lines.append(L["explain_high"].format(name=name, value=fmt_num(val), lo=fmt_num(lo), hi=fmt_num(hi)))
        else:
            lines.append(L["explain_low"].format(name=name, value=fmt_num(val), lo=fmt_num(lo), hi=fmt_num(hi)))
    return lines


def build_export_csv(
    target: str,
    risk: float,
    level: str,
    features: list[str],
    core_features: list[str],
    user_values: dict[str, float | None],
    value_trace: dict[str, dict[str, Any]],
    feat_dict: dict[str, Any],
) -> bytes:
    miss_all = sum(1 for f in features if user_values.get(f) is None)
    miss_core = sum(1 for f in core_features if user_values.get(f) is None)
    rows = []
    for feat in features:
        meta = feat_dict.get(feat, {})
        rows.append(
            {
                "target": target,
                "risk_probability": risk,
                "risk_level": level,
                "total_features": len(features),
                "missing_all": miss_all,
                "missing_rate_all": miss_all / max(1, len(features)),
                "core_features": len(core_features),
                "missing_core": miss_core,
                "missing_rate_core": miss_core / max(1, len(core_features)),
                "feature": feat,
                "display_name": meta.get("display_name_en", meta.get("display_name", feat)),
                "model_unit": meta.get("unit"),
                "value_input": user_values.get(feat),
                "value_model_unit": value_trace.get(feat, {}).get("value"),
                "status": value_trace.get(feat, {}).get("source", "provided"),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


@st.cache_data
def load_benchmark_metrics(target: str) -> dict[str, dict[str, float]]:
    metrics = {"internal": {}, "external": {}}

    # Internal benchmark from artifacts/models/<target>/internal_diagnostic_perf.csv
    internal_path = os.path.join(MODELS_DIR, target, "internal_diagnostic_perf.csv")
    if Path(internal_path).exists():
        try:
            df_i = pd.read_csv(internal_path)
            sub = df_i[
                (df_i["Algorithm"].astype(str).str.lower() == "xgboost")
                & (df_i["Group"].astype(str).str.contains("Full", case=False, na=False))
            ]
            if not sub.empty:
                row = sub.iloc[0]
                metrics["internal"] = {
                    "auc": float(row.get("AUC", np.nan)),
                    "sensitivity": float(row.get("Sensitivity", np.nan)),
                    "specificity": float(row.get("Specificity", np.nan)),
                }
        except Exception:
            pass

    # External benchmark from results/main/tables/Table4_external_validation.csv
    if Path(TABLE4_PATH).exists():
        try:
            df_e = pd.read_csv(TABLE4_PATH)
            sub = df_e[
                (df_e["Target"].astype(str).str.lower() == target.lower())
                & (df_e["Algorithm"].astype(str).str.lower() == "xgboost")
            ]
            if not sub.empty:
                row = sub.iloc[0]
                metrics["external"] = {
                    "auc": float(row.get("AUC", np.nan)),
                    "sensitivity": float(row.get("Sensitivity", np.nan)),
                    "specificity": float(row.get("Specificity", np.nan)),
                }
        except Exception:
            pass
    return metrics


def _fmt_pct_or_na(v: float, na_text: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na_text
    return f"{v:.1%}"


def _fmt_num_or_na(v: float, na_text: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na_text
    return f"{v:.3f}"


def compute_value_trace(
    bundle: dict[str, Any],
    user_values: dict[str, float | None],
    features: list[str],
) -> dict[str, dict[str, Any]]:
    trace: dict[str, dict[str, Any]] = {}
    imputer = bundle.get("imputer")
    imputer_cols = bundle.get("imputer_feature_order") or list(getattr(imputer, "feature_names_in_", []))
    skewed_cols = set(bundle.get("skewed_cols", []))

    df = pd.DataFrame([{c: np.nan for c in imputer_cols}]) if imputer_cols else pd.DataFrame([{}])
    for k, v in user_values.items():
        if v is not None and k in df.columns:
            df.at[0, k] = v
    for c in skewed_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    df_imp = pd.DataFrame(index=[0])
    if imputer_cols and imputer is not None:
        arr = imputer.transform(df)
        df_imp = pd.DataFrame(arr, columns=imputer_cols, index=[0])

    for f in features:
        if user_values.get(f) is not None:
            trace[f] = {"source": "provided", "value": float(user_values[f])}
        elif f in df_imp.columns:
            v = float(df_imp.at[0, f])
            if f in skewed_cols:
                v = float(np.expm1(v))
            trace[f] = {"source": "imputed", "value": v}
        else:
            trace[f] = {"source": "default_zero", "value": 0.0}
    return trace


def get_feature_importance_weights(bundle: dict[str, Any], features: list[str]) -> dict[str, float]:
    default = {f: 1.0 for f in features}
    model = bundle.get("best_model")
    if model is None:
        return default
    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cal_clf = model.calibrated_classifiers_[0]
            base = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
        else:
            base = model
        imps = getattr(base, "feature_importances_", None)
        if imps is None or len(imps) != len(features):
            return default
        weights = {f: float(max(0.0, w)) for f, w in zip(features, imps)}
        s = sum(weights.values())
        if s <= 0:
            return default
        return {k: v / s for k, v in weights.items()}
    except Exception:
        return default


def weighted_completeness_pct(
    user_values: dict[str, float | None],
    features: list[str],
    weights: dict[str, float],
) -> float:
    if not features:
        return 0.0
    total = sum(weights.get(f, 0.0) for f in features)
    if total <= 0:
        return 0.0
    filled = sum(weights.get(f, 0.0) for f in features if user_values.get(f) is not None)
    return 100.0 * filled / total


def _extract_xgb_params(bundle: dict[str, Any]) -> dict[str, Any]:
    model = bundle.get("best_model")
    default = {"random_state": 42, "eval_metric": "logloss", "n_jobs": 1}
    if model is None:
        return default
    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cal_clf = model.calibrated_classifiers_[0]
            base = getattr(cal_clf, "estimator", getattr(cal_clf, "base_estimator", None))
        else:
            base = model
        params = dict(base.get_params())
        params["n_jobs"] = 1
        return params
    except Exception:
        return default


@st.cache_data
def dynamic_subset_vs_full(target: str, selected_features: tuple[str, ...]) -> dict[str, float]:
    if not DYNAMIC_DATA_AVAILABLE:
        return {}
    if len(selected_features) < 2:
        return {}

    bundle = load_bundle(target)
    full_features = list(bundle["features"])

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    y_train = df_train[target].dropna().astype(int)
    y_test = df_test[target].dropna().astype(int)

    X_train_full_raw = df_train.loc[y_train.index, full_features]
    X_test_full_raw = df_test.loc[y_test.index, full_features]
    X_test_full = bundle["scaler"].transform(X_test_full_raw)
    p_full = bundle["best_model"].predict_proba(X_test_full)[:, 1]
    y_test_arr = y_test.values
    auc_full = float(roc_auc_score(y_test_arr, p_full))
    acc_full = float(accuracy_score(y_test_arr, (p_full >= 0.5).astype(int)))

    feats = [f for f in selected_features if f in df_train.columns and f in df_test.columns]
    if len(feats) < 2:
        return {}

    X_train_sub = df_train.loc[y_train.index, feats]
    X_test_sub = df_test.loc[y_test.index, feats]

    scaler = StandardScaler()
    X_train_sub_s = scaler.fit_transform(X_train_sub)
    X_test_sub_s = scaler.transform(X_test_sub)

    params = _extract_xgb_params(bundle)
    xgb = XGBClassifier(**params)
    model_sub = CalibratedClassifierCV(xgb, cv=3, method="isotonic", n_jobs=1)
    model_sub.fit(X_train_sub_s, y_train.values)
    p_sub = model_sub.predict_proba(X_test_sub_s)[:, 1]
    auc_sub = float(roc_auc_score(y_test_arr, p_sub))
    acc_sub = float(accuracy_score(y_test_arr, (p_sub >= 0.5).astype(int)))

    # bootstrap CI (internal test, paired resampling)
    rng = np.random.default_rng(42)
    n = len(y_test_arr)
    auc_sub_boot = []
    auc_full_boot = []
    auc_ratio_boot = []
    for _ in range(300):
        idx = rng.integers(0, n, size=n)
        y_b = y_test_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue
        af = float(roc_auc_score(y_b, p_full[idx]))
        asb = float(roc_auc_score(y_b, p_sub[idx]))
        auc_full_boot.append(af)
        auc_sub_boot.append(asb)
        if af > 0:
            auc_ratio_boot.append(asb / af)

    def _ci(arr: list[float]) -> tuple[float, float]:
        if not arr:
            return (np.nan, np.nan)
        return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))

    auc_sub_lo, auc_sub_hi = _ci(auc_sub_boot)
    auc_full_lo, auc_full_hi = _ci(auc_full_boot)
    ratio_lo, ratio_hi = _ci(auc_ratio_boot)

    return {
        "k": float(len(feats)),
        "auc_sub": auc_sub,
        "auc_sub_ci_low": auc_sub_lo,
        "auc_sub_ci_high": auc_sub_hi,
        "auc_full": auc_full,
        "auc_full_ci_low": auc_full_lo,
        "auc_full_ci_high": auc_full_hi,
        "auc_ratio": auc_sub / auc_full if auc_full > 0 else np.nan,
        "auc_ratio_ci_low": ratio_lo,
        "auc_ratio_ci_high": ratio_hi,
        "auc_delta": auc_sub - auc_full,
        "acc_sub": acc_sub,
        "acc_full": acc_full,
        "acc_ratio": acc_sub / acc_full if acc_full > 0 else np.nan,
        "acc_delta": acc_sub - acc_full,
    }


def main() -> None:
    st.set_page_config(page_title=I18N["en"]["page_title"], page_icon="🩺", layout="wide")
    apply_custom_theme()

    _, lang_col = st.columns([8, 2])
    with lang_col:
        st.markdown('<div class="lang-top">', unsafe_allow_html=True)
        lang = st.selectbox(
            "Language / 语言",
            options=["English", "中文"],
            index=0,
            label_visibility="collapsed",
            key="lang_top_right",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    lang_key = "en" if lang == "English" else "zh"
    L = I18N[lang_key]

    target_labels = {
        L["target_pof"]: "pof",
        L["target_mortality"]: "mortality",
        L["target_composite"]: "composite",
    }

    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-kicker">{L["hero_kicker"]}</div>
            <h1 class="hero-title">{L["title"]}</h1>
            <p class="hero-desc">{L["hero_desc"]}</p>
            <div class="hero-meta">
                <span>ID: AP-RISK-ML</span>
                <span>ENGINE: XGBOOST</span>
                <span>MODE: SINGLE CASE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    target_label = st.selectbox(L["endpoint"], list(target_labels.keys()), index=0)
    target = target_labels[target_label]

    try:
        bundle = load_bundle(target)
    except Exception as e:
        st.error(L["load_failed"].format(err=e))
        st.stop()

    feat_dict = load_feature_dict()
    features = list(bundle["features"])
    core_features, optional_features, smart_source, smart_k = get_smart_subset_features(target, bundle)
    feat_weights = get_feature_importance_weights(bundle, features)
    bench = load_benchmark_metrics(target)

    st.info(L["model_info"].format(target=target, n=len(features)))
    with st.expander(f"{L['bench_toggle']}: {L['bench_title']}", expanded=False):
        b1, b2 = st.columns(2)
        with b1:
            st.caption(L["bench_internal"])
            i1, i2, i3 = st.columns(3)
            i1.metric(L["bench_auc"], _fmt_num_or_na(bench["internal"].get("auc", np.nan), L["bench_na"]))
            i2.metric(L["bench_sens"], _fmt_pct_or_na(bench["internal"].get("sensitivity", np.nan), L["bench_na"]))
            i3.metric(L["bench_spec"], _fmt_pct_or_na(bench["internal"].get("specificity", np.nan), L["bench_na"]))
        with b2:
            st.caption(L["bench_external"])
            e1, e2, e3 = st.columns(3)
            e1.metric(L["bench_auc"], _fmt_num_or_na(bench["external"].get("auc", np.nan), L["bench_na"]))
            e2.metric(L["bench_sens"], _fmt_pct_or_na(bench["external"].get("sensitivity", np.nan), L["bench_na"]))
            e3.metric(L["bench_spec"], _fmt_pct_or_na(bench["external"].get("specificity", np.nan), L["bench_na"]))
    st.markdown(f'<div class="panel-title">{L["section_inputs"]}</div>', unsafe_allow_html=True)

    user_values: dict[str, float | None] = {}
    parse_errors: list[str] = []

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    with st.form("predict_form", clear_on_submit=False):
        mode = st.radio(
            L["input_mode"],
            options=[L["mode_smart"], L["mode_full"]],
            horizontal=True,
        )

        for f in features:
            user_values[f] = None

        st.markdown(f"**{L['group_title']}**")
        if mode == L["mode_smart"]:
            st.caption(L["smart_hint"])
            st.caption(L["smart_source"].format(source=smart_source, k=smart_k))
            if target == "composite":
                st.caption(L["composite_fast_note"])
            render_grouped_inputs(
                core_features,
                feat_dict,
                lang_key,
                L,
                user_values,
                parse_errors,
                key_prefix="core",
            )
            if optional_features:
                with st.expander(f"{L['optional_vars']} ({len(optional_features)})", expanded=False):
                    render_grouped_inputs(
                        optional_features,
                        feat_dict,
                        lang_key,
                        L,
                        user_values,
                        parse_errors,
                        key_prefix="optional",
                    )
        else:
            render_grouped_inputs(
                features,
                feat_dict,
                lang_key,
                L,
                user_values,
                parse_errors,
                key_prefix="full",
            )

        submitted = st.form_submit_button(L["submit"], type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    result_ready_key = f"result_ready_{target}_{lang_key}"
    if submitted:
        st.session_state[result_ready_key] = True

    if submitted or st.session_state.get(result_ready_key, False):
        if parse_errors:
            st.error(L["parse_failed"].format(errors="\n- ".join(parse_errors)))
            st.stop()
        try:
            risk = predict_risk(bundle, user_values)
        except Exception as e:
            st.error(L["predict_failed"].format(err=e))
            st.stop()

        if risk < 0.30:
            level = L["risk_low"]
        elif risk < 0.60:
            level = L["risk_mid"]
        else:
            level = L["risk_high"]

        risk_class = "risk-low" if risk < 0.30 else ("risk-mid" if risk < 0.60 else "risk-high")
        missing_all = sum(1 for f in features if user_values.get(f) is None)
        missing_core = sum(1 for f in core_features if user_values.get(f) is None)
        missing_all_pct = 100.0 * missing_all / max(1, len(features))
        missing_core_pct = 100.0 * missing_core / max(1, len(core_features))
        completeness_pct = weighted_completeness_pct(user_values, features, feat_weights)
        value_trace = compute_value_trace(bundle, user_values, features)

        st.markdown(
            f"""
            <div class="result-card">
                <p class="result-title">{L["section_result"]}</p>
                <span class="risk-chip {risk_class}">{L["risk_level"]}: {level}</span>
                <div class="summary-note">{L["summary_note"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns([1, 1])
        m1.metric(L["prob"], f"{risk:.1%}")
        m2.metric(L["risk_level"], level)
        st.progress(risk, text=f"{L['risk_bar']}: {risk:.1%}")
        t1, t2, t3, t4 = st.tabs([L["details_tab_1"], L["details_tab_2"], L["details_tab_3"], L["details_tab_4"]])

        with t1:
            st.markdown(f"**{L['explain_title']}**")
            explain_lines = build_risk_explanations(user_values, feat_dict, lang_key, L, top_n=3)
            if explain_lines:
                for line in explain_lines:
                    st.markdown(f"- {line}")
            else:
                st.caption(L["explain_none"])
            st.markdown(f"**{L['shap_title']}**")
            st.caption(L["shap_note"])
            shap_mode_key = f"shap_mode_{target}_{lang_key}"
            shap_mode_sync_key = f"shap_mode_sync_{target}_{lang_key}"
            desired_mode = (
                L["shap_mode_smart"]
                if (mode == L["mode_smart"] and DYNAMIC_DATA_AVAILABLE)
                else L["shap_mode_full"]
            )
            if st.session_state.get(shap_mode_sync_key) != mode:
                st.session_state[shap_mode_key] = desired_mode
                st.session_state[shap_mode_sync_key] = mode

            ctrl_col, note_col = st.columns([1.3, 2.0])
            with ctrl_col:
                shap_mode = st.radio(
                    L["shap_mode_label"],
                    options=[L["shap_mode_full"], L["shap_mode_smart"]],
                    horizontal=True,
                    key=shap_mode_key,
                    label_visibility="visible",
                )
            with note_col:
                use_subset_shap = shap_mode == L["shap_mode_smart"]
                if use_subset_shap:
                    st.caption(L["shap_mode_smart_note"])
                    st.caption(L["shap_mode_source"].format(source=smart_source, k=smart_k))
                    if not DYNAMIC_DATA_AVAILABLE:
                        st.caption(L["shap_mode_smart_unavailable"])
                else:
                    st.caption(L["shap_mode_full_note"])

            shap_feats = core_features if use_subset_shap else features
            shap_sig = (shap_mode, tuple((f, user_values.get(f)) for f in shap_feats))
            shap_state_key = f"shap_payload_{target}_{lang_key}"
            if st.button(L["shap_run"], key=f"run_shap_{target}_{lang_key}"):
                try:
                    if use_subset_shap and not DYNAMIC_DATA_AVAILABLE:
                        raise FileNotFoundError(L["shap_mode_smart_unavailable"])
                    exp = build_single_case_shap_explanation(
                        target=target,
                        bundle=bundle,
                        input_values=user_values,
                        feat_dict=feat_dict,
                        lang_key=lang_key,
                        feature_subset=core_features if use_subset_shap else None,
                    )
                    img_wf, img_force = render_shap_images(exp, max_display=min(12, len(shap_feats)))
                    st.session_state[shap_state_key] = {
                        "sig": shap_sig,
                        "waterfall": img_wf,
                        "force": img_force,
                        "top": top_shap_contributions(exp, top_n=5),
                    }
                except Exception as err:
                    st.error(L["shap_failed"].format(err=err))

            payload = st.session_state.get(shap_state_key)
            if payload:
                if payload.get("sig") == shap_sig:
                    st.markdown(f"- **{L['shap_top']}**")
                    for line in payload.get("top", []):
                        st.markdown(f"- {line}")
                    c1, c2 = st.columns(2)
                    c1.image(payload["waterfall"], caption=L["shap_waterfall"], use_container_width=True)
                    c2.image(payload["force"], caption=L["shap_force"], use_container_width=True)
                else:
                    st.caption(L["shap_stale"])

        with t2:
            st.caption(L["completeness_weighted"].format(pct=f"{completeness_pct:.1f}"))
            st.progress(min(1.0, max(0.0, completeness_pct / 100.0)))
            st.caption(L["completeness_hint"])
            st.caption(L["missing_rate_core"].format(pct=f"{missing_core_pct:.1f}", miss=missing_core, total=len(core_features)))
            st.caption(L["missing_rate_all"].format(pct=f"{missing_all_pct:.1f}", miss=missing_all, total=len(features)))
            st.caption(L["missing_hint"])

        with t3:
            st.markdown(f"**{L['dynamic_title']}**")
            if not DYNAMIC_DATA_AVAILABLE:
                st.caption(L["dynamic_unavailable"])
            else:
                filled_feats = tuple([f for f in features if user_values.get(f) is not None])
                if len(filled_feats) < 2:
                    st.caption(L["dynamic_need_more"])
                else:
                    dyn = dynamic_subset_vs_full(target, filled_feats)
                    if dyn:
                        feat_names = [feature_title(f, feat_dict.get(f, {}), lang_key) for f in filled_feats]
                        st.markdown(f"- **{L['dynamic_feats']} ({int(dyn['k'])})**: " + ", ".join(feat_names))
                        st.markdown(
                            f"- **{L['dynamic_auc_subset']}**: `{_fmt_num_or_na(dyn['auc_sub'], L['bench_na'])}`; "
                            + L["dynamic_ci"].format(
                                lo=_fmt_num_or_na(dyn["auc_sub_ci_low"], L["bench_na"]),
                                hi=_fmt_num_or_na(dyn["auc_sub_ci_high"], L["bench_na"]),
                            )
                        )
                        st.markdown(
                            f"- **{L['dynamic_auc_full']}**: `{_fmt_num_or_na(dyn['auc_full'], L['bench_na'])}`; "
                            + L["dynamic_ci"].format(
                                lo=_fmt_num_or_na(dyn["auc_full_ci_low"], L["bench_na"]),
                                hi=_fmt_num_or_na(dyn["auc_full_ci_high"], L["bench_na"]),
                            )
                        )
                        st.markdown(
                            f"- **{L['dynamic_auc_ratio']}**: `{_fmt_num_or_na(dyn['auc_ratio'], L['bench_na'])}`; "
                            + L["dynamic_ci"].format(
                                lo=_fmt_num_or_na(dyn["auc_ratio_ci_low"], L["bench_na"]),
                                hi=_fmt_num_or_na(dyn["auc_ratio_ci_high"], L["bench_na"]),
                            )
                        )
                        st.markdown(f"- **{L['dynamic_delta_auc']}**: `{_fmt_num_or_na(dyn['auc_delta'], L['bench_na'])}`")
                        st.markdown(f"- **{L['dynamic_ratio_acc']}**: `{_fmt_num_or_na(dyn['acc_ratio'], L['bench_na'])}`")
                        st.markdown(f"- **{L['dynamic_delta_acc']}**: `{_fmt_num_or_na(dyn['acc_delta'], L['bench_na'])}`")
                        st.caption(L["dynamic_note"])
                    else:
                        st.caption(L["dynamic_need_more"])

        with t4:
            csv_data = build_export_csv(target, risk, level, features, core_features, user_values, value_trace, feat_dict)
            st.download_button(
                label=L["export_btn"],
                data=csv_data,
                file_name=L["export_file"].format(target=target),
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander(L["input_review"], expanded=False):
                display_df = pd.DataFrame(
                    [
                        {
                            "feature": k,
                            "value_input": v if v is not None else L["missing"],
                            "value_model_used": fmt_num(value_trace.get(k, {}).get("value")) if value_trace.get(k, {}).get("value") is not None else L["missing"],
                            "source": L.get(value_trace.get(k, {}).get("source", "provided"), value_trace.get(k, {}).get("source", "provided")),
                        }
                        for k, v in user_values.items()
                    ]
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <style>
            .main .block-container {{
                padding-bottom: 4rem;
            }}
            .app-footer {{
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                text-align: center;
                background: #f8f9fb;
                border-top: 1px solid #d9dbe0;
                color: #2b2f36;
                padding: 8px 12px;
                font-size: 12px;
                z-index: 9999;
            }}
        </style>
        <div class="app-footer">{L["footer"]}</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
