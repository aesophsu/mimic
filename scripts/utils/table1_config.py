"""
Table 1 基线特征表配置
- 变量分组与顺序
- 展示名覆盖（论文规范）
- 分类变量仅显示阳性类别
- 单位标注
"""

# 变量分组：Demographics → Severity → Organ Support → Laboratory → Outcomes
# 每组内变量顺序
TABLE1_GROUPS = {
    "Demographics & Comorbidities": [
        "admission_age",
        "weight_admit",
        "gender",
        "heart_failure",
        "chronic_kidney_disease",
        "malignant_tumor",
    ],
    "Severity Scores at Admission": [
        "sofa_score",
        "apsiii",
        "sapsii",
        "oasis",
        "lods",
    ],
    "Organ Support": [
        "mechanical_vent_flag",
        "vaso_flag",
    ],
    "Laboratory Parameters": [
        "wbc_max",
        "hemoglobin_min",
        "platelets_min",
        "bun_max",
        "creatinine_max",
        "bilirubin_total_max",
        "alt_max",
        "ast_max",
        "alp_max",
        "lactate_max",
        "pao2fio2ratio_min",
        "spo2_min",
        "ph_min",
        "sodium_max",
        "potassium_max",
        "bicarbonate_min",
    ],
    "Outcomes": [
        "pof",
        "mortality",
    ],
}

# Table 1 专用展示名覆盖（论文规范）
# Mech vent → Mechanical ventilation, Creat → Creatinine 等
TABLE1_DISPLAY_OVERRIDES = {
    "mechanical_vent_flag": "Mechanical ventilation",
    "vaso_flag": "Vasopressors",
    "creatinine_max": "Creatinine",
    "creatinine_min": "Creatinine (min)",
    "admission_age": "Age",
    "weight_admit": "Weight",
    "gender": "Male",
    "heart_failure": "CHF",
    "chronic_kidney_disease": "CKD",
    "malignant_tumor": "Malignancy",
    "pao2fio2ratio_min": "P/F ratio",
    "bicarbonate_min": "HCO3",
    "sodium_max": "Sodium",
    "potassium_max": "Potassium",
    "lactate_max": "Lactate",
    "pof": "POF",
    "mortality": "28-day death",
}

# 单位标注（与 feature_dictionary 一致，用于 Table 1 列名）
# 注：×10⁹/L 在 CSV 中显示为 10^9/L
TABLE1_UNITS = {
    "admission_age": "years",
    "weight_admit": "kg",
    "sofa_score": None,
    "apsiii": None,
    "sapsii": None,
    "oasis": None,
    "lods": None,
    "wbc_max": "10^9/L",
    "hemoglobin_min": "g/dL",
    "platelets_min": "10^9/L",
    "bun_max": "mg/dL",
    "creatinine_max": "mg/dL",
    "bilirubin_total_max": "mg/dL",
    "alt_max": "U/L",
    "ast_max": "U/L",
    "alp_max": "U/L",
    "lactate_max": "mmol/L",
    "pao2fio2ratio_min": "mmHg",
    "spo2_min": "%",
    "ph_min": None,
    "sodium_max": "mmol/L",
    "potassium_max": "mmol/L",
    "bicarbonate_min": "mmol/L",
}

# 二分类变量：仅显示阳性类别 (1) 的 n (%)，变量名改为阳性标签
# 如 gender → Male, mechanical_vent_flag → Mechanical ventilation
BINARY_SHOW_POSITIVE_ONLY = [
    "gender",  # Male
    "heart_failure",  # CHF
    "chronic_kidney_disease",  # CKD
    "malignant_tumor",  # Malignancy
    "mechanical_vent_flag",  # Mechanical ventilation
    "vaso_flag",  # Vasopressors
    "pof",  # POF
    "mortality",  # 28-day death
]

# 表格脚注（SCI 投稿规范）
TABLE1_FOOTNOTES = [
    "Abbreviations: POF, Persistent Organ Failure; SOFA, Sequential Organ Failure Assessment; APS-III, Acute Physiology Score III; SAPS-II, Simplified Acute Physiology Score II; OASIS, Oxford Acute Severity of Illness Score; LODS, Logistic Organ Dysfunction Score; P/F ratio, PaO2/FiO2 ratio; CHF, Congestive Heart Failure; CKD, Chronic Kidney Disease.",
    "SMD > 0.1 indicates a significant imbalance between groups.",
    "SMD (MIMIC vs eICU) quantifies distributional drift between development and external validation cohorts.",
    "— indicates feature not collected in eICU validation cohort; handled via cross-center alignment in model deployment.",
    "P/F ratio: Both MIMIC and eICU use PaO2/FiO2 from arterial blood gas only; eICU sets NULL when PaO2 unavailable (aligned definition).",
]

# 分层缩进：子项在类别名下缩进（2 空格）
TABLE1_USE_INDENT = True
