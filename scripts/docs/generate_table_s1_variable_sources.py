#!/usr/bin/env python3
"""
Generate Table S1: provenance mapping for the 62 candidate variables.

Outputs:
- docs/tables/TableS1_62_variable_sources.csv
- docs/tables/TableS1_62_variable_sources.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DICT_PATH = PROJECT_ROOT / "artifacts" / "features" / "feature_dictionary.json"
MIMIC_TRAIN_PATH = PROJECT_ROOT / "data" / "cleaned" / "mimic_train_processed.csv"
EICU_RAW_PATH = PROJECT_ROOT / "data" / "external" / "eicu_raw_scale.csv"
OUT_DIR = PROJECT_ROOT / "docs" / "tables"
OUT_CSV = OUT_DIR / "TableS1_62_variable_sources.csv"
OUT_MD = OUT_DIR / "TableS1_62_variable_sources.md"


PROTECTED_COLS = [
    "pof",
    "mortality",
    "composite",
    "subgroup_no_renal",
    "resp_pof",
    "cv_pof",
    "renal_pof",
    "sofa_score",
    "apsiii",
    "sapsii",
    "oasis",
    "lods",
    "subject_id",
    "hadm_id",
    "stay_id",
    "los",
    "mechanical_vent_flag",
    "vaso_flag",
]


GROUP_ORDER = [
    "Demographics & Comorbidities",
    "Vital Signs",
    "Laboratory Measurements",
    "Organ Support / Interventions",
    "Scoring Systems",
]


DEMOGRAPHIC_VARS = {
    "gender",
    "admission_age",
    "weight_admit",
    "heart_failure",
    "chronic_kidney_disease",
    "malignant_tumor",
}
VITAL_VARS = {"spo2_min", "spo2_max", "spo2_slope"}
ORGAN_SUPPORT_VARS = {"mechanical_vent_flag", "vaso_flag"}
SCORING_VARS = {"sofa_score", "apsiii", "sapsii", "oasis", "lods"}


MIMIC_EXPLICIT_SOURCE = {
    "gender": "mimiciv_hosp.patients / gender",
    "admission_age": "mimiciv_hosp.patients / (anchor_age, anchor_year) -> derived at ICU intime",
    "weight_admit": "mimiciv_derived.first_day_weight / weight",
    "heart_failure": "mimiciv_hosp.diagnoses_icd / ICD-9 428* or ICD-10 I50*",
    "chronic_kidney_disease": "mimiciv_hosp.diagnoses_icd / ICD-9 585* or ICD-10 N18*",
    "malignant_tumor": "mimiciv_hosp.diagnoses_icd / ICD-9 14*/15* or ICD-10 C*",
    "spo2_min": "mimiciv_icu.chartevents / itemid 220277 (SpO2)",
    "spo2_max": "mimiciv_icu.chartevents / itemid 220277 (SpO2)",
    "spo2_slope": "mimiciv_icu.chartevents / itemid 220277 (SpO2 trend 0-24h)",
    "ph_min": "mimiciv_derived.first_day_bg / ph_min",
    "ph_max": "mimiciv_derived.first_day_bg / ph_max",
    "pao2fio2ratio_min": "mimiciv_derived.first_day_bg / pao2fio2ratio_min",
    "glucose_lab_min": "mimiciv_hosp.labevents / itemid 50931, 50809, 52569",
    "glucose_lab_max": "mimiciv_hosp.labevents / itemid 50931, 50809, 52569",
    "glucose_slope": "mimiciv_hosp.labevents / itemid 50931, 50809, 52569 (trend 0-24h)",
    "rdw_max": "mimiciv_hosp.labevents / itemid 51277, 52172",
    "phosphate_min": "mimiciv_hosp.labevents / itemid 50970",
    "lipase_max": "mimiciv_hosp.labevents / itemid 50956",
    "tbar": "Derived / bilirubin_total_max / albumin_min",
}


EICU_EXPLICIT_SOURCE = {
    "gender": "eicu_derived.icustay_detail / gender",
    "admission_age": "eicu_derived.icustay_detail / age",
    "weight_admit": "eicu_derived.icustay_detail / admissionweight",
    "heart_failure": "eicu_crd.diagnosis / diagnosisstring pattern (CHF/heart failure)",
    "chronic_kidney_disease": "eicu_crd.diagnosis / diagnosisstring pattern (CKD/ESRD)",
    "malignant_tumor": "eicu_crd.diagnosis / diagnosisstring pattern (malignant/cancer)",
    "spo2_min": "eicu_derived.pivoted_vital / spo2",
    "spo2_max": "eicu_derived.pivoted_vital / spo2",
    "spo2_slope": "eicu_derived.pivoted_vital / spo2 trend (0-24h)",
    "ph_min": "Derived chain / pivoted_bg.ph + lab(pH) + apacheapsvar.ph",
    "ph_max": "Derived chain / pivoted_bg.ph + lab(pH) + apacheapsvar.ph",
    "pao2fio2ratio_min": "eicu_derived.pivoted_bg / pao2, fio2 -> P/F ratio",
    "glucose_slope": "eicu_crd.lab / labname~glucose (offset 0-1440)",
    "tbar": "Derived / bilirubin_total_max / albumin_min",
}


REMARKS_EXPLICIT = {
    "gender": "Binary coding aligned across cohorts (Male=1, Female=0).",
    "admission_age": "In eICU, age '> 89' was recoded to 90 before alignment.",
    "heart_failure": "Binary indicator from diagnosis-pattern mapping; harmonized to 0/1.",
    "chronic_kidney_disease": "Binary indicator from diagnosis-pattern mapping; harmonized to 0/1.",
    "malignant_tumor": "Binary indicator from diagnosis-pattern mapping; harmonized to 0/1.",
    "creatinine_min": "eICU creatinine values in umol/L were converted to mg/dL when needed.",
    "creatinine_max": "eICU creatinine values in umol/L were converted to mg/dL when needed.",
    "bilirubin_total_min": "eICU bilirubin values in umol/L were converted to mg/dL when needed.",
    "bilirubin_total_max": "eICU bilirubin values in umol/L were converted to mg/dL when needed.",
    "albumin_min": "eICU albumin values in g/L were converted to g/dL when needed.",
    "albumin_max": "eICU albumin values in g/L were converted to g/dL when needed.",
    "pao2fio2ratio_min": "P/F ratio derived from PaO2 and FiO2; no SpO2 fallback in eICU extraction.",
    "ph_min": "Multi-source rescue in eICU (blood gas, lab, then APACHE).",
    "ph_max": "Multi-source rescue in eICU (blood gas, lab, then APACHE).",
}


def get_group(var: str) -> str:
    if var in DEMOGRAPHIC_VARS:
        return "Demographics & Comorbidities"
    if var in VITAL_VARS:
        return "Vital Signs"
    if var in ORGAN_SUPPORT_VARS:
        return "Organ Support / Interventions"
    if var in SCORING_VARS:
        return "Scoring Systems"
    return "Laboratory Measurements"


def unit_text(cfg: Dict) -> str:
    unit = cfg.get("unit")
    if unit in (None, "", "None"):
        if cfg.get("standard_name") in {"sex", "gender"}:
            return "binary (0/1)"
        if cfg.get("category") in {"comorbidity", "intervention"}:
            return "binary (0/1)"
        return "-"
    return str(unit)


def variable_label(var: str, cfg: Dict) -> str:
    display = cfg.get("display_name_en") or cfg.get("display_name") or var
    return f"{display} [{var}]"


def mimic_source(var: str, cfg: Dict) -> str:
    if var in MIMIC_EXPLICIT_SOURCE:
        return MIMIC_EXPLICIT_SOURCE[var]

    source_col = cfg.get("mimic_source_col", var)
    if var.startswith("spo2_"):
        return "mimiciv_icu.chartevents / itemid 220277 (SpO2)"
    if var in {"ph_min", "ph_max", "pao2fio2ratio_min", "pao2fio2ratio_max"}:
        return f"mimiciv_derived.first_day_bg / {source_col}"
    if var in {"glucose_lab_min", "glucose_lab_max", "glucose_slope", "rdw_max", "phosphate_min", "lipase_max"}:
        return f"temp_lab_slopes (from mimiciv_hosp.labevents) / {source_col}"
    return f"mimiciv_derived.first_day_lab / {source_col}"


def eicu_source(var: str, cfg: Dict) -> str:
    if var in EICU_EXPLICIT_SOURCE:
        return EICU_EXPLICIT_SOURCE[var]

    source_col = cfg.get("eicu_source_col", var)
    if var.startswith("spo2_"):
        return "eicu_derived.pivoted_vital / spo2"
    if var in {"ph_min", "ph_max"}:
        return "Derived chain / pivoted_bg.ph + lab(pH) + apacheapsvar.ph"
    if var in {"pao2fio2ratio_min", "pao2fio2ratio_max"}:
        return "eicu_derived.pivoted_bg / pao2, fio2 -> P/F ratio"
    if var in {"heart_failure", "chronic_kidney_disease", "malignant_tumor"}:
        return "eicu_crd.diagnosis / diagnosisstring pattern"
    return f"eicu_crd.lab / {source_col}"


def build_table() -> pd.DataFrame:
    with FEATURE_DICT_PATH.open("r", encoding="utf-8") as f:
        feat_dict = json.load(f)

    df0 = pd.read_csv(MIMIC_TRAIN_PATH, nrows=1)
    candidate_vars = [c for c in df0.columns if c not in PROTECTED_COLS]
    if len(candidate_vars) != 62:
        raise ValueError(f"Expected 62 candidate variables, got {len(candidate_vars)}")

    mimic_df = pd.read_csv(MIMIC_TRAIN_PATH, usecols=candidate_vars)
    eicu_df = pd.read_csv(EICU_RAW_PATH)

    def miss_pct(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return 100.0
        return float(df[col].isna().mean() * 100.0)

    rows: List[Dict[str, str]] = []
    for var in candidate_vars:
        cfg = feat_dict.get(var, {})
        mimic_missing = miss_pct(mimic_df, var)
        eicu_missing = miss_pct(eicu_df, var)
        remark = REMARKS_EXPLICIT.get(var, "")
        if var not in eicu_df.columns:
            extra = "Variable not explicitly retained in eICU raw-scale export; missingness shown as 100%."
            remark = f"{remark} {extra}".strip()
        rows.append(
            {
                "Category": get_group(var),
                "Variable Name": variable_label(var, cfg),
                "Unit": unit_text(cfg),
                "MIMIC-IV Missing (%)": f"{mimic_missing:.1f}",
                "eICU-CRD Missing (%)": f"{eicu_missing:.1f}",
                "MIMIC-IV Source (Table / Item)": mimic_source(var, cfg),
                "eICU-CRD Source (Table / Item)": eicu_source(var, cfg),
                "Definition / Processing": remark,
            }
        )

    out = pd.DataFrame(rows)
    out["__group_rank"] = out["Category"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    out["__var_rank"] = out.index
    out = out.sort_values(["__group_rank", "__var_rank"]).drop(columns=["__group_rank", "__var_rank"])
    return out


def write_markdown(df: pd.DataFrame) -> None:
    headers = [
        "Category",
        "Variable Name",
        "Unit",
        "MIMIC-IV Missing (%)",
        "eICU-CRD Missing (%)",
        "MIMIC-IV Source (Table / Item)",
        "eICU-CRD Source (Table / Item)",
        "Definition / Processing",
    ]

    lines = []
    lines.append("# Table S1. Source Mapping of 62 Candidate Variables")
    lines.append("")
    lines.append(
        "This table reports variable-level harmonized provenance between MIMIC-IV and eICU-CRD "
        "for the 62 candidate predictors used in preselection."
    )
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for group in GROUP_ORDER:
        gdf = df[df["Category"] == group]
        if gdf.empty:
            continue
        group_cells = [f"**{group}**"] + [""] * (len(headers) - 1)
        lines.append("| " + " | ".join(group_cells) + " |")
        for _, row in gdf.iterrows():
            vals = [str(row[h]) if pd.notna(row[h]) else "" for h in headers]
            vals = [v.replace("|", "\\|") for v in vals]
            lines.append("| " + " | ".join(vals) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_markdown(table)
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved Markdown: {OUT_MD}")
    print(f"Rows: {len(table)}")


if __name__ == "__main__":
    main()
