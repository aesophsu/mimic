"""Generate docs/main/tables/Table1_baseline.csv with strict MIMIC-vs-eICU p-values.

Rules:
- Continuous variables: Kruskal-Wallis test
- Binary variables: chi-square test
- Missing variable in one cohort: value/p-value set to "—"
- SOFA row is intentionally excluded
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_PATH = PROJECT_ROOT / "data" / "cleaned" / "mimic_raw_scale.csv"
EICU_PATH = PROJECT_ROOT / "data" / "external" / "eicu_raw_scale.csv"
OUT_PATH = PROJECT_ROOT / "docs" / "main" / "tables" / "Table1_baseline.csv"


def normalize_gender(series: pd.Series) -> pd.Series:
    def to_binary(v: object) -> float:
        if pd.isna(v):
            return np.nan
        text = str(v).strip().lower()
        if text in {"m", "male", "1", "1.0"}:
            return 1.0
        if text in {"f", "female", "0", "0.0"}:
            return 0.0
        return np.nan

    series = series.map(to_binary)
    mode = series.dropna().mode()
    fill = mode.iloc[0] if len(mode) else 0
    return series.fillna(fill).astype(int)


def median_iqr(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "—"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def identity(series: pd.Series) -> pd.Series:
    return series


def n_pct(series: pd.Series, total: int) -> str:
    n = int((series == 1).sum())
    return f"{n} ({100 * n / total:.1f}%)"


def p_cont(mimic_col: pd.Series, eicu_col: pd.Series) -> str:
    x = mimic_col.dropna()
    y = eicu_col.dropna()
    if len(x) < 2 or len(y) < 2:
        return "—"
    try:
        _, p = stats.kruskal(x, y)
    except Exception:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_bin(mimic_col: pd.Series, eicu_col: pd.Series) -> str:
    xm = mimic_col.dropna()
    ye = eicu_col.dropna()
    if len(xm) == 0 or len(ye) == 0:
        return "—"

    m1 = int((xm == 1).sum())
    e1 = int((ye == 1).sum())
    m0 = int(len(xm) - m1)
    e0 = int(len(ye) - e1)
    table = np.array([[m0, m1], [e0, e1]])
    try:
        _, p, _, _ = stats.chi2_contingency(table)
    except Exception:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_table1(mimic: pd.DataFrame, eicu: pd.DataFrame) -> pd.DataFrame:
    n_mimic = len(mimic)
    n_eicu = len(eicu)

    spec = [
        ("group", "Demographics & Comorbidities", None, identity),
        ("cont", "Age (years)", "admission_age", identity),
        ("bin", "Male, n (%)", "gender", identity),
        ("cont", "Weight (kg)", "weight_admit", identity),
        ("bin", "Congestive heart failure, n (%)", "heart_failure", identity),
        ("bin", "Chronic kidney disease, n (%)", "chronic_kidney_disease", identity),
        ("bin", "Malignancy, n (%)", "malignant_tumor", identity),
        ("group", "Laboratory Parameters (extreme values within first 24h)", None, identity),
        ("group", "[Renal / Metabolic]", None, identity),
        ("cont", "Creatinine (max, umol/L) *", "creatinine_max", lambda s: s * 88.4),
        ("cont", "Urea (max, mmol/L)", "bun_max", lambda s: s * 0.357),
        ("cont", "Lactate (max, mmol/L)", "lactate_max", identity),
        ("group", "[Respiratory / Acid-Base]", None, identity),
        ("cont", "P/F ratio (min, mmHg) *", "pao2fio2ratio_min", identity),
        ("cont", "pH (min) *", "ph_min", identity),
        ("cont", "HCO3 (min, mmol/L)", "bicarbonate_min", identity),
        ("group", "[Hematology / Others]", None, identity),
        ("cont", "WBC (max, 10^9/L)", "wbc_max", identity),
        ("cont", "Hemoglobin (min, g/L)", "hemoglobin_min", lambda s: s * 10.0),
        ("cont", "Platelet (min, 10^9/L)", "platelets_min", identity),
        ("cont", "Total bilirubin (max, umol/L)", "bilirubin_total_max", lambda s: s * 17.104),
        ("cont", "Sodium (max, mmol/L)", "sodium_max", identity),
        ("group", "Organ Support", None, identity),
        ("bin", "Mechanical ventilation, n (%)", "mechanical_vent_flag", identity),
        ("bin", "Vasopressors, n (%)", "vaso_flag", identity),
        ("group", "Clinical Outcomes", None, identity),
        ("bin", "Persistent organ failure (POF), n (%)", "pof", identity),
        ("bin", "28-day mortality, n (%)", "mortality", identity),
    ]

    rows: list[dict[str, str]] = []
    for row_type, label, col, transform in spec:
        if row_type == "group":
            rows.append(
                {
                    "Characteristic": label,
                    "MIMIC-IV, n=1,095": "",
                    "eICU-CRD, n=1,112": "",
                    "p-value": "",
                }
            )
            continue

        mimic_has = col in mimic.columns
        eicu_has = col in eicu.columns

        if mimic_has:
            mimic_series = transform(mimic[col]) if row_type == "cont" else mimic[col]
            mimic_val = median_iqr(mimic_series) if row_type == "cont" else n_pct(mimic_series, n_mimic)
        else:
            mimic_val = "—"
            mimic_series = None

        if eicu_has:
            eicu_series = transform(eicu[col]) if row_type == "cont" else eicu[col]
            eicu_val = median_iqr(eicu_series) if row_type == "cont" else n_pct(eicu_series, n_eicu)
        else:
            eicu_val = "—"
            eicu_series = None

        if (not mimic_has) or (not eicu_has):
            pval = "—"
        else:
            pval = p_cont(mimic_series, eicu_series) if row_type == "cont" else p_bin(mimic_series, eicu_series)

        rows.append(
            {
                "Characteristic": label,
                "MIMIC-IV, n=1,095": mimic_val,
                "eICU-CRD, n=1,112": eicu_val,
                "p-value": pval,
            }
        )

    rows.append(
        {
            "Characteristic": "Note: Laboratory variables are extreme values within first 24h (max/min as labeled).",
            "MIMIC-IV, n=1,095": "",
            "eICU-CRD, n=1,112": "",
            "p-value": "",
        }
    )
    rows.append(
        {
            "Characteristic": "Note: * denotes prespecified key variables: Creatinine (max), P/F ratio (min), and pH (min).",
            "MIMIC-IV, n=1,095": "",
            "eICU-CRD, n=1,112": "",
            "p-value": "",
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    mimic = pd.read_csv(MIMIC_PATH)
    eicu = pd.read_csv(EICU_PATH)

    if "gender" in mimic.columns:
        mimic["gender"] = normalize_gender(mimic["gender"])
    if "gender" in eicu.columns:
        eicu["gender"] = normalize_gender(eicu["gender"])

    out_df = build_table1(mimic, eicu)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
