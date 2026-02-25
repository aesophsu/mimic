"""Generate Table 2 (internal vs external validation matrix) in docs/main/tables."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs" / "main" / "tables"

IN_PERF = PROJECT_ROOT / "artifacts" / "models" / "performance_report.csv"
IN_SLIM = {
    "pof": PROJECT_ROOT / "artifacts" / "models" / "pof" / "xgb_internal_pruning_curve.csv",
    "mortality": PROJECT_ROOT / "artifacts" / "models" / "mortality" / "xgb_internal_pruning_curve.csv",
    "composite": PROJECT_ROOT / "artifacts" / "models" / "composite" / "xgb_internal_pruning_curve.csv",
}
EXTERNAL_FULL = PROJECT_ROOT / "results" / "main" / "tables" / "Table4_external_validation.csv"
EXTERNAL_SLIM = PROJECT_ROOT / "results" / "main" / "tables" / "Table4_external_validation_slim.csv"

ENDPOINTS = [
    ("Primary Endpoint (POF)", "pof"),
    ("Secondary Endpoint (28-day Mortality)", "mortality"),
    ("Composite Endpoint", "composite"),
]
ALL_MODELS = ["XGBoost", "Random Forest", "SVM", "Logistic Regression", "Decision Tree"]
ENDPOINT_MODELS = {
    "pof": ["XGBoost", "Random Forest", "SVM", "Logistic Regression", "Decision Tree"],
    "mortality": ["XGBoost", "Logistic Regression"],
    "composite": ["XGBoost"],
}

FEATURE_NAME_MAP = {
    "pao2fio2ratio_min": "P/F ratio (minimum)",
    "creatinine_max": "Creatinine (maximum)",
    "ph_min": "pH (minimum)",
    "admission_age": "Age at admission",
    "ptt_min": "PTT (minimum)",
    "bun_max": "Blood urea nitrogen (maximum)",
    "lactate_max": "Lactate (maximum)",
    "hemoglobin_min": "Hemoglobin (minimum)",
    "albumin_max": "Albumin (maximum)",
    "bun_min": "Blood urea nitrogen (minimum)",
}


def parse_ci_text(text: str) -> tuple[float | None, float | None]:
    if "(" not in text or ")" not in text:
        return None, None
    core = text.split("(", 1)[1].split(")", 1)[0]
    if "-" not in core:
        return None, None
    lo, hi = core.split("-", 1)
    try:
        return float(lo), float(hi)
    except ValueError:
        return None, None


def format_auc_ci(auc: float, low: float | None, high: float | None) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return f"{auc:.3f} (CI N/A)"
    return f"{auc:.3f} ({float(low):.3f}-{float(high):.3f})"


def load_internal_full() -> pd.DataFrame:
    df = pd.read_csv(IN_PERF)
    df = df[df["Algorithm"].isin(ALL_MODELS)].copy()
    df["Outcome"] = df["Outcome"].str.lower()
    # performance_report.csv stores CI in text column Main CI
    df[["AUC_Low", "AUC_High"]] = df["Main CI"].apply(lambda x: pd.Series(parse_ci_text(str(x))))
    return df


def load_external_full() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_FULL)
    df = df[df["Algorithm"].isin(ALL_MODELS)].copy()
    df["Target"] = df["Target"].str.lower()
    return df


def load_external_slim() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_SLIM)
    df["Target"] = df["Target"].str.lower()
    return df


def load_internal_slim(target: str, k: int) -> pd.Series:
    curve = pd.read_csv(IN_SLIM[target])
    row = curve[curve["k"] == k]
    if row.empty:
        raise ValueError(f"{target}: missing k={k} in {IN_SLIM[target]}")
    return row.iloc[0]


def format_feature_list(raw_features: str) -> str:
    names = []
    for f in str(raw_features).split(";"):
        key = f.strip()
        if not key:
            continue
        names.append(FEATURE_NAME_MAP.get(key, key))
    return "; ".join(names)


def build_table2() -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    internal_full = load_internal_full()
    external_full = load_external_full()
    external_slim = load_external_slim()

    rows: list[dict[str, str]] = []
    slim_notes: dict[str, dict[str, str]] = {}

    for section_name, target in ENDPOINTS:
        rows.append(
            {
                "Endpoints & Model": section_name,
                "Internal AUROC (95% CI)": "",
                "Internal Brier": "",
                "External AUROC (95% CI)": "",
                "External Brier": "",
            }
        )

        # Full-feature XGBoost
        in_xgb = internal_full[(internal_full["Outcome"] == target) & (internal_full["Algorithm"] == "XGBoost")].iloc[0]
        ex_xgb = external_full[(external_full["Target"] == target) & (external_full["Algorithm"] == "XGBoost")].iloc[0]
        rows.append(
            {
                "Endpoints & Model": "Full-feature XGBoost (n=12)",
                "Internal AUROC (95% CI)": format_auc_ci(float(in_xgb["Main AUC"]), in_xgb["AUC_Low"], in_xgb["AUC_High"]),
                "Internal Brier": f"{float(in_xgb['Brier']):.3f}",
                "External AUROC (95% CI)": format_auc_ci(float(ex_xgb["AUC"]), ex_xgb["AUC_Low"], ex_xgb["AUC_High"]),
                "External Brier": f"{float(ex_xgb['Brier']):.3f}",
            }
        )

        # Parsimonious XGBoost (k from external slim artifact)
        ex_slim = external_slim[external_slim["Target"] == target].iloc[0]
        k = int(ex_slim["K"])
        in_slim = load_internal_slim(target, k)
        rows.append(
            {
                "Endpoints & Model": f"Parsimonious XGBoost (n={k})",
                "Internal AUROC (95% CI)": format_auc_ci(float(in_slim["auc"]), in_slim.get("auc_low"), in_slim.get("auc_high")),
                "Internal Brier": f"{float(in_slim['brier']):.3f}",
                "External AUROC (95% CI)": format_auc_ci(float(ex_slim["AUC"]), ex_slim["AUC_Low"], ex_slim["AUC_High"]),
                "External Brier": f"{float(ex_slim['Brier']):.3f}",
            }
        )
        slim_notes[target] = {"k": str(k), "features": str(ex_slim["Features"])}

        # Other baselines (endpoint-specific)
        for model in ENDPOINT_MODELS[target]:
            if model == "XGBoost":
                continue
            in_row = internal_full[(internal_full["Outcome"] == target) & (internal_full["Algorithm"] == model)].iloc[0]
            ex_row = external_full[(external_full["Target"] == target) & (external_full["Algorithm"] == model)].iloc[0]
            rows.append(
                {
                    "Endpoints & Model": f"{model} (n=12)",
                    "Internal AUROC (95% CI)": format_auc_ci(float(in_row["Main AUC"]), in_row["AUC_Low"], in_row["AUC_High"]),
                    "Internal Brier": f"{float(in_row['Brier']):.3f}",
                    "External AUROC (95% CI)": format_auc_ci(float(ex_row["AUC"]), ex_row["AUC_Low"], ex_row["AUC_High"]),
                    "External Brier": f"{float(ex_row['Brier']):.3f}",
                }
            )

    return pd.DataFrame(rows), slim_notes


def write_outputs(df: pd.DataFrame, slim_notes: dict[str, dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "Table2_model_performance.csv"
    md_path = OUT_DIR / "Table2_model_performance.md"
    df.to_csv(csv_path, index=False)

    lines = [
        "# Table 2. Internal and External Validation Performance Matrix",
        "",
        "| Endpoints & Model | Internal Validation (MIMIC-IV) AUROC (95% CI) | Internal Validation Brier | External Validation (eICU-CRD) AUROC (95% CI) | External Validation Brier |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        name = str(row["Endpoints & Model"])
        is_section = row["Internal AUROC (95% CI)"] == "" and row["Internal Brier"] == ""
        if is_section:
            lines.append(f"| **{name}** |  |  |  |  |")
            continue

        model = name
        ext_auc = str(row["External AUROC (95% CI)"])
        ext_brier = str(row["External Brier"])
        if model.startswith("Parsimonious XGBoost"):
            ext_auc = f"**{ext_auc}**"
            ext_brier = f"**{ext_brier}**"

        lines.append(
            f"| &nbsp;&nbsp;&nbsp;&nbsp;{model} | {row['Internal AUROC (95% CI)']} | {row['Internal Brier']} | {ext_auc} | {ext_brier} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- For the primary endpoint (POF), all developed algorithms are presented for a comprehensive comparison. For secondary and composite endpoints, only the top-performing models and their corresponding parsimonious versions are shown to maintain conciseness and emphasize clinical utility.",
            "- AUROC values are shown with 95% CI.",
            "- Brier score reflects probabilistic calibration (lower is better).",
            "- Abbreviations: AUROC, area under the receiver operating characteristic curve; LR, logistic regression.",
            "- Parsimonious feature sets:",
            f"  - POF (n={slim_notes['pof']['k']}): `{format_feature_list(slim_notes['pof']['features'])}`",
            f"  - 28-day mortality (n={slim_notes['mortality']['k']}): `{format_feature_list(slim_notes['mortality']['features'])}`",
            f"  - Composite endpoint (n={slim_notes['composite']['k']}): `{format_feature_list(slim_notes['composite']['features'])}`",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


def main() -> None:
    df, slim_notes = build_table2()
    write_outputs(df, slim_notes)


if __name__ == "__main__":
    main()
