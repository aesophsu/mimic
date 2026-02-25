#!/usr/bin/env python3
"""
Generate Table S2: LASSO-selected non-zero features and coefficients.

Outputs:
- docs/tables/TableS2_lasso_selected_features.csv
- docs/tables/TableS2_lasso_selected_features.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DICT_PATH = PROJECT_ROOT / "artifacts" / "features" / "feature_dictionary.json"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
OUT_DIR = PROJECT_ROOT / "docs" / "tables"
OUT_CSV = OUT_DIR / "TableS2_lasso_selected_features.csv"
OUT_MD = OUT_DIR / "TableS2_lasso_selected_features.md"


TARGETS = [
    ("pof", "Primary endpoint (POF)"),
    ("mortality", "Secondary endpoint (28-day mortality)"),
    ("composite", "Composite endpoint"),
]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _display_name(feature: str, feat_dict: Dict) -> str:
    cfg = feat_dict.get(feature, {})
    return cfg.get("display_name_en") or cfg.get("display_name") or feature


def build_table() -> pd.DataFrame:
    feat_dict = _load_json(FEATURE_DICT_PATH)
    rows: List[Dict] = []

    for target_key, target_label in TARGETS:
        sel_path = MODELS_DIR / target_key / "selected_features.json"
        sel = _load_json(sel_path)
        weights = sel.get("weights", {})
        feats = sel.get("features", [])

        for feat in feats:
            coef = float(weights.get(feat, 0.0))
            rows.append(
                {
                    "Endpoint": target_label,
                    "Feature": feat,
                    "Feature Display Name": _display_name(feat, feat_dict),
                    "Coefficient (beta)": round(coef, 4),
                    "Abs Coefficient": round(abs(coef), 4),
                }
            )

    df = pd.DataFrame(rows)
    df["Endpoint Order"] = df["Endpoint"].map({label: i for i, (_, label) in enumerate(TARGETS)})
    df = df.sort_values(["Endpoint Order", "Abs Coefficient"], ascending=[True, False]).drop(columns=["Endpoint Order"])
    return df


def write_markdown(df: pd.DataFrame) -> None:
    headers = ["Endpoint", "Feature", "Feature Display Name", "Coefficient (beta)"]
    lines = []
    lines.append("# Table S2. LASSO-Selected Features and Coefficients")
    lines.append("")
    lines.append(
        "Non-zero coefficients from LASSO logistic regression for each endpoint. "
        "Coefficients are reported on standardized training features."
    )
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for endpoint in [x[1] for x in TARGETS]:
        gdf = df[df["Endpoint"] == endpoint]
        if gdf.empty:
            continue
        lines.append("| " + " | ".join([f"**{endpoint}**", "", "", ""]) + " |")
        for _, r in gdf.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r["Endpoint"]),
                        str(r["Feature"]),
                        str(r["Feature Display Name"]),
                        f"{float(r['Coefficient (beta)']):.4f}",
                    ]
                )
                + " |"
            )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_markdown(df)
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved Markdown: {OUT_MD}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
