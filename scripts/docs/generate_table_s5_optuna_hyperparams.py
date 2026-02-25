#!/usr/bin/env python3
"""
Generate Table S5: Optuna search space and best hyperparameter combinations.

Outputs:
- docs/tables/TableS5_optuna_hyperparameter_search_space.csv
- docs/tables/TableS5_optuna_best_hyperparameters.csv
- docs/tables/TableS5_optuna_hyperparameters.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
OUT_DIR = PROJECT_ROOT / "docs" / "tables"

OUT_SPACE_CSV = OUT_DIR / "TableS5_optuna_hyperparameter_search_space.csv"
OUT_BEST_CSV = OUT_DIR / "TableS5_optuna_best_hyperparameters.csv"
OUT_MD = OUT_DIR / "TableS5_optuna_hyperparameters.md"


TARGETS = [
    ("pof", "Primary endpoint (POF)"),
    ("mortality", "Secondary endpoint (28-day mortality)"),
    ("composite", "Composite endpoint"),
]


SPACE_DEF = {
    "XGBoost": {
        "n_trials": 100,
        "params": [
            ("n_estimators", "int", "[100, 500]", "uniform integer"),
            ("max_depth", "int", "[3, 7]", "uniform integer"),
            ("learning_rate", "float", "[0.01, 0.2]", "log-uniform"),
            ("subsample", "float", "[0.5, 1.0]", "uniform"),
        ],
    },
    "Random Forest": {
        "n_trials": 100,
        "params": [
            ("n_estimators", "int", "[100, 500]", "uniform integer"),
            ("max_depth", "int", "[5, 15]", "uniform integer"),
            ("min_samples_split", "int", "[2, 20]", "uniform integer"),
            ("min_samples_leaf", "int", "[1, 10]", "uniform integer"),
        ],
    },
    "SVM": {
        "n_trials": 50,
        "params": [
            ("C", "float", "[0.1, 10.0]", "log-uniform"),
            ("gamma", "categorical", "{scale, auto}", "categorical"),
            ("kernel", "fixed", "rbf", "fixed"),
        ],
    },
    "Decision Tree": {
        "n_trials": 50,
        "params": [
            ("max_depth", "int", "[3, 15]", "uniform integer"),
            ("min_samples_leaf", "int", "[1, 20]", "uniform integer"),
        ],
    },
}


BEST_KEYS = {
    "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample"],
    "Random Forest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
    "SVM": ["C", "gamma", "kernel"],
    "Decision Tree": ["max_depth", "min_samples_leaf"],
}


def _extract_estimator(calibrated_clf):
    cc = calibrated_clf.calibrated_classifiers_[0]
    return getattr(cc, "estimator", getattr(cc, "base_estimator", None))


def build_search_space_table() -> pd.DataFrame:
    rows: List[Dict] = []
    for model_name, cfg in SPACE_DEF.items():
        for p, ptype, pspace, dist in cfg["params"]:
            rows.append(
                {
                    "Model": model_name,
                    "Hyperparameter": p,
                    "Type": ptype,
                    "Search Space": pspace,
                    "Sampling Distribution": dist,
                    "Optuna Trials (per endpoint)": cfg["n_trials"],
                }
            )
    return pd.DataFrame(rows)


def build_best_params_table() -> pd.DataFrame:
    rows: List[Dict] = []
    for target_key, target_label in TARGETS:
        model_dict = joblib.load(MODELS_DIR / target_key / "all_models_dict.pkl")
        for model_name in ["XGBoost", "Random Forest", "SVM", "Decision Tree"]:
            est = _extract_estimator(model_dict[model_name])
            params = est.get_params()
            picked = {k: params.get(k) for k in BEST_KEYS[model_name]}
            rows.append(
                {
                    "Endpoint": target_label,
                    "Model": model_name,
                    "Best Hyperparameter Combination": json.dumps(picked, ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


def write_markdown(df_space: pd.DataFrame, df_best: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Table S5. Optuna Search Space and Best Hyperparameters")
    lines.append("")
    lines.append("## A. Hyperparameter Search Space")
    lines.append("")
    h1 = [
        "Model",
        "Hyperparameter",
        "Type",
        "Search Space",
        "Sampling Distribution",
        "Optuna Trials (per endpoint)",
    ]
    lines.append("| " + " | ".join(h1) + " |")
    lines.append("|" + "|".join(["---"] * len(h1)) + "|")
    for _, r in df_space.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["Model"]),
                    str(r["Hyperparameter"]),
                    str(r["Type"]),
                    str(r["Search Space"]),
                    str(r["Sampling Distribution"]),
                    str(r["Optuna Trials (per endpoint)"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## B. Best Hyperparameter Combination")
    lines.append("")
    h2 = ["Endpoint", "Model", "Best Hyperparameter Combination"]
    lines.append("| " + " | ".join(h2) + " |")
    lines.append("|" + "|".join(["---"] * len(h2)) + "|")
    for endpoint in [x[1] for x in TARGETS]:
        e = df_best[df_best["Endpoint"] == endpoint]
        lines.append("| " + " | ".join([f"**{endpoint}**", "", ""]) + " |")
        for _, r in e.iterrows():
            combo = str(r["Best Hyperparameter Combination"]).replace("|", "\\|")
            lines.append(f"| {r['Endpoint']} | {r['Model']} | `{combo}` |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_space = build_search_space_table()
    df_best = build_best_params_table()
    df_space.to_csv(OUT_SPACE_CSV, index=False, encoding="utf-8-sig")
    df_best.to_csv(OUT_BEST_CSV, index=False, encoding="utf-8-sig")
    write_markdown(df_space, df_best)
    print(f"Saved: {OUT_SPACE_CSV}")
    print(f"Saved: {OUT_BEST_CSV}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
