#!/usr/bin/env python3
"""Generate SF4b_drift and SF5b_Dep legend .md files with targeted content."""
import os
from pathlib import Path

FEATURE_NAMES = {
    "admission_age": "年龄",
    "albumin_max": "白蛋白最大值",
    "albumin_min": "白蛋白最小值",
    "alp_min": "碱性磷酸酶最小值",
    "aniongap_min": "阴离子间隙最小值",
    "bun_max": "血尿素氮最大值",
    "creatinine_max": "血肌酐最大值",
    "hemoglobin_min": "血红蛋白最小值",
    "lactate_max": "乳酸最大值",
    "pao2fio2ratio_min": "氧合指数（PaO2/FiO2）最小值",
    "ph_min": "pH 最小值",
    "phosphate_min": "磷酸盐最小值",
    "ptt_min": "凝血酶原时间（PTT）最小值",
    "spo2_max": "血氧饱和度（SpO2）最大值",
    "spo2_min": "血氧饱和度（SpO2）最小值",
    "spo2_slope": "血氧饱和度斜率",
    "tbar": "TBAR",
    "wbc_max": "白细胞最大值",
}

OUTCOME_NAMES = {
    "pof": "持续性器官功能衰竭（POF）",
    "mortality": "28 天全因死亡率",
    "composite": "混合终点（POF 或 28 天死亡）",
}

BASE = Path(__file__).resolve().parent.parent / "docs" / "figures" / "supplementary"


def sf4b_content(feature: str, outcome: str) -> str:
    fname = FEATURE_NAMES.get(feature, feature)
    oname = OUTCOME_NAMES.get(outcome, outcome)
    return f"""# Supplementary Figure S4b 分布漂移（{fname}，{oname}）

**Supplementary Figure S4b.** {fname}（{feature}）在{oname}终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
"""


def sf5b_content(outcome: str, feature: str) -> str:
    fname = FEATURE_NAMES.get(feature, feature)
    oname = OUTCOME_NAMES.get(outcome, outcome)
    return f"""# Supplementary Figure S5b SHAP 依赖图（{oname}，{fname}）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示{oname}终点下{fname}（{feature}）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
"""


def main():
    sf4b_dir = BASE / "SF4_comparison" / "SF4b_drift"
    for f in sf4b_dir.glob("SF4b_drift_*.pdf"):
        name = f.stem.replace("SF4b_drift_", "")
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            feature, outcome = parts
            md_path = f.with_suffix(".pdf.md")
            md_path.write_text(sf4b_content(feature, outcome), encoding="utf-8")
            print(f"Created {md_path.name}")
    for f in sf4b_dir.glob("SF4b_drift_*.png"):
        name = f.stem.replace("SF4b_drift_", "")
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            feature, outcome = parts
            md_path = f.with_suffix(".png.md")
            md_path.write_text(sf4b_content(feature, outcome), encoding="utf-8")
            print(f"Created {md_path.name}")

    sf5b_dir = BASE / "SF5_interpretation" / "SF5b_Dep"
    for f in sf5b_dir.glob("SF5b_Dep_*.pdf"):
        name = f.stem.replace("SF5b_Dep_", "")
        parts = name.split("_", 1)
        if len(parts) == 2:
            outcome, feature = parts
            md_path = f.with_suffix(".pdf.md")
            md_path.write_text(sf5b_content(outcome, feature), encoding="utf-8")
            print(f"Created {md_path.name}")
    for f in sf5b_dir.glob("SF5b_Dep_*.png"):
        name = f.stem.replace("SF5b_Dep_", "")
        parts = name.split("_", 1)
        if len(parts) == 2:
            outcome, feature = parts
            md_path = f.with_suffix(".png.md")
            md_path.write_text(sf5b_content(outcome, feature), encoding="utf-8")
            print(f"Created {md_path.name}")


if __name__ == "__main__":
    main()
