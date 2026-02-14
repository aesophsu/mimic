# 稿件图表索引（Supplementary Materials Manifest）

> 论文投稿用表格已统一置于 `docs/tables/`：正文表格见 `docs/tables/main/`，补充表格见 `docs/tables/supplementary/`。图形见 `docs/figures/main/` 与 `docs/figures/supplementary/`。本文档列出补充材料 S1–S5 的文件路径及主文引用方式。

**图表说明文件**：每个图表均配有同名 `.md` 说明文件（figure/table legend），内容为医学论文导向、针对性描述的中文 legend，与对应图片/表格位于同一目录。

---

## 主文引用方式

在 Methods 与 Results 中引用补充材料时，使用格式：**详见补充材料 S1**、**Supplementary Figure S2** 等。

---

## Supplementary Figure S1：LASSO 特征选择

**路径**：`docs/figures/supplementary/SF1_lasso/`

| 文件 | 内容 |
|------|------|
| `SF1a_diag_{pof,mortality,composite}.{pdf,png}` | LASSO 路径诊断图 |
| `SF1b_importance_{pof,mortality,composite}.{pdf,png}` | LASSO 特征重要性 |

**主文引用**：Methods 特征选择段落 →「详见补充材料 S1」

---

## Supplementary Figure S2：内部验证 ROC 与校准曲线

**路径**：`docs/figures/supplementary/SF2_internal_ROC/`

| 文件 | 内容 |
|------|------|
| `SF2a_ROC_{pof,mortality,composite}.{pdf,png}` | 内部 ROC 曲线 |
| `SF2b_Calibration_{pof,mortality,composite}.{pdf,png}` | 内部校准曲线（全人群） |

**主文引用**：Results 内部验证段落 →「详见补充材料 S2」

> 说明：按肾功能亚组（Subgroup No Renal=1/0）的内部校准曲线由 `scripts/modeling/06_model_training_main.py` 生成，文件命名为 `{target}_Calibration_subgroupNoRenal_{1,0}.{pdf,png}`，位于 `results/supplementary/figures/S2_internal_ROC/{target}/`，如有需要可作为补充图 S2c 引用。

---

## Supplementary Figure S3：诊断四格表与特征重要性

**路径**：`docs/figures/supplementary/SF3_cutoff/`

| 内容 | 文件 |
|------|------|
| 诊断四格表 | `SF3a_Diagnostic_{pof,mortality,composite}.{pdf,png}` |
| 特征重要性 | `SF3b_feature_importance.{pdf,png}` |
| 森林图 | `SF3c_forest_plot.{pdf,png}` |

**主文引用**：Results 内部验证或可解释性段落 →「详见补充材料 S3」

---

## Supplementary Figure S4：漂移分析（MIMIC vs eICU）

**路径**：`docs/figures/supplementary/SF4_comparison/`

| 内容 | 文件 |
|------|------|
| Table 4 可视化 | `SF4a_Table4_visualization.{pdf,png}` |
| 特征分布漂移 | `SF4b_drift/SF4b_drift_{feature}_{outcome}.{pdf,png}` |

**漂移数据**：`validation/eicu_vs_mimic_drift.json`

**主文引用**：Methods 外部验证或 Results 外部验证段落 →「详见补充材料 S4」

---

## Supplementary Figure S5：SHAP 可解释性（依赖图、Force 图）

**路径**：`docs/figures/supplementary/SF5_interpretation/`

| 内容 | 文件 |
|------|------|
| 个体 SHAP Force 图 | `SF5a_Force_{pof,mortality,composite}.{pdf,png}` |
| 依赖图 | `SF5b_Dep/SF5b_Dep_{outcome}_{feature}.{pdf,png}` |

**主文引用**：Results 可解释性段落 →「详见补充材料 S5」

---

## Supplementary Tables（论文用）

**路径**：`docs/tables/supplementary/`

| 内容 | 文件 |
|------|------|
| ST1 OR 统计 | `ST1_OR_{pof,mortality,composite}.csv` |
| ST2 LASSO 入选特征 | `ST2_LASSO_selected_features.csv` |
| ST3 漂移分析 | `ST3_drift_analysis.csv` |
| ST4 DCA 净获益汇总 | `ST4_DCA_summary.csv` |
| ST5 内部验证明细（按终点） | `ST5_Table3_per_endpoint.csv` |

原始脚本产出见 `results/supplementary/tables/`（含 OR_Json_*.json、DCA_Data_*.csv 等）。

> 说明：DCA 分层结果文件命名为 `DCA_Data_{target}_subgroupNoRenal_{0,1}.csv`，由 `scripts/audit_eval/12_clinical_calibration_dca.py` 生成，可在撰写不同肾功能状态下的临床净获益分析时引用。

---

## 主文表格速查（论文用）

**路径**：`docs/tables/main/`

| 类型 | 路径 |
|------|------|
| Table 1 基线 | `docs/tables/main/Table1_baseline.csv` |
| Table 2 肾亚组 | `docs/tables/main/Table2_renal_subgroup.csv` |
| Table 3 内部效能 | `docs/tables/main/Table3_performance.csv` |
| Table 4 外部效能 | `docs/tables/main/Table4_external_validation.csv` |

原始脚本产出见 `results/main/tables/`。

---

## 主文图形速查（论文用）

**路径**：`docs/figures/main/`（投稿用副本见 `results/main/figures/`）

| 类型 | 路径 |
|------|------|
| **Figure 1（合并版）** | `results/main/figures/Fig1_study_overview.{pdf,png}` — 脚本 `scripts/audit_eval/04b_fig1_study_overview.py` 生成；Panel A 流程图 + Panel B 缺失热图（仅纳入标准所涉关键生理指标）。 |
| Figure 1 CONSORT 流程图 | `docs/figures/main/Fig1_flowchart.svg` |
| Figure 1 缺失热图 | `docs/figures/main/Fig1_missing_heatmap.{pdf,png}`（04 全变量热图） |
| Figure 2 外部 ROC | `docs/figures/main/Fig2_ROC_external_{pof,mortality,composite}.{pdf,png}` |
| Figure 3 DCA/校准 | `docs/figures/main/Fig3_DCA_calibration_{pof,mortality,composite}.{pdf,png}` |
| Figure 4 SHAP | `docs/figures/main/Fig4_SHAP_summary_{pof,mortality,composite}.{pdf,png}` |
| Figure 5 列线图 | `docs/figures/main/Fig5_nomogram_{pof,mortality,composite}.{pdf,png}` |
| Figure 5 森林图 | `docs/figures/main/Fig5_forest_{pof,mortality,composite}.{pdf,png}` |
