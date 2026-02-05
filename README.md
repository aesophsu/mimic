# 重症急性胰腺炎预测模型：基于 MIMIC-IV v3.1 与 eICU 的机器学习开发与外部验证

> 本研究以**完成一篇医学专业论文**为导向。本文档提供：**研究流程总览**、**数据提取方法**（MIMIC/eICU 纳入排除、时间窗、结局定义）、**数据清洗与预处理步骤**（审计、缺失、变换、划分、MICE、标准化）、**方法学与稿件图表对应**，便于直接撰写 Methods、Results 及 Supplementary Materials，并满足 TRIPOD 报告规范。

---

## 一、研究概要

### 1.1 标题（建议）

**Development and External Validation of Machine Learning Models for Predicting Persistent Organ Failure and Mortality in Severe Acute Pancreatitis: A Multicenter Cohort Study**

### 1.2 结构化摘要（Structured Abstract）

| 段落 | 内容要点 |
|------|----------|
| **Background** | 重症急性胰腺炎（SAP）预后预测对临床决策至关重要；现有模型多为单中心、缺乏外部验证。 |
| **Methods** | 基于 MIMIC-IV v3.1 开发预测模型，采用 LASSO 特征选择、多算法竞赛（XGBoost、RF、SVM、DT、LR）、概率校准；在 eICU 进行外部验证；评估临床净获益（DCA）并构建列线图。 |
| **Results** | 主要终点 POF、次要终点 28-day mortality、混合终点 Composite 的 AUC、校准度及 DCA 净获益；外部验证效能；SHAP 可解释性分析。 |
| **Conclusions** | 模型在独立队列中表现稳定，具有临床转化潜力。 |

### 1.3 研究设计

| 项目 | 说明 |
|------|------|
| **研究类型** | 回顾性、多中心队列研究 |
| **开发队列** | MIMIC-IV v3.1（训练 80% + 内部验证 20%，按 composite 分层） |
| **验证队列** | eICU（完全独立、盲测） |
| **主要终点** | POF（持续性器官功能衰竭） |
| **次要终点** | 28-day mortality |
| **混合终点** | Composite（POF ∪ 28-day mortality） |
| **报告规范** | TRIPOD（Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis） |

---

## 二、研究流程总览（论文 Methods 路线图）

本研究从原始数据库到可投稿图表，分为 **数据层 → 预处理层 → 建模层 → 验证与解释层**，对应稿件的 **Study population、Variables、Model development、Validation、Clinical utility**。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 数据层（需在本地/云端数据库执行 SQL）                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  MIMIC-IV v3.1                    │  eICU CRD                                │
│  01_mimic_extraction.sql          │  08_eicu_extraction.sql                   │
│  → mimic_raw_data.csv            │  → eicu_raw_data.csv                      │
│  01_mimic_flowchart_counts.sql   │  08_eicu_flowchart_counts.sql             │
│  → Figure 1 纳入排除人数          │  → Figure 1 纳入排除人数                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 预处理层（Python：防泄露，仅训练集 fit）                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  01_mimic_cleaning.py     → 结局对齐、缺失剔除、生理范围、1%-99% 盖帽           │
│                            → mimic_raw_scale.csv                             │
│  02_mimic_standardization.py → 划分 80/20、log 变换、MICE、Z-Score            │
│                            → mimic_train/test_processed.csv + 资产(scaler等)  │
│  08_eicu_alignment_cleaning.py → eICU 与 MIMIC 对齐 + deploy_bundle 推理      │
│                            → eicu_processed_{pof,mortality,composite}.csv    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 建模层                                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  05 LASSO 特征选择(仅训练集) → selected_features.json                         │
│  06 多算法训练+校准 → deploy_bundle.pkl（含 imputer/scaler/模型）              │
│  07 最优切点、诊断图、森林图 → Table 3、补充图                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 验证与解释层                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  03 Table 1 基线（MIMIC/eICU、SMD）  09 漂移分析  10 外部验证(Table 4, Fig2)  │
│  11 SHAP  12 DCA/校准(Fig3)  13 列线图/森林图(Fig5)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**防泄露原则**：划分 train/test 在 MICE 与 Scaler 之前；LASSO 与所有超参优化仅用训练集；eICU 仅使用开发阶段产出的 `deploy_bundle`，无任何再拟合。

---

## 三、方法学详述（Methods）

### 3.1 数据来源与队列构建

| 数据源 | 版本 | 用途 |
|--------|------|------|
| **MIMIC-IV** | v3.1 | 开发队列：训练 80% + 内部验证 20% |
| **eICU** | Collaborative Research Database | 外部验证队列（完全独立、盲测） |

纳入排除标准在 **3.2 数据提取方法** 中按 MIMIC / eICU 分别详述；流程图人数由 `scripts/sql/01_mimic_flowchart_counts.sql` 与 `08_eicu_flowchart_counts.sql` 产出，可直接填入 Figure 1（CONSORT）。纳入排除顺序采用 **先 LOS 再首次** 的临床逻辑：先剔除 ICU 住院时间 &lt; 24 小时者，再在符合 LOS 的入住中保留每位患者首次 ICU 入住。

### 3.2 数据提取方法（Data extraction）

#### 3.2.1 MIMIC-IV 开发队列

- **诊断标准**：急性胰腺炎（AP）由住院诊断编码界定——ICD-9: `5770`，ICD-10: `K85%`（`mimiciv_hosp.diagnoses_icd`）。
- **纳入顺序（与 flowchart 一致）**：  
  1) 在 AP 相关住院中，仅保留 **ICU 住院时间（LOS）≥ 24 小时** 的 ICU 入住；  
  2) 在上述入住中，按患者 **首次 ICU 入住**（按 `intime` 排序取第一条）保留一例；  
  3) **年龄 ≥ 18 岁**；  
  4) **关键生理指标缺失 ≤ 80%**：在 10 项指标（creatinine_max, bun_max, lactate_max, pao2fio2ratio_min, ph_min, wbc_max, hemoglobin_min, bilirubin_total_max, sodium_max, albumin_min）中，缺失比例超过 80% 者剔除。
- **时间窗（基线特征）**：  
  - 实验室：住院时间前 6 小时至 ICU 入室后 24 小时（`labevents`）；  
  - 血气/生命体征/葡萄糖·乳酸·SpO2 斜率：ICU 入室至入室后 24 小时（`chartevents`、`first_day_bg` 等）；  
  - 机械通气/血管加压药（基线）：入室至入室后 24 小时（`ventilation`、`vasoactive_agent`）。
- **结局定义**：  
  - **POF（持续性器官功能衰竭）**：基于 SOFA 评分，在 ICU 入室后 24 小时至 7 天内，呼吸/心血管/肾脏任一器官 SOFA ≥ 2 且持续 ≥ 2 天则记为 POF；  
  - **28 天死亡**：院内死亡时间或患者死亡日期 ≤ 入室后 28 天；  
  - **24–48 小时早期死亡**：若死亡发生在入室后 24–48 小时，则结局覆盖为 POF=1、mortality=1，用于竞争风险处理。  
- **产出**：`scripts/sql/01_mimic_extraction.sql` 生成 `my_custom_schema.ap_final_analysis_cohort`，导出为 `data/raw/mimic_raw_data.csv`。

#### 3.2.2 eICU 外部验证队列

- **诊断标准**：诊断文本包含 “pancreatit” 且不包含 “chronic”（`eicu_crd.diagnosis`），与 MIMIC 的 AP 定义相对应。
- **纳入顺序**：与 MIMIC 一致——先 LOS ≥ 24 小时，再每位患者首次 ICU 入住，年龄 ≥ 18 岁；此外排除 **关键生理指标缺失 &gt; 80%** 或 **POF 无法判定（pof 为 NULL）** 者。
- **时间窗对齐**：实验室、血气、生命体征、P/F 比、葡萄糖/乳酸/SpO2 斜率、首日机械通气和血管加压药均限定在 **入 ICU 后 0–1440 分钟（0–24 小时）**，与 MIMIC 的 intime 至 intime+24h 对齐（详见 [EXTRACTION_ALIGNMENT](docs/EXTRACTION_ALIGNMENT.md)）。
- **P/F 比**：与 MIMIC 一致，**仅使用动脉血气 PaO2/FiO2**；eICU 中无 PaO2 时该例 P/F 记为 NULL（详见 [P/F 提取对齐](docs/PF_RATIO_ANALYSIS.md)）。
- **POF 代理（eICU 无 SOFA）**：在 0–24h 外延至第 1–7 天的时间窗内，根据机械通气、血管加压药、透析、P/F&lt;300、肌酐&gt;1.9 等规则构造 POF 代理；**28 天死亡**与 **24–48 小时早期死亡**定义与 MIMIC 对齐（早期死亡同样覆盖 POF/死亡率）。
- **单位转换**：肌酐、胆红素、白蛋白、FiO2 等已按 MIMIC 单位统一（如 μmol/L→mg/dL、g/L→g/dL）。
- **产出**：`scripts/sql/08_eicu_extraction.sql` 生成 `eicu_cview.ap_external_validation`，导出为 `data/raw/eicu_raw_data.csv`。

Figure 1 所需纳入排除人数由 `01_mimic_flowchart_counts.sql`、`08_eicu_flowchart_counts.sql` 输出，详见 [FLOWCHART_COUNTS](docs/FLOWCHART_COUNTS.md)。

### 3.3 数据清洗与预处理（Cleaning and preprocessing）

清洗与标准化在 Python 中完成，**严格先划分训练/测试集，再在训练集上拟合所有变换**，避免信息泄露。

#### 3.3.1 第一步：临床特征审计与清洗（01_mimic_cleaning.py）

- **结局对齐**：对 24–48 小时内死亡者统一标记为 POF=1、mortality=1；结局列名与三终点（pof, mortality, composite）对齐。  
- **特征缺失剔除**：除白名单（如 subject_id、stay_id、pof、mortality、composite、lactate_max、pao2fio2ratio_min 等）外，**缺失率 &gt; 30%** 的特征予以剔除（阈值见 `utils/study_config.py` 中 `MISSING_THRESHOLD`）。  
- **生理范围审计**：依据 `artifacts/features/feature_dictionary.json` 中的 `ref_range`（logical_min/logical_max）与 `conversion_factor`，对超出生理合理范围的值置为缺失；必要时应用单位换算（如原单位为 μmol/L 的指标乘以 conversion_factor）。  
- **统计盖帽**：对数值型特征（排除二分类结局及白名单）做 **1%–99% 分位数盖帽**，减轻极端值对建模的影响。  
- **产出**：`data/cleaned/mimic_raw_scale.csv`（清洗后、未插补、未标准化的 MIMIC 表），供下一步划分与标准化。

#### 3.3.2 第二步：划分、变换与标准化（02_mimic_standardization.py）

- **训练/测试划分**：按 **composite** 分层（若缺失则按 mortality），80% 训练 / 20% 测试，`random_state=42`；仅对结局非缺失的样本参与划分，划分前不进行插补。  
- **偏态变换**：对特征字典中标记为 `needs_log_transform` 的变量在训练集上做 **log1p** 变换，测试集使用相同变换；偏态列列表保存至 `artifacts/scalers/skewed_cols_config.pkl`，供 eICU 推理复用。  
- **缺失插补**：**MICE 多重插补**（`IterativeImputer`，max_iter=25，仅训练集 fit）；训练集与测试集均用同一拟合的 imputer 变换，eICU 外部验证时使用同一 imputer。  
- **标准化**：**Z-Score 标准化**（StandardScaler，仅训练集 fit）；训练/测试/eICU 均使用同一 scaler。  
- **产出**：`mimic_train_processed.csv`、`mimic_test_processed.csv`、`mimic_processed.csv`；以及 `mimic_mice_imputer.joblib`、`mimic_scaler.joblib`、`train_assets_bundle.pkl` 等，供 LASSO、模型训练与 eICU 部署使用。

#### 3.3.3 eICU 与开发队列对齐（08_eicu_alignment_cleaning.py）

- eICU 原始表与 MIMIC 特征空间对齐（列名、单位、时间窗已由 SQL 与 08 脚本保证）。  
- 使用开发阶段产出的 **deploy_bundle**（含 MICE、scaler、log 变换配置等）对 eICU 进行 **仅 transform、不 fit** 的预处理，得到 `eicu_processed_{pof,mortality,composite}.csv`，用于外部验证与 Table 1 的 eICU 列。

### 3.4 变量与结局定义

| 变量类型 | 内容 |
|----------|------|
| **预测变量** | 人口学、实验室、生命体征、器官支持等（详见 `artifacts/features/feature_dictionary.json`）；单位与展示名见特征字典，Table 1 与图表均由此统一。 |
| **结局** | POF、28-day mortality、Composite（二分类）；定义见 3.2.1 与 3.2.2。 |

### 3.5 特征选择

- **方法**：LASSO 回归，1-SE 准则
- **范围**：仅训练集
- **产出**：`artifacts/features/selected_features.json`

### 3.6 模型开发

| 算法 | 超参优化 | 概率校准 |
|------|----------|----------|
| XGBoost | Optuna | Platt scaling |
| Random Forest | Optuna | Platt scaling |
| SVM | Optuna | Platt scaling |
| Decision Tree | Optuna | Platt scaling |
| Logistic Regression | - | - |

**统一部署**：`deploy_bundle.pkl` 含 imputer、scaler、模型，确保 eICU 与 MIMIC 预处理一致。

### 3.7 内部验证

- 测试集盲测
- AUC、AUPRC、Brier、灵敏度、特异度
- ROC、校准曲线
- Youden 指数确定最优切点

### 3.8 外部验证

- eICU 盲测，无任何再拟合
- 使用 deploy_bundle 进行克隆式预处理
- 报告 AUC、95% CI、校准度、DCA

### 3.9 临床效用与可解释性

- **DCA**：净获益、Treat All/None 参考线
- **SHAP**：全局特征贡献、个体预测解释
- **列线图**：Logistic 回归系数、Bootstrap OR 及 95% CI

---

## 四、稿件图表对应

**路径说明**：新运行产出写入 `results/main/` 与 `results/supplementary/`；历史产出已归档于 `results/archive/main/`、`results/archive/supplementary/`。

### 4.1 主文表格（Main Text Tables）

| 表格 | 文件路径 | 内容 |
|------|----------|------|
| **Table 1** | `results/main/tables/Table1_baseline.csv` | 基线特征（MIMIC Non-POF/POF、eICU 外部验证、SMD） |
| **Table 2** | `results/main/tables/Table2_renal_subgroup.csv` | 肾亚组分析（如适用） |
| **Table 3** | `results/main/tables/Table3_performance.csv` | 内部验证效能（AUC、灵敏度、特异度、最优切点） |
| **Table 4** | `results/main/tables/Table4_external_validation.csv` | 外部验证效能（eICU） |

### 4.2 主文插图（Main Text Figures）

| 图号 | 文件路径 | 建议图注 |
|------|----------|----------|
| **Figure 1** | 流程图人数：运行 `01_mimic_flowchart_counts.sql`、`08_eicu_flowchart_counts.sql`；热图 `Fig1_missing_heatmap.png` | CONSORT 纳入排除流程图 + 缺失模式热图 |
| **Figure 2** | `results/main/figures/Fig2_ROC_external_{pof,mortality,composite}.png` | 外部验证 ROC 曲线（eICU，多模型对比） |
| **Figure 3** | `results/main/figures/Fig3_DCA_calibration_{target}.png` | DCA 与校准曲线（临床净获益） |
| **Figure 4** | `results/main/figures/Fig4_SHAP_summary_{target}.png` | SHAP 特征贡献摘要 |
| **Figure 5** | `results/main/figures/Fig5_nomogram_{target}_en.png` | 临床列线图 |

### 4.3 补充材料（Supplementary Materials）

| 类型 | 路径/内容 |
|------|-----------|
| **LASSO 路径** | `results/supplementary/figures/S1_lasso/` |
| **ROC/校准（内部）** | `results/supplementary/figures/S2_internal_ROC/{pof,mortality,composite}/` |
| **诊断四格表** | `results/supplementary/figures/S3_cutoff/` |
| **特征重要性** | `results/supplementary/figures/S3_cutoff/sci_feature_importance.png` |
| **森林图（OR）** | `results/main/figures/Fig5_forest_{target}_en.png` |
| **SHAP 依赖图** | `results/supplementary/figures/S5_interpretation/shap_values/Fig4C_Dep_*.png` |
| **漂移分析** | `validation/eicu_vs_mimic_drift.json`、`results/supplementary/figures/S4_comparison/dist_drift_*.png` |
| **OR 统计** | `results/supplementary/tables/OR_Statistics_{target}.csv`、`OR_Json_{target}.json` |

---

## 五、技术实现与复现

### 5.1 环境

- **Python**：3.10+（推荐 Nix + uv）
- **依赖**：`uv pip install -r requirements.txt`

### 5.2 一键运行（除 SQL）

```bash
cd <project_root>
uv run python run_all.py              # 全流程
uv run python run_all.py --skip-04   # 跳过可选审计
uv run python run_all.py --mimic-only   # 仅 MIMIC 阶段 (01-07)
uv run python run_all.py --eicu-only    # 仅 eICU+解释 (08-13)
```

**运行日志**：脚本通过 `utils.logger` 同时输出到控制台和日志文件。默认路径为 `logs/run.log`（追加写入）；可通过环境变量 `MIMIC_LOG_FILE` 指定其他路径，例如 `export MIMIC_LOG_FILE=/tmp/mimic_run.log`。

### 5.3 分步运行顺序

| 阶段 | 步骤 | 命令 |
|------|:---:|------|
| **SQL** | 01 | 在 MIMIC 执行 `scripts/sql/01_mimic_extraction.sql` |
| | 01a | 在 MIMIC 执行 `scripts/sql/01_mimic_flowchart_counts.sql`（Figure 1 流程图计数，需先执行 01） |
| **MIMIC** | 01 | `cd scripts/preprocess && uv run python 01_mimic_cleaning.py` |
| | 02 | `cd scripts/preprocess && uv run python 02_mimic_standardization.py` |
| | 03 | `cd scripts/audit_eval && uv run python 03_table1_baseline.py`（mimic-only 时；全流程在 08 后） |
| | 04 | `cd scripts/audit_eval && uv run python 04_mimic_stat_audit.py`（可选） |
| | 05 | `cd scripts/modeling && uv run python 05_feature_selection_lasso.py` |
| | 06 | `cd scripts/modeling && uv run python 06_model_training_main.py` |
| | 07 | `cd scripts/modeling && uv run python 07_optimal_cutoff_analysis.py` |
| **SQL** | 08 | 在 eICU 执行 `scripts/sql/08_eicu_extraction.sql`（详见 [eICU 提取指南](docs/EICU_EXTRACTION_GUIDE.md)） |
| | 08a | 在 eICU 执行 `scripts/sql/08_eicu_flowchart_counts.sql`（Figure 1 流程图计数，需先执行 08） |
| **eICU** | 08 | `cd scripts/preprocess && uv run python 08_eicu_alignment_cleaning.py` |
| | 03 | `cd scripts/audit_eval && uv run python 03_table1_baseline.py`（全流程：含 eICU 列） |
| | 09 | `cd scripts/audit_eval && uv run python 09_cross_cohort_audit.py` |
| | 10 | `cd scripts/audit_eval && uv run python 10_external_validation_perf.py` |
| **解释** | 11 | `cd scripts/audit_eval && uv run python 11_model_interpretation_shap.py` |
| | 12 | `cd scripts/audit_eval && uv run python 12_clinical_calibration_dca.py` |
| | 13 | `cd scripts/audit_eval && uv run python 13_nomogram_odds_ratio.py` |

### 5.4 项目目录树

```
<project_root>/
├── .python-version
├── run_all.py                  # 一键运行（除 SQL）
├── requirements.txt
│
├── data/
│   ├── raw/                    # 原始数据（SQL 产出）
│   │   ├── mimic_raw_data.csv          ← Step 01 SQL
│   │   └── eicu_raw_data.csv           ← Step 08 SQL
│   │
│   ├── cleaned/                # MIMIC 中间产物
│   │   ├── mimic_raw_scale.csv         ← Step 01
│   │   ├── mimic_train_processed.csv   ← Step 02
│   │   ├── mimic_test_processed.csv    ← Step 02
│   │   └── mimic_processed.csv         ← Step 02（合并）
│   │
│   └── external/               # eICU 验证产物
│       ├── eicu_processed_pof.csv      ← Step 08
│       ├── eicu_processed_mortality.csv
│       ├── eicu_processed_composite.csv
│       └── eicu_raw_scale.csv
│
├── docs/                       # 文档
│   ├── EICU_EXTRACTION_GUIDE.md       # eICU 数据提取指南
│   ├── EXTRACTION_ALIGNMENT.md         # MIMIC vs eICU 提取逻辑对齐
│   ├── FLOWCHART_COUNTS.md             # Figure 1 流程图纳入排除计数
│   ├── PF_RATIO_ANALYSIS.md            # P/F ratio 提取差异分析
│   └── TABLE1_FLOW.md                  # Table 1 生成流程
│
├── artifacts/
│   ├── features/
│   │   ├── feature_dictionary.json     # 特征定义与展示名
│   │   └── selected_features.json      ← Step 05 LASSO
│   │
│   ├── scalers/                 # Step 02 产出
│   │   ├── mimic_mice_imputer.joblib
│   │   ├── mimic_scaler.joblib
│   │   ├── skewed_cols_config.pkl
│   │   └── train_assets_bundle.pkl
│   │
│   └── models/                  # Step 06 产出
│       ├── pof/
│       │   ├── deploy_bundle.pkl
│       │   ├── all_models_dict.pkl
│       │   ├── eval_data.pkl
│       │   ├── feature_importance.csv
│       │   ├── thresholds.json
│       │   └── ...
│       ├── mortality/
│       │   └── ...
│       └── composite/
│           └── ...
│
├── scripts/
│   ├── sql/
│   │   ├── 01_mimic_extraction.sql
│   │   ├── 01_mimic_flowchart_counts.sql   # Figure 1 流程图计数（MIMIC）
│   │   ├── 08_eicu_extraction.sql
│   │   └── 08_eicu_flowchart_counts.sql   # Figure 1 流程图计数（eICU）
│   │
│   ├── preprocess/
│   │   ├── 01_mimic_cleaning.py
│   │   ├── 02_mimic_standardization.py
│   │   └── 08_eicu_alignment_cleaning.py
│   │
│   ├── modeling/
│   │   ├── 05_feature_selection_lasso.py
│   │   ├── 06_model_training_main.py
│   │   └── 07_optimal_cutoff_analysis.py
│   │
│   ├── audit_eval/
│   │   ├── 03_table1_baseline.py
│   │   ├── 04_mimic_stat_audit.py
│   │   ├── 09_cross_cohort_audit.py
│   │   ├── 10_external_validation_perf.py
│   │   ├── 11_model_interpretation_shap.py
│   │   ├── 12_clinical_calibration_dca.py
│   │   └── 13_nomogram_odds_ratio.py
│   │
│   └── utils/
│       ├── study_config.py      # 终点、常量 (MISSING_THRESHOLD, SPLIT_SEED)
│       ├── paths.py             # 统一路径 (get_project_root, get_raw_path 等)
│       ├── logger.py            # 统一日志 (log, log_header)
│       ├── outcome_utils.py     # 结局对齐 (早期死亡覆盖、列重命名)
│       ├── feature_formatter.py
│       ├── feature_formatter_config.py
│       ├── table1_config.py     # Table 1 配置与脚注
│       ├── deploy_utils.py      # 模型部署与推理
│       ├── plot_config.py
│       └── plot_utils.py
│
├── results/
│   ├── main/                   # 主文（投稿用）
│   │   ├── tables/
│   │   │   ├── Table1_baseline.csv           # 基线特征（MIMIC Non-POF/POF、eICU、SMD）
│   │   │   ├── Table2_renal_subgroup.csv     # 肾亚组分析
│   │   │   ├── Table3_performance.csv        # 内部验证效能（AUC、灵敏度、特异度、最优切点）
│   │   │   └── Table4_external_validation.csv # 外部验证效能（eICU）
│   │   └── figures/
│   │       ├── Fig1_missing_heatmap.{pdf,png}           # 缺失模式热图
│   │       ├── Fig2_ROC_external_{pof,mortality,composite}.{pdf,png}  # 外部验证 ROC
│   │       ├── Fig3_DCA_calibration_{pof,mortality,composite}.{pdf,png} # DCA 与校准曲线
│   │       ├── Fig4_SHAP_summary_{pof,mortality,composite}.{pdf,png}   # SHAP 特征贡献摘要
│   │       ├── Fig5_nomogram_{pof,mortality,composite}_en.{pdf,png}   # 列线图
│   │       └── Fig5_forest_{pof,mortality,composite}_en.{pdf,png}     # 森林图（OR）
│   │
│   ├── supplementary/          # 补充材料
│   │   ├── tables/
│   │   │   ├── OR_Statistics_{pof,mortality,composite}.csv   # OR 统计
│   │   │   ├── OR_Json_{pof,mortality,composite}.json       # OR JSON
│   │   │   ├── DCA_Data_{pof,mortality,composite}.csv        # DCA 净获益数据
│   │   │   └── Table3_Perf_{pof,mortality,composite}.csv    # Table 3 明细（按终点）
│   │   └── figures/
│   │       ├── S1_lasso/                      # LASSO 路径
│   │       │   ├── lasso_diag_{pof,mortality,composite}.{pdf,png}
│   │       │   └── lasso_importance_{pof,mortality,composite}.{pdf,png}
│   │       ├── S2_internal_ROC/               # 内部 ROC/校准
│   │       │   ├── pof/{pof_ROC,pof_Calibration}.{pdf,png}
│   │       │   ├── mortality/{mortality_ROC,mortality_Calibration}.{pdf,png}
│   │       │   └── composite/{composite_ROC,composite_Calibration}.{pdf,png}
│   │       ├── S3_cutoff/                     # 诊断四格表与特征重要性
│   │       │   ├── {pof,mortality,composite}/07_Diagnostic_*.{pdf,png}
│   │       │   ├── sci_feature_importance.{pdf,png}
│   │       │   └── sci_forest_plot.{pdf,png}
│   │       ├── S4_comparison/                  # 漂移分析（MIMIC vs eICU）
│   │       │   ├── dist_drift_{feature}_{outcome}.{pdf,png}  # 特征分布漂移图
│   │       │   └── Table4_Performance_Visualization.{pdf,png}
│   │       └── S5_interpretation/             # SHAP 可解释性
│   │           └── shap_values/
│   │               ├── Fig4B_Force_{pof,mortality,composite}.{pdf,png}   # 个体 SHAP
│   │               ├── Fig4C_Dep_*_{feature}.{pdf,png}                   # 依赖图
│   │               ├── SHAP_Data_Export_{pof,mortality,composite}.csv
│   │               └── SHAP_BaseValue_*.txt
│   │
│   └── MANIFEST.md             # 稿件图表索引（可选，见 results/MANIFEST.md）
│
└── validation/
    └── eicu_vs_mimic_drift.json     # Step 10 漂移分析
```

---

## 六、关键方法学说明

### 6.1 防泄露

- **Step 03**：先划分 train/test，再在训练集上 fit MICE 与 Scaler
- **Step 05**：LASSO 仅用训练集
- **Step 09**：eICU 使用 deploy_bundle，无任何再拟合

### 6.2 Table 1 基线表规范

Table 1 由 `scripts/audit_eval/03_table1_baseline.py` 生成，符合医学论文规范：

- **结构分组**：Demographics & Comorbidities → Severity Scores at Admission → Organ Support → Laboratory Parameters → Outcomes
- **SMD (POF vs Non-POF)**：MIMIC 内部组间均衡性，SMD > 0.1 表示存在显著差异
- **SMD (MIMIC vs eICU)**：开发队列与外部验证队列的分布漂移，量化人群差异
- **单位统一**：实验室指标附单位（如 Creatinine, mg/dL; Lactate, mmol/L）
- **分类变量合并**：仅显示阳性类别（如 Male n(%)、Mechanical ventilation n(%)）
- **eICU 外部验证列**：展示人群漂移，反衬模型泛化能力（P/F ratio 与 MIMIC 定义一致，均仅用 PaO2）
- **Outcomes**：POF 发生率 + 28-day death，eICU 列补充完整
- **脚注**：Abbreviations、SMD 释义、eICU 缺失项说明（— 表示未采集，经跨中心对齐处理）

配置：`scripts/utils/table1_config.py`

### 6.3 特征名称统一

所有输出均从 `artifacts/features/feature_dictionary.json` 读取 `display_name_en` / `display_name_cn`，保证全文一致。

**切换语言**：修改 `scripts/utils/feature_formatter_config.py` 中 `DISPLAY_LANG`（`'en'` / `'cn'`）。

### 6.4 图片规范

- DPI：600（投稿）
- 尺寸：双栏 7.2 in，单栏 3.5 in
- 格式：PDF + PNG
- 配置：`scripts/utils/plot_config.py`

---

## 七、TRIPOD 对照（建议）

| TRIPOD 条目 | 对应内容 |
|-------------|----------|
| 标题/摘要 | 见 一、1.1–1.2 |
| 背景/目标 | 见 一、1.3 与 三、3.1 |
| 数据源 | 三、3.1（MIMIC-IV v3.1、eICU） |
| 参与者 | 三、3.2（数据提取与纳入排除） |
| 结局 | 三、3.2.1 / 3.2.2、3.4 |
| 预测变量 | 三、3.4，feature_dictionary.json |
| 缺失数据 | 三、3.3（MICE，仅训练集 fit） |
| 样本量 | Table 1、Figure 1 流程图 |
| 模型开发 | 三、3.5–3.6（LASSO + 多算法 + 校准） |
| 模型验证 | 三、3.7–3.8（内部测试集 + 外部 eICU） |
| 模型性能 | Table 3、Table 4 |
| 临床效用 | 三、3.9（DCA、列线图） |
| 局限性 | 回顾性、单次外部验证等 |
| 可解释性 | SHAP、森林图 |

---

## 八、引用与致谢

若使用 MIMIC-IV v3.1 / eICU，请按官方要求完成培训与数据使用协议，并在稿件中引用相应文献。
