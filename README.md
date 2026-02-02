# 重症急性胰腺炎预测模型：基于 MIMIC-IV v3.1 与 eICU 的机器学习开发与外部验证

> 本研究流程支持转化为医学 SCI 论文。本文档提供方法学详述、产出与稿件图表对应关系，便于撰写 Methods、Results 及 Supplementary Materials。

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

## 二、方法学（Methods）

### 2.1 数据来源与纳入排除

| 数据源 | 版本 | 用途 |
|--------|------|------|
| **MIMIC-IV** | v3.1 | 开发队列：训练 + 内部验证 |
| **eICU** | Collaborative Research Database | 外部验证队列 |

**纳入/排除**：由 `scripts/sql/01_mimic_extraction.sql` 与 `08_eicu_extraction.sql` 定义，包括 SAP 诊断标准、ICU 入住时长、结局可及性等。

**eICU 与 MIMIC 特征对齐**：08 SQL 已按 MIMIC 结构补齐以下特征：
- **人口学/合并症**：Weight、CHF、CKD（diagnosis 表）
- **实验室**：Hgb、Plt、Bili、ALT、AST、SpO2、HCO3、Na、K、Cl、Ca、Amylase、Lipase、CRP、Phosphate、D-dimer、Fibrinogen、LDH、Triglycerides、Cholesterol、Neutrophils、Lymphocytes、NLR
- **血气/趋势**：P/F ratio (min/max)、Lactate (min/max)、Glucose/Lactate/SpO2 slope、LAR
- **P/F ratio 定义**：MIMIC 与 eICU 均仅使用动脉血气 PaO2/FiO2；eICU 无 PaO2 时设为 NULL（详见 [P/F 提取对齐说明](docs/PF_RATIO_ANALYSIS.md)）

09 脚本保留 `TABLE1_BASELINE_COLS` 确保 eicu_raw_scale 含完整基线列。

### 2.2 变量与结局定义

| 变量类型 | 内容 |
|----------|------|
| **预测变量** | 人口学、实验室、生命体征、器官支持等（详见 `artifacts/features/feature_dictionary.json`） |
| **结局** | POF、28-day mortality、Composite（二分类） |

### 2.3 数据预处理

| 步骤 | 操作 | 防泄露措施 |
|------|------|------------|
| **划分** | 按 composite 分层 80/20，random_state=42 | 先划分再拟合 |
| **缺失** | MICE 插补（仅训练集 fit） | 测试集/验证集使用训练集参数 |
| **偏态** | 对数变换（skewed 变量） | 仅对训练集 fit 的列 |
| **标准化** | Z-Score（仅训练集 fit） | 验证集使用训练集 mean/std |

**产出**：`mimic_train_processed.csv`、`mimic_test_processed.csv`、`eicu_processed_{target}.csv`

### 2.4 特征选择

- **方法**：LASSO 回归，1-SE 准则
- **范围**：仅训练集
- **产出**：`artifacts/features/selected_features.json`

### 2.5 模型开发

| 算法 | 超参优化 | 概率校准 |
|------|----------|----------|
| XGBoost | Optuna | Platt scaling |
| Random Forest | Optuna | Platt scaling |
| SVM | Optuna | Platt scaling |
| Decision Tree | Optuna | Platt scaling |
| Logistic Regression | - | - |

**统一部署**：`deploy_bundle.pkl` 含 imputer、scaler、模型，确保 eICU 与 MIMIC 预处理一致。

### 2.6 内部验证

- 测试集盲测
- AUC、AUPRC、Brier、灵敏度、特异度
- ROC、校准曲线
- Youden 指数确定最优切点

### 2.7 外部验证

- eICU 盲测，无任何再拟合
- 使用 deploy_bundle 进行克隆式预处理
- 报告 AUC、95% CI、校准度、DCA

### 2.8 临床效用与可解释性

- **DCA**：净获益、Treat All/None 参考线
- **SHAP**：全局特征贡献、个体预测解释
- **列线图**：Logistic 回归系数、Bootstrap OR 及 95% CI

---

## 三、稿件图表对应

### 3.1 主文表格（Main Text Tables）

| 表格 | 文件路径 | 内容 |
|------|----------|------|
| **Table 1** | `results/main/tables/Table1_baseline.csv` | 基线特征（MIMIC Non-POF/POF、eICU 外部验证、SMD） |
| **Table 2** | `results/main/tables/Table2_renal_subgroup.csv` | 肾亚组分析（如适用） |
| **Table 3** | `results/main/tables/Table3_performance.csv` | 内部验证效能（AUC、灵敏度、特异度、最优切点） |
| **Table 4** | `results/main/tables/Table4_external_validation.csv` | 外部验证效能（eICU） |

### 3.2 主文插图（Main Text Figures）

| 图号 | 文件路径 | 建议图注 |
|------|----------|----------|
| **Figure 1** | `results/main/figures/Fig1_missing_heatmap.png` | 缺失模式热图 |
| **Figure 2** | `results/main/figures/Fig2_ROC_external_{pof,mortality,composite}.png` | 外部验证 ROC 曲线（eICU，多模型对比） |
| **Figure 3** | `results/main/figures/Fig3_DCA_calibration_{target}.png` | DCA 与校准曲线（临床净获益） |
| **Figure 4** | `results/main/figures/Fig4_SHAP_summary_{target}.png` | SHAP 特征贡献摘要 |
| **Figure 5** | `results/main/figures/Fig5_nomogram_{target}_en.png` | 临床列线图 |

### 3.3 补充材料（Supplementary Materials）

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

## 四、技术实现与复现

### 4.1 环境

- **Python**：3.10+（推荐 Nix + uv）
- **依赖**：`uv pip install -r requirements.txt`

### 4.2 一键运行（除 SQL）

```bash
cd <project_root>
uv run python run_all.py              # 全流程
uv run python run_all.py --skip-04   # 跳过可选审计
uv run python run_all.py --mimic-only   # 仅 MIMIC 阶段 (01-07)
uv run python run_all.py --eicu-only    # 仅 eICU+解释 (08-13)
```

### 4.3 分步运行顺序

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

### 4.4 项目目录树

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

## 五、关键方法学说明

### 5.1 防泄露

- **Step 03**：先划分 train/test，再在训练集上 fit MICE 与 Scaler
- **Step 05**：LASSO 仅用训练集
- **Step 09**：eICU 使用 deploy_bundle，无任何再拟合

### 5.2 Table 1 基线表规范

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

### 5.3 特征名称统一

所有输出均从 `artifacts/features/feature_dictionary.json` 读取 `display_name_en` / `display_name_cn`，保证全文一致。

**切换语言**：修改 `scripts/utils/feature_formatter_config.py` 中 `DISPLAY_LANG`（`'en'` / `'cn'`）。

### 5.4 图片规范

- DPI：600（投稿）
- 尺寸：双栏 7.2 in，单栏 3.5 in
- 格式：PDF + PNG
- 配置：`scripts/utils/plot_config.py`

---

## 六、TRIPOD 对照（建议）

| TRIPOD 条目 | 对应内容 |
|-------------|----------|
| 标题/摘要 | 见 1.1、1.2 |
| 背景/目标 | 见 1.3、2.1 |
| 数据源 | MIMIC-IV v3.1、eICU |
| 参与者 | SQL 纳入排除 |
| 结局 | POF、mortality、composite |
| 预测变量 | feature_dictionary.json |
| 缺失数据 | MICE |
| 样本量 | Table1_baseline |
| 模型开发 | LASSO + 多算法 + 校准 |
| 模型验证 | 内部 20% 测试集 + 外部 eICU |
| 模型性能 | Table 3、Table 4 |
| 临床效用 | DCA、列线图 |
| 局限性 | 回顾性、单次验证等 |
| 可解释性 | SHAP、森林图 |

---

## 七、引用与致谢

若使用 MIMIC-IV v3.1 / eICU，请按官方要求完成培训与数据使用协议，并在稿件中引用相应文献。
