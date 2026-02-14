# Methods（方法）

> 适用于 Critical Care 投稿，基于项目脚本与 README 撰写。报告符合 TRIPOD 规范。正文表格见 `docs/tables/main/`，补充表格见 `docs/tables/supplementary/`。

---

## 研究设计与数据来源

本研究为回顾性多中心队列研究，旨在开发并外部验证重症急性胰腺炎（severe acute pancreatitis, SAP）患者持续性器官功能衰竭（persistent organ failure, POF）及死亡风险的预测模型。开发队列来自 MIMIC-IV 数据库（v3.1），按 80% : 20% 比例划分训练集与内部验证集（约 876 例训练、219 例测试，按 composite 分层）；外部验证队列来自 eICU 协作研究数据库（Collaborative Research Database），完全独立、盲测。主要终点为 POF，次要终点为 28 天死亡率，混合终点为 Composite（POF 或 28 天死亡的任一发生）。

MIMIC-IV 与 eICU 的使用均已获得相应数据库官方培训与数据使用协议批准，引用方式遵循其要求。

---

## 纳入排除标准与流程图

### MIMIC-IV 开发队列

急性胰腺炎（AP）由住院诊断编码界定：ICD-9 为 5770，ICD-10 为 K85%。纳入排除顺序采用先 LOS 再首次的临床逻辑：（1）在 AP 相关住院的 ICU 入住中，仅保留 ICU 住院时间（LOS）≥ 24 小时的入住；（2）在上述入住中，按患者首次 ICU 入住（按入室时间排序取第一条）保留一例；（3）年龄 ≥ 18 岁；（4）关键生理指标（肌酐、尿素氮、乳酸、P/F 比、pH、白细胞、血红蛋白、胆红素、钠、白蛋白）缺失比例 ≤ 80%。流程图人数依次为：初始池 1686 人 → 排除 LOS < 24h 后 1405 人 → 排除非首次入住后 1100 人 → 排除年龄 < 18 岁后 1100 人 → 排除关键指标缺失后最终 1095 人。

### eICU 外部验证队列

AP 由诊断文本包含 “pancreatit” 且不包含 “chronic” 界定。纳入排除顺序与 MIMIC 一致；此外排除关键生理指标缺失 > 80% 或 POF 无法判定者。流程图人数依次为：初始池 1658 人 → 排除 LOS < 24h 后 1322 人 → 排除非首次入住后 1143 人 → 排除年龄 < 18 岁后 1142 人 → 排除关键指标缺失或 POF 无法判定后最终 1112 人。

**MIMIC-IV Flowchart Counts（Figure 1）**

| 步骤 | 说明 | 剩余人数 | 排除人数 |
|------|------|----------|----------|
| 初始池 | AP 诊断的 ICU 入住 | 1686 | — |
| 排除1 | ICU LOS < 24 h（先 LOS） | 1405 | 281 |
| 排除2 | 非首次 ICU（保留首次，再首次） | 1100 | 305 |
| 排除3 | 年龄 < 18 岁 | 1100 | 0 |
| 排除4 | 关键生理指标缺失 > 80% | 1095 | 5 |
| **最终纳入** | 开发队列 | **1095** | — |

**eICU Flowchart Counts（Figure 1）**

| 步骤 | 说明 | 剩余人数 | 排除人数 |
|------|------|----------|----------|
| 初始池 | AP 诊断的 ICU 入住 | 1658 | — |
| 排除1 | ICU LOS < 24 h（先 LOS） | 1322 | 336 |
| 排除2 | 非首次 ICU（保留首次，再首次） | 1143 | 179 |
| 排除3 | 年龄 < 18 岁 | 1142 | 1 |
| 排除4 | 关键指标缺失 > 80% 或 POF 无法判定 | 1112 | 30 |
| **最终纳入** | 外部验证队列 | **1112** | — |

Figure 1 展示纳入排除流程图（CONSORT）及缺失模式热图。CONSORT 流程图见 `docs/figures/main/Fig1_flowchart.svg`，缺失热图见 `docs/figures/main/Fig1_missing_heatmap.{pdf,png}`。人数由 `scripts/sql/01_mimic_flowchart_counts.sql` 与 `08_eicu_flowchart_counts.sql` 产出。

---

## 时间窗与预测变量

### 时间窗

基线特征的时间窗如下：实验室指标取住院前 6 小时至 ICU 入室后 24 小时（labevents）；血气、生命体征、葡萄糖/乳酸/SpO2 斜率、机械通气与血管加压药取 ICU 入室至入室后 24 小时（chartevents、first_day_bg 等）。

eICU 的时间窗与 MIMIC 对齐：实验室、血气、生命体征、P/F 比、葡萄糖/乳酸/SpO2 斜率、首日机械通气与血管加压药均限定在入 ICU 后 0–1440 分钟（0–24 小时）。

### 预测变量

预测变量包括人口学特征、合并症、入住时严重程度评分（SOFA、APS-III、SAPS-II、OASIS、LODS）、器官支持（机械通气、血管加压药）及实验室参数（白细胞、血红蛋白、血小板、尿素氮、肌酐、胆红素、转氨酶、乳酸、P/F 比、SpO2、pH、电解质、碳酸氢盐等）。Table 1（`docs/tables/main/Table1_baseline.csv`）变量分组为：Demographics & Comorbidities → Severity Scores at Admission → Organ Support → Laboratory Parameters → Outcomes。变量定义、单位与展示名由特征字典统一。

### 结局定义

- **POF**：基于 SOFA 评分，在 ICU 入室后 24 小时至 7 天内，呼吸、心血管或肾脏任一器官 SOFA ≥ 2 且持续 ≥ 2 天则记为 POF。eICU 无 SOFA，在入室后 1–7 天时间窗内，采用机械通气、血管加压药、透析、P/F < 300 mmHg、肌酐 > 1.9 mg/dL 等规则构造 POF 代理。
- **28 天死亡**：院内死亡时间或患者死亡日期 ≤ 入室后 28 天。
- **24–48 小时早期死亡**：若死亡发生在入室后 24–48 小时，则结局覆盖为 POF=1、mortality=1，用于竞争风险处理。

P/F 比在 MIMIC 与 eICU 均仅使用动脉血气 PaO2/FiO2；eICU 中无 PaO2 时记为 NULL。

---

## 数据清洗与预处理

**防泄露原则**：严格先划分训练集与测试集，再在训练集上拟合所有变换；eICU 仅使用开发阶段产出的 deploy_bundle，无任何再拟合。

### 第一步：临床特征审计与清洗（01_mimic_cleaning.py）

对 24–48 小时内死亡者统一标记为 POF=1、mortality=1。除白名单变量（如 subject_id、stay_id、pof、mortality、composite、lactate_max、pao2fio2ratio_min 等）外，缺失率 > 30% 的特征予以剔除（阈值见 `study_config.py`）。依据特征字典中的生理范围（ref_range）对超出生理合理范围的值置为缺失；必要时应用单位换算。对数值型特征（排除二分类结局及白名单）做 1%–99% 分位数盖帽，减轻极端值影响。

### 第二步：划分、变换与标准化（02_mimic_standardization.py）

按 composite 分层（若缺失则按 mortality），80% 训练 / 20% 测试，随机种子为 42；仅对结局非缺失的样本参与划分，划分前不进行插补。对特征字典中标记为需对数变换的变量做 log1p 变换。采用 MICE 多重插补（IterativeImputer，max_iter=25，random_state=42），仅训练集拟合；训练集与测试集均用同一拟合的 imputer 变换。采用 Z-Score 标准化（StandardScaler），仅训练集拟合；训练/测试/eICU 均使用同一 scaler。

### eICU 对齐（08_eicu_alignment_cleaning.py）

eICU 原始表与 MIMIC 特征空间对齐（列名、单位、时间窗已由 SQL 与脚本保证）。使用 deploy_bundle 对 eICU 进行仅 transform、不 fit 的预处理，得到外部验证用数据。

---

## 特征选择

采用 LASSO 回归（LogisticRegressionCV，1-SE 准则，max_iter=1000，random_state=42）进行特征选择，仅在训练集上执行。各终点（POF、28 天死亡、Composite）分别得到 12 个入选特征；入选特征见 Supplementary Table S2（`docs/tables/supplementary/ST2_LASSO_selected_features.csv`）。主要终点 POF 关键特征包括 creatinine_max、pao2fio2ratio_min、ph_min 等；次要终点 28 天死亡包括 admission_age、bun_max、lactate_max、ptt_min 等。详见补充材料 S1。

---

## 模型开发

采用五种算法：XGBoost、随机森林（RF）、支持向量机（SVM）、决策树（DT）、逻辑回归（LR）。除逻辑回归外，其余算法均采用 Optuna 进行超参数优化（XGBoost 与 RF 各 n_trials=100，SVM 与 DT 各 n_trials=50）。SVM 与逻辑回归均设置 class_weight='balanced'。对输出概率采用 Platt scaling 进行校准。统一部署包 deploy_bundle.pkl 含 imputer、scaler、模型，用于 eICU 盲测。

---

## 内部验证

在 MIMIC 测试集（20%）上进行盲测。评价指标包括 AUC、AUPRC、Brier 评分、灵敏度、特异度；采用 Youden 指数确定最优切点。详见 Table 3（`docs/tables/main/Table3_performance.csv`）及 Supplementary Table S5（`docs/tables/supplementary/ST5_Table3_per_endpoint.csv`）、补充材料 S2。

---

## 外部验证

eICU 队列使用 deploy_bundle 进行预处理，无任何再拟合。报告 AUC（95% CI）、Brier、灵敏度、特异度、AUPRC 及阈值。跨队列分布漂移分析见 Supplementary Table S3（`docs/tables/supplementary/ST3_drift_analysis.csv`）及补充材料 S4。

作为附加分析，我们增加了精简版 XGBoost 外部验证（`scripts/audit_eval/10b_external_validation_slim.py`）：基于开发集内特征重要性排序固定特征数（POF=3、28 天死亡=8、Composite=4），在 MIMIC 训练集重训并校准精简模型，在 MIMIC 测试集用 Youden 指数确定阈值后固定到 eICU 队列评估。结果单独输出为 `results/main/tables/Table4_external_validation_slim.csv`，不替代主分析的多模型 Table 4。

---

## 临床效用与可解释性

采用决策曲线分析（DCA）评估净获益，与 Treat All、Treat None 参考线比较。采用 SHAP（SHapley Additive exPlanations）进行全局特征贡献与个体预测解释。基于 logistic 回归系数构建列线图，并报告 Bootstrap odds ratio（OR）及 95% CI。OR 统计见 Supplementary Table S1（`docs/tables/supplementary/ST1_OR_*.csv`）。详见 Figure 3、Figure 4、Figure 5 及补充材料 S5。

此外，部署端 `deploy_min/app.py` 支持单病例推理与单例 SHAP 面板（waterfall 与 force 图），并统一调用特征字典展示名（含单位）以保证临床显示一致性。该功能用于研究演示，不替代主文统计分析。

**软件与环境**：Python 3.10+，scikit-learn，XGBoost，Optuna，SHAP；复现所需依赖见 `requirements.txt`。
