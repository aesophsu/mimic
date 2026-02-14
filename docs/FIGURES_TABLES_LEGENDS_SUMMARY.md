# 稿件图表说明汇总（Figure & Table Legends）

> 本文档汇总所有图片和表格的说明（legend），便于复制、修改后用于论文投稿。
> 原始说明文件位于各图表同目录下的同名 `.md` 文件中。

---

## 一、主文表格（Main Tables）

### Table1_baseline

```
# Table 1 基线特征

**Table 1.** 急性胰腺炎（AP）患者开发队列与外部验证队列基线特征。MIMIC-IV 开发队列（n=1095）按持续性器官功能衰竭（POF）分组：Non-POF 652 例，POF 443 例；eICU 外部验证队列 1112 例。列含 Overall/Non-POF/POF/eICU、P 值、标准化均数差（SMD，POF vs Non-POF）、SMD（MIMIC vs eICU）。变量分组：人口学与合并症（Demographics & Comorbidities）、入住时严重程度评分（SOFA、APS-III、SAPS-II、OASIS、LODS）、器官支持（机械通气、血管加压药）、实验室参数。连续变量以中位数[四分位距]表示，分类变量以 n（%）表示。eICU 队列中 SOFA、APS-III、SAPS-II、OASIS、LODS、ALP 未采集，表内以—表示。SMD > 0.1 表示组间存在显著不平衡。
```

---

### Table2_renal_subgroup

```
# Table 2 肾亚组分析

**Table 2.** MIMIC 开发队列按慢性肾脏病（CKD）分层的肾亚组基线特征与结局。有肾损害亚组（CKD=1）490 例，无肾损害亚组（CKD=0）605 例。有肾亚组 POF 发生率 66.3%，28 天全因死亡率 24.5%；无肾亚组分别为 26.9%、8.1%。两亚组在年龄、SOFA、APS-III、SAPS-II、OASIS、LODS、血尿素氮、血肌酐、乳酸、氧合指数（P/F 比）、pH、碳酸氢盐等均有显著差异（P<0.001）。缩写同 Table 1。
```

---

### Table3_performance

```
# Table 3 内部验证效能

**Table 3.** MIMIC 测试集（20%）内部验证效能。列含终点（Primary POF、Secondary 28 天死亡、Composite）、人群（Full Population、Subgroup No Renal）、模型、AUC（95% 置信区间）、灵敏度、特异度、F1、最优切点。主要终点 POF：SVM AUC 0.853（0.795–0.903），灵敏度 0.897，特异度 0.720；次要终点 28 天死亡：逻辑回归 AUC 0.876；混合终点 Composite：SVM AUC 0.881。无肾亚组各终点 AUC 较全人群下降（POF 最佳约 0.77，Mortality 约 0.80），提示肾功能相关指标对预测有贡献。
```

---

### Table4_external_validation

```
# Table 4 外部验证效能

**Table 4.** eICU 外部验证队列盲测效能。列含终点（Primary POF、Secondary 28 天死亡、Composite）、算法、AUC（95% CI）、Brier 评分、灵敏度、特异度、AUPRC、阈值。主要终点 POF：Random Forest AUC 0.859（0.837–0.881），AUPRC 0.822；SVM 0.849，XGBoost 0.847；Logistic Regression 0.736（泛化较差）。次要终点 28 天死亡：XGBoost AUC 0.846。混合终点 Composite：Random Forest AUC 0.864。Brier 评分 POF/Composite 约 0.16–0.21，Mortality 约 0.06–0.07。
```

---

### Table4_external_validation_slim

```
# Table 4b 精简版外部验证效能（附加分析）

**Table 4b.** eICU 外部验证队列中精简版 XGBoost 的附加分析结果。特征数固定为 POF=3、28 天死亡=8、Composite=4（按开发集内特征重要性排序确定）。列含终点、算法（XGBoost-Slim）、K、特征组合、AUC（95% CI）、AUPRC、Brier、灵敏度、特异度、阈值。该表用于量化“模型精简后”与主分析（Table 4）的性能差异，不替代主分析多模型比较结果。
```

---

## 二、主文图形（Main Figures）

### Fig1_flowchart

```
# Figure 1A CONSORT 流程图

**Figure 1A.** CONSORT 纳入排除流程图。MIMIC-IV 开发队列：初始池 1686 例 AP 诊断的 ICU 入住 → 排除 ICU LOS<24 h 后 1405 例 → 排除非首次 ICU 入住后 1100 例 → 排除年龄<18 岁后 1100 例 → 排除关键生理指标缺失>80% 后最终 1095 例。eICU 外部验证队列：初始池 1658 例 → 1322 → 1143 → 1142 → 最终 1112 例。AP，急性胰腺炎；LOS，住院时长。关键生理指标包括血肌酐、血尿素氮、乳酸、氧合指数（PaO2/FiO2）、pH、白细胞、血红蛋白、胆红素、钠、白蛋白。
```

---

### Fig1_missing_heatmap

```
# Figure 1B 缺失模式热图

**Figure 1B.** 关键生理指标缺失模式热图。展示 MIMIC-IV 开发队列与 eICU 外部验证队列中纳入排除标准所涉变量的缺失比例及分布模式。行通常为患者或样本，列为变量（如血肌酐、血尿素氮、乳酸、氧合指数、pH、白细胞、血红蛋白、胆红素、钠、白蛋白等）。缺失比例>80% 的样本已被排除；热图用于评估剩余样本的缺失模式及多中心间一致性。
```

---

### Fig2_ROC_external_composite

```
# Figure 2 外部验证 ROC 曲线（Composite）

**Figure 2（混合终点 Composite）.** eICU 外部验证队列各模型 ROC 曲线。混合终点（POF 或 28 天死亡）：Random Forest AUC 0.864（95% CI 0.844–0.884），AUPRC 0.839；XGBoost 0.859，SVM 0.845，Decision Tree 0.830，Logistic Regression 0.741。Brier 评分约 0.15–0.21。总体而言，模型在 eICU 外部验证队列中保持较好区分能力。
```

---

### Fig2_ROC_external_mortality

```
# Figure 2 外部验证 ROC 曲线（28 天死亡）

**Figure 2（次要终点 28 天死亡）.** eICU 外部验证队列各模型 ROC 曲线。次要终点 28 天全因死亡率：XGBoost AUC 0.846（95% CI 0.809–0.881），Random Forest 0.836，SVM 0.836，Logistic Regression 0.810，Decision Tree 0.779。Brier 评分约 0.06–0.07。模型在 eICU 外部验证队列中保持较好区分能力。
```

---

### Fig2_ROC_external_pof

```
# Figure 2 外部验证 ROC 曲线（POF）

**Figure 2（主要终点 POF）.** eICU 外部验证队列各模型 ROC 曲线。主要终点持续性器官功能衰竭（POF）：Random Forest AUC 0.859（95% CI 0.837–0.881），AUPRC 0.822；SVM 0.849，XGBoost 0.847，Decision Tree 0.821，Logistic Regression 0.736（泛化较差）。Brier 评分 0.16–0.21。机器学习模型在外部队列中相对稳定（SVM 内部 0.853 → 外部 0.849）。
```

---

### Fig3_DCA_calibration_composite

```
# Figure 3 DCA 与校准曲线（Composite）

**Figure 3（混合终点 Composite）.** 决策曲线分析（DCA）与校准曲线。混合终点（POF 或 28 天死亡）下，预测模型在阈值 0–1 合理范围内较 Treat All 和 Treat None 策略具有净获益。DCA 展示各模型净获益；校准曲线评估预测概率与观察频率的一致性。具体净获益数值见 Supplementary Table S4。
```

---

### Fig3_DCA_calibration_mortality

```
# Figure 3 DCA 与校准曲线（28 天死亡）

**Figure 3（次要终点 28 天死亡）.** 决策曲线分析（DCA）与校准曲线。次要终点 28 天全因死亡率下，预测模型在阈值 0–1 合理范围内较 Treat All 和 Treat None 策略具有净获益。DCA 展示净获益；校准曲线评估预测概率校准程度。具体净获益数值见 Supplementary Table S4。
```

---

### Fig3_DCA_calibration_pof

```
# Figure 3 DCA 与校准曲线（POF）

**Figure 3（主要终点 POF）.** 决策曲线分析（DCA）与校准曲线。主要终点持续性器官功能衰竭（POF）下，预测模型在阈值 0–1 合理范围内（临床常用 0–0.5）较 Treat All 和 Treat None 策略具有净获益。DCA 展示各模型净获益随概率阈值变化；校准曲线评估预测概率与观察频率的一致性。具体净获益数值见 Supplementary Table S4。
```

---

### Fig4_SHAP_summary_composite

```
# Figure 4 SHAP 摘要图（Composite）

**Figure 4（混合终点 Composite）.** SHAP 摘要图，展示混合终点（POF 或 28 天死亡）下各预测因子的贡献。关键贡献特征包括血肌酐最大值（creatinine_max）、氧合指数最小值（pao2fio2ratio_min）、pH 最小值（ph_min）、血尿素氮最小值（bun_min）、白蛋白最小值（albumin_min）等，与 LASSO 入选特征一致。颜色表示特征值高低，横轴表示对预测的贡献方向与大小。
```

---

### Fig4_SHAP_summary_mortality

```
# Figure 4 SHAP 摘要图（28 天死亡）

**Figure 4（次要终点 28 天死亡）.** SHAP 摘要图，展示次要终点 28 天全因死亡率下各预测因子的贡献。关键贡献特征包括年龄（admission_age）、血尿素氮最大值（bun_max）、凝血酶原时间最小值（ptt_min）、乳酸最大值（lactate_max）、血红蛋白最小值（hemoglobin_min）等，与 LASSO 入选特征一致。颜色表示特征值高低，横轴表示对预测的贡献方向与大小。
```

---

### Fig4_SHAP_summary_pof

```
# Figure 4 SHAP 摘要图（POF）

**Figure 4（主要终点 POF）.** SHAP（SHapley Additive exPlanations）摘要图，展示主要终点持续性器官功能衰竭（POF）下各预测因子的贡献。关键贡献特征包括血肌酐最大值（creatinine_max）、氧合指数最小值（pao2fio2ratio_min）、pH 最小值（ph_min）、白蛋白（albumin_min/max）、乳酸最大值（lactate_max）等，与 LASSO 入选特征一致。颜色表示特征值高低，横轴表示对预测的贡献方向与大小。
```

---

### Fig5_forest_composite

```
# Figure 5 森林图（Composite）

**Figure 5（混合终点 Composite）.** 森林图展示逻辑回归 odds ratio（OR）及 95% 置信区间。混合终点（POF 或 28 天死亡）入选特征：氧合指数最小值 OR 0.52、pH 最小值 OR 0.59、血肌酐最大值 OR 1.37、血尿素氮最小值 OR 1.55、白蛋白最小值 OR 0.71 等（见 Supplementary Table S1）。OR>1 为危险因素，OR<1 为保护因素。用于临床决策支持及与主文列线图配套解读。
```

---

### Fig5_forest_mortality

```
# Figure 5 森林图（28 天死亡）

**Figure 5（次要终点 28 天死亡）.** 森林图展示逻辑回归 odds ratio（OR）及 95% 置信区间。次要终点 28 天全因死亡率入选特征：年龄 OR 1.96、血尿素氮最大值 OR 2.05、乳酸最大值 OR 1.50、凝血酶原时间最小值 OR 1.41、血红蛋白最小值 OR 0.63、恶性肿瘤 OR 1.74 等（见 Supplementary Table S1）。OR>1 为危险因素，OR<1 为保护因素。用于临床决策支持及与主文列线图配套解读。
```

---

### Fig5_forest_pof

```
# Figure 5 森林图（POF）

**Figure 5（主要终点 POF）.** 森林图展示逻辑回归 odds ratio（OR）及 95% 置信区间。主要终点持续性器官功能衰竭（POF）入选特征：血肌酐最大值 OR 1.75、氧合指数最小值 OR 0.50、pH 最小值 OR 0.65、白蛋白最小值 OR 0.74、白细胞最大值 OR 1.29 等（见 Supplementary Table S1）。OR>1 为危险因素，OR<1 为保护因素。用于临床决策支持及与主文列线图配套解读。
```

---

### Fig5_nomogram_composite

```
# Figure 5 列线图（Composite）

**Figure 5（混合终点 Composite）.** 基于逻辑回归系数的列线图（nomogram），用于混合终点（POF 或 28 天死亡）的个体风险估计。入选特征包括氧合指数最小值、pH 最小值、血肌酐最大值、血尿素氮最小值、白蛋白最小值等（见 Supplementary Table S1）。可用于临床决策支持及个体化风险分层。
```

---

### Fig5_nomogram_mortality

```
# Figure 5 列线图（28 天死亡）

**Figure 5（次要终点 28 天死亡）.** 基于逻辑回归系数的列线图（nomogram），用于次要终点 28 天全因死亡率的个体风险估计。入选特征包括年龄、血尿素氮最大值、乳酸最大值、凝血酶原时间最小值、血红蛋白最小值、恶性肿瘤等（见 Supplementary Table S1）。可用于临床决策支持及个体化风险分层。
```

---

### Fig5_nomogram_pof

```
# Figure 5 列线图（POF）

**Figure 5（主要终点 POF）.** 基于逻辑回归系数的列线图（nomogram），用于主要终点持续性器官功能衰竭（POF）的个体风险估计。入选特征包括血肌酐最大值、氧合指数最小值、pH 最小值、白蛋白、血尿素氮、乳酸、白细胞等（见 Supplementary Table S1）。可用于临床决策支持及个体化风险分层。
```

---

## 三、补充表格（Supplementary Tables）

### ST1_OR_composite

```
# Supplementary Table S1 OR 统计（Composite）

**Supplementary Table S1c.** 混合终点（POF 或 28 天死亡）的逻辑回归系数与 odds ratio（OR）及 95% 置信区间。入选特征包括：氧合指数最小值（P/F min）OR 0.52、pH 最小值（pH min）OR 0.59、血肌酐最大值（Creat max）OR 1.37、血尿素氮最小值（BUN min）OR 1.55、白蛋白最小值（Alb min）OR 0.71 等。用于 Figure 5 列线图与森林图构建。
```

---

### ST1_OR_mortality

```
# Supplementary Table S1 OR 统计（28 天死亡）

**Supplementary Table S1b.** 次要终点 28 天全因死亡率的逻辑回归系数与 odds ratio（OR）及 95% 置信区间。入选特征包括：年龄（admission_age）OR 1.96、血尿素氮最大值（BUN max）OR 2.05、乳酸最大值（Lac max）OR 1.50、凝血酶原时间最小值（PTT min）OR 1.41、血红蛋白最小值（Hgb min）OR 0.63、恶性肿瘤 OR 1.74 等。用于 Figure 5 列线图与森林图构建。
```

---

### ST1_OR_pof

```
# Supplementary Table S1 OR 统计（POF）

**Supplementary Table S1a.** 主要终点持续性器官功能衰竭（POF）的逻辑回归系数与 odds ratio（OR）及 95% 置信区间。入选特征（LASSO 1-SE 准则）包括：血肌酐最大值（Creat max）OR 1.75、氧合指数最小值（P/F min）OR 0.50、pH 最小值（pH min）OR 0.65、白蛋白最小值（Alb min）OR 0.74、白细胞最大值（WBC max）OR 1.29、血尿素氮最小值（BUN min）OR 1.23 等。用于 Figure 5 列线图与森林图构建。
```

---

### ST2_LASSO_selected_features

```
# Supplementary Table S2 LASSO 入选特征

**Supplementary Table S2.** LASSO 回归（1-SE 准则）特征选择结果。各终点分别入选 12 个特征。POF：creatinine_max、pao2fio2ratio_min、ph_min、albumin_min/max、wbc_max、bun_min、lactate_max、spo2_max、spo2_slope、alp_min、phosphate_min。28 天死亡：admission_age、bun_max、ptt_min、lactate_max、hemoglobin_min、wbc_min、pao2fio2ratio_min、spo2_min、tbar、albumin_max、malignant_tumor、glucose_min。Composite：与 POF 及 Mortality 特征集合相似。Display_name 用于图表标注；LASSO_Weight 为系数，Rank 为重要性排序。
```

---

### ST3_drift_analysis

```
# Supplementary Table S3 漂移分析

**Supplementary Table S3.** MIMIC 开发队列与 eICU 外部验证队列间特征分布漂移分析。列含 Outcome（pof、mortality、composite）、Feature、KS_statistic、p_value、Drift_significant。KS 检验 p<0.05 标记为 Yes。多数 LASSO 入选特征存在显著漂移（如 spo2_slope、pao2fio2ratio_min、albumin_max、creatinine_max、lactate_max 等）。用于 Supplementary Figure S4b 漂移图解读及外部验证结果讨论。
```

---

### ST4_DCA_summary

```
# Supplementary Table S4 DCA 净获益汇总

**Supplementary Table S4.** 决策曲线分析（DCA）净获益数值。列含 Outcome（POF、Mortality、Composite）、Model、Threshold（0.1–0.5）、Net_benefit。展示各模型在临床常用阈值范围内的净获益，与 Treat All、Treat None 参考策略比较。用于 Figure 3 DCA 曲线及补充材料 S4 解读。
```

---

### ST5_Table3_per_endpoint

```
# Supplementary Table S5 内部验证明细（按终点）

**Supplementary Table S5.** MIMIC 内部验证按终点展开的详细效能表。列含 Outcome、Algorithm、Group（Full Population、Subgroup No Renal）、AUC、Threshold、Sensitivity、Specificity、PPV、NPV、F1_Score、Accuracy。与 Table 3 对应，提供阳性/阴性预测值及准确度。无肾亚组（CKD=0）各模型 AUC 较全人群有所下降。
```

---

## 四、补充图形（Supplementary Figures）

### SF1_lasso / SF1a_diag_composite

```
# Supplementary Figure S1a LASSO 路径诊断（Composite）

**Supplementary Figure S1a（混合终点 Composite）.** LASSO 回归路径诊断图（正则化系数 λ 与系数路径）。展示混合终点（POF 或 28 天死亡）下 12 个入选特征的系数随 λ 变化轨迹。入选特征包括血肌酐最大值、氧合指数最小值、pH 最小值、血尿素氮最小值、白蛋白最小值等。1-SE 准则用于确定最优 λ。详见补充材料 S1。
```

---

### SF1_lasso / SF1a_diag_mortality

```
# Supplementary Figure S1a LASSO 路径诊断（28 天死亡）

**Supplementary Figure S1a（次要终点 28 天死亡）.** LASSO 回归路径诊断图（正则化系数 λ 与系数路径）。展示次要终点 28 天全因死亡率下 12 个入选特征的系数随 λ 变化轨迹。入选特征包括年龄、血尿素氮最大值、凝血酶原时间最小值、乳酸最大值等。1-SE 准则用于确定最优 λ。详见补充材料 S1。
```

---

### SF1_lasso / SF1a_diag_pof

```
# Supplementary Figure S1a LASSO 路径诊断（POF）

**Supplementary Figure S1a（主要终点 POF）.** LASSO 回归路径诊断图（正则化系数 λ 与系数路径）。展示主要终点持续性器官功能衰竭（POF）下 12 个入选特征的系数随 λ 变化轨迹。1-SE 准则用于确定最优 λ；用于特征选择及 Supplementary Table S2 入选特征对应。详见补充材料 S1。
```

---

### SF1_lasso / SF1b_importance_composite

```
# Supplementary Figure S1b LASSO 特征重要性（Composite）

**Supplementary Figure S1b（混合终点 Composite）.** LASSO 回归特征重要性排序。混合终点（POF 或 28 天死亡）入选特征按 LASSO 系数绝对值排序：血肌酐最大值、氧合指数最小值、pH 最小值、血尿素氮最小值、白蛋白最小值等。与 Supplementary Table S2 及 Figure 4 SHAP 摘要图对应。详见补充材料 S1。
```

---

### SF1_lasso / SF1b_importance_mortality

```
# Supplementary Figure S1b LASSO 特征重要性（28 天死亡）

**Supplementary Figure S1b（次要终点 28 天死亡）.** LASSO 回归特征重要性排序。次要终点 28 天全因死亡率入选特征按 LASSO 系数绝对值排序：年龄、血尿素氮最大值、凝血酶原时间最小值、乳酸最大值、血红蛋白最小值等。与 Supplementary Table S2 及 Figure 4 SHAP 摘要图对应。详见补充材料 S1。
```

---

### SF1_lasso / SF1b_importance_pof

```
# Supplementary Figure S1b LASSO 特征重要性（POF）

**Supplementary Figure S1b（主要终点 POF）.** LASSO 回归特征重要性排序。主要终点持续性器官功能衰竭（POF）入选特征按 LASSO 系数绝对值排序：血肌酐最大值、氧合指数最小值、pH 最小值、白蛋白、血尿素氮、乳酸、白细胞等。与 Supplementary Table S2 及 Figure 4 SHAP 摘要图对应。详见补充材料 S1。
```

---

### SF2_internal_ROC / SF2a_ROC_composite

```
# Supplementary Figure S2a 内部 ROC 曲线（Composite）

**Supplementary Figure S2a（混合终点 Composite）.** MIMIC 测试集内部验证 ROC 曲线。混合终点（POF 或 28 天死亡）：SVM AUC 0.881，XGBoost 0.868，逻辑回归与 Random Forest 均 0.867，Decision Tree 0.840。与 Table 3 内部验证效能对应。详见补充材料 S2。
```

---

### SF2_internal_ROC / SF2a_ROC_mortality

```
# Supplementary Figure S2a 内部 ROC 曲线（28 天死亡）

**Supplementary Figure S2a（次要终点 28 天死亡）.** MIMIC 测试集内部验证 ROC 曲线。次要终点 28 天全因死亡率：逻辑回归 AUC 0.876，XGBoost 0.865，SVM 0.864，Random Forest 0.861，Decision Tree 0.791。与 Table 3 内部验证效能对应。详见补充材料 S2。
```

---

### SF2_internal_ROC / SF2a_ROC_pof

```
# Supplementary Figure S2a 内部 ROC 曲线（POF）

**Supplementary Figure S2a（主要终点 POF）.** MIMIC 测试集（20%）内部验证 ROC 曲线。主要终点持续性器官功能衰竭（POF）：SVM AUC 0.853，XGBoost 0.847，逻辑回归 0.846，Random Forest 0.842，Decision Tree 0.818。与 Table 3 内部验证效能对应。详见补充材料 S2。
```

---

### SF2_internal_ROC / SF2b_Calibration_composite

```
# Supplementary Figure S2b 内部校准曲线（Composite）

**Supplementary Figure S2b（混合终点 Composite）.** MIMIC 测试集内部验证校准曲线（calibration curve）。评估各模型对混合终点（POF 或 28 天死亡）预测概率与观察频率的一致性。Platt scaling 用于概率校准。详见补充材料 S2。
```

---

### SF2_internal_ROC / SF2b_Calibration_mortality

```
# Supplementary Figure S2b 内部校准曲线（28 天死亡）

**Supplementary Figure S2b（次要终点 28 天死亡）.** MIMIC 测试集内部验证校准曲线（calibration curve）。评估各模型对次要终点 28 天全因死亡率预测概率与观察频率的一致性。Platt scaling 用于概率校准。详见补充材料 S2。
```

---

### SF2_internal_ROC / SF2b_Calibration_pof

```
# Supplementary Figure S2b 内部校准曲线（POF）

**Supplementary Figure S2b（主要终点 POF）.** MIMIC 测试集内部验证校准曲线（calibration curve）。评估各模型对主要终点持续性器官功能衰竭（POF）预测概率与观察频率的一致性。Platt scaling 用于概率校准。对角线表示完美校准。详见补充材料 S2。按肾功能良好与否（Subgroup No Renal=1/0）的分层校准曲线见脚本 06（`{target}_Calibration_subgroupNoRenal_{1,0}`），可根据篇幅作为 Supplementary Figure S2c 披露。
```

---

### SF3_cutoff / SF3a_Diagnostic_composite

```
# Supplementary Figure S3a 诊断四格表（Composite）

**Supplementary Figure S3a（混合终点 Composite）.** 混合终点（POF 或 28 天死亡）诊断四格表（混淆矩阵）。基于 Youden 指数确定的最优切点，展示各模型（如 SVM）在 MIMIC 测试集上的真阳性、假阳性、真阴性、假阴性。用于内部验证灵敏度、特异度、阳性/阴性预测值计算。详见补充材料 S3。
```

---

### SF3_cutoff / SF3a_Diagnostic_mortality

```
# Supplementary Figure S3a 诊断四格表（28 天死亡）

**Supplementary Figure S3a（次要终点 28 天死亡）.** 次要终点 28 天全因死亡率诊断四格表（混淆矩阵）。基于 Youden 指数确定的最优切点，展示各模型（如 Logistic Regression）在 MIMIC 测试集上的真阳性、假阳性、真阴性、假阴性。用于内部验证灵敏度、特异度计算。详见补充材料 S3。
```

---

### SF3_cutoff / SF3a_Diagnostic_pof

```
# Supplementary Figure S3a 诊断四格表（POF）

**Supplementary Figure S3a（主要终点 POF）.** 主要终点持续性器官功能衰竭（POF）诊断四格表（混淆矩阵）。基于 Youden 指数确定的最优切点，展示各模型（如 SVM）在 MIMIC 测试集上的真阳性、假阳性、真阴性、假阴性。用于内部验证灵敏度、特异度、阳性/阴性预测值计算。详见补充材料 S3。
```

---

### SF3_cutoff / SF3b_feature_importance

```
# Supplementary Figure S3b 特征重要性（cutoff 分析）

**Supplementary Figure S3b.** 基于最优切点诊断分析的特征重要性。展示各预测因子在区分阳性/阴性结局中的贡献，与 LASSO 特征选择及 SHAP 可解释性结果互补。用于内部验证可解释性及补充材料 S3 解读。
```

---

### SF3_cutoff / SF3c_forest_plot

```
# Supplementary Figure S3c 森林图（cutoff 分析）

**Supplementary Figure S3c.** 基于最优切点诊断分析的森林图。展示各预测因子与结局关联的 odds ratio（OR）及 95% 置信区间，与 Figure 5 主文森林图及 Supplementary Table S1 对应。用于内部验证可解释性及补充材料 S3 解读。
```

---

### SF4_comparison / SF4a_Table4_visualization

```
# Supplementary Figure S4a Table 4 可视化

**Supplementary Figure S4a.** Table 4 外部验证效能的视觉化呈现。将 eICU 外部队列各模型在各终点（POF、28 天死亡、Composite）下的 AUC、Brier、灵敏度、特异度、AUPRC 等指标进行可视化（如雷达图或柱状图），便于跨模型、跨终点比较。用于补充材料 S4 及外部验证结果解读。
```

---

### SF4_comparison / SF4b_drift_admission_age_mortality

```
# Supplementary Figure S4b 分布漂移（年龄，28 天全因死亡率）

**Supplementary Figure S4b.** 年龄（admission_age）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_albumin_max_composite

```
# Supplementary Figure S4b 分布漂移（白蛋白最大值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 白蛋白最大值（albumin_max）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_albumin_max_mortality

```
# Supplementary Figure S4b 分布漂移（白蛋白最大值，28 天全因死亡率）

**Supplementary Figure S4b.** 白蛋白最大值（albumin_max）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_albumin_max_pof

```
# Supplementary Figure S4b 分布漂移（白蛋白最大值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 白蛋白最大值（albumin_max）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_albumin_min_composite

```
# Supplementary Figure S4b 分布漂移（白蛋白最小值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 白蛋白最小值（albumin_min）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_albumin_min_pof

```
# Supplementary Figure S4b 分布漂移（白蛋白最小值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 白蛋白最小值（albumin_min）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_alp_min_pof

```
# Supplementary Figure S4b 分布漂移（碱性磷酸酶最小值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 碱性磷酸酶最小值（alp_min）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_aniongap_min_composite

```
# Supplementary Figure S4b 分布漂移（阴离子间隙最小值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 阴离子间隙最小值（aniongap_min）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_bun_max_mortality

```
# Supplementary Figure S4b 分布漂移（血尿素氮最大值，28 天全因死亡率）

**Supplementary Figure S4b.** 血尿素氮最大值（bun_max）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_creatinine_max_composite

```
# Supplementary Figure S4b 分布漂移（血肌酐最大值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 血肌酐最大值（creatinine_max）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_creatinine_max_pof

```
# Supplementary Figure S4b 分布漂移（血肌酐最大值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 血肌酐最大值（creatinine_max）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_hemoglobin_min_composite

```
# Supplementary Figure S4b 分布漂移（血红蛋白最小值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 血红蛋白最小值（hemoglobin_min）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_hemoglobin_min_mortality

```
# Supplementary Figure S4b 分布漂移（血红蛋白最小值，28 天全因死亡率）

**Supplementary Figure S4b.** 血红蛋白最小值（hemoglobin_min）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_lactate_max_composite

```
# Supplementary Figure S4b 分布漂移（乳酸最大值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 乳酸最大值（lactate_max）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_lactate_max_mortality

```
# Supplementary Figure S4b 分布漂移（乳酸最大值，28 天全因死亡率）

**Supplementary Figure S4b.** 乳酸最大值（lactate_max）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_lactate_max_pof

```
# Supplementary Figure S4b 分布漂移（乳酸最大值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 乳酸最大值（lactate_max）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_pao2fio2ratio_min_composite

```
# Supplementary Figure S4b 分布漂移（氧合指数（PaO2/FiO2）最小值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 氧合指数（PaO2/FiO2）最小值（pao2fio2ratio_min）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_pao2fio2ratio_min_mortality

```
# Supplementary Figure S4b 分布漂移（氧合指数（PaO2/FiO2）最小值，28 天全因死亡率）

**Supplementary Figure S4b.** 氧合指数（PaO2/FiO2）最小值（pao2fio2ratio_min）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_pao2fio2ratio_min_pof

```
# Supplementary Figure S4b 分布漂移（氧合指数（PaO2/FiO2）最小值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 氧合指数（PaO2/FiO2）最小值（pao2fio2ratio_min）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_ph_min_composite

```
# Supplementary Figure S4b 分布漂移（pH 最小值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** pH 最小值（ph_min）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_ph_min_pof

```
# Supplementary Figure S4b 分布漂移（pH 最小值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** pH 最小值（ph_min）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_phosphate_min_pof

```
# Supplementary Figure S4b 分布漂移（磷酸盐最小值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 磷酸盐最小值（phosphate_min）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_ptt_min_composite

```
# Supplementary Figure S4b 分布漂移（凝血酶原时间（PTT）最小值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 凝血酶原时间（PTT）最小值（ptt_min）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_ptt_min_mortality

```
# Supplementary Figure S4b 分布漂移（凝血酶原时间（PTT）最小值，28 天全因死亡率）

**Supplementary Figure S4b.** 凝血酶原时间（PTT）最小值（ptt_min）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_spo2_max_pof

```
# Supplementary Figure S4b 分布漂移（血氧饱和度（SpO2）最大值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 血氧饱和度（SpO2）最大值（spo2_max）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_spo2_min_mortality

```
# Supplementary Figure S4b 分布漂移（血氧饱和度（SpO2）最小值，28 天全因死亡率）

**Supplementary Figure S4b.** 血氧饱和度（SpO2）最小值（spo2_min）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_spo2_slope_composite

```
# Supplementary Figure S4b 分布漂移（血氧饱和度斜率，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 血氧饱和度斜率（spo2_slope）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_spo2_slope_pof

```
# Supplementary Figure S4b 分布漂移（血氧饱和度斜率，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 血氧饱和度斜率（spo2_slope）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_tbar_mortality

```
# Supplementary Figure S4b 分布漂移（TBAR，28 天全因死亡率）

**Supplementary Figure S4b.** TBAR（tbar）在28 天全因死亡率终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_wbc_max_composite

```
# Supplementary Figure S4b 分布漂移（白细胞最大值，混合终点（POF 或 28 天死亡））

**Supplementary Figure S4b.** 白细胞最大值（wbc_max）在混合终点（POF 或 28 天死亡）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF4_comparison / SF4b_drift_wbc_max_pof

```
# Supplementary Figure S4b 分布漂移（白细胞最大值，持续性器官功能衰竭（POF））

**Supplementary Figure S4b.** 白细胞最大值（wbc_max）在持续性器官功能衰竭（POF）终点下的 MIMIC 开发队列与 eICU 外部验证队列分布对比。展示跨队列分布漂移（distributional drift），用于解释模型外推时的特征分布差异。该特征为 LASSO 入选变量。KS 检验详见 Supplementary Table S3。详见补充材料 S4。
```

---

### SF5_interpretation / SF5a_Force_composite

```
# Supplementary Figure S5a SHAP 力导向图（Composite）

**Supplementary Figure S5a（混合终点 Composite）.** 个体 SHAP 力导向图（Force plot）。展示单个患者预测中，各特征对混合终点（POF 或 28 天死亡）预测的推动方向与大小。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5a_Force_mortality

```
# Supplementary Figure S5a SHAP 力导向图（28 天死亡）

**Supplementary Figure S5a（次要终点 28 天死亡）.** 个体 SHAP 力导向图（Force plot）。展示单个患者预测中，各特征对次要终点 28 天全因死亡率预测的推动方向与大小。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5a_Force_pof

```
# Supplementary Figure S5a SHAP 力导向图（POF）

**Supplementary Figure S5a（主要终点 POF）.** 个体 SHAP 力导向图（Force plot）。展示单个患者预测中，各特征对主要终点持续性器官功能衰竭（POF）预测的推动方向与大小。红色推动朝向阳性，蓝色朝向阴性。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_composite_creatinine_max

```
# Supplementary Figure S5b SHAP 依赖图（混合终点（POF 或 28 天死亡），血肌酐最大值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示混合终点（POF 或 28 天死亡）终点下血肌酐最大值（creatinine_max）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_composite_pao2fio2ratio_min

```
# Supplementary Figure S5b SHAP 依赖图（混合终点（POF 或 28 天死亡），氧合指数（PaO2/FiO2）最小值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示混合终点（POF 或 28 天死亡）终点下氧合指数（PaO2/FiO2）最小值（pao2fio2ratio_min）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_composite_ph_min

```
# Supplementary Figure S5b SHAP 依赖图（混合终点（POF 或 28 天死亡），pH 最小值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示混合终点（POF 或 28 天死亡）终点下pH 最小值（ph_min）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_mortality_admission_age

```
# Supplementary Figure S5b SHAP 依赖图（28 天全因死亡率，年龄）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示28 天全因死亡率终点下年龄（admission_age）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_mortality_bun_max

```
# Supplementary Figure S5b SHAP 依赖图（28 天全因死亡率，血尿素氮最大值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示28 天全因死亡率终点下血尿素氮最大值（bun_max）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_mortality_ptt_min

```
# Supplementary Figure S5b SHAP 依赖图（28 天全因死亡率，凝血酶原时间（PTT）最小值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示28 天全因死亡率终点下凝血酶原时间（PTT）最小值（ptt_min）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_pof_creatinine_max

```
# Supplementary Figure S5b SHAP 依赖图（持续性器官功能衰竭（POF），血肌酐最大值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示持续性器官功能衰竭（POF）终点下血肌酐最大值（creatinine_max）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_pof_pao2fio2ratio_min

```
# Supplementary Figure S5b SHAP 依赖图（持续性器官功能衰竭（POF），氧合指数（PaO2/FiO2）最小值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示持续性器官功能衰竭（POF）终点下氧合指数（PaO2/FiO2）最小值（pao2fio2ratio_min）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---

### SF5_interpretation / SF5b_Dep_pof_ph_min

```
# Supplementary Figure S5b SHAP 依赖图（持续性器官功能衰竭（POF），pH 最小值）

**Supplementary Figure S5b.** SHAP 依赖图（dependence plot）。展示持续性器官功能衰竭（POF）终点下pH 最小值（ph_min）与 SHAP 值的关系，反映该特征对预测的边际贡献及与其他特征的交互。用于个体化可解释性及补充材料 S5 解读。
```

---
