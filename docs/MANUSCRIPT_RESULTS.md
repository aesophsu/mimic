# Results（结果）

> 适用于 Critical Care 投稿，基于 `docs/tables/main/` 与 `docs/tables/supplementary/` 表格及 `docs/figures/main/`、`docs/figures/supplementary/` 图形撰写。

---

## 队列描述

根据纳入排除标准，MIMIC-IV 开发队列最终纳入 1095 例急性胰腺炎患者，eICU 外部验证队列纳入 1112 例。Figure 1 展示 CONSORT 流程图（`docs/figures/main/Fig1_flowchart.svg`）及缺失模式热图（`docs/figures/main/Fig1_missing_heatmap.{pdf,png}`）。

MIMIC 队列中，POF 发生率为 40.5%（443/1095），28 天死亡率为 15.4%（169/1095）。eICU 队列中，POF 发生率为 41.6%（463/1112），28 天死亡率为 8.4%（93/1112）。

基线特征见 Table 1（`docs/tables/main/Table1_baseline.csv`）。MIMIC 队列按 POF 分组：Non-POF 652 例，POF 443 例。POF 组与 Non-POF 组比较，年龄中位数相近（58 vs 59 岁），但 POF 组体重更高（86.0 vs 80.5 kg）、合并 CKD 比例更高（26.2% vs 14.3%）、机械通气率更高（58.0% vs 19.0%）、血管加压药使用率更高（49.9% vs 13.8%）。严重程度评分在 POF 组均显著升高：SOFA 中位数 4 vs 9，APS-III 42 vs 68，SAPS-II 30 vs 46，OASIS 29 vs 39，LODS 3 vs 8。实验室指标方面，POF 组肌酐（1.0 vs 2.1 mg/dL）、BUN、乳酸升高，P/F 比（157 vs 104 mmHg）、pH、HCO3 降低。标准化均数差（SMD）> 0.1 的变量表明组间存在显著不平衡；eICU 中 SOFA、APS-III、SAPS-II、OASIS、LODS、ALP 未采集（表内以—表示），见表注。

eICU 外部验证队列与 MIMIC 开发队列相比，存在一定分布漂移（SMD MIMIC vs eICU）：年龄中位数 54 岁，男性 61.1%；机械通气率显著更高（96.6% vs 34.8%，SMD 1.30），P/F 比分布差异较大（366.7 vs 120.0 mmHg 中位数，SMD 2.09）；CHF、CKD、恶性肿瘤等合并症比例及部分实验室指标亦有差异（见表注）。

---

## 内部验证效能

内部验证结果见 Table 3（`docs/tables/main/Table3_performance.csv`）。各终点全人群效能如下。

**主要终点（POF）**：SVM AUC 0.853（95% CI 0.795–0.903），灵敏度 0.897，特异度 0.720，F1 0.772，最优切点 0.370；XGBoost AUC 0.847，逻辑回归与 RF 均在 0.84 左右；决策树 AUC 0.818。

**次要终点（28 天死亡）**：逻辑回归 AUC 0.876（0.812–0.929），灵敏度 0.789，特异度 0.834；XGBoost AUC 0.865，SVM 与 RF 均在 0.86 左右；决策树 AUC 0.791。

**混合终点（Composite）**：SVM AUC 0.881（0.835–0.924），灵敏度 0.816，特异度 0.785，F1 0.784；XGBoost AUC 0.868，逻辑回归与 RF 均在 0.86–0.87；决策树 AUC 0.840。

内部 ROC 与校准曲线见补充材料 S2。

**肾亚组分析（Table 2，`docs/tables/main/Table2_renal_subgroup.csv`）**：无肾亚组（CKD=0）605 例，POF 发生率 26.9%，28 天死亡 8.1%；有肾亚组（CKD=1）490 例，POF 66.3%，28 天死亡 24.5%。两亚组在年龄、SOFA、BUN、肌酐、乳酸、pH、HCO3 等均有显著差异。无肾亚组各终点 AUC 较全人群有所下降（POF 最佳约 0.77，Mortality 约 0.80，Composite 约 0.79），提示肾功能相关指标对预测有贡献。

---

## 外部验证效能

eICU 盲测结果见 Table 4（`docs/tables/main/Table4_external_validation.csv`）及 Figure 2。

**主要终点（POF）**：Random Forest AUC 0.859（95% CI 0.837–0.881），AUPRC 0.822；SVM 0.849，XGBoost 0.847，Decision Tree 0.821，Logistic Regression 0.736。Brier 评分 0.16–0.21。

**次要终点（28 天死亡）**：XGBoost AUC 0.846（0.809–0.881），Random Forest 0.836，SVM 0.836，Logistic Regression 0.810，Decision Tree 0.779。Brier 评分约 0.06–0.07。

**混合终点（Composite）**：Random Forest AUC 0.864（0.844–0.884），AUPRC 0.839；XGBoost 0.859，SVM 0.845，Decision Tree 0.830，Logistic Regression 0.741。Brier 评分约 0.15–0.21。

**内外部 AUC 对比**：机器学习模型在外部队列中相对稳定（如 POF SVM 内部 0.853 → 外部 0.849）；逻辑回归泛化较差（如 POF 内部 0.846 → 外部 0.736）。总体而言，模型在 eICU 外部验证队列中保持较好区分能力。跨队列漂移分析（补充材料 S4）显示，多数学入选特征在 MIMIC 与 eICU 间存在显著分布漂移（KS 检验 p < 0.05），如 wbc_max、albumin_min、creatinine_max、lactate_max、pao2fio2ratio_min 等；在存在漂移情况下模型仍保持较好外推能力。校准曲线与 DCA 见 Figure 3。

---

## 临床效用

决策曲线分析（DCA）显示，在阈值 0–1 的合理范围内（临床常用 0–0.5），预测模型较 Treat All 和 Treat None 策略具有净获益（Figure 3）；具体净获益数值见 Supplementary Table S4（`docs/tables/supplementary/ST4_DCA_summary.csv`）。

列线图与森林图（OR 及 95% CI）见 Figure 5，可用于个体风险估计与临床决策支持。各终点入选特征及其 OR 详见 Supplementary Table S1（`docs/tables/supplementary/ST1_OR_*.csv`）。

---

## 可解释性

**SHAP 可解释性**：SHAP 摘要图（Figure 4）展示各终点下关键预测因子的贡献。主要终点 POF 中，肌酐、P/F 比、pH、白蛋白、乳酸等贡献较大；次要终点 28 天死亡中，年龄、BUN、PTT、乳酸、血红蛋白等贡献突出；混合终点 Composite 中，creatinine_max、pao2fio2ratio_min、ph_min、bun_min、albumin_min 等贡献突出，与 LASSO 入选特征一致。个体 SHAP 力导向图与依赖图见补充材料 S5。
