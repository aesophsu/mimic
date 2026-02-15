# 流程方法说明（实现版）

> 文件名沿用历史命名，仅用于记录当前仓库的实现流程与参数设置。

## 1. 数据来源与队列
- 开发域：MIMIC-IV v3.1。
- 外部域：eICU-CRD。
- 目标人群：成人 AP ICU 患者，统一按 LOS、首次 ICU、年龄与关键变量缺失率规则筛选。

## 2. 数据提取与时间窗
- MIMIC：`scripts/sql/01_mimic_extraction.sql`
- eICU：`scripts/sql/08_eicu_extraction.sql`
- 特征窗口：以 ICU 入室后前 24h 为主，实验室包含入室前 6h 到入室后 24h。

## 3. 清洗与预处理
- 清洗脚本：`scripts/01_mimic_cleaning.py`
- 标准化脚本：`scripts/02_mimic_standardization.py`
- 原则：
  - 先划分训练/测试，再 fit 插补器与标准化器。
  - eICU 仅使用开发阶段参数做 transform，不再拟合。
- 关键资产：
  - `artifacts/scalers/mimic_mice_imputer.joblib`
  - `artifacts/scalers/mimic_scaler.joblib`
  - `artifacts/deployment/deploy_bundle.pkl`

## 4. 特征筛选与建模
- 初筛：LASSO（`scripts/05_lasso_feature_selection.py`）。
- 多模型训练：`scripts/06_多算法训练+校准.py`。
- 可选精简：
  - XGBoost 重要性筛选：`scripts/06b_xgb_feature_pruning.py`
  - SHAP + bootstrap 稳定性筛选：`scripts/06c_shap_bootstrap_feature_pruning.py`

## 5. 评估与验证
- 内部评估：`scripts/07_*`、`scripts/audit_eval/03_*`。
- 外部评估：`scripts/audit_eval/10_external_validation.py`。
- 精简模型外部评估：`scripts/audit_eval/10b_external_validation_slim.py`。

## 6. 解释与可视化
- SHAP 解释：`scripts/11_shap_analysis.py`
- DCA/校准：`scripts/12_dca_calibration.py`
- 风险展示：`app.py` 与 `deploy_min/app.py`

## 7. 默认运行链路
- 主流程入口：`run_all.py`
- 默认包含主分析步骤。
- 06c/10b 为可选增强步骤，可通过参数启用。
