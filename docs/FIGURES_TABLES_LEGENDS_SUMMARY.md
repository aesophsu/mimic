# 输出映射总览

本文件用于快速定位结果文件，不提供结果式图注模板。

## 1. 主结果表（`results/main/tables`）
- `Table1_baseline.csv`：基线统计。
- `Table2_renal_subgroup.csv`：肾功能分层统计。
- `Table3_performance.csv`：内部验证性能。
- `Table4_external_validation.csv`：外部验证性能。
- `Table4_external_validation_slim.csv`：精简模型外部验证。

## 2. 主结果图（`results/main/figures`）
- `Fig1_*`：样本筛选与缺失可视化。
- `Fig2_*`：外部 ROC/校准图。
- `Fig3_*`：DCA 与校准。
- `Fig4_*`：SHAP 总结图。
- `Fig5_*`：列线图与 OR 可视化。

## 3. 扩展结果（`results/supplementary`）
- `figures/S1_lasso`：LASSO 相关图。
- `figures/S2_internal_ROC`：内部 ROC/校准。
- `figures/S3_cutoff`：阈值与诊断图。
- `figures/S4_comparison`：跨队列分布对比。
- `figures/S5_interpretation`：SHAP 个体解释与依赖图。
- `tables/`：补充统计表。

## 4. 使用建议
1. 先看 `Table3` + `Table4`，确认内外部一致性。
2. 再看 `Table4_external_validation_slim`，评估精简模型折损。
3. 最后结合 `Fig4` 与 `S5` 进行可解释性核查。
