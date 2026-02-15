# 结果说明（实现版）

> 文件名沿用历史命名，仅用于沉淀当前版本的运行结果解读。

## 1. 结果文件位置
- 主结果表：`results/main/tables/`
- 主结果图：`results/main/figures/`
- 扩展结果：`results/supplementary/`

## 2. 推荐优先查看
- `results/main/tables/Table3_performance.csv`：内部性能。
- `results/main/tables/Table4_external_validation.csv`：外部性能。
- `results/main/tables/Table4_external_validation_slim.csv`：精简模型外部性能。

## 3. 解读顺序建议
1. 先看内部与外部 AUC、Sensitivity、Specificity 是否一致。
2. 再看精简模型相对全量模型的性能损失是否在可接受范围。
3. 结合 SHAP 输出判断关键变量是否稳定。

## 4. 常见结论模板
- 当精简模型外部 AUC 仅轻度下降时，可优先用于快速评估场景。
- 当外部下降明显时，优先保留更多变量或使用更高 k 的方案。
- mortality 常对变量压缩更敏感，需要单独评估。

## 5. Web 端对应关系
- 默认风险计算读取 `deploy_bundle.pkl`。
- Smart-subset 模式优先读取 SHAP 推荐特征（若可用）。
- 单例解释支持 waterfall/force 图，并复用特征字典显示名。
