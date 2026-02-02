"""
研究设计配置：重症AP预测模型
- 主要终点：POF (持续性器官功能衰竭)
- 次要终点：28天死亡率
- 混合终点：POF + 28天死亡率
- MIMIC-IV：训练 + 内部验证
- eICU：外部验证
"""

# 终点顺序：主要 → 次要 → 混合
OUTCOMES = ['pof', 'mortality', 'composite']

# 预处理常量
MISSING_THRESHOLD = 0.3  # 缺失率超过此值剔除特征
SPLIT_SEED = 42
TEST_SIZE = 0.2

# 终点类型（用于表格/图表标注）
OUTCOME_TYPE = {
    'pof': 'Primary',
    'mortality': 'Secondary',
    'composite': 'Composite',
}

# 终点展示名
OUTCOME_LABEL = {
    'pof': 'POF',
    'mortality': '28-day mortality',
    'composite': 'POF + 28-day mortality',
}

# 分层划分依据（混合终点）
STRATIFY_COL = 'composite'
