# AP Risk Modeling Pipeline (MIMIC-IV + eICU)

本仓库用于重症 AP 风险预测的工程化流程管理，覆盖数据抽取、清洗、训练、外部验证与 Web 工具部署。

## 1. 目标
- 统一运行 MIMIC-IV 开发流程与 eICU 外部验证流程。
- 输出可复核的模型评估结果与图表产物。
- 提供可部署的 Streamlit 风险计算器（研究演示用途）。

## 2. 目录概览
- `scripts/sql/`：MIMIC/eICU 抽取 SQL。
- `scripts/`：清洗、建模、验证、解释脚本。
- `artifacts/`：模型与特征资产（含 `deploy_bundle.pkl`、特征字典、筛选结果）。
- `results/`：表格与图形输出。
- `validation/`：跨队列漂移与外部验证辅助结果。
- `deploy_min/`：最小化 Streamlit 在线部署目录。

## 3. 环境准备
```bash
cd /Users/sue/Documents/mimic
uv sync
```

如需本地运行 Streamlit，请使用 `uv` 启动（避免找不到全局命令）：
```bash
uv run streamlit run app.py
# 或
cd deploy_min
uv run streamlit run app.py
```

## 4. 主流程运行
### 4.1 一键运行
```bash
uv run python run_all.py
```

### 4.2 常用选项
```bash
uv run python run_all.py --mimic-only
uv run python run_all.py --eicu-only
uv run python run_all.py --with-xgb-pruning
uv run python run_all.py --with-shap-pruning
uv run python run_all.py --with-slim-external
```

说明：
- `--with-shap-pruning`：启用 `06c_shap_bootstrap_feature_pruning.py`。
- `10b_external_validation_slim.py` 默认读取 SHAP 推荐特征文件（若存在）。

## 5. 关键输入输出
### 5.1 核心输入
- `data/raw/mimic_raw_data.csv`
- `data/raw/eicu_raw_data.csv`

### 5.2 核心资产
- `artifacts/deployment/deploy_bundle.pkl`
- `artifacts/features/feature_dictionary.json`
- `artifacts/features/xgb_shap_pruning_recommendation.json`

### 5.3 关键结果
- `results/main/tables/Table3_performance.csv`
- `results/main/tables/Table4_external_validation.csv`
- `results/main/tables/Table4_external_validation_slim.csv`

## 6. Web 工具
### 6.1 本地完整版
```bash
uv run streamlit run app.py
```

### 6.2 最小部署版
```bash
cd deploy_min
uv run streamlit run app.py
```

部署到 Streamlit Cloud 时建议：
- Repo：当前仓库
- Branch：部署分支
- Main file path：`deploy_min/app.py`

## 7. 说明文档
- `docs/EICU_EXTRACTION_GUIDE.md`：eICU 抽取与对齐要点。
- `docs/FLOWCHART_COUNTS.md`：队列筛选计数脚本说明。
- `docs/TABLE1_FLOW.md`：基线表生成流程。
- `docs/MANUSCRIPT_METHODS.md`：流程实现说明（历史文件名保留）。
- `docs/MANUSCRIPT_RESULTS.md`：结果解读说明（历史文件名保留）。

## 8. 合规声明
本项目及 Web 工具仅用于学术与工程演示，不直接用于临床诊疗决策。
