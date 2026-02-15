# 基线表生成逻辑说明

## 1. 输入依赖
- `data/cleaned/mimic_raw_scale.csv`
- `data/external/eicu_raw_scale.csv`

## 2. 生成脚本
- `scripts/audit_eval/03_table1_baseline.py`

## 3. 运行时机
- `mimic-only`：可先生成 MIMIC 列。
- 全流程：在 eICU 处理完成后再生成，得到完整基线表。

## 4. 输出文件
- `results/main/tables/Table1_baseline.csv`

## 5. 备注
`run_all.py` 已按依赖顺序编排，无需手动调整步骤顺序。
