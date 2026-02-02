# Table 1 生成逻辑说明

## 数据依赖

| 输入 | 来源 | 说明 |
|------|------|------|
| `mimic_raw_scale.csv` | Step 01 | MIMIC 清洗后（data/cleaned/） |
| `eicu_raw_scale.csv` | Step 08 | eICU 临床审计后（data/external/） |

## 03_table1_baseline 运行时机

| 场景 | 03 运行时机 | Table 1 内容 |
|------|-------------|--------------|
| **MIMIC-only** (01-07) | 03 在 02 之后 | 仅 MIMIC 列（无 eICU 外部验证列） |
| **全流程** (01-13) | 03 在 08 之后 | MIMIC + eICU 列 + SMD (MIMIC vs eICU) |
| **eICU-only** (08-13) | 03 在 08 之后 | MIMIC + eICU 列 + SMD |

## 结论

- **完整 Table 1**（含 eICU 外部验证列）必须在 **08 运行之后** 才能生成
- **MIMIC-only Table 1** 可在 01 之后独立生成
- `run_all.py` 已按上述逻辑安排 03 的执行顺序
