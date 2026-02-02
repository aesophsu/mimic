# Figure 1 Flowchart 纳入排除计数

本文档说明如何获取 Figure 1 (CONSORT 流程图) 所需的人数统计。

## 一、MIMIC-IV 开发队列

### 执行顺序

1. 在 MIMIC 数据库中执行 `scripts/sql/01_mimic_extraction.sql`（生成 `my_custom_schema.ap_final_analysis_cohort`）
2. 执行 `scripts/sql/01_mimic_flowchart_counts.sql`

### 输出字段（填入 Figure 1）

纳入排除顺序：**先 LOS 再首次**（临床逻辑）

| 步骤 | 说明 | 输出 |
|------|------|------|
| **初始池** | MIMIC-IV 中诊断为急性胰腺炎（AP）的 ICU 入住 | ______ 人 |
| **排除1** | ICU LOS < 24 小时（先 LOS） | 剔除 ______ 人 |
| **排除2** | 非首次入 ICU（在 LOS≥24h 的入住中保留首次，再首次） | 剔除 ______ 人 |
| **排除3** | 年龄 < 18 岁 | 剔除 ______ 人 |
| **排除4** | 关键生理指标缺失严重（>80% 缺失） | 剔除 ______ 人 |
| **最终纳入** | — | 与提取脚本一致 |

### 排除标准 4 说明

关键生理指标（10 项）：creatinine_max, bun_max, lactate_max, pao2fio2ratio_min, ph_min, wbc_max, hemoglobin_min, bilirubin_total_max, sodium_max, albumin_min。若某患者 >80% 上述指标缺失，则剔除。

---

## 二、eICU 外部验证队列

### 执行顺序

1. 在 eICU 数据库中执行 `scripts/sql/08_eicu_extraction.sql`（生成 `eicu_cview.ap_external_validation`）
2. 执行 `scripts/sql/08_eicu_flowchart_counts.sql`

### 输出字段（填入 Figure 1）

纳入排除顺序：**先 LOS 再首次**（与 MIMIC 一致）

| 步骤 | 说明 | 输出 |
|------|------|------|
| **初始池** | eICU 中诊断为 AP 的 ICU 入住 | ______ 人 |
| **排除1** | ICU LOS < 24 小时（先 LOS） | 剔除 ______ 人 |
| **排除2** | 非首次入 ICU（在 LOS≥24h 的入住中保留首次，再首次） | 剔除 ______ 人 |
| **排除3** | 年龄 < 18 岁 | 剔除 ______ 人 |
| **排除4** | 关键指标缺失 >80% 或 POF 无法判定 | 剔除 ______ 人 |
| **最终纳入** | 外部验证 | 与提取脚本一致 |

---

## 三、SQL 脚本位置

```
scripts/sql/
├── 01_mimic_extraction.sql      # MIMIC 主提取（先执行）
├── 01_mimic_flowchart_counts.sql # MIMIC 流程图计数
├── 08_eicu_extraction.sql       # eICU 主提取（先执行）
└── 08_eicu_flowchart_counts.sql # eICU 流程图计数
```
