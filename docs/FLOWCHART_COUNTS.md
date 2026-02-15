# 流程计数说明

本文档说明如何获取队列纳入排除流程所需的人数统计。

## 1. MIMIC 计数
1. 执行 `scripts/sql/01_mimic_extraction.sql`。
2. 执行 `scripts/sql/01_mimic_flowchart_counts.sql`。
3. 将输出人数用于流程图或结果记录。

## 2. eICU 计数
1. 执行 `scripts/sql/08_eicu_extraction.sql`。
2. 执行 `scripts/sql/08_eicu_flowchart_counts.sql`。
3. 将输出人数用于流程图或结果记录。

## 3. 输出字段建议
- 初始池人数
- LOS 过滤后人数
- 首次 ICU 过滤后人数
- 年龄过滤后人数
- 关键变量缺失过滤后最终人数

## 4. 脚本位置
- `scripts/sql/01_mimic_flowchart_counts.sql`
- `scripts/sql/08_eicu_flowchart_counts.sql`
