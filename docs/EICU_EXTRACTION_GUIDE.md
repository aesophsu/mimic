# eICU 数据提取使用指南

本指南说明如何在本地 eICU 数据库上运行 `08_eicu_extraction.sql` 并导出数据供项目使用。

---

## 一、前置条件

### 1.1 数据库环境

- **PostgreSQL**：eICU 使用 PostgreSQL 存储
- **eICU-CRD**：已按官方流程完成安装，包含：
  - `eicu_crd` schema：原始表（diagnosis, lab, patient, treatment, careplangeneral, apacheapsvar 等）
  - `eicu_derived` schema：衍生表（icustay_detail, pivoted_bg, pivoted_vital）

> 若尚未安装 eICU，请参考 [eICU-CRD 官方教程](https://eicu-crd.mit.edu/tutorials/install_eicu_locally/) 完成数据导入与衍生表构建。

### 1.2 创建输出 schema

脚本会将结果写入 `eicu_cview.ap_external_validation`，需先创建 schema：

```sql
CREATE SCHEMA IF NOT EXISTS eicu_cview;
```

---

## 二、执行步骤

### 2.1 连接数据库

使用 `psql` 或图形化工具（如 pgAdmin、DBeaver）连接到你的 eICU 数据库。

```bash
# 示例：使用 psql 连接
psql -h localhost -U your_username -d eicu
```

### 2.2 创建 schema（若尚未创建）

```sql
CREATE SCHEMA IF NOT EXISTS eicu_cview;
```

### 2.3 运行提取脚本

**方式 A：psql 命令行（推荐）**

```bash
cd /Users/sue/Documents/mimic
psql -h localhost -U your_username -d eicu -f scripts/sql/08_eicu_extraction.sql
```

将 `your_username` 替换为你的 PostgreSQL 用户名，`eicu` 替换为你的 eICU 数据库名。

**方式 B：在 psql 交互环境中**

```bash
psql -h localhost -U your_username -d eicu
```

连接成功后，在 psql 提示符下执行：

```sql
\i /Users/sue/Documents/mimic/scripts/sql/08_eicu_extraction.sql
```

**方式 C：图形化工具（pgAdmin / DBeaver）**

1. 连接到 eICU 数据库
2. 打开 `scripts/sql/08_eicu_extraction.sql`
3. 选择全部内容并执行（Execute）

### 2.3.1 队列纳入与排除（与 MIMIC 对齐）

当前脚本内已实现与 MIMIC 一致的纳入逻辑，便于在 Methods 中直接描述：

- **AP 诊断**：`diagnosisstring ILIKE '%pancreatit%' AND NOT ILIKE '%chronic%'`（排除慢性胰腺炎）  
- **ICU 住院时间**：先按 `icu_los_hours >= 24` 过滤 ICU 入住（相当于 LOS≥1 天，**先 LOS**）  
- **首次 ICU 入住**：在满足 LOS 条件的入住中，按 `hospitaladmitoffset` + `unitadmitoffset` 排序取 `stay_rank = 1`（**再首次**）  
- **年龄**：`age_num >= 18`（将 `'> 89'` 统一记为 90 岁）  
- **关键生理指标缺失**：在 10 项核心实验室指标中（creatinine_max, bun_max, lactate_max, pao2fio2ratio_min, ph_min, wbc_max, hemoglobin_min, bilirubin_total_max, sodium_max, albumin_min），若缺失比例 >80% 或 POF 无法判定（`pof IS NULL`），则排除。  
- **P/F 比**：仅使用动脉血气 PaO2/FiO2，时间窗为入 ICU 后 0–24 小时；无 PaO2 时 P/F 记为 NULL，不再使用 SpO2 回退。

实际纳入/排除人数可通过运行 `08_eicu_flowchart_counts.sql` 获得（详见 `docs/FLOWCHART_COUNTS.md`），对应稿件 Figure 1（CONSORT 流程图）。

### 2.4 运行流程图计数（Figure 1）

主提取完成后，执行流程图计数脚本获取纳入排除人数：

```bash
psql -h localhost -U your_username -d eicu -f scripts/sql/08_eicu_flowchart_counts.sql
```

或在 psql 中：

```sql
\i /Users/sue/Documents/mimic/scripts/sql/08_eicu_flowchart_counts.sql
```

输出将显示初始池、各排除步骤剔除人数及最终纳入人数。

### 2.5 导出为 CSV

脚本执行完成后，会生成表 `eicu_cview.ap_external_validation`。需将其导出为 CSV 并放到项目指定位置：

**方式 A：psql \copy（推荐）**

```bash
psql -h localhost -U your_username -d eicu -c "\copy (SELECT * FROM eicu_cview.ap_external_validation) TO '/path/to/mimic/data/raw/eicu_raw_data.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
```

**方式 B：在 psql 交互环境中**

```sql
\copy (SELECT * FROM eicu_cview.ap_external_validation) TO '/path/to/mimic/data/raw/eicu_raw_data.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
```

> 将 `/path/to/mimic` 替换为你的项目根目录，例如 `/Users/sue/Documents/mimic`。

**方式 C：SQL COPY（需 PostgreSQL 超级用户或文件权限）**

```sql
COPY eicu_cview.ap_external_validation TO '/path/to/mimic/data/raw/eicu_raw_data.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
```

---

## 三、验证输出

1. 确认文件存在：`data/raw/eicu_raw_data.csv`
2. 检查行数：应包含表头 + 若干行数据
3. 后续运行 Python 流程：

```bash
uv run python run_all.py --eicu-only
```

---

## 四、常见问题

### Q1：报错 `relation "eicu_derived.icustay_detail" does not exist`

**原因**：`eicu_derived` 中的衍生表未构建。

**解决**：使用 [eICU-CRD 官方代码](https://github.com/MIT-LCP/eicu-code) 中的 `build-db/postgres/` 脚本生成 `icustay_detail`、`pivoted_bg`、`pivoted_vital` 等表。

### Q2：\copy 报错 "could not open file for writing"

**原因**：路径不存在或权限不足。

**解决**：先创建目录 `mkdir -p data/raw`，并确保路径为绝对路径；Windows 下路径格式为 `C:/path/to/mimic/data/raw/eicu_raw_data.csv`。

### Q3：导出文件编码问题

**解决**：使用 `ENCODING 'UTF8'`，Excel 打开时选择 UTF-8 编码；或使用 `ENCODING 'UTF8'` 并保存为带 BOM 的 UTF-8。

---

## 五、一键脚本示例（可选）

可将连接信息与路径写入脚本，便于重复执行：

```bash
#!/bin/bash
# extract_eicu.sh - 需根据本机修改以下变量
DB_HOST=localhost
DB_USER=your_username
DB_NAME=eicu
PROJECT_ROOT=/Users/sue/Documents/mimic

psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "CREATE SCHEMA IF NOT EXISTS eicu_cview;"
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f "$PROJECT_ROOT/scripts/sql/08_eicu_extraction.sql"
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f "$PROJECT_ROOT/scripts/sql/08_eicu_flowchart_counts.sql"
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "\copy (SELECT * FROM eicu_cview.ap_external_validation) TO '$PROJECT_ROOT/data/raw/eicu_raw_data.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
echo "Done. Output: $PROJECT_ROOT/data/raw/eicu_raw_data.csv"
```

保存后执行：`chmod +x extract_eicu.sh && ./extract_eicu.sh`
