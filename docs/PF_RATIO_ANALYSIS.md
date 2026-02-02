# P/F Ratio 提取逻辑差异分析

## 一、现象

Table 1 基线表显示 MIMIC 与 eICU 的 P/F ratio 存在**极大差异**：

| 队列 | P/F ratio (median [Q1, Q3]) | SMD (MIMIC vs eICU) |
|------|-----------------------------|---------------------|
| MIMIC Overall | 120.0 [75.0, 191.7] | — |
| MIMIC Non-POF | 157.0 [96.0, 248.1] | — |
| MIMIC POF | 104.0 [68.3, 168.0] | — |
| **eICU (External Validation)** | **394.3 [366.4, 411.4]** | **2.68** |

- **KS 统计量**：0.73（漂移分析中最高，见 `validation/eicu_vs_mimic_drift.json`）
- **临床意义**：MIMIC 中位数 120 提示严重低氧（ARDS 范围）；eICU 394 接近正常氧合（>300 为正常）

---

## 二、提取逻辑对比

### 2.1 MIMIC (`scripts/sql/01_mimic_extraction.sql`)

```sql
LEFT JOIN mimiciv_derived.first_day_bg bg ON c.stay_id = bg.stay_id
-- 使用 bg.pao2fio2ratio_min, bg.pao2fio2ratio_max
```

- **数据源**：`mimiciv_derived.first_day_bg`（MIMIC-IV 官方衍生表）
- **计算方式**：**仅基于动脉血气** PaO2/FiO2，入 ICU 后首 24 小时内
- **覆盖人群**：**仅在有 ABG 记录的患者** 有 P/F 值；无 ABG 者为 NULL
- **选择偏倚**：ICU 中通常对病情较重者（如机械通气、呼吸窘迫）行 ABG，故 MIMIC 的 P/F 多来自病情更重人群

### 2.2 eICU (`scripts/sql/08_eicu_extraction.sql` 第 272–304 行)

```sql
-- PaO2 可用时：pf_val = pao2 / FiO2
oxy_data: pao2 / (CASE WHEN fio2 IS NULL THEN 0.21 ... END)

-- PaO2 不可用时：SpO2 回退
vital_oxy: spo2 / (CASE WHEN b.fio2 IS NULL THEN 0.21 ... END)  -- sf_val
-- LEFT JOIN pivoted_bg b ON ... AND v.chartoffset = b.chartoffset

-- 最终取值
COALESCE(MIN(o.pf_val), MIN(v.sf_val) * 0.9) AS pao2fio2ratio_min
```

- **数据源**：`eicu_derived.pivoted_bg`（PaO2）与 `pivoted_vital`（SpO2）
- **回退逻辑**：若无 PaO2，则用 **SpO2/FiO2 × 0.9** 近似
- **FiO2 默认**：FiO2 为 NULL 时按 **0.21（室内空气）** 处理
- **vital_oxy 的 JOIN**：SpO2 与 FiO2 按 `chartoffset` 精确匹配；无匹配时 `b.fio2 IS NULL` → 0.21

---

## 三、问题根因分析

### 3.1 SpO2 回退导致系统性高估

当 eICU 患者**无首 24 小时内 PaO2** 时，会使用 SpO2 回退：

- 公式：`SpO2 / 0.21 × 0.9`（FiO2 默认 0.21）
- 示例：SpO2 92% → 92/0.21×0.9 ≈ **394**
- eICU 中位数 394.3 与上述计算高度一致，提示**大量 eICU 患者实际使用了 SpO2 回退**，而非真实 PaO2/FiO2

### 3.2 FiO2 = 0.21 的默认值放大偏差

- `vital_oxy` 中 SpO2 与 FiO2 通过 `chartoffset` 精确匹配
- 无血气记录时，`LEFT JOIN` 无匹配 → FiO2 固定为 0.21
- 吸氧或机械通气患者若 FiO2 未同步记录，会被当作室内空气，**P/F 被系统性高估**

### 3.3 人群与测量差异

| 维度 | MIMIC | eICU |
|------|-------|------|
| P/F 来源 | 仅动脉血气 | PaO2 或 SpO2 回退 |
| 有 P/F 的人群 | 有 ABG 者（多病情较重） | 几乎全部（含无 ABG 者） |
| 机械通气比例 | 34.9% | 96.7% |

eICU 机械通气比例更高，按理应有更多血气，但 P/F 反而更高，说明：

1. eICU `pivoted_bg` 中 PaO2 覆盖率可能低于预期，或
2. 时间窗/匹配逻辑导致大量患者实际走 SpO2 回退

### 3.4 SpO2/FiO2 与 PaO2/FiO2 的换算局限

文献中 SpO2/FiO2 与 PaO2/FiO2 的关系为**非线性**（氧解离曲线），简单乘以 0.9 仅适用于特定 SpO2 范围，且：

- SpO2 90–100% 时，对应 PaO2 范围很宽（约 60–500+ mmHg）
- 固定系数 0.9 在 SpO2 较高时会**系统性高估** P/F

---

## 四、结论：是否为提取逻辑问题？

**是的，主要是提取逻辑差异导致的系统性偏差**，而非单纯人群差异：

1. **eICU 的 SpO2 回退**：在无 PaO2 时用 SpO2/FiO2×0.9，且 FiO2 常默认为 0.21，使 P/F 系统性偏高。
2. **MIMIC 无回退**：仅使用真实 PaO2/FiO2，且多来自病情较重、有 ABG 的患者。
3. **定义不一致**：两库的 P/F 在“有/无血气”“是否使用 SpO2”上定义不同，直接比较会夸大差异。

---

## 五、已实施的修正（方案 A）

**eICU 仅使用真实 PaO2**（`scripts/sql/08_eicu_extraction.sql` 已修改）：

- 在 eICU 中**仅保留有 PaO2 记录**患者的 P/F
- 无 PaO2 者将 `pao2fio2ratio_min`、`pao2fio2ratio_max` 设为 NULL
- 已移除 SpO2/FiO2×0.9 回退逻辑
- 与 MIMIC 定义一致，可比性提升

**注意**：需重新执行 08 SQL 及后续预处理（08_eicu_alignment_cleaning.py 等）以更新数据。

### 方案 B：放宽 FiO2 匹配条件

- 当前：SpO2 与 FiO2 按 `chartoffset` 精确匹配
- 改进：在 ±30–60 分钟内取最近 FiO2，减少“无匹配 → 默认 0.21”的情况
- 需检查 eICU `pivoted_bg` 与 `pivoted_vital` 的时间粒度

### 方案 C：敏感性分析

- 主分析：采用方案 A（仅 PaO2）
- 敏感性分析：包含 SpO2 回退，并在文中明确标注“含 SpO2 估算”
- 或：排除 P/F 特征后重新建模，评估其对模型的影响

### 方案 D：文档化现状

- 在 Table 1 脚注和 Methods 中明确说明：
  - MIMIC：仅动脉血气
  - eICU：无 PaO2 时用 SpO2/FiO2×0.9，FiO2 缺失时默认 0.21
- 将大 SMD 解释为**测量与定义差异**，而非单纯人群差异

---

## 六、建议的验证查询（在 eICU 中执行）

```sql
-- 1. 首 24 小时内有 PaO2 的患者比例
SELECT 
  COUNT(DISTINCT c.patientunitstayid) AS total,
  COUNT(DISTINCT o.patientunitstayid) AS with_pao2,
  COUNT(DISTINCT c.patientunitstayid) - COUNT(DISTINCT o.patientunitstayid) AS spo2_fallback_only
FROM cohort_base c
LEFT JOIN (
  SELECT DISTINCT patientunitstayid 
  FROM eicu_derived.pivoted_bg 
  WHERE pao2 > 0 AND chartoffset BETWEEN 0 AND 1440
) o ON c.patientunitstayid = o.patientunitstayid;

-- 2. PaO2 组 vs SpO2 回退组的 P/F 分布
-- 若有显著差异，可支持“回退导致高估”的结论
```

执行上述查询可量化 PaO2 覆盖率及回退使用比例，为选择修正方案提供依据。
