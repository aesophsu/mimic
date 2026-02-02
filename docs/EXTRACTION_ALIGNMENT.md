# MIMIC vs eICU 提取逻辑对齐说明

本文档记录 MIMIC-IV 与 eICU 数据提取逻辑的对齐状态，便于跨中心验证与审稿说明。

## 实施摘要

| 对齐项 | 状态 | 说明 |
|--------|------|------|
| 实验室时间窗 | ✅ | eICU 0-1440 min (入ICU 0-24h) |
| 血气/BG/SpO2/Vitals | ✅ | eICU 0-1440 min |
| 葡萄糖/乳酸斜率 | ✅ | eICU 0-1440 min |
| 基线干预 (vent/vaso) | ✅ | eICU 新增 mechanical_vent_flag_day1, vaso_flag_day1 |
| POF 代理 | 近似 | eICU 无 SOFA，用 vent/vaso/P/F/creat 代理 |
| 单位转换 | ✅ | Creat/Bili/Alb/FiO2 已统一 |

---

## 1. 队列纳入标准

| 项目 | MIMIC | eICU | 对齐状态 |
|------|-------|------|----------|
| AP 诊断 | ICD-9 5770, ICD-10 K85% | diagnosisstring ILIKE '%pancreatit%' NOT '%chronic%' | 近似（eICU 无 ICD，用文本） |
| 年龄 | ≥18 | ≥18 | ✅ |
| ICU 时长 | los ≥ 1 day | icu_los_hours ≥ 24 | ✅ |
| 首次住院 | ap_stay_seq = 1 | stay_rank = 1 | ✅ |
| 体重/身高 | first_day_weight/height | admissionweight 30-300, height 120-250 | ✅ |

## 2. 时间窗口（关键对齐点）

| 数据类型 | MIMIC | eICU (offset 单位: 分钟) | 对齐状态 |
|----------|-------|-------------------------|----------|
| **实验室** | admittime-6h 至 intime+24h (labevents) | 0 至 1440 (入ICU 0-24h) | ✅ 已收紧 |
| **血气/BG** | intime 至 intime+24h | 0 至 1440 | ✅ 已对齐 |
| **P/F ratio** | first_day_bg (PaO2 only) | 0 至 1440 (PaO2 only, NULL when unavailable) | ✅ 已对齐 |
| **SpO2/Vitals** | intime 至 intime+24h | 0 至 1440 | ✅ 已对齐 |
| **葡萄糖/乳酸斜率** | intime 至 intime+24h | 0 至 1440 | ✅ 已对齐 |
| **机械通气 (基线)** | intime 至 intime+24h (首日) | 0 至 1440 (treatment/careplan) | ✅ 已增加 mechanical_vent_flag_day1 |
| **血管加压药 (基线)** | intime 至 intime+24h | 0 至 1440 | ✅ 已增加 vaso_flag_day1 |
| **机械通气/血管加压药 (POF)** | — | 1440-10080 (day1-7) | eICU POF 代理仍用 day1-7 |
| **POF 判定** | SOFA day1-7 (intime+24h 至 +7d) | 代理规则 (vent/vaso/P/F/creat) | 定义不同，eICU 为简化代理 |

## 3. 单位转换

| 指标 | MIMIC | eICU | 对齐 |
|------|-------|------|------|
| Creatinine | mg/dL | >30 时 /88.4 (umol/L→mg/dL) | ✅ |
| Bilirubin | mg/dL | >10 时 /17.1 (umol/L→mg/dL) | ✅ |
| Albumin | g/dL | >10 时 /10 (g/L→g/dL) | ✅ |
| BUN | mg/dL | 保持 | ✅ |
| FiO2 | 0-1 | ≥21 时 /100 (百分数→小数) | ✅ |

## 4. POF 定义

**MIMIC** (SOFA 标准):
- resp_pof: 呼吸 SOFA≥2 持续 ≥2 天 (day1-7, intime+24h 至 +7d)
- cv_pof: 心血管 SOFA≥2 持续 ≥2 天
- renal_pof: 肾脏 SOFA≥2 持续 ≥2 天

**eICU** (代理规则，无 SOFA，时间窗 day1-7 与 MIMIC 概念一致):
- vaso_flag=1 且 los≥1 (day1-7 treatment)
- mechanical_vent_flag=1 且 los≥2 (day1-7 treatment/careplan)
- pao2fio2ratio_min<300 且 los≥2
- dialysis_flag=1
- creatinine_max>1.9

**说明**：eICU 无 SOFA，POF 为简化代理；基线干预 (Table 1、模型输入) 已用首日 (0-1440 min) 与 MIMIC 对齐。

## 5. 28 天死亡与早期死亡

| 项目 | MIMIC | eICU |
|------|-------|------|
| 28天死亡 | deathtime/dod ≤ intime+28d | hosp_mort=1 且 hospitaldischargeoffset≤40320 (28×24×60 min) |
| 24-48h 早期死亡 | deathtime/dod ∈ [intime+24h, intime+48h] | unitdischargestatus='Expired' 且 unitdischargeoffset ∈ [1440, 2880] min |
| 早期死亡→POF 覆盖 | 01_mimic_cleaning 中 pof=1, mortality=1 | 08_eicu_alignment 中 pof=1, mortality=1 |
| 对齐 | ✅ | ✅ |

## 6. 干预时间窗（已实施）

- **基线/Table 1**：eICU 已增加 `mechanical_vent_flag_day1`、`vaso_flag_day1`（0-1440 min），与 MIMIC intime 至 intime+24h 对齐；预处理脚本 09 将其映射为 `mechanical_vent_flag`、`vaso_flag` 供模型推理。
- **POF 代理**：eICU 仍使用 day1-7 (1440-10080 min) 的 vent/vaso 判定，与 MIMIC SOFA day1-7 时间窗概念一致。
