--------------------------------------------------------------------------------
-- Figure 1 Flowchart: eICU 纳入排除计数
-- 在 eICU 数据库中执行，输出 CONSORT 流程图所需的人数
--
-- 执行顺序：
--   1. 先运行 08_eicu_extraction.sql（生成 eicu_cview.ap_external_validation）
--   2. 再运行本脚本
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Step 0: 初始池 - AP 诊断的 ICU 入住
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_ap;
CREATE TEMP TABLE temp_flowchart_ap AS
SELECT patientunitstayid
FROM eicu_crd.diagnosis
WHERE diagnosisstring ILIKE '%pancreatit%'
  AND diagnosisstring NOT ILIKE '%chronic%'
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_flowchart_step0;
CREATE TEMP TABLE temp_flowchart_step0 AS
SELECT i.patientunitstayid, i.uniquepid,
       i.icu_los_hours / 24.0 AS los,
       CASE WHEN i.age = '> 89' THEN 90 
            WHEN i.age ~ '^[0-9]+$' THEN CAST(i.age AS INT) 
            ELSE 0 END AS admission_age
FROM eicu_derived.icustay_detail i
INNER JOIN temp_flowchart_ap ap ON i.patientunitstayid = ap.patientunitstayid;

--------------------------------------------------------------------------------
-- Step 1: 排除 ICU LOS < 24 小时（先 LOS）
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step1;
CREATE TEMP TABLE temp_flowchart_step1 AS
SELECT * FROM temp_flowchart_step0
WHERE los >= 1;

--------------------------------------------------------------------------------
-- Step 2: 排除非首次 ICU - 在 LOS>=24h 的入住中保留每位患者首次入住（再首次）
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step2;
CREATE TEMP TABLE temp_flowchart_step2 AS
SELECT t.patientunitstayid, t.uniquepid, t.los, t.admission_age
FROM (
    SELECT i.patientunitstayid, i.uniquepid, i.icu_los_hours/24.0 AS los,
           CASE WHEN i.age = '> 89' THEN 90 WHEN i.age ~ '^[0-9]+$' THEN CAST(i.age AS INT) ELSE 0 END AS admission_age,
           ROW_NUMBER() OVER (PARTITION BY i.uniquepid ORDER BY i.hospitaladmitoffset ASC, i.unitadmitoffset ASC) AS stay_rank
    FROM eicu_derived.icustay_detail i
    INNER JOIN temp_flowchart_ap ap ON i.patientunitstayid = ap.patientunitstayid
    WHERE i.icu_los_hours >= 24  -- 先 LOS
) t
WHERE t.stay_rank = 1;

--------------------------------------------------------------------------------
-- Step 3: 排除年龄 < 18 岁
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step3;
CREATE TEMP TABLE temp_flowchart_step3 AS
SELECT * FROM temp_flowchart_step2
WHERE admission_age >= 18;

--------------------------------------------------------------------------------
-- Step 4: 排除关键生理指标缺失严重（>80%）及 pof=NULL 无法判定者
-- 与 MIMIC 对齐的 10 项关键指标；eICU 最终表还排除 pof IS NULL
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step4;
CREATE TEMP TABLE temp_flowchart_step4 AS
SELECT c.patientunitstayid
FROM temp_flowchart_step3 c
INNER JOIN eicu_cview.ap_external_validation f ON c.patientunitstayid = f.patientunitstayid
WHERE f.pof IS NOT NULL  -- 排除无法判定 POF 者（与 08 主脚本一致）
  AND (
    (CASE WHEN f.creatinine_max IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.bun_max IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.lactate_max IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.pao2fio2ratio_min IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.ph_min IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.wbc_max IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.hemoglobin_min IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.bilirubin_total_max IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.sodium_max IS NULL THEN 1 ELSE 0 END) +
    (CASE WHEN f.albumin_min IS NULL THEN 1 ELSE 0 END)
  )::float / 10.0 <= 0.8;

--------------------------------------------------------------------------------
-- 汇总输出：Figure 1 Flowchart 计数（eICU 外部验证）
--------------------------------------------------------------------------------
SELECT '=== eICU Flowchart Counts (Figure 1 - External Validation) ===' AS section;

SELECT 
    '初始池: AP诊断的ICU入住' AS step,
    (SELECT COUNT(*)::int FROM temp_flowchart_step0) AS n,
    NULL::int AS excluded
UNION ALL
SELECT 
    '排除1: ICU LOS<24h（先LOS）',
    (SELECT COUNT(*)::int FROM temp_flowchart_step1),
    (SELECT COUNT(*)::int FROM temp_flowchart_step0) - (SELECT COUNT(*)::int FROM temp_flowchart_step1)
UNION ALL
SELECT 
    '排除2: 非首次ICU（保留首次，再首次）',
    (SELECT COUNT(*)::int FROM temp_flowchart_step2),
    (SELECT COUNT(*)::int FROM temp_flowchart_step1) - (SELECT COUNT(*)::int FROM temp_flowchart_step2)
UNION ALL
SELECT 
    '排除3: 年龄<18岁',
    (SELECT COUNT(*)::int FROM temp_flowchart_step3),
    (SELECT COUNT(*)::int FROM temp_flowchart_step2) - (SELECT COUNT(*)::int FROM temp_flowchart_step3)
UNION ALL
SELECT 
    '排除4: 关键指标缺失>80% 或 pof无法判定',
    (SELECT COUNT(*)::int FROM temp_flowchart_step4),
    (SELECT COUNT(*)::int FROM temp_flowchart_step3) - (SELECT COUNT(*)::int FROM temp_flowchart_step4);

SELECT '--- 最终纳入（外部验证） ---' AS section;
SELECT COUNT(*)::int AS final_n FROM temp_flowchart_step4;
