--------------------------------------------------------------------------------
-- Figure 1 Flowchart: MIMIC-IV 纳入排除计数
-- 在 MIMIC 数据库中执行，输出 CONSORT 流程图所需的人数
--
-- 执行顺序：
--   1. 先运行 01_mimic_extraction.sql（生成 my_custom_schema.ap_final_analysis_cohort）
--   2. 再运行本脚本
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Step 0: 初始池 - AP 诊断的 ICU 入住（ICD-9: 5770, ICD-10: K85%）
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_ap;
CREATE TEMP TABLE temp_flowchart_ap AS
SELECT DISTINCT hadm_id, subject_id
FROM mimiciv_hosp.diagnoses_icd
WHERE (icd_version = 9 AND icd_code = '5770')
   OR (icd_version = 10 AND icd_code LIKE 'K85%');

DROP TABLE IF EXISTS temp_flowchart_step0;
CREATE TEMP TABLE temp_flowchart_step0 AS
SELECT icu.stay_id, icu.subject_id, icu.hadm_id, icu.intime, icu.los,
       (EXTRACT(YEAR FROM icu.intime) - pat.anchor_year + pat.anchor_age) AS admission_age
FROM mimiciv_icu.icustays icu
INNER JOIN temp_flowchart_ap ap ON icu.hadm_id = ap.hadm_id
INNER JOIN mimiciv_hosp.patients pat ON icu.subject_id = pat.subject_id;

--------------------------------------------------------------------------------
-- Step 1: 排除非首次 ICU - 仅保留每位患者首次入住
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step1;
CREATE TEMP TABLE temp_flowchart_step1 AS
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS stay_seq
    FROM temp_flowchart_step0
) t WHERE stay_seq = 1;

--------------------------------------------------------------------------------
-- Step 2: 排除年龄 < 18 岁
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step2;
CREATE TEMP TABLE temp_flowchart_step2 AS
SELECT * FROM temp_flowchart_step1
WHERE admission_age >= 18;

--------------------------------------------------------------------------------
-- Step 3: 排除 ICU LOS < 24 小时（los 单位为天，>=1 即 >=24h）
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step3;
CREATE TEMP TABLE temp_flowchart_step3 AS
SELECT * FROM temp_flowchart_step2
WHERE los >= 1;

--------------------------------------------------------------------------------
-- Step 4: 排除关键生理指标缺失严重（>80% 缺失）
-- 关键指标（10项）：creatinine_max, bun_max, lactate_max, pao2fio2ratio_min, 
-- ph_min, wbc_max, hemoglobin_min, bilirubin_total_max, sodium_max, albumin_min
-- 依赖 ap_final_analysis_cohort（由 01_mimic_extraction.sql 生成）
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_flowchart_step4;
CREATE TEMP TABLE temp_flowchart_step4 AS
SELECT c.stay_id
FROM temp_flowchart_step3 c
INNER JOIN my_custom_schema.ap_final_analysis_cohort f ON c.stay_id = f.stay_id
WHERE (
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
-- 汇总输出：Figure 1 Flowchart 计数（可直接填入 Figure 1）
--------------------------------------------------------------------------------
SELECT '=== MIMIC-IV Flowchart Counts (Figure 1) ===' AS section;

SELECT 
    '初始池: AP诊断的ICU入住' AS step,
    (SELECT COUNT(*)::int FROM temp_flowchart_step0) AS n,
    NULL::int AS excluded
UNION ALL
SELECT 
    '排除1: 非首次ICU（保留首次）',
    (SELECT COUNT(*)::int FROM temp_flowchart_step1),
    (SELECT COUNT(*)::int FROM temp_flowchart_step0) - (SELECT COUNT(*)::int FROM temp_flowchart_step1)
UNION ALL
SELECT 
    '排除2: 年龄<18岁',
    (SELECT COUNT(*)::int FROM temp_flowchart_step2),
    (SELECT COUNT(*)::int FROM temp_flowchart_step1) - (SELECT COUNT(*)::int FROM temp_flowchart_step2)
UNION ALL
SELECT 
    '排除3: ICU LOS<24h',
    (SELECT COUNT(*)::int FROM temp_flowchart_step3),
    (SELECT COUNT(*)::int FROM temp_flowchart_step2) - (SELECT COUNT(*)::int FROM temp_flowchart_step3)
UNION ALL
SELECT 
    '排除4: 关键生理指标缺失>80%',
    (SELECT COUNT(*)::int FROM temp_flowchart_step4),
    (SELECT COUNT(*)::int FROM temp_flowchart_step3) - (SELECT COUNT(*)::int FROM temp_flowchart_step4);

SELECT '--- 最终纳入 ---' AS section;
SELECT COUNT(*)::int AS final_n FROM temp_flowchart_step4;
