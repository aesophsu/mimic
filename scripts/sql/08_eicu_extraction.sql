--------------------------------------------------------------------------------
-- 1. 识别 AP 患者
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_ap_patients;
CREATE TEMP TABLE temp_ap_patients AS
SELECT patientunitstayid
FROM eicu_crd.diagnosis
WHERE diagnosisstring ILIKE '%pancreatit%'
  AND diagnosisstring NOT ILIKE '%chronic%'
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 2. 构建核心队列 (18岁以上, ICU >= 24h)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
WITH ranked_stays AS (
    SELECT i.*,
        ROW_NUMBER() OVER (
            PARTITION BY i.uniquepid 
            ORDER BY i.hospitaladmitoffset ASC, i.unitadmitoffset ASC
        ) AS stay_rank
    FROM eicu_derived.icustay_detail i
    INNER JOIN temp_ap_patients ap ON i.patientunitstayid = ap.patientunitstayid
),
age_parsed AS (
    SELECT *,
        CASE WHEN age = '> 89' THEN 90 
             WHEN age ~ '^[0-9]+$' THEN CAST(age AS INT) 
             ELSE 0 END AS age_num
    FROM ranked_stays
)
SELECT 
    patientunitstayid,
    uniquepid,
    age_num AS admission_age,
    gender,
    CASE WHEN admissionheight BETWEEN 120 AND 250 THEN admissionheight ELSE NULL END AS height_admit,
    CASE WHEN admissionweight BETWEEN 30 AND 300 THEN admissionweight ELSE NULL END AS weight_admit,
    icu_los_hours / 24.0 AS los,
    hosp_mort,
    CASE 
        WHEN (admissionheight BETWEEN 120 AND 250) AND (admissionweight BETWEEN 30 AND 300) 
        THEN (admissionweight / POWER(admissionheight / 100.0, 2)) 
        ELSE NULL 
    END AS bmi
FROM age_parsed
WHERE stay_rank = 1 
  AND icu_los_hours >= 24
  AND age_num >= 18;
CREATE INDEX IF NOT EXISTS idx_cohort_pid ON cohort_base(patientunitstayid);
ANALYZE cohort_base;

--------------------------------------------------------------------------------
-- 3. 深度打捞实验室指标 (补全 RDW, PT, INR, Hct, Bilirubin_Max)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_raw_all;
CREATE TEMP TABLE temp_lab_raw_all AS
WITH lab_filt AS (
    SELECT 
        patientunitstayid, labname, labresult,
        CASE 
            -- 1. BUN 单位 mg/dL (保持原始)
            WHEN labname ILIKE '%BUN%' AND labresult BETWEEN 1 AND 200 
                THEN labresult
            
            -- 2. Creatinine 单位转换补丁 (umol/L -> mg/dL)
            WHEN labname ILIKE '%creatinine%' THEN 
                CASE 
                    WHEN labresult > 30 THEN labresult / 88.4 
                    WHEN labresult BETWEEN 0.1 AND 30 THEN labresult 
                    ELSE NULL 
                END

            -- 3. Hemoglobin & Hematocrit (Hgb: 4-25, Hct: 12-75)
            WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%', '%total hemoglobin%']) 
                AND labname NOT ILIKE '%A1c%' 
                AND labresult BETWEEN 4 AND 25 THEN labresult 
            WHEN labname ILIKE '%Hct%' AND labresult BETWEEN 12 AND 75 THEN labresult

            -- 4. pH 生理性过滤 (6.7-7.8)
            WHEN labname ILIKE ANY(ARRAY['%pH%', '%arterial pH%']) AND labname NOT ILIKE ANY(ARRAY['%urine%','%fluid%'])
                AND labresult BETWEEN 6.7 AND 7.8 AND labresult NOT IN (7.0, 8.0) THEN labresult
            
            -- 5. 凝血功能 (PTT: 10-150, PT: 5-150, INR: 0.5-20)
            WHEN labname ILIKE ANY(ARRAY['%PTT%', '%Partial Thromboplastin Time%', '%aPTT%']) 
                AND labresult BETWEEN 10 AND 150 THEN labresult
            WHEN labname ILIKE '%PT%' AND labname NOT ILIKE '%PTT%' AND labresult BETWEEN 5 AND 150 THEN labresult
            WHEN labname ILIKE '%INR%' AND labresult BETWEEN 0.5 AND 20 THEN labresult

            -- 6. 乳酸打捞 (0.1-30 mmol/L)
            WHEN labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%', '%lac%']) 
                AND labresult BETWEEN 0.1 AND 30 THEN labresult

            -- 7. RDW (10-35)
            WHEN labname ILIKE '%RDW%' AND labresult BETWEEN 10 AND 35 THEN labresult

            -- 8. 其他常规指标 (带单位兼容性转换)
            WHEN labname ILIKE '%paCO2%' AND labresult BETWEEN 5 AND 150 THEN labresult
            WHEN (labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%') AND labname NOT ILIKE '%total co2%'
                AND labresult BETWEEN 2 AND 60 THEN labresult
            WHEN labname ILIKE '%WBC%' AND labresult BETWEEN 0.1 AND 500 THEN labresult
            
            -- 8a. Albumin (兼容 g/L 转 g/dL)
            WHEN labname ILIKE '%albumin%' THEN 
                CASE 
                    WHEN labresult > 10 AND (labresult / 10.0) BETWEEN 1.0 AND 6.0 THEN labresult / 10.0
                    WHEN labresult BETWEEN 1.0 AND 6.0 THEN labresult 
                    ELSE NULL 
                END
            
            WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' AND labresult BETWEEN 4 AND 15 THEN labresult
            WHEN labname ILIKE '%AST%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            WHEN labname ILIKE '%ALT%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            WHEN labname ILIKE '%platelet%' AND labresult BETWEEN 1 AND 1000 THEN labresult
            WHEN labname ILIKE '%anion gap%' AND labresult BETWEEN 2 AND 50 THEN labresult
            
            -- 8b. Total Bilirubin (兼容 umol/L 转 mg/dL)
            WHEN labname ILIKE '%total bilirubin%' THEN 
                CASE 
                    WHEN labresult > 10 AND (labresult / 17.1) BETWEEN 0.1 AND 70 THEN labresult / 17.1
                    WHEN labresult BETWEEN 0.1 AND 70 THEN labresult 
                    ELSE NULL 
                END
                
            WHEN labname ILIKE '%glucose%' AND labresult BETWEEN 10 AND 2000 THEN labresult
            WHEN labname ILIKE '%alkaline phos%' AND labresult BETWEEN 5 AND 2500 THEN labresult

            -- 8c. Chloride (90-130 mmol/L)
            WHEN labname ILIKE '%chloride%' AND labresult BETWEEN 90 AND 130 THEN labresult

            -- 8d. Phosphate (0.5-15 mg/dL)
            WHEN labname ILIKE '%phosphate%' AND labresult BETWEEN 0.5 AND 15 THEN labresult

            -- 8e. CRP (0-500 mg/L)
            WHEN labname ILIKE '%CRP%' AND labname NOT ILIKE '%creatinine%' AND labresult BETWEEN 0 AND 500 THEN labresult

            -- 8f. Amylase (0-5000 IU/L)
            WHEN labname ILIKE '%amylase%' AND labresult BETWEEN 0 AND 5000 THEN labresult

            -- 8g. Lipase (0-5000 U/L)
            WHEN labname ILIKE '%lipase%' AND labresult BETWEEN 0 AND 5000 THEN labresult

            -- 8h. D-dimer (0.1-50 ug/mL)
            WHEN labname ILIKE '%D-dimer%' AND labresult BETWEEN 0.1 AND 50 THEN labresult

            -- 8i. Fibrinogen (g/L 2-10 转 mg/dL; 或 50-1000 已是 mg/dL)
            WHEN labname ILIKE '%fibrinogen%' THEN 
                CASE WHEN labresult > 0 AND labresult < 15 THEN labresult * 100
                     WHEN labresult BETWEEN 50 AND 1000 THEN labresult ELSE NULL END

            -- 8j. LDH (50-2000 IU/L)
            WHEN labname ILIKE '%LDH%' AND labresult BETWEEN 50 AND 2000 THEN labresult

            -- 8k. Triglycerides (10-2000 mg/dL)
            WHEN labname ILIKE '%triglyceride%' AND labresult BETWEEN 10 AND 2000 THEN labresult

            -- 8l. Total cholesterol (20-500 mg/dL)
            WHEN labname ILIKE '%total cholesterol%' AND labresult BETWEEN 20 AND 500 THEN labresult

            -- 8m. Neutrophils/Lymphocytes (% 0-100)
            WHEN labname ILIKE ANY(ARRAY['%neutrophil%', '%neutrophils%']) AND labresult BETWEEN 0 AND 100 THEN labresult
            WHEN labname ILIKE ANY(ARRAY['%lymphocyte%', '%lymphocytes%']) AND labresult BETWEEN 0 AND 100 THEN labresult

            -- 9. 电解质 (Sodium 120-180 mmol/L, Potassium 1.5-10 mmol/L)
            WHEN labname ILIKE '%sodium%' AND labresult BETWEEN 120 AND 180 THEN labresult
            WHEN labname ILIKE '%potassium%' AND labname NOT ILIKE '%urine%' AND labresult BETWEEN 1.5 AND 10 THEN labresult
            ELSE NULL 
        END AS labresult_clean
    FROM eicu_crd.lab
    WHERE labresultoffset BETWEEN 0 AND 1440 
      AND labresult IS NOT NULL
      AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
)
SELECT 
    patientunitstayid,
    -- 基础血气打捞源
    MIN(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_direct,
    MAX(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_max,
    AVG(CASE WHEN labname ILIKE '%paCO2%' THEN labresult_clean END) AS lab_paco2,
    AVG(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS lab_hco3,
    MIN(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS bicarbonate_min,
    MAX(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS bicarbonate_max,

    -- 肾功
    MAX(CASE WHEN labname ILIKE '%creatinine%' THEN labresult_clean END) AS creatinine_max,
    MIN(CASE WHEN labname ILIKE '%creatinine%' THEN labresult_clean END) AS creatinine_min,
    MAX(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_max,
    MIN(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_min,

    -- 血常规
    MAX(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_max,
    MIN(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_min,
    MAX(CASE WHEN labname ILIKE '%RDW%' THEN labresult_clean END) AS rdw_max,
    MAX(CASE WHEN labname ILIKE '%Hct%' THEN labresult_clean END) AS hematocrit_max,
    MIN(CASE WHEN labname ILIKE '%Hct%' THEN labresult_clean END) AS hematocrit_min,
    MIN(CASE WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%']) THEN labresult_clean END) AS hemoglobin_min,
    MAX(CASE WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%']) THEN labresult_clean END) AS hemoglobin_max,
    MIN(CASE WHEN labname ILIKE '%platelet%' THEN labresult_clean END) AS platelets_min,
    MAX(CASE WHEN labname ILIKE '%platelet%' THEN labresult_clean END) AS platelets_max,

    -- 肝功
    MIN(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_min,
    MAX(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_max,
    MAX(CASE WHEN labname ILIKE '%total bilirubin%' THEN labresult_clean END) AS bilirubin_total_max,
    MIN(CASE WHEN labname ILIKE '%total bilirubin%' THEN labresult_clean END) AS bilirubin_total_min,
    MAX(CASE WHEN labname ILIKE '%AST%' THEN labresult_clean END) AS ast_max,
    MIN(CASE WHEN labname ILIKE '%AST%' THEN labresult_clean END) AS ast_min,
    MAX(CASE WHEN labname ILIKE '%ALT%' THEN labresult_clean END) AS alt_max,
    MIN(CASE WHEN labname ILIKE '%ALT%' THEN labresult_clean END) AS alt_min,
    MIN(CASE WHEN labname ILIKE '%alkaline phos%' THEN labresult_clean END) AS alp_min,
    MAX(CASE WHEN labname ILIKE '%alkaline phos%' THEN labresult_clean END) AS alp_max,

    -- 凝血
    MIN(CASE WHEN labname ILIKE ANY(ARRAY['%PTT%', '%aPTT%']) THEN labresult_clean END) AS ptt_min,
    MAX(CASE WHEN labname ILIKE ANY(ARRAY['%PTT%', '%aPTT%']) THEN labresult_clean END) AS ptt_max,
    MAX(CASE WHEN labname ILIKE '%PT%' AND labname NOT ILIKE '%PTT%' THEN labresult_clean END) AS pt_max,
    MIN(CASE WHEN labname ILIKE '%PT%' AND labname NOT ILIKE '%PTT%' THEN labresult_clean END) AS pt_min,
    MAX(CASE WHEN labname ILIKE '%INR%' THEN labresult_clean END) AS inr_max,
    MIN(CASE WHEN labname ILIKE '%INR%' THEN labresult_clean END) AS inr_min,

    -- 血糖与电解质
    MAX(CASE WHEN labname ILIKE '%glucose%' THEN labresult_clean END) AS glucose_max,
    MIN(CASE WHEN labname ILIKE '%glucose%' THEN labresult_clean END) AS glucose_min,
    MAX(CASE WHEN labname ILIKE '%lactate%' THEN labresult_clean END) AS lactate_max,
    MIN(CASE WHEN labname ILIKE '%lactate%' THEN labresult_clean END) AS lactate_min,
    MAX(CASE WHEN labname ILIKE '%anion gap%' THEN labresult_clean END) AS aniongap_max,
    MIN(CASE WHEN labname ILIKE '%anion gap%' THEN labresult_clean END) AS aniongap_min,
    MAX(CASE WHEN labname ILIKE '%sodium%' THEN labresult_clean END) AS sodium_max,
    MIN(CASE WHEN labname ILIKE '%sodium%' THEN labresult_clean END) AS sodium_min,
    MAX(CASE WHEN labname ILIKE '%potassium%' AND labname NOT ILIKE '%urine%' THEN labresult_clean END) AS potassium_max,
    MIN(CASE WHEN labname ILIKE '%potassium%' AND labname NOT ILIKE '%urine%' THEN labresult_clean END) AS potassium_min,

    -- Chloride, Calcium (与 MIMIC 对齐)
    MIN(CASE WHEN labname ILIKE '%chloride%' THEN labresult_clean END) AS chloride_min,
    MAX(CASE WHEN labname ILIKE '%chloride%' THEN labresult_clean END) AS chloride_max,
    MIN(CASE WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' THEN labresult_clean END) AS lab_calcium_min,
    MAX(CASE WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' THEN labresult_clean END) AS calcium_max,

    -- AP 特异性指标
    MIN(CASE WHEN labname ILIKE '%amylase%' THEN labresult_clean END) AS lab_amylase_min,
    MAX(CASE WHEN labname ILIKE '%amylase%' THEN labresult_clean END) AS lab_amylase_max,
    MAX(CASE WHEN labname ILIKE '%lipase%' THEN labresult_clean END) AS lipase_max,
    MAX(CASE WHEN labname ILIKE '%CRP%' AND labname NOT ILIKE '%creatinine%' THEN labresult_clean END) AS crp_max,
    MIN(CASE WHEN labname ILIKE '%phosphate%' THEN labresult_clean END) AS phosphate_min,
    MAX(CASE WHEN labname ILIKE '%D-dimer%' THEN labresult_clean END) AS d_dimer_max,
    MAX(CASE WHEN labname ILIKE '%fibrinogen%' THEN labresult_clean END) AS fibrinogen_max,
    MAX(CASE WHEN labname ILIKE '%LDH%' THEN labresult_clean END) AS ldh_max,
    MAX(CASE WHEN labname ILIKE '%triglyceride%' THEN labresult_clean END) AS triglycerides_max,
    MIN(CASE WHEN labname ILIKE '%total cholesterol%' THEN labresult_clean END) AS total_cholesterol_min,
    AVG(CASE WHEN labname ILIKE ANY(ARRAY['%neutrophil%', '%neutrophils%']) THEN labresult_clean END) AS neutrophils_mean,
    AVG(CASE WHEN labname ILIKE ANY(ARRAY['%lymphocyte%', '%lymphocytes%']) THEN labresult_clean END) AS lymphocytes_mean
FROM lab_filt
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 4. 血气增强打捞
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_bg;
CREATE TEMP TABLE temp_bg AS
SELECT patientunitstayid, 
    MIN(CASE WHEN ph BETWEEN 6.7 AND 7.8 AND ph NOT IN (7, 8) THEN ph END) AS bg_ph_min,
    MAX(CASE WHEN ph BETWEEN 6.7 AND 7.8 AND ph NOT IN (7, 8) THEN ph END) AS bg_ph_max,
    AVG(CASE WHEN paco2 BETWEEN 5 AND 150 THEN paco2 END) AS bg_paco2
FROM eicu_derived.pivoted_bg 
WHERE chartoffset BETWEEN 0 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;


--------------------------------------------------------------------------------
-- P/F ratio: 与 MIMIC 对齐逻辑（仅使用真实 PaO2）
-- - FiO2 单位: 若 fio2 >= 21 视为百分数(21-100%)，除以100；否则用 0.21
-- - 时间窗: chartoffset 0 至 1440 (入ICU 0-24h，与 MIMIC first_day_bg 一致)
-- - 若无 PaO2: 设为 NULL（与 MIMIC 定义一致，不使用 SpO2 回退）
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_respiratory_support;
CREATE TEMP TABLE temp_respiratory_support AS
WITH oxy_data AS (
    SELECT 
        patientunitstayid,
        pao2 / (CASE WHEN fio2 IS NULL THEN 0.21 WHEN fio2 >= 21 THEN fio2/100.0 ELSE 0.21 END) AS pf_val
    FROM eicu_derived.pivoted_bg 
    WHERE pao2 > 0 AND chartoffset BETWEEN 0 AND 1440
      AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
)
SELECT 
    c.patientunitstayid,
    MIN(o.pf_val) AS pao2fio2ratio_min,
    MAX(o.pf_val) AS pao2fio2ratio_max
FROM cohort_base c
LEFT JOIN oxy_data o ON c.patientunitstayid = o.patientunitstayid
GROUP BY c.patientunitstayid;
--------------------------------------------------------------------------------
-- 4a. 葡萄糖与乳酸斜率 (与 MIMIC 对齐，单次 lab 扫描合并计算)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_glucose_slope;
DROP TABLE IF EXISTS temp_lactate_slope;
DROP TABLE IF EXISTS lab_slope_agg;
CREATE TEMP TABLE lab_slope_agg AS
SELECT patientunitstayid,
    COUNT(*) FILTER (WHERE labname ILIKE '%glucose%' AND labname NOT ILIKE '%A1c%' AND labresult BETWEEN 10 AND 2000) AS glu_n,
    MIN(labresult) FILTER (WHERE labname ILIKE '%glucose%' AND labname NOT ILIKE '%A1c%' AND labresult BETWEEN 10 AND 2000) AS glu_min,
    MAX(labresult) FILTER (WHERE labname ILIKE '%glucose%' AND labname NOT ILIKE '%A1c%' AND labresult BETWEEN 10 AND 2000) AS glu_max,
    MIN(labresultoffset) FILTER (WHERE labname ILIKE '%glucose%' AND labname NOT ILIKE '%A1c%' AND labresult BETWEEN 10 AND 2000) AS glu_t_min,
    MAX(labresultoffset) FILTER (WHERE labname ILIKE '%glucose%' AND labname NOT ILIKE '%A1c%' AND labresult BETWEEN 10 AND 2000) AS glu_t_max,
    COUNT(*) FILTER (WHERE labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%']) AND labresult BETWEEN 0.1 AND 30) AS lac_n,
    MIN(labresult) FILTER (WHERE labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%']) AND labresult BETWEEN 0.1 AND 30) AS lac_min,
    MAX(labresult) FILTER (WHERE labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%']) AND labresult BETWEEN 0.1 AND 30) AS lac_max,
    MIN(labresultoffset) FILTER (WHERE labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%']) AND labresult BETWEEN 0.1 AND 30) AS lac_t_min,
    MAX(labresultoffset) FILTER (WHERE labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%']) AND labresult BETWEEN 0.1 AND 30) AS lac_t_max
FROM eicu_crd.lab
WHERE labresultoffset BETWEEN 0 AND 1440
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
  AND (
      (labname ILIKE '%glucose%' AND labname NOT ILIKE '%A1c%')
      OR labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%'])
  )
GROUP BY patientunitstayid;

CREATE TEMP TABLE temp_glucose_slope AS
SELECT patientunitstayid,
    CASE WHEN glu_n >= 3 AND (glu_t_max - glu_t_min) > 30
         THEN (glu_max - glu_min) / ((glu_t_max - glu_t_min) / 60.0) ELSE NULL END AS glucose_slope
FROM lab_slope_agg
WHERE glu_n > 0;

CREATE TEMP TABLE temp_lactate_slope AS
SELECT patientunitstayid,
    CASE WHEN lac_n >= 3 AND (lac_t_max - lac_t_min) > 30
         THEN (lac_max - lac_min) / ((lac_t_max - lac_t_min) / 60.0) ELSE NULL END AS lactate_slope
FROM lab_slope_agg
WHERE lac_n > 0;

DROP TABLE lab_slope_agg;

--------------------------------------------------------------------------------
-- 补充：计算 SpO2 斜率 (前24小时趋势，使用 array_agg 避免双 ROW_NUMBER)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_spo2_trend;
CREATE TEMP TABLE temp_spo2_trend AS
SELECT patientunitstayid,
    CASE WHEN time_span_hr > 0 THEN (spo2_last - spo2_first) / time_span_hr ELSE 0 END AS spo2_slope
FROM (
    SELECT patientunitstayid,
        (array_agg(spo2 ORDER BY chartoffset))[1] AS spo2_first,
        (array_agg(spo2 ORDER BY chartoffset DESC))[1] AS spo2_last,
        (MAX(chartoffset) - MIN(chartoffset)) / 60.0 AS time_span_hr
    FROM eicu_derived.pivoted_vital
    WHERE chartoffset BETWEEN 0 AND 1440 
      AND spo2 BETWEEN 50 AND 100
      AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
    GROUP BY patientunitstayid
) t;

DROP TABLE IF EXISTS temp_vital_full;
CREATE TEMP TABLE temp_vital_full AS
SELECT patientunitstayid, 
    MAX(NULLIF(heartrate, -1)) AS heart_rate_max, MIN(NULLIF(heartrate, -1)) AS heart_rate_min,
    MAX(NULLIF(respiratoryrate, -1)) AS resp_rate_max, MIN(NULLIF(respiratoryrate, -1)) AS resp_rate_min,
    MIN(COALESCE(NULLIF(ibp_mean, -1), NULLIF(nibp_mean, -1))) AS mbp_min, MAX(NULLIF(spo2, -1)) AS spo2_max, MIN(NULLIF(spo2, -1)) AS spo2_min,
    MAX(CASE WHEN temperature BETWEEN 80 AND 115 THEN (temperature-32)*5/9 WHEN temperature BETWEEN 30 AND 45 THEN temperature END) AS temp_max,
    MIN(CASE WHEN temperature BETWEEN 80 AND 115 THEN (temperature-32)*5/9 WHEN temperature BETWEEN 30 AND 45 THEN temperature END) AS temp_min
FROM eicu_derived.pivoted_vital 
WHERE chartoffset BETWEEN 0 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 5. 合并症 (CHF, CKD, Malignancy) 来自 diagnosis 表
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_comorbidity;
CREATE TEMP TABLE temp_comorbidity AS
SELECT patientunitstayid,
    MAX(CASE WHEN diagnosisstring ILIKE ANY(ARRAY['%malignant%','%cancer%','%metastas%']) THEN 1 ELSE 0 END) AS malignant_tumor,
    MAX(CASE WHEN diagnosisstring ILIKE ANY(ARRAY['%congestive heart failure%','%CHF%','%heart failure%','%cardiac failure%']) THEN 1 ELSE 0 END) AS heart_failure,
    MAX(CASE WHEN diagnosisstring ILIKE ANY(ARRAY['%chronic kidney%','%CKD%','%renal failure%','%ESRD%','%end stage renal%']) THEN 1 ELSE 0 END) AS chronic_kidney_disease
FROM eicu_crd.diagnosis 
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 6. POF 定义相关 (Interventions, CarePlan, APACHE, Early Death)
--------------------------------------------------------------------------------
-- 6a. Day1-7 干预 (用于 POF 代理判定)
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT patientunitstayid, 
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%vasopressor%','%dopamine%','%norepinephrine%','%vasopressin%','%epinephrine%']) THEN 1 ELSE 0 END) AS vaso_flag,
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%dialysis%','%CRRT%','%hemofiltration%']) THEN 1 ELSE 0 END) AS dialysis_flag,
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%ventilation%','%intubat%']) THEN 1 ELSE 0 END) AS vent_treatment_flag
FROM eicu_crd.treatment 
WHERE treatmentoffset BETWEEN 1440 AND 10080 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_vent_careplan;
CREATE TEMP TABLE temp_vent_careplan AS
SELECT patientunitstayid, 1 AS vent_careplan_flag FROM eicu_crd.careplangeneral
WHERE cplitemoffset BETWEEN 1440 AND 10080 
  AND cplitemvalue ILIKE ANY(ARRAY['%ventilat%','%intubat%','%ET tube%'])
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

-- 6b. 首日干预 (0-1440 min，与 MIMIC intime 至 intime+24h 对齐，用于基线/Table 1)
DROP TABLE IF EXISTS temp_interventions_firstday;
CREATE TEMP TABLE temp_interventions_firstday AS
SELECT patientunitstayid, 
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%vasopressor%','%dopamine%','%norepinephrine%','%vasopressin%','%epinephrine%']) THEN 1 ELSE 0 END) AS vaso_flag_day1,
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%ventilation%','%intubat%']) THEN 1 ELSE 0 END) AS vent_treatment_flag_day1
FROM eicu_crd.treatment 
WHERE treatmentoffset BETWEEN 0 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_vent_careplan_firstday;
CREATE TEMP TABLE temp_vent_careplan_firstday AS
SELECT patientunitstayid, 1 AS vent_careplan_flag_day1 FROM eicu_crd.careplangeneral
WHERE cplitemoffset BETWEEN 0 AND 1440 
  AND cplitemvalue ILIKE ANY(ARRAY['%ventilat%','%intubat%','%ET tube%'])
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_apache_aps;
CREATE TEMP TABLE temp_apache_aps AS
SELECT patientunitstayid, 
    MAX(CASE WHEN ph BETWEEN 6.8 AND 7.8 AND ph NOT IN (7, 8) THEN ph END) AS apache_ph, 
    MAX(CASE WHEN creatinine BETWEEN 0.1 AND 20 THEN creatinine END) AS apache_creatinine, 
    MAX(vent) AS apache_vent_flag, 
    MAX(dialysis) AS apache_dialysis_flag
FROM eicu_crd.apacheapsvar
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

-- 与 MIMIC 对齐: 24-48h 内死亡 (unitdischargeoffset 单位: 分钟)
DROP TABLE IF EXISTS temp_early_death;
CREATE TEMP TABLE temp_early_death AS
SELECT patientunitstayid, 1 AS early_death_24_48h FROM eicu_crd.patient 
WHERE unitdischargestatus = 'Expired' AND unitdischargeoffset BETWEEN 1440 AND 2880
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base);

--------------------------------------------------------------------------------
-- 7 最终整合与结局逻辑判定 (修正语法顺序)
--------------------------------------------------------------------------------
-- 更新 temp 表统计信息以优化最终 JOIN 的执行计划
ANALYZE temp_lab_raw_all;
ANALYZE temp_bg;
ANALYZE temp_respiratory_support;
ANALYZE temp_vital_full;

-- 确保 schema 存在
CREATE SCHEMA IF NOT EXISTS eicu_cview;
DROP TABLE IF EXISTS eicu_cview.ap_external_validation;
-- 第二步：开始创建表
CREATE TABLE eicu_cview.ap_external_validation AS
WITH tier_calc AS (
    SELECT
        c.patientunitstayid, 
        c.los,
        c.uniquepid, 
        c.admission_age, 
        c.gender, 
        c.bmi,
        c.weight_admit,
        c.height_admit, 
        
        -- 实验室指标 (与 MIMIC 对齐)
        l.creatinine_max, l.creatinine_min, l.bun_max, l.bun_min,
        l.wbc_max, l.wbc_min, l.rdw_max, l.hematocrit_max, l.hematocrit_min,
        l.hemoglobin_min, l.hemoglobin_max, l.platelets_min, l.platelets_max,
        l.albumin_min, l.albumin_max, l.bilirubin_total_max, l.bilirubin_total_min,
        l.ast_max, l.ast_min, l.alt_max, l.alt_min, l.alp_min, l.alp_max,
        l.ptt_min, l.ptt_max, l.pt_max, l.pt_min, l.inr_max, l.inr_min,
        l.glucose_max, l.glucose_min, l.glucose_min AS glucose_lab_min, l.glucose_max AS glucose_lab_max,
        l.lactate_max, l.lactate_min,
        l.aniongap_max, l.aniongap_min, l.sodium_max, l.sodium_min,
        l.potassium_max, l.potassium_min, l.chloride_min, l.chloride_max,
        l.lab_calcium_min, l.calcium_max, l.bicarbonate_min, l.bicarbonate_max,
        l.lab_amylase_min, l.lab_amylase_max, l.lipase_max, l.crp_max, l.phosphate_min,
        l.d_dimer_max, l.fibrinogen_max, l.ldh_max, l.triglycerides_max, l.total_cholesterol_min,
        l.neutrophils_mean, l.lymphocytes_mean,
        CASE WHEN l.lymphocytes_mean > 0 THEN l.neutrophils_mean / l.lymphocytes_mean ELSE NULL END AS nlr,
        slp.spo2_slope, gslp.glucose_slope, lslp.lactate_slope,
        
        -- pH 打捞链
        bg.bg_ph_min AS ph_t1, 
        l.lab_ph_direct AS ph_t2, 
        CASE 
            WHEN l.lab_hco3 > 0 AND COALESCE(bg.bg_paco2, l.lab_paco2) > 0 
            THEN (6.1 + LOG10(l.lab_hco3 / (0.0301 * COALESCE(bg.bg_paco2, l.lab_paco2)))) 
            ELSE NULL 
        END AS ph_t3, 
        aps.apache_ph AS ph_t4, 
        COALESCE(bg.bg_ph_max, l.lab_ph_max, bg.bg_ph_min, l.lab_ph_direct) AS ph_max_raw,

        -- 衍生比值 (与 MIMIC 对齐)
        (l.bilirubin_total_max / NULLIF(GREATEST(l.albumin_min, 1.0), 0)) AS tbar,
        (l.lactate_max / NULLIF(GREATEST(l.albumin_min, 1.0), 0)) AS lar,

        -- 呼吸指标 (与 MIMIC 对齐)
        res.pao2fio2ratio_min, res.pao2fio2ratio_max,
        
        COALESCE(comb.malignant_tumor, 0) AS malignant_tumor,
        COALESCE(comb.heart_failure, 0) AS heart_failure,
        COALESCE(comb.chronic_kidney_disease, 0) AS chronic_kidney_disease,
        v.heart_rate_max, v.heart_rate_min, v.resp_rate_max, v.resp_rate_min,
        v.mbp_min, v.spo2_max, v.spo2_min, v.temp_max, v.temp_min,
        
        COALESCE(intv.vaso_flag, 0) AS vaso_flag, 
        COALESCE(aps.apache_vent_flag, cp.vent_careplan_flag, intv.vent_treatment_flag, 0) AS mechanical_vent_flag, 
        COALESCE(intv.dialysis_flag, aps.apache_dialysis_flag, 0) AS dialysis_flag,
        -- 首日干预 (与 MIMIC intime 至 intime+24h 对齐，用于 Table 1 基线)
        COALESCE(intv1.vaso_flag_day1, 0) AS vaso_flag_day1, 
        COALESCE(cp1.vent_careplan_flag_day1, intv1.vent_treatment_flag_day1, 0) AS mechanical_vent_flag_day1,
        -- 早期死亡 (与 MIMIC 24-48h 内死亡对齐)
        COALESCE(ed.early_death_24_48h, 0) AS early_death_24_48h,
        
        CASE WHEN c.hosp_mort = 1 AND pat.hospitaldischargeoffset <= 40320 THEN 1 ELSE 0 END AS mortality_28d
    FROM cohort_base c
    INNER JOIN eicu_crd.patient pat ON c.patientunitstayid = pat.patientunitstayid
    LEFT JOIN temp_lab_raw_all l ON c.patientunitstayid = l.patientunitstayid
    LEFT JOIN temp_spo2_trend slp ON c.patientunitstayid = slp.patientunitstayid
    LEFT JOIN temp_glucose_slope gslp ON c.patientunitstayid = gslp.patientunitstayid
    LEFT JOIN temp_lactate_slope lslp ON c.patientunitstayid = lslp.patientunitstayid
    LEFT JOIN temp_apache_aps aps ON c.patientunitstayid = aps.patientunitstayid
    LEFT JOIN temp_bg bg ON c.patientunitstayid = bg.patientunitstayid
    LEFT JOIN temp_vital_full v ON c.patientunitstayid = v.patientunitstayid
    LEFT JOIN temp_interventions intv ON c.patientunitstayid = intv.patientunitstayid
    LEFT JOIN temp_vent_careplan cp ON c.patientunitstayid = cp.patientunitstayid
    LEFT JOIN temp_interventions_firstday intv1 ON c.patientunitstayid = intv1.patientunitstayid
    LEFT JOIN temp_vent_careplan_firstday cp1 ON c.patientunitstayid = cp1.patientunitstayid
    LEFT JOIN temp_early_death ed ON c.patientunitstayid = ed.patientunitstayid
    LEFT JOIN temp_respiratory_support res ON c.patientunitstayid = res.patientunitstayid
    LEFT JOIN temp_comorbidity comb ON c.patientunitstayid = comb.patientunitstayid
),
base_final AS (
    SELECT *,
        CASE 
            WHEN ph_t1 BETWEEN 6.7 AND 7.8 THEN ph_t1
            WHEN ph_t2 BETWEEN 6.7 AND 7.8 THEN ph_t2
            WHEN ph_t3 BETWEEN 6.7 AND 7.8 THEN CAST(ph_t3 AS NUMERIC)
            WHEN ph_t4 BETWEEN 6.7 AND 7.8 THEN ph_t4
            ELSE NULL 
        END AS ph_min,
        CASE 
            WHEN ph_max_raw BETWEEN 6.7 AND 7.8 THEN ph_max_raw 
            ELSE NULL 
        END AS ph_max
    FROM tier_calc
),
label_calc AS (
    SELECT *,
        CASE  
            WHEN (vaso_flag = 1 AND los >= 1) 
              OR (mechanical_vent_flag = 1 AND los >= 2) 
              OR (pao2fio2ratio_min < 300 AND los >= 2) 
              OR (dialysis_flag = 1) 
              OR (creatinine_max > 1.9) THEN 1
            WHEN creatinine_max IS NULL AND pao2fio2ratio_min IS NULL THEN NULL
            ELSE 0 
        END AS pof,
        CASE 
            WHEN (vaso_flag = 1 AND los >= 1) 
              OR (mechanical_vent_flag = 1 AND los >= 2)
              OR (pao2fio2ratio_min < 300 AND los >= 2) 
              OR (dialysis_flag = 1)
              OR (creatinine_max > 1.9) 
              OR (mortality_28d = 1) THEN 1
            WHEN creatinine_max IS NULL AND pao2fio2ratio_min IS NULL AND mortality_28d IS NULL THEN NULL
            ELSE 0
        END AS composite_outcome
    FROM base_final
)
-- 最终选择：排除 pof 为 NULL 的那 1 例
SELECT * FROM label_calc 
WHERE pof IS NOT NULL;
