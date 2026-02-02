import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.study_config import OUTCOMES
from utils.paths import get_project_root, get_raw_path, get_artifact_path, get_external_dir, get_model_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.outcome_utils import apply_early_death_override, align_outcome_columns
from utils.deploy_utils import load_deploy_bundle

BASE_DIR = get_project_root()
EICU_RAW_PATH = get_raw_path("eicu")
DICT_PATH = get_artifact_path("features", "feature_dictionary.json")
SELECTED_FEAT_PATH = get_artifact_path("features", "selected_features.json")
MODEL_ROOT = get_model_dir()
SAVE_DIR = get_external_dir()
ensure_dirs(SAVE_DIR)

PROTECTED_COLS = [
    'pof', 'mortality', 'composite', 'gender', 'malignant_tumor', 
    'mechanical_vent_flag', 'vaso_flag', 'dialysis_flag', 'subgroup_no_renal'
]

ESSENTIAL_COLS = ['patientunitstayid', 'uniquepid', 'los'] + PROTECTED_COLS
EXCLUDE_CLIPPING = ['patientunitstayid', 'uniquepid', 'los'] + PROTECTED_COLS

# Table 1 基线表所需列（与 MIMIC 对齐，即使不在模型特征中也保留）
TABLE1_BASELINE_COLS = [
    'weight_admit', 'heart_failure', 'chronic_kidney_disease',
    'hemoglobin_min', 'platelets_min', 'bilirubin_total_max', 'alt_max', 'ast_max',
    'spo2_min', 'spo2_max', 'bicarbonate_min', 'sodium_max', 'sodium_min',
    'potassium_max', 'potassium_min', 'chloride_min', 'chloride_max',
    'lab_calcium_min', 'calcium_max', 'lab_amylase_min', 'lab_amylase_max',
    'lipase_max', 'crp_max', 'phosphate_min', 'd_dimer_max', 'fibrinogen_max',
    'ldh_max', 'triglycerides_max', 'total_cholesterol_min',
    'neutrophils_mean', 'lymphocytes_mean', 'nlr', 'glucose_lab_min', 'glucose_lab_max',
    'glucose_slope', 'lactate_min', 'lactate_slope', 'pao2fio2ratio_max', 'lar',
]

os.makedirs(SAVE_DIR, exist_ok=True)

def audit_clinical_limits(df, feature_dict):
    df_temp = df.copy()
    for col, config in feature_dict.items():
        if col not in df_temp.columns or not pd.api.types.is_numeric_dtype(df_temp[col]):
            continue
        
        ref = config.get('ref_range', {})
        log_min, log_max = ref.get('logical_min'), ref.get('logical_max')
        factor = config.get('conversion_factor', 1.0)
        
        series_valid = df_temp[col].dropna()
        if series_valid.empty: continue
        
        if log_min is not None and factor != 1.0:
            if series_valid.median() < (log_min * 0.2): 
                df_temp[col] *= factor
        
        if log_min is not None and log_max is not None:
            mask = (df_temp[col] < log_min) | (df_temp[col] > log_max)
            df_temp.loc[mask, col] = np.nan
            
    return df_temp

def apply_clinical_audit_workflow(df, auditor_config):
    # 1. 早期死亡与 MIMIC 对齐（与 01_mimic_cleaning 一致）
    df = apply_early_death_override(df)
    df = align_outcome_columns(df)

    # 2. 基线干预与 MIMIC 对齐：eICU 使用首日 (0-1440 min) 干预列；缺失时保守为 0
    if 'mechanical_vent_flag_day1' in df.columns:
        df['mechanical_vent_flag'] = df['mechanical_vent_flag_day1'].fillna(0).astype(int)
    if 'vaso_flag_day1' in df.columns:
        df['vaso_flag'] = df['vaso_flag_day1'].fillna(0).astype(int)

    # 3. 白名单过滤（含 Table 1 基线列、early_death，确保 eICU 列可补齐）
    allowed_cols = list(set(list(auditor_config.keys()) + ESSENTIAL_COLS + TABLE1_BASELINE_COLS + ['early_death_24_48h']))
    df_cleaned = df[[c for c in allowed_cols if c in df.columns]].copy()
    
    _log(f"[审计] 原始 {df.shape[1]} 列 -> 目标 {df_cleaned.shape[1]} 列", "INFO")

    # 4. 亚组标记
    if 'creatinine_max' in df_cleaned.columns:
        df_cleaned['subgroup_no_renal'] = (df_cleaned['creatinine_max'] < 1.5).astype(int)

    # 5. 生理规则校验
    return audit_clinical_limits(df_cleaned, auditor_config)

def _load_bundle_for_target(target):
    bundle = load_deploy_bundle(target, fill_missing=True)
    if bundle is None:
        _log(f"deploy_bundle 缺失: {os.path.abspath(os.path.join(get_model_dir(target), 'deploy_bundle.pkl'))}，请先运行 06_model_training_main.py", "ERR")
        return None
    _log(f"deploy_bundle 加载成功 [{target}] (含 {len(bundle.get('skewed_cols', []))} 个偏态特征)", "OK")
    return bundle

def get_union_feature_config():
    if not (os.path.exists(SELECTED_FEAT_PATH) and os.path.exists(DICT_PATH)):
        _log("错误: 配置文件缺失", "ERR")
        return None

    with open(SELECTED_FEAT_PATH, 'r', encoding='utf-8') as f:
        selected_json = json.load(f)
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        full_physio_dict = json.load(f)

    union_features = {feat for target in selected_json.values() for feat in target['features']}
    
    auditor_config = {k: v for k, v in full_physio_dict.items() if k in union_features}
    _log(f"[配置] 识别到 {len(union_features)} 个唯一特征，已匹配审计规则", "INFO")
    return auditor_config

def run_clinical_audit(df_raw, auditor_config):
    df_audited = apply_clinical_audit_workflow(df_raw, auditor_config)
    
    # 2. 1%-99% 盖帽
    clinical_features = [c for c in df_audited.columns if c in auditor_config and c not in EXCLUDE_CLIPPING]
    
    clipped_count = 0
    for col in clinical_features:
        q = df_audited[col].quantile([0.01, 0.99])
        if q.isnull().any(): continue
        
        df_audited[col] = df_audited[col].clip(lower=q[0.01], upper=q[0.99])
        clipped_count += 1
    
    _log(f"盖帽处理完成: {clipped_count} 个临床特征已约束", "OK")
    return df_audited

def align_feature_space(df_audited, required_features):
    df_aligned = pd.DataFrame(index=df_audited.index)
    for col in required_features:
        if col in df_audited.columns:
            val = df_audited[col]
            df_aligned[col] = val.iloc[:, 0] if isinstance(val, pd.DataFrame) else val
        else:
            df_aligned[col] = np.nan
    return df_aligned

def apply_mimic_transform(df_aligned, bundle):
    imputer = bundle['imputer']
    mimic_scaler = bundle['mimic_scaler']
    scaler_pre = bundle['scaler']
    skewed_cols = bundle.get('skewed_cols', [])
    imputer_cols = bundle.get('imputer_feature_order') or getattr(imputer, 'feature_names_in_', df_aligned.columns.tolist())
    df_trans = df_aligned.copy()
    
    # 1. Log1p
    for col in skewed_cols:
        if col in df_trans.columns:
            df_trans[col] = np.log1p(df_trans[col].clip(lower=0))
    
    # 2. MICE + mimic_scaler（传入带列名的 DataFrame 避免 StandardScaler 警告）
    _log("执行 Transform (MICE + mimic_scaler + scaler_pre)...", "INFO")
    imputer_cols = list(imputer_cols) if imputer_cols is not None else list(df_trans.columns)
    imputed_data = imputer.transform(df_trans)
    df_imputed = pd.DataFrame(imputed_data, columns=imputer_cols, index=df_trans.index)
    mimic_scaled = mimic_scaler.transform(df_imputed)
    
    # 3. scaler_pre
    features_in = bundle['feature_names']
    df_mimic_scaled = pd.DataFrame(mimic_scaled, columns=imputer_cols, index=df_aligned.index)
    X_selected = df_mimic_scaled[[c for c in features_in if c in df_mimic_scaled.columns]]
    for c in features_in:
        if c not in X_selected.columns:
            X_selected[c] = 0
    X_selected = X_selected[features_in]
    final_scaled = scaler_pre.transform(X_selected)
    return pd.DataFrame(final_scaled, columns=features_in, index=df_aligned.index)

def audit_final_distribution(df):
    _log("关键特征 (Z-score) 审计:", "INFO")
    for col in ['creatinine_max', 'lactate_max', 'pao2fio2ratio_min', 'ph_min']:
        if col in df.columns:
            _log(f"   {col:<20}: Mean={df[col].mean():.3f} | Std={df[col].std():.3f}", "INFO")

def generate_eicu_processed(target, df_audited, bundle):
    _log(f"处理结局: {target}", "INFO")
    
    # 1. 对齐 + 变换
    imputer_cols = bundle.get('imputer_feature_order') or getattr(bundle['imputer'], 'feature_names_in_', None)
    if imputer_cols is None:
        raise ValueError("deploy_bundle 缺少 imputer 列顺序，请检查 Step 06 输出")
    df_aligned = align_feature_space(df_audited, imputer_cols)
    df_scaled = apply_mimic_transform(df_aligned, bundle)

    # 2. 恢复保护列
    df_final = df_scaled.copy()
    for col in PROTECTED_COLS:
        if col in df_audited.columns:
            source = df_audited[col]
            df_final[col] = (source.iloc[:, 0] if isinstance(source, pd.DataFrame) else source).fillna(0).astype(int).values

    # 3. 导出
    audit_final_distribution(df_final)
    save_path = os.path.join(SAVE_DIR, f"eicu_processed_{target}.csv")
    df_final.to_csv(save_path, index=False)
    _log(f"保存推理集: {os.path.abspath(save_path)}", "OK")

def main():
    log_header("🚀 08_eicu_alignment_cleaning: eICU 预处理与结局对齐")
    _log(f"eICU 原始数据: {os.path.abspath(EICU_RAW_PATH)}", "INFO")
    _log(f"输出目录: {os.path.abspath(SAVE_DIR)}", "INFO")

    auditor_config = get_union_feature_config()
    if not auditor_config:
        return

    if not os.path.exists(EICU_RAW_PATH):
        _log(f"Raw data not found: {os.path.abspath(EICU_RAW_PATH)}", "ERR")
        return

    df_raw = pd.read_csv(EICU_RAW_PATH)
    _log(f"Loaded eICU: N={df_raw.shape[0]} patients", "OK")
    
    # 3. 临床审计
    df_audited = run_clinical_audit(df_raw, auditor_config)

    # 4. 保存中间态
    scale_save_path = os.path.join(SAVE_DIR, "eicu_raw_scale.csv")
    df_audited.to_csv(scale_save_path, index=False)
    _log(f"中间产物: {os.path.abspath(scale_save_path)}", "OK")

    _log("开始多结局对齐与 Transform (deploy_bundle)...", "INFO")
    for target in OUTCOMES:
        bundle = _load_bundle_for_target(target)
        if bundle:
            generate_eicu_processed(target, df_audited, bundle)

    # Table1 由 run_all 在 08 之后调用 03 生成（含 eICU 外部验证列）

    _log("08 步完成！", "OK")
    _log("下一步: 03_table1_baseline.py（更新 Table 1 含 eICU 列）→ 09_cross_cohort_audit.py", "INFO")

if __name__ == "__main__":
    main()
