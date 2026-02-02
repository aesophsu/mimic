"""01: MIMIC 临床特征审计与清洗"""
import os
import json
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import get_project_root, get_raw_path, get_artifact_path, get_cleaned_path, ensure_dirs
from utils.logger import log as _log, log_header
from utils.study_config import MISSING_THRESHOLD
from utils.outcome_utils import apply_early_death_override, align_outcome_columns

BASE_DIR = get_project_root()
INPUT_PATH = get_raw_path("mimic")
DICT_PATH = get_artifact_path("features", "feature_dictionary.json")
SAVE_DIR = get_cleaned_path(".")
ensure_dirs(SAVE_DIR)

class FeatureAuditor:
    def __init__(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.feature_dict = json.load(f)

    def audit_units_and_ranges(self, df):
        print(f"\n📋 原始数据探测: {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"{'Feature Name':<25} | {'Missing%':<10} | {'Median':<10} | {'Mean':<10} | {'Max':<10}")
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                missing = df[col].isnull().mean() * 100
                med = series.median() if not series.empty else 0
                mean = series.mean() if not series.empty else 0
                v_max = series.max() if not series.empty else 0
                print(f"{col:<25} | {missing:>8.2f}% | {med:>10.2f} | {mean:>10.2f} | {v_max:>10.2f}")
               
        df_cleaned = df.copy()
        print(f"\n{'Feature':<20} | {'Action':<40} | {'Status'}")
        print("-" * 80)
        
        for col, config in self.feature_dict.items():
            if col not in df_cleaned.columns:
                continue
           
            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue
            
            ref = config.get('ref_range', {})
            log_min = ref.get('logical_min')
            log_max = ref.get('logical_max')
            factor = config.get('conversion_factor', 1.0)
           
            series_curr = df_cleaned[col].dropna()
            if series_curr.empty:
                continue
           
            med = series_curr.median()
            
            if log_min is not None and factor != 1.0:
                if med < (log_min * 0.2):
                    df_cleaned[col] = df_cleaned[col] * factor
                    print(f"{col:<20} | Applied conversion factor x{factor:<10} | ✅")
            
            if log_min is not None and log_max is not None:
                mask = (df_cleaned[col] < log_min) | (df_cleaned[col] > log_max)
                if mask.any():
                    num_removed = mask.sum()
                    df_cleaned.loc[mask, col] = np.nan
                    print(f"{col:<20} | Removed {num_removed:>3} physiologic outliers | ⚠️")
        
        return df_cleaned

def run_cross_database_alignment():
    log_header("🚀 01_mimic_cleaning: 临床特征空间审计与清洗 (MIMIC-IV)")
    _log(f"输入: {os.path.abspath(INPUT_PATH)}", "INFO")
    _log(f"输出目录: {os.path.abspath(SAVE_DIR)}", "INFO")

    if not os.path.exists(INPUT_PATH):
        _log(f"找不到输入文件: {INPUT_PATH}", "ERR")
        return

    df = pd.read_csv(INPUT_PATH)
    auditor = FeatureAuditor(DICT_PATH)

    df = apply_early_death_override(df)
    df = align_outcome_columns(df)
    _log("[Labels] 结局指标对齐完成", "OK")

    white_list = [
        'subject_id', 'hadm_id', 'stay_id', 'database',
        'pof', 'mortality', 'composite', 'early_death_24_48h',
        'lactate_max', 'pao2fio2ratio_min', 'lipase_max', 'creatinine_max', 'bun_min'
    ]
   
    missing_pct = df.isnull().mean()
    cols_to_drop = [c for c in missing_pct[missing_pct > MISSING_THRESHOLD].index if c not in white_list]
    if cols_to_drop:
        _log(f"[Filter] 剔除缺失率 >{MISSING_THRESHOLD*100:.0f}% 的特征 ({len(cols_to_drop)} 个): {', '.join(cols_to_drop)}", "WARN")
    else:
        _log("[Filter] 无特征因缺失率超标被剔除", "OK")
    df = df.drop(columns=cols_to_drop)
    
    df = auditor.audit_units_and_ranges(df)
    
    _log("[Clipping] 执行 1%-99% 统计盖帽处理", "INFO")
    binary_cols = ['gender', 'vaso_flag', 'mechanical_vent_flag', 'composite', 'pof']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
   
    for col in numeric_cols:
        if col not in white_list and col not in binary_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            if pd.notnull(lower) and pd.notnull(upper):
                df[col] = df[col].clip(lower, upper)
    
    print("\n" + "-" * 70)
    _log(f"清洗完成: 样本 N={df.shape[0]}, 特征 P={df.shape[1]}", "OK")

    report_cols = ['bun_min', 'creatinine_max', 'lactate_max', 'spo2_max', 'pao2fio2ratio_min', 'rdw_max', 'mortality', 'composite']
    print("\n🔍 关键特征统计审计:")
    for c in report_cols:
        if c in df.columns:
            col = df[c].squeeze() if hasattr(df[c], 'squeeze') else df[c]
            ser = col.dropna() if hasattr(col, 'dropna') else pd.Series(col).dropna()
            med = ser.median() if not ser.empty else 0.0
            miss = (col.isnull().mean() * 100) if hasattr(col, 'isnull') else 0.0
            med_val = float(med) if np.isscalar(med) else 0.0
            miss_val = float(miss) if np.isscalar(miss) else 0.0
            print(f" > {c:<20}: Median={med_val:>8.2f} | Missing={miss_val:>6.2f}%")
    
    save_path = os.path.join(SAVE_DIR, "mimic_raw_scale.csv")
    df.to_csv(save_path, index=False)
    abs_path = os.path.abspath(save_path)
    _log(f"已保存: {abs_path}", "OK")
    _log("下一步: 02_mimic_standardization.py", "INFO")

if __name__ == "__main__":
    run_cross_database_alignment()
