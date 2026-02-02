import os
import sys
import warnings
import joblib
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore', message='.*Early stopping criterion not reached.*', module='sklearn')

from utils.study_config import STRATIFY_COL, SPLIT_SEED, TEST_SIZE
from utils.feature_formatter import FeatureFormatter
from utils.paths import get_project_root, get_cleaned_path, get_artifact_path, get_main_table_dir, ensure_dirs
from utils.logger import log as _log, log_header
from utils.outcome_utils import normalize_gender

from tableone import TableOne
from sklearn.preprocessing import StandardScaler

# Fix TableOne + pandas 2.x: _insert_n_row puts int into StringDtype column
_orig_insert_n_row = TableOne._insert_n_row
def _patched_insert_n_row(self, table, data):
    n_row = pd.DataFrame(columns=['variable', 'value', 'Missing'])
    n_row = n_row.set_index(['variable', 'value'])
    n_row.loc['n', 'Missing'] = None
    try:
        table = pd.concat([n_row, table], sort=False)
    except TypeError:
        table = pd.concat([n_row, table])
    if self._groupbylvls == ['Overall']:
        table.loc['n', 'Overall'] = str(len(data.index))
    else:
        if self._overall:
            table.loc['n', 'Overall'] = str(len(data.index))
        for g in self._groupbylvls:
            ct = data[self._groupby][data[self._groupby] == g].count()
            table.loc['n', '{}'.format(g)] = str(ct)
    return table
TableOne._insert_n_row = _patched_insert_n_row
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

BASE_DIR = get_project_root()
INPUT_PATH = get_cleaned_path("mimic_raw_scale.csv")
SAVE_DIR = get_cleaned_path(".")
ARTIFACT_DIR = get_artifact_path("scalers")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "mimic_scaler.joblib")
IMPUTER_PATH = os.path.join(ARTIFACT_DIR, "mimic_mice_imputer.joblib")
SKEW_CONFIG_PATH = os.path.join(ARTIFACT_DIR, "skewed_cols_config.pkl")
REPORT_DIR = get_main_table_dir()
ensure_dirs(SAVE_DIR, ARTIFACT_DIR, REPORT_DIR)


def run_mimic_standardization():
    log_header("🚀 02_mimic_standardization: Log 转换、MICE 插补与标准化 (MIMIC-IV)")
    abs_input = os.path.abspath(INPUT_PATH)
    abs_artifact = os.path.abspath(ARTIFACT_DIR)
    _log(f"输入: {abs_input}", "INFO")
    _log(f"资产目录: {abs_artifact}", "INFO")

    if not os.path.exists(INPUT_PATH):
        _log(f"输入文件不存在: {abs_input}", "ERR")
        return

    try:
        with open(get_artifact_path("features", "feature_dictionary.json"), 'r', encoding='utf-8') as f:
            feat_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _log(f"特征字典读取失败: {e}", "ERR")
        return

    df = pd.read_csv(INPUT_PATH)

    if 'creatinine_max' in df.columns and 'chronic_kidney_disease' in df.columns:
        df['subgroup_no_renal'] = (
            (df['creatinine_max'] < 1.5) & (df['chronic_kidney_disease'] == 0)
        ).astype(int)

    df = normalize_gender(df)

    # Table 1 基线特征表：由 03_table1_baseline.py 生成（Step 03 独立运行）

    # Table 2 肾亚组基线表（仍用 TableOne）
    if 'subgroup_no_renal' in df.columns:
        clinical_features = [
            'admission_age', 'weight_admit', 'gender', 'sofa_score', 'apsiii', 'sapsii', 'oasis', 'lods',
            'heart_failure', 'chronic_kidney_disease', 'malignant_tumor', 'mechanical_vent_flag', 'vaso_flag',
            'wbc_max', 'hemoglobin_min', 'platelets_min', 'bun_max', 'creatinine_max', 'bilirubin_total_max',
            'alt_max', 'ast_max', 'alp_max', 'lactate_max', 'pao2fio2ratio_min', 'spo2_min', 'ph_min',
            'sodium_max', 'potassium_max', 'bicarbonate_min'
        ]
        outcome_cols = ['pof', 'mortality', 'composite']
        cols_for_t2 = [c for c in (clinical_features + outcome_cols) if c in df.columns and c != 'subgroup_no_renal']
        cat_for_t2 = [c for c in ['gender', 'heart_failure', 'chronic_kidney_disease', 'malignant_tumor',
            'mechanical_vent_flag', 'vaso_flag', 'mortality', 'composite'] if c in cols_for_t2]
        nonnormal_for_t2 = [c for c in cols_for_t2 if c not in cat_for_t2]
        formatter = FeatureFormatter()
        rename_t2 = {c: formatter.get_label(c) for c in cols_for_t2}
        t2 = TableOne(df, columns=cols_for_t2, categorical=cat_for_t2, nonnormal=nonnormal_for_t2,
                      groupby='subgroup_no_renal', pval=True, rename=rename_t2)
        t2_path = os.path.join(REPORT_DIR, "Table2_renal_subgroup.csv")
        t2.to_csv(t2_path)
        _log(f"Table2 已保存: {os.path.abspath(t2_path)}", "OK")

    _log("数值特征审计 (清洗后):", "INFO")
    print(f"  {'Feature':<22} | {'Miss%':>7} | {'Median':>9} | {'Mean':>9} | {'Max':>9}")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            missing = df[col].isnull().mean() * 100
            med = series.median() if not series.empty else 0
            mean = series.mean() if not series.empty else 0
            v_max = series.max() if not series.empty else 0
            print(f"  {col:<22} | {missing:>6.2f}% | {med:>9.2f} | {mean:>9.2f} | {v_max:>9.2f}")

    drop_from_modeling = [
        'subject_id', 'hadm_id', 'stay_id', 
        'admittime', 'dischtime', 'intime', 'deathtime', 'dod',
        'early_death_24_48h', 'hosp_mortality', 'los'
    ]
    
    protected_cols = [
        'pof', 'resp_pof', 'cv_pof', 'renal_pof', 
        'mortality', 'composite', 'subgroup_no_renal',
        'gender', 'heart_failure', 'chronic_kidney_disease', 
        'malignant_tumor', 'mechanical_vent_flag', 'vaso_flag'
    ]
    outcome_cols_protected = ['pof', 'mortality', 'composite']
    non_outcome_protected = [c for c in protected_cols if c not in outcome_cols_protected]

    df_model = df.drop(columns=[c for c in drop_from_modeling if c in df.columns])
    for col in non_outcome_protected:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0).astype(int)

    remaining_text = df_model.select_dtypes(include=['object']).columns.tolist()
    if remaining_text:
        df_model = df_model.drop(columns=remaining_text)

    numeric_features = [c for c in df_model.select_dtypes(include=[np.number]).columns
                        if c not in protected_cols]

    skewed_cols = [
        col for col in numeric_features
        if col in feat_dict and feat_dict[col].get("needs_log_transform", False)
    ]

    if skewed_cols:
        _log(f"对 {len(skewed_cols)} 个特征应用 log1p: {', '.join(skewed_cols)}", "INFO")
        for col in skewed_cols:
            df_model[col] = np.log1p(df_model[col].clip(lower=0))

    joblib.dump(skewed_cols, SKEW_CONFIG_PATH)

    strat_col = STRATIFY_COL if STRATIFY_COL in df_model.columns else "mortality"
    df_split = df_model.dropna(subset=[strat_col])
    if len(df_split) < len(df_model):
        _log(f"剔除 {len(df_model) - len(df_split)} 行（{strat_col} 缺失）用于划分", "WARN")
    train_idx, test_idx = train_test_split(
        df_split.index,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=df_split[strat_col],
    )
    df_train = df_model.loc[train_idx].copy()
    df_test = df_model.loc[test_idx].copy()
    _log(f"划分: 训练 N={len(df_train)}, 测试 N={len(df_test)} (stratify={strat_col})", "OK")

    _log("MICE 多重插补（仅训练集 fit）...", "INFO")
    imputer = IterativeImputer(max_iter=25, random_state=42, verbose=0)
    df_train[numeric_features] = imputer.fit_transform(df_train[numeric_features])
    df_test[numeric_features] = imputer.transform(df_test[numeric_features])
    _log(f"MICE 完成，迭代次数: {imputer.n_iter_}", "OK")
    joblib.dump(imputer, IMPUTER_PATH)

    _log("StandardScaler 标准化（仅训练集 fit）...", "INFO")
    scaler = StandardScaler()
    df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
    df_test[numeric_features] = scaler.transform(df_test[numeric_features])
    joblib.dump(scaler, SCALER_PATH)

    raw_medians = df_train[numeric_features].median().to_dict()
    train_assets = {
        'skewed_cols': skewed_cols,
        'medians': raw_medians,
        'feature_order': numeric_features,
        'n_samples': len(df_train),
        'train_idx': train_idx.tolist(),
        'test_idx': test_idx.tolist()
    }
    bundle_path = os.path.join(ARTIFACT_DIR, "train_assets_bundle.pkl")
    joblib.dump(train_assets, bundle_path)

    train_path = os.path.join(SAVE_DIR, "mimic_train_processed.csv")
    test_path = os.path.join(SAVE_DIR, "mimic_test_processed.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_merged = pd.concat([df_train, df_test], ignore_index=True)
    processed_path = os.path.join(SAVE_DIR, "mimic_processed.csv")
    df_merged.to_csv(processed_path, index=False)


    print("\n" + "-" * 50)
    _log("产物核查:", "INFO")
    artifacts = {
        "Skew Config": SKEW_CONFIG_PATH,
        "MICE Imputer": IMPUTER_PATH,
        "Standard Scaler": SCALER_PATH,
        "Asset Bundle": bundle_path,
        "Train Data": train_path,
        "Test Data": test_path,
        "Merged Data": processed_path
    }
    for name, path in artifacts.items():
        status = "✅" if os.path.exists(path) else "❌"
        size = f"{os.path.getsize(path)/1024:.1f} KB" if os.path.exists(path) else "N/A"
        print(f"  {status} {name:<18} | {size}")

    _log(f"建模特征数: {len(numeric_features)}, 张量: {df_model.shape}", "OK")
    _log(f"中间产物: {os.path.abspath(processed_path)}", "OK")
    _log(f"核心资产: {os.path.abspath(ARTIFACT_DIR)}", "OK")
    _log("下一步: 03_table1_baseline.py 或 04_mimic_stat_audit.py 或 05_feature_selection_lasso.py", "INFO")

if __name__ == "__main__":
    run_mimic_standardization()
