import os
import sys
import joblib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.paths import get_model_dir, get_main_figure_dir, get_supplementary_figure_dir, ensure_dirs
from utils.feature_formatter import FeatureFormatter
from utils.study_config import OUTCOMES
from utils.plot_config import apply_medical_style, SAVE_DPI, FIG_WIDTH_DOUBLE, save_fig_medical
from utils.logger import log as _log, log_header

warnings.filterwarnings("ignore")

MODEL_ROOT = get_model_dir()
FIGURE_DIR = get_main_figure_dir()  # Fig4A Summary
INTERP_DIR = get_supplementary_figure_dir("S5_interpretation", "shap_values")  # Fig4B/C, SHAP data
ensure_dirs(INTERP_DIR)

TARGETS = OUTCOMES
RANDOM_STATE = 42

def _patch_shap_for_xgboost31():
    """修补 SHAP 以兼容 XGBoost 3.1+ 的 base_score 数组格式（如 '[4.0614334E-1]'）"""
    import shap.explainers._tree as _tree_mod

    if getattr(_patch_shap_for_xgboost31, "_patched", False):
        return

    _orig_decode = _tree_mod.decode_ubjson_buffer

    def _patched_decode(fd):
        result = _orig_decode(fd)
        learner = result.get("learner", {})
        lmp = learner.get("learner_model_param", {})
        bs = lmp.get("base_score")
        if isinstance(bs, str) and bs.strip().startswith("[") and bs.strip().endswith("]"):
            lmp["base_score"] = bs.strip()[1:-1]
        return result

    _tree_mod.decode_ubjson_buffer = _patched_decode
    _patch_shap_for_xgboost31._patched = True


def compute_shap_values(X, model, explainer_type):
    """计算 SHAP 值：统一处理 Tree 和 Linear 解释器"""
    _log(f"Calculating SHAP values ({explainer_type})...", "INFO")
    if explainer_type == "tree":
        _patch_shap_for_xgboost31()
        explainer = shap.TreeExplainer(model)
        # 兼容不同版本的 SHAP 返回格式
        shap_res = explainer(X)
        if len(shap_res.shape) == 3: # 针对某些二分类输出
            shap_res = shap_res[:, :, 1]
    else:
        explainer = shap.Explainer(model, X)
        shap_res = explainer(X)
    return shap_res

def save_shap_data(shap_values, X, target):
    """保存 SHAP 值及对应的原始特征值，便于后续统计建模"""
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    raw_df = X.reset_index(drop=True)
    raw_df.columns = [f"raw_{col}" for col in raw_df.columns]
    combined_df = pd.concat([shap_df, raw_df], axis=1)
    save_path = os.path.join(INTERP_DIR, f"SHAP_Data_Export_{target}.csv")
    combined_df.to_csv(save_path, index=False)
    base_val = shap_values.base_values[0] if isinstance(shap_values.base_values, np.ndarray) else shap_values.base_values
    with open(os.path.join(INTERP_DIR, f"SHAP_BaseValue_{target}.txt"), "w") as f:
        f.write(str(base_val))
    _log(f"SHAP 原始数据已导出: {save_path}", "OK")

def _get_forest_style_feature_names(cols):
    """获取与 Forest 图一致的特征展示名：带单位，与 feature_dictionary 一致"""
    formatter = FeatureFormatter()
    return formatter.format_features(cols, with_unit=True)


def plot_shap_summary(shap_values, X, target):
    """医学蜂群图：全局特征贡献排序，使用与 Forest 图一致的特征展示名（带单位）"""
    apply_medical_style()
    feature_names = _get_forest_style_feature_names(X.columns.tolist())
    X_display = X.copy()
    X_display.columns = feature_names
    plt.figure(figsize=(FIG_WIDTH_DOUBLE, 6), dpi=300, facecolor='white')
    
    shap.summary_plot(
        shap_values, X_display, feature_names=feature_names, plot_type="dot", max_display=15,
        show=False, color_bar=True, plot_size=None
    )
    
    # 细节微调
    plt.title(f"Impact on {target.upper()} Risk", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=11, labelpad=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 自动保存 PDF 和 PNG
    base_n = os.path.join(FIGURE_DIR, f"Fig4_SHAP_summary_{target}")
    save_fig_medical(base_n)
    plt.close()

def plot_shap_force(shap_values, X, target):
    """个体决策图：解析高风险患者的特征贡献"""
    apply_medical_style()
    risk_scores = shap_values.values.sum(axis=1)
    idx = np.argmax(risk_scores)
    plt.figure(figsize=(FIG_WIDTH_DOUBLE, 3), facecolor='white')
    shap.force_plot(
        shap_values[idx].base_values,
        shap_values[idx].values,
        X.iloc[idx].round(2),
        matplotlib=True, show=False,
        text_rotation=15, contribution_threshold=0.03
    )
    
    plt.title(f"Individual Prediction Logic: High-Risk {target.upper()}", fontsize=12, pad=25)
    
    base_n = os.path.join(INTERP_DIR, f"Fig4B_Force_{target}")
    save_fig_medical(base_n)
    plt.close()

def plot_shap_dependence(shap_values, X, target):
    """依赖关系图：分析关键特征的非线性趋势与交互影响，使用与 Forest 图一致的特征展示名（带单位）"""
    apply_medical_style()
    feature_names = _get_forest_style_feature_names(X.columns.tolist())
    X_display = X.copy()
    X_display.columns = feature_names
    imp = np.abs(shap_values.values).mean(0)
    top_indices = np.argsort(imp)[-3:][::-1]
    top_feats = X.columns[top_indices]
    top_display = _get_forest_style_feature_names(top_feats.tolist())

    for feat, disp in zip(top_feats, top_display):
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
        # dependence_plot 需用 X_display 的列名（展示名），feat 对应 X_display 中同位置的列
        col_display = X_display.columns[list(X.columns).index(feat)]
        shap.dependence_plot(
            col_display, shap_values.values, X_display, show=False, ax=ax,
            interaction_index='auto', alpha=0.7, dot_size=25
        )
        
        ax.set_title(f"Non-linear Risk Trend: {disp}", fontsize=12, fontweight='bold')
        ax.grid(color='whitesmoke', linestyle='-', linewidth=0.5, zorder=0)
        ax.set_facecolor('white')
        
        # 保存
        f_clean = str(feat).replace("/", "_").replace(" ", "_").replace(">", "gt").replace("<", "lt")
        base_n = os.path.join(INTERP_DIR, f"Fig4C_Dep_{target}_{f_clean}")
        save_fig_medical(base_n)
        plt.close()

def load_eval_and_model(target):
    """加载测试集和最佳模型，适配 CalibratedClassifierCV 结构"""
    target_dir = get_model_dir(target)
    
    eval_path = os.path.join(target_dir, "eval_data.pkl")
    models_path = os.path.join(target_dir, "all_models_dict.pkl")
    
    if not all(os.path.exists(p) for p in [eval_path, models_path]):
        raise FileNotFoundError(f"缺少资产: {target}")
    
    eval_data = joblib.load(eval_path)
    models = joblib.load(models_path)
    
    # 优先使用 XGBoost（TreeExplainer 性能最佳）
    if "XGBoost" in models:
        cal_model = models["XGBoost"]
        # [关键修复]：从校准容器中提取原始 XGBoost 实例
        if hasattr(cal_model, "calibrated_classifiers_"):
            model = cal_model.calibrated_classifiers_[0].estimator
        else:
            model = cal_model
        explainer_type = "tree"
    else:
        _log(f"{target} 无 XGBoost，使用 Logistic 替代", "WARN")
        model = models.get("Logistic Regression")
        explainer_type = "linear"
    
    X_test = eval_data['X_test_raw']  # 使用原始尺度进行解释，提高临床可读性
    y_test = eval_data['y_test']
    
    return X_test, y_test, model, explainer_type

def main():
    log_header("启动模块 12: SHAP 模型解释性分析 (适配校准模型)")
    
    for target in TARGETS:
        _log(f"正在生成结局解释: {target.upper()}", "INFO")
        try:
            # 1. 加载资产并处理嵌套的校准模型
            X_test, y_test, model, explainer_type = load_eval_and_model(target)
            
            # 2. 计算 SHAP 值
            shap_vals = compute_shap_values(X_test, model, explainer_type)
            save_shap_data(shap_vals, X_test, target)
            # 3. 生成三大标准解释图
            plot_shap_summary(shap_vals, X_test, target)
            plot_shap_force(shap_vals, X_test, target)
            plot_shap_dependence(shap_vals, X_test, target)
            
        except Exception as e:
            _log(f"失败 {target}: {str(e)}", "ERR")
            import traceback
            traceback.print_exc()
            continue
    
    _log("12 步完成！所有结局的 SHAP 临床解释已保存至结果目录。", "OK")
    _log("下一步：执行 12_clinical_calibration_dca.py 进行获益评估。", "INFO")

if __name__ == "__main__":
    main()
