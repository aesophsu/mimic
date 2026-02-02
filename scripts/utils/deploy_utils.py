"""部署资产加载：deploy_bundle 等（09、11 共用）"""
import os
import joblib

from .paths import get_model_dir, get_artifact_path


def load_deploy_bundle(target, fill_missing=True):
    """
    加载指定结局的 deploy_bundle。
    fill_missing: 若 True，自动补齐 imputer/mimic_scaler/skewed_cols（09 变换用）
    """
    bundle_path = os.path.join(get_model_dir(target), "deploy_bundle.pkl")
    if not os.path.exists(bundle_path):
        return None
    try:
        bundle = joblib.load(bundle_path)
    except Exception:
        return None

    if not fill_missing:
        return bundle

    artifact_dir = get_artifact_path("scalers")
    if "imputer" not in bundle:
        bundle["imputer"] = joblib.load(os.path.join(artifact_dir, "mimic_mice_imputer.joblib"))
    if "mimic_scaler" not in bundle:
        bundle["mimic_scaler"] = joblib.load(os.path.join(artifact_dir, "mimic_scaler.joblib"))
    if "skewed_cols" not in bundle:
        bundle["skewed_cols"] = joblib.load(os.path.join(artifact_dir, "skewed_cols_config.pkl"))
    if "imputer_feature_order" not in bundle and "imputer" in bundle:
        imp = bundle["imputer"]
        order = getattr(imp, "feature_names_in_", None)
        if order is None:
            order = bundle.get("train_assets_bundle", {}).get("feature_order", [])
        bundle["imputer_feature_order"] = list(order) if order is not None else []
    return bundle
