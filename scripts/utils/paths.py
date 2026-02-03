"""项目路径配置：统一管理数据、模型、结果目录

论文导向产出结构：
  results/main/          主文（投稿用）
    tables/              Table 1-4
    figures/             Figure 1-5
  results/supplementary/ 补充材料
    tables/              OR 统计、SHAP 数据、DCA 数据等
    figures/             S1-S5 补充图
"""
import os

# 项目根目录（scripts/utils 的上级的上级）
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))


def get_project_root() -> str:
    """返回项目根目录"""
    return _PROJECT_ROOT


def get_data_dir() -> str:
    """data/ 目录"""
    return os.path.join(_PROJECT_ROOT, "data")


def get_raw_path(name: str = "mimic") -> str:
    """data/raw/ 下原始数据路径"""
    return os.path.join(_PROJECT_ROOT, "data", "raw", f"{name}_raw_data.csv")


def get_cleaned_path(name: str) -> str:
    """data/cleaned/ 下清洗后数据路径"""
    return os.path.join(_PROJECT_ROOT, "data", "cleaned", name)


def get_external_dir() -> str:
    """data/external/ 目录"""
    return os.path.join(_PROJECT_ROOT, "data", "external")


def get_artifact_path(*parts: str) -> str:
    """artifacts/ 下路径"""
    return os.path.join(_PROJECT_ROOT, "artifacts", *parts)


def get_result_path(*parts: str) -> str:
    """results/ 下路径"""
    return os.path.join(_PROJECT_ROOT, "results", *parts)


def get_model_dir(target: str | None = None) -> str:
    """artifacts/models/ 或 artifacts/models/{target}/"""
    base = os.path.join(_PROJECT_ROOT, "artifacts", "models")
    return os.path.join(base, target.lower()) if target else base


# ---------- 论文导向产出路径 ----------

def get_main_table_dir() -> str:
    """主文表格目录 results/main/tables/（Table 1-4）"""
    return os.path.join(_PROJECT_ROOT, "results", "main", "tables")


def get_main_figure_dir() -> str:
    """主文插图目录 results/main/figures/（Figure 1-5）"""
    return os.path.join(_PROJECT_ROOT, "results", "main", "figures")


def get_supplementary_table_dir() -> str:
    """补充材料表格目录 results/supplementary/tables/"""
    return os.path.join(_PROJECT_ROOT, "results", "supplementary", "tables")


def get_supplementary_figure_dir(*subdirs: str) -> str:
    """补充材料插图目录 results/supplementary/figures/ 或带子目录"""
    base = os.path.join(_PROJECT_ROOT, "results", "supplementary", "figures")
    return os.path.join(base, *subdirs) if subdirs else base


# ---------- 兼容旧接口（指向主文，便于迁移） ----------

def get_figure_dir(*subdirs: str) -> str:
    """兼容：按子目录路由到 main 或 supplementary。
    主文图：audit→main, clinical(DCA/Nomogram)→main, comparison(ROC_External)→main, interpretation(Fig4A)→main
    补充图：lasso→S1, pof/mortality/composite→S2, cutoff→S3, comparison(drift)→S4, interpretation(其他)→S5
    脚本内显式使用 get_main_figure_dir / get_supplementary_figure_dir 更清晰。
    """
    base = os.path.join(_PROJECT_ROOT, "results", "figures")
    return os.path.join(base, *subdirs) if subdirs else base


def get_table_dir() -> str:
    """兼容：返回主文表格目录（Table 1-4 为主文）"""
    return get_main_table_dir()


def get_external_path(name: str) -> str:
    """data/external/{name}"""
    return os.path.join(_PROJECT_ROOT, "data", "external", name)


def get_validation_dir() -> str:
    """validation/"""
    return os.path.join(_PROJECT_ROOT, "validation")


def get_log_file() -> str:
    """运行日志文件路径，默认 logs/run.log（可通过环境变量 MIMIC_LOG_FILE 覆盖）"""
    return os.environ.get("MIMIC_LOG_FILE", os.path.join(_PROJECT_ROOT, "logs", "run.log"))


def ensure_dirs(*paths: str) -> None:
    """确保目录存在"""
    for p in paths:
        os.makedirs(p, exist_ok=True)
