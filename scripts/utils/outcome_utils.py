"""结局变量对齐：早期死亡覆盖、列重命名（MIMIC 与 eICU 共用）"""
import pandas as pd


def apply_early_death_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    24-48h 内早期死亡视为 POF，覆盖 pof 与 mortality_28d。
    与 01_mimic_cleaning、08_eicu_alignment 逻辑一致。
    """
    if "early_death_24_48h" not in df.columns:
        return df
    mask = df["early_death_24_48h"] == 1
    if not mask.any():
        return df
    df = df.copy()
    df.loc[mask, "pof"] = 1
    if "mortality_28d" in df.columns:
        df.loc[mask, "mortality_28d"] = 1
    return df


def align_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    结局列重命名：mortality_28d -> mortality, composite_outcome -> composite
    并重新计算 composite = pof | mortality（覆盖早期死亡后的最新状态）
    """
    df = df.copy()
    mort_col = "mortality_28d" if "mortality_28d" in df.columns else "mortality"
    if "pof" in df.columns and mort_col in df.columns:
        df["composite"] = ((df["pof"] == 1) | (df[mort_col] == 1)).astype(int)
    if "mortality_28d" in df.columns:
        df = df.rename(columns={"mortality_28d": "mortality"})
    if "composite_outcome" in df.columns:
        df = df.drop(columns=["composite_outcome"], errors="ignore")
    return df


def normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    """性别列标准化：M/Male/1 -> 1, F/Female/0 -> 0"""
    if "gender" not in df.columns:
        return df
    df = df.copy()
    df["gender"] = (
        df["gender"]
        .replace(["M", "Male", "MALE", 1, 1.0], 1)
        .replace(["F", "Female", "FEMALE", 0, 0.0], 0)
        .fillna(df["gender"].mode()[0] if not df["gender"].dropna().empty else 0)
        .astype(int)
    )
    return df
