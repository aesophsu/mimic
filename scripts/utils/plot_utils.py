"""绘图工具（供 Step 14 等使用）"""
import numpy as np
import pandas as pd


class PlotUtils:
    """绘图辅助：特征标签格式化、OR 误差计算等"""

    def __init__(self, formatter, lang='en'):
        self.formatter = formatter
        self.lang = lang

    def format_feature_labels(self, feature_list, with_unit=True):
        """将原始特征名转为展示名"""
        return self.formatter.format_features(feature_list, with_unit=with_unit, lang=self.lang)

    @staticmethod
    def format_or_ci(or_val, lower, upper):
        """格式化 OR (95% CI) 文本"""
        return f"{or_val:.2f} ({lower:.2f}–{upper:.2f})"

    def compute_or_error(self, or_df):
        """计算 OR 误差条（对数尺度），确保非负以满足 matplotlib xerr 要求"""
        left_err = np.maximum(0, or_df['OR'] - or_df['OR_Lower'])
        right_err = np.maximum(0, or_df['OR_Upper'] - or_df['OR'])
        return left_err.values, right_err.values

    def compute_or_xlim(self, or_df):
        """计算 OR 森林图 x 轴范围"""
        vals = pd.concat([or_df['OR_Lower'], or_df['OR_Upper']])
        vmin, vmax = vals.min(), vals.max()
        margin = (np.log10(vmax) - np.log10(max(vmin, 0.01))) * 0.15
        return 10 ** (np.log10(max(vmin, 0.01)) - margin), 10 ** (np.log10(vmax) + margin)
