"""
统一特征名称格式化模块
从 feature_dictionary.json 加载 display_name，供表格、图片输出统一使用。
"""
import os
import json

# 默认语言：从 config 读取，若无则 'en'
try:
    from .feature_formatter_config import DISPLAY_LANG as DEFAULT_LANG
except (ImportError, AttributeError):
    DEFAULT_LANG = 'en'

# 字典缓存
_dict_cache = None
_dict_path = None


def _get_feature_dict():
    """懒加载 feature_dictionary.json"""
    global _dict_cache, _dict_path
    if _dict_cache is not None:
        return _dict_cache
    # 支持从项目根或 scripts 目录运行
    for base in [
        os.path.join(os.path.dirname(__file__), '../..'),
        os.path.join(os.path.dirname(__file__), '..'),
        '.'
    ]:
        path = os.path.join(base, 'artifacts/features/feature_dictionary.json')
        if os.path.exists(path):
            _dict_path = os.path.abspath(path)
            with open(path, 'r', encoding='utf-8') as f:
                _dict_cache = json.load(f)
            return _dict_cache
    _dict_cache = {}
    return _dict_cache


class FeatureFormatter:
    """
    特征名称格式化器：将原始特征名（如 creatinine_max）转为展示名（如 Creatinine (Max)）
    """

    def __init__(self, lang=None):
        self.lang = lang or DEFAULT_LANG
        self._dict = _get_feature_dict()

    def get_label(self, feature_name, with_unit=False, lang=None):
        """
        获取单个特征的展示名称
        :param feature_name: 原始特征名（如 creatinine_max）
        :param with_unit: 是否附加单位（如 mg/dL）
        :param lang: 可选，覆盖实例语言 ('en'|'cn')
        :return: 展示名称
        """
        if feature_name not in self._dict:
            return str(feature_name)
        cfg = self._dict[feature_name]
        _lang = lang if lang is not None else self.lang
        key = 'display_name_cn' if _lang == 'cn' else 'display_name_en'
        label = cfg.get(key) or cfg.get('display_name') or str(feature_name)
        if with_unit and cfg.get('unit'):
            unit = cfg['unit']
            # 将 Unicode 上标 (⁹等) 转为 ^n，避免 Arial 等字体缺失 glyph 警告
            unit = unit.replace('⁹', '^9').replace('⁶', '^6').replace('⁵', '^5')
            unit = unit.replace('⁴', '^4').replace('³', '^3').replace('²', '^2').replace('¹', '^1').replace('⁰', '^0')
            label = f"{label} ({unit})"
        return label

    def format_feature(self, feature_name, with_unit=False):
        """别名：与 get_label 一致"""
        return self.get_label(feature_name, with_unit)

    def format_features(self, feature_list, with_unit=False, lang=None):
        """
        批量格式化特征名
        :param feature_list: 原始特征名列表
        :param with_unit: 是否附加单位
        :param lang: 可选，覆盖实例语言
        :return: 展示名列表
        """
        return [self.get_label(f, with_unit, lang) for f in feature_list]

    def format_dataframe_columns(self, df, columns=None, with_unit=False):
        """
        将 DataFrame 列名转为展示名，返回 {raw: display} 映射
        :param df: DataFrame
        :param columns: 指定列，默认全部
        :param with_unit: 是否附加单位
        :return: dict 列名映射
        """
        cols = columns or df.columns.tolist()
        return {c: self.get_label(c, with_unit) for c in cols if c in df.columns}


def format_feature(feature_name, lang=None, with_unit=False):
    """
    便捷函数：格式化单个特征名
    """
    return FeatureFormatter(lang=lang).get_label(feature_name, with_unit)


def format_features(feature_list, lang=None, with_unit=False):
    """
    便捷函数：批量格式化特征名
    """
    return FeatureFormatter(lang=lang).format_features(feature_list, with_unit)
