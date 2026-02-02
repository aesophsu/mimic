"""统一日志输出：便于各脚本一致展示"""
from typing import Literal

LogLevel = Literal["INFO", "WARN", "OK", "ERR"]

_PREFIX = {"INFO": "  ", "WARN": "⚠️ ", "OK": "✅ ", "ERR": "❌ "}


def log(msg: str, level: LogLevel = "INFO") -> None:
    """统一日志格式"""
    prefix = _PREFIX.get(level, "  ")
    print(f"{prefix}{msg}")


def log_header(title: str, width: int = 70) -> None:
    """打印步骤标题"""
    print("=" * width)
    print(title)
    print("=" * width)
