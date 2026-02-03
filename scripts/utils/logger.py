"""统一日志输出：控制台 + 单独 log 文件"""
import os
import sys
from typing import Literal

from .paths import get_log_file, ensure_dirs

LogLevel = Literal["INFO", "WARN", "OK", "ERR"]

_PREFIX = {"INFO": "  ", "WARN": "⚠️ ", "OK": "✅ ", "ERR": "❌ "}

# 模块级文件句柄，首次写入时打开（追加）
_log_file_handle = None


def _get_log_stream():
    """获取日志文件流，首次调用时创建 logs 目录并打开文件（追加）"""
    global _log_file_handle
    if _log_file_handle is None:
        log_path = get_log_file()
        ensure_dirs(os.path.dirname(log_path))
        _log_file_handle = open(log_path, "a", encoding="utf-8")
    return _log_file_handle


def _write_log(line: str) -> None:
    """同时输出到控制台和日志文件"""
    print(line)
    try:
        stream = _get_log_stream()
        stream.write(line + "\n")
        stream.flush()
    except OSError:
        pass  # 写文件失败时不影响控制台


def log(msg: str, level: LogLevel = "INFO") -> None:
    """统一日志格式：控制台 + 写入 log 文件"""
    prefix = _PREFIX.get(level, "  ")
    line = f"{prefix}{msg}"
    _write_log(line)


def log_header(title: str, width: int = 70) -> None:
    """打印步骤标题：控制台 + 写入 log 文件"""
    sep = "=" * width
    _write_log(sep)
    _write_log(title)
    _write_log(sep)
