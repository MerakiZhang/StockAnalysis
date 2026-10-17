"""股池文件位置：data/pool/{pool}/{YYYY-MM-DD}.csv 与 {YYYY-MM-DD}_brief.md（写入见 selection/points_daily.py）。

日期为**信号日**（数据最新交易日 T），计划于 T 之后首个交易日执行。
"""
from __future__ import annotations

from pathlib import Path

from ..paths import POOL_OUTPUT_DIR


def pool_dir(name: str) -> Path:
    return POOL_OUTPUT_DIR / name
