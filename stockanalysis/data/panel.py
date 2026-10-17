"""派生紧凑表（data/panel/<name>.parquet）：由 data/raw 生成、可随时重建，读取比逐个原始文件快得多。

    is_st         交易日 × 股票的 ST 宽表（来自 raw/baostock/daily，见 baostock_data.is_st_panel）
    risk_notice   命中风险关键词的公告（来自 raw/akshare/notice，见 features/risk.risk_events）

每张表旁有 <name>.meta.json，记录源文件签名（个数、文件名校验和、最大 mtime）与额外键（如关键词正则）；
签名或额外键变化即重建，否则直接读取。删除 data/panel/ 不丢任何信息。
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import Callable

import pandas as pd

from ..paths import PANEL_DIR


def signature(files: list[Path]) -> dict:
    names = "\n".join(sorted(p.name for p in files)).encode("utf-8")
    mtime = max((os.stat(p).st_mtime for p in files), default=0.0)
    return {"n": len(files), "names_crc": zlib.crc32(names), "max_mtime": mtime}


def build_if_stale(name: str, src_files: list[Path], builder: Callable[[], pd.DataFrame],
                   extra: dict | None = None) -> pd.DataFrame:
    path, meta_path = PANEL_DIR / f"{name}.parquet", PANEL_DIR / f"{name}.meta.json"
    meta = {**signature(src_files), **(extra or {})}
    if path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")) == meta:
                return pd.read_parquet(path)
        except (OSError, json.JSONDecodeError):
            pass
    df = builder()
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return df
