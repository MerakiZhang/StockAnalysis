"""原始缓存的公共约定（data/raw/<来源>/<数据集>/<单元>.parquet）。

快照类数据（全量历史一次返回、或只有当前状态，如增减持、回购、行业、当日行情快照）以抓取日为单元：
保存新一份时删掉同数据集的旧文件，只留最新一份；读取时取最新一份。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def today_tag() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d")


def snapshot_path(folder: Path) -> Path:
    """今天的快照文件路径（不保证存在）。"""
    return Path(folder) / f"{today_tag()}.parquet"


def latest_snapshot(folder: Path) -> Path | None:
    files = sorted(Path(folder).glob("*.parquet"))
    return files[-1] if files else None


def save_snapshot(df: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """写入今天的快照并删除同目录的旧快照。"""
    path = snapshot_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    for old in Path(folder).glob("*.parquet"):
        if old != path:
            old.unlink()
    return df
