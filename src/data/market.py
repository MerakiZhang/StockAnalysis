"""qlib 初始化与行情数据读取封装。"""
from __future__ import annotations

import os

import pandas as pd
import qlib
from qlib.data import D

DEFAULT_PROVIDER = os.path.expanduser(r"~\.qlib\qlib_data\cn_data")  # 可用 QLIB_DATA 环境变量覆盖

_initialized = False


def data_dir() -> str:
    """qlib 数据目录：环境变量 QLIB_DATA 优先，否则 DEFAULT_PROVIDER。"""
    return os.environ.get("QLIB_DATA", DEFAULT_PROVIDER)


def init_qlib(provider_uri: str | None = None, kernels: int | None = None) -> None:
    """初始化 qlib（幂等）。

    kernels: 并行加载的工作进程数。本机内存有限，默认 2，
    可用环境变量 QLIB_KERNELS 覆盖（设为 1 完全串行）。
    """
    global _initialized
    if _initialized:
        return
    uri = provider_uri or data_dir()
    k = kernels or int(os.environ.get("QLIB_KERNELS", "2"))
    qlib.init(provider_uri=uri, region="cn", kernels=k)
    _initialized = True


def load_ohlcv(
    codes: list[str] | str,
    start: str,
    end: str,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount", "factor"),
) -> pd.DataFrame:
    """读取日线行情，返回 MultiIndex(instrument, datetime) 的 DataFrame。

    fields 为不含 $ 前缀的字段名。
    """
    init_qlib()
    exprs = [f"${f}" for f in fields]
    df = D.features(codes, exprs, start_time=start, end_time=end)
    df.columns = list(fields[: len(df.columns)])
    return df


def latest_trade_date() -> pd.Timestamp:
    """qlib 日历中最后一个有数据的交易日（用于校验数据新鲜度）。"""
    init_qlib()
    cal = D.calendar(start_time="2000-01-01")
    return pd.Timestamp(cal[-1])
