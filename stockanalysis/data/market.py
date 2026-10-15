"""qlib 初始化与行情数据读取：日历、OHLCV、宽表、行情上下文。

约定：宽表 DataFrame（index=交易日，columns=qlib 代码），float32。
"""
from __future__ import annotations

import os

import numpy as np
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


def trade_calendar(start: str | None = None, end: str | None = None) -> pd.DatetimeIndex:
    """qlib 交易日历（含两端）。"""
    init_qlib()
    return pd.DatetimeIndex(D.calendar(start_time=start, end_time=end))


def latest_trade_date() -> pd.Timestamp:
    """qlib 日历中最后一个有数据的交易日。"""
    return trade_calendar()[-1]


def load_wide(
    codes: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount"),
) -> dict[str, pd.DataFrame]:
    """加载行情并透视成 {field: 宽表(date x code)}，float32 节省内存。"""
    df = load_ohlcv(codes, start, end, fields)
    return {f: df[f].unstack(level="instrument").astype(np.float32).sort_index()
            for f in fields if f in df.columns}


CTX_FIELDS = ("open", "high", "low", "close", "volume", "amount")


def market_ctx(codes: list[str], start: str, end: str,
               fields: tuple[str, ...] = CTX_FIELDS) -> dict[str, pd.DataFrame]:
    """行情上下文：{field: 宽表}，供过滤规则与打分器共用、只加载一次。"""
    return load_wide(codes, start, end, fields)
