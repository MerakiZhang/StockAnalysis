"""技术面因子：动量/波动（稳健池）+ 短线情绪/反转（短线池）。

注意：方向（动量 vs 反转）不预设，由 IC 检验实证决定；
此处按原始定义输出，取反统一在注册表（registry.py）或调用方完成。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import daily_returns


# ---------------------------------------------------------------- 动量/波动
def mom_n(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 日动量：close/close.shift(n) - 1。"""
    return close / close.shift(n) - 1


def vol_n(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """N 日收益率波动率。"""
    return daily_returns(close).rolling(n).std()


def risk_adj_momentum(close: pd.DataFrame, mom_window: int = 60, vol_window: int = 20) -> pd.DataFrame:
    """风险调整动量：60日动量 / 20日波动率。"""
    m = mom_n(close, mom_window)
    v = vol_n(close, vol_window)
    return m / v.replace(0, np.nan)


def dist_high_n(close: pd.DataFrame, n: int = 250) -> pd.DataFrame:
    """距 N 日高点的距离：close / rolling_max - 1（≤0，越接近 0 离高点越近）。"""
    return close / close.rolling(n, min_periods=n // 2).max() - 1


def max_ret_n(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """MAX 彩票因子：近 N 日最大单日涨幅。预期负向（IC 检验实证）。"""
    return daily_returns(close).rolling(n, min_periods=n // 2).max()


def downside_vol_n(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """下行波动率：近 N 日负收益的标准差（全日收益不足时退化为全样本波动）。"""
    ret = daily_returns(close)
    down = ret.where(ret < 0)
    return down.rolling(n, min_periods=n // 2).std()


# ---------------------------------------------------------------- 短线因子
def mom_5d(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.shift(5) - 1


def mom_10d(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.shift(10) - 1


def volume_ratio(volume: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """量比：当日成交量 / 前 N 日均量。"""
    return volume / volume.shift(1).rolling(n).mean()


def breakout_20d(close: pd.DataFrame, high: pd.DataFrame | None = None) -> pd.DataFrame:
    """突破 20 日新高：收盘距 20 日最高价的相对距离（<=0，越接近 0 越强）。"""
    ref = high if high is not None else close
    rolling_max = ref.rolling(20).max()
    return close / rolling_max - 1


def turnover_surge(volume: pd.DataFrame, short: int = 5, long: int = 60) -> pd.DataFrame:
    """换手率突变代理：短均量 / 长均量 - 1。"""
    return volume.rolling(short).mean() / volume.rolling(long).mean() - 1


def limit_up_threshold(code: str, is_st: bool = False) -> float:
    """分板块涨跌停幅度：主板 10%、创业板/科创板 20%、北交所 30%、ST 5%。"""
    if is_st:
        return 0.05
    c = code[-6:]
    if c.startswith(("30", "68")):
        return 0.20
    if c.startswith(("4", "8", "92")):
        return 0.30
    return 0.10


def limit_up_count(close: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """涨停基因：近 N 日涨停次数（按分板块阈值，容差 0.5%）。"""
    ret = daily_returns(close)
    cols = close.columns
    thr = pd.Series({c: limit_up_threshold(c) for c in cols}, dtype=np.float32)
    is_limit = ret.ge(thr - 0.005)
    return is_limit.rolling(n).sum()


def dist_from_high(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """距 N 日高点的相对距离（<=0）。方向由 IC 检验决定。"""
    return close / close.rolling(n).max() - 1


def price_position(close: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """N 日价格区间中的相对位置：0=最低 1=最高。"""
    lo, hi = close.rolling(n).min(), close.rolling(n).max()
    return (close - lo) / (hi - lo).replace(0, np.nan)


def lower_shadow(open_: pd.DataFrame, close: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """下影线比例：(min(open,close)-low)/close，探底回升信号。"""
    return (np.minimum(open_, close) - low) / close


def gap(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """隔夜跳空：open / 昨收 - 1。"""
    return open_ / close.shift(1) - 1


def intraday_ret(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """日内收益：close / open - 1。"""
    return close / open_ - 1


def range_n(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """N 日平均振幅：(high-low)/昨收 的 N 日均值。"""
    return ((high - low) / close.shift(1)).rolling(n).mean()


def volume_cv(volume: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """成交量变异系数：std/mean，越小越缩量平稳。"""
    m = volume.rolling(n).mean()
    return volume.rolling(n).std() / m.replace(0, np.nan)


def max_drawdown_n(close: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """N 日内最大回撤（<=0）：close / N日内rolling max 累计回撤的最小值。"""
    ret = close.pct_change(fill_method=None)
    cum = (1 + ret.fillna(0)).rolling(n).apply(np.prod, raw=True)
    peak = cum.rolling(n, min_periods=1).max()
    return cum / peak - 1
