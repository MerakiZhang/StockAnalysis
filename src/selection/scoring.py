"""因子预处理与打分合成：MAD 去极值 → zscore 标准化 → 加权合成。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_mad(s: pd.Series, n: float = 5.0) -> pd.Series:
    """MAD 去极值（按行/截面调用）。"""
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return s
    upper, lower = med + n * 1.4826 * mad, med - n * 1.4826 * mad
    return s.clip(lower, upper)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def normalize_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """对宽表逐行（交易日截面）做 MAD 去极值 + zscore。"""
    return df.apply(lambda row: zscore(winsorize_mad(row)), axis=1)


def combine(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """多因子加权合成总分。

    factors: {因子名: 宽表}，全部为"越大越好"。
    权重为 0 或缺失的因子不参与。返回总分宽表。
    """
    total: pd.DataFrame | None = None
    wsum = 0.0
    for name, df in factors.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        normed = normalize_cross_section(df)
        total = normed * w if total is None else total.add(normed * w, fill_value=np.nan)
        wsum += abs(w)
    if total is None or wsum == 0:
        raise ValueError("没有有效因子或权重全为 0")
    return total / wsum
