"""研究指标（唯一定义处）：逐日 RankIC、IC 汇总。

输入均为宽表（index=交易日, columns=代码）。前向收益见 features/labels.py，仅限研究用途。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(score: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """逐日截面 RankIC：只在两者都非空的股票上计算秩相关。"""
    fwd = fwd.reindex_like(score)
    both = score.notna() & fwd.notna()
    return score.where(both).rank(axis=1).corrwith(fwd.where(both).rank(axis=1), axis=1)


def summarize_ic(ic: pd.Series) -> dict[str, float]:
    ic = ic.dropna()
    std = ic.std()
    return {
        "mean_ic": float(ic.mean()),
        "ic_std": float(std),
        "icir": float(ic.mean() / std) if std else np.nan,
        "positive_rate": float((ic > 0).mean()),
    }
