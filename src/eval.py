"""评估工具：单因子 IC 检验（IC/RankIC、分层收益、因子相关性）+ 进程内存监控。

内存约定（本机 32G）：重任务脚本在关键节点打印 RSS，
超过 ~12G 应主动降 QLIB_KERNELS 或缩小加载范围。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .selection.scoring import normalize_cross_section


# ------------------------------------------------------------ 因子检验
def forward_returns(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """未来 N 日收益（用于 IC 检验，注意仅限研究用途，不可用于选股信号）。"""
    return close.shift(-n) / close - 1


def daily_rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """逐日截面 RankIC（因子值与未来收益的 Spearman 相关）。"""
    fr = factor.rank(axis=1)
    rr = fwd.rank(axis=1)
    ic = fr.corrwith(rr, axis=1)
    return ic.dropna()


def rolling_rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, window: int = 250) -> pd.Series:
    """滚动窗口均值 RankIC（因子衰减监控）。"""
    ic = daily_rank_ic(factor, fwd)
    return ic.rolling(window, min_periods=window // 2).mean().dropna()


def summarize_ic(ic: pd.Series) -> dict[str, float]:
    return {
        "mean_ic": float(ic.mean()),
        "ic_std": float(ic.std()),
        "icir": float(ic.mean() / ic.std()) if ic.std() else np.nan,
        "positive_rate": float((ic > 0).mean()),
    }


def quantile_returns(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    q: int = 5,
) -> pd.DataFrame:
    """分层收益：按因子值分 q 组，统计各组平均未来收益（检验单调性）。"""
    results: dict[int, list[float]] = {i: [] for i in range(1, q + 1)}
    common_dates = factor.dropna(how="all").index.intersection(fwd.index)
    for dt in common_dates:
        f = factor.loc[dt].dropna()
        r = fwd.loc[dt].reindex(f.index)  # 因子有而行情无的代码收益为 NaN，不计入分组
        if len(f) < q * 10:
            continue
        grp = pd.qcut(f, q, labels=False, duplicates="drop") + 1
        for g, sub in r.groupby(grp):
            results[int(g)].append(sub.mean())
    rows = {g: np.nanmean(v) if v else np.nan for g, v in results.items()}
    return pd.Series(rows, name="mean_fwd_return").to_frame()


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """因子间平均截面相关（对 z-score 后的因子逐日求相关再平均）。"""
    normed = {k: normalize_cross_section(v) for k, v in factors.items()}
    names = list(normed)
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                mat.loc[a, b] = 1.0
            elif pd.isna(mat.loc[b, a]):
                daily = normed[a].corrwith(normed[b], axis=1)
                mat.loc[a, b] = mat.loc[b, a] = float(daily.mean())
            else:
                mat.loc[a, b] = mat.loc[b, a]
    return mat


# ------------------------------------------------------------ 内存监控
def rss_gb() -> float | None:
    """当前进程 RSS（GB）；未安装 psutil 时返回 None。"""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def log_rss(tag: str = "") -> None:
    g = rss_gb()
    if g is not None:
        print(f"[mem] {tag} RSS={g:.2f} GB", flush=True)
