"""研究指标（唯一定义处）：逐日 RankIC、TopK 未来超额、IC 汇总、特征间相关。

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


def topk_excess(score: pd.DataFrame, eligible: pd.DataFrame, fwd: pd.DataFrame,
                ks=(5, 10, 20)) -> dict[int, pd.Series]:
    """每日在可选范围（eligible & 有前向收益）内按分数取前 K 只，其未来收益相对可选范围均值的超额。"""
    fwd = fwd.reindex_like(score)
    ok = eligible.reindex_like(score).fillna(False).astype(bool) & fwd.notna() & score.notna()
    fwd = fwd.where(ok)
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    rank = score.where(ok).rank(axis=1, ascending=False, method="first")
    return {k: excess.where(rank <= k).mean(axis=1) for k in ks}


def summarize_ic(ic: pd.Series) -> dict[str, float]:
    ic = ic.dropna()
    std = ic.std()
    return {
        "mean_ic": float(ic.mean()),
        "ic_std": float(std),
        "icir": float(ic.mean() / std) if std else np.nan,
        "positive_rate": float((ic > 0).mean()),
    }


def score_ties(score: pd.DataFrame) -> pd.DataFrame:
    """逐日分数去重比例与最高分并列只数（检查模型是否退化）。"""
    return pd.DataFrame({
        "去重比例": score.apply(lambda r: r.dropna().nunique() / max(r.notna().sum(), 1), axis=1),
        "最高分并列": score.apply(lambda r: int((r == r.max()).sum()), axis=1),
    })


def winsorize_mad(s: pd.Series, n: float = 5.0) -> pd.Series:
    """MAD 去极值（按截面调用）。"""
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return s
    return s.clip(med - n * 1.4826 * mad, med + n * 1.4826 * mad)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def normalize_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """对宽表逐行（交易日截面）做 MAD 去极值 + zscore。"""
    return df.apply(lambda row: zscore(winsorize_mad(row)), axis=1)


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """特征间平均截面相关（对标准化后的特征逐日求相关再平均）。"""
    normed = {k: normalize_cross_section(v) for k, v in factors.items()}
    names = list(normed)
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        mat.loc[a, a] = 1.0
        for b in names[i + 1:]:
            mat.loc[a, b] = mat.loc[b, a] = float(normed[a].corrwith(normed[b], axis=1).mean())
    return mat
