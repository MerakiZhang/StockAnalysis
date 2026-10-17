"""补充基本面特征（Alpha158 没有的）：估值 / 股息 / 质量 / 成长。

作为 ML 特征经 StaticDataLoader 并入 Alpha158（不写入 cn_data），方向不需处理（模型自学），
但统一为"有经济含义的比率"，便于截面标准化。

防未来函数：报告期数据从 available_date（实际披露日次日，缺失时为披露截止日）起可见；分红从预案公告日起可见。

价格口径（重要）：qlib $close 是复权价，真实成交价 = $close / $factor。
每股指标（EPS / 每股净资产 / 每股分红）按当时股本计，送转后与新股价不可比；
这里统一换算到复权口径：每股值 × 报告期末（或公告日）的 $factor，再除以复权收盘价，
送转、分红前后保持连续。

    ep_ttm              EPS_TTM / 价格（盈利收益率，亏损为负）
    bp                  每股净资产 / 价格（PB 倒数）
    dividend_yield      过去 365 天已公告每股现金分红之和 / 价格
    dividend_continuity 过去 5 年有现金分红的公告次数
    roe / gross_margin / revenue_yoy / netprofit_yoy   业绩报表原值（%）
    ocf_to_netprofit    经营现金流 TTM / 净利润 TTM（净利润 ≤0 为缺失）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import fundamental as fd

FEATURES = ("ep_ttm", "bp", "dividend_yield", "dividend_continuity", "roe", "gross_margin",
            "ocf_to_netprofit", "revenue_yoy", "netprofit_yoy")
START_YEAR = 2012  # 报告期起始年（TTM 需要前一年）


def _asof(wide: pd.DataFrame, dates: pd.Series, codes: pd.Series) -> np.ndarray:
    """逐行取 wide 在 (date 当日或之前最近交易日, code) 的值。"""
    idx = wide.index.searchsorted(pd.DatetimeIndex(dates), side="right") - 1
    col = wide.columns.get_indexer(codes)
    ok = (idx >= 0) & (col >= 0)
    out = np.full(len(dates), np.nan, dtype="float64")
    vals = wide.to_numpy()
    out[ok] = vals[idx[ok], col[ok]]
    return out


def _latest_visible(df: pd.DataFrame, field: str, cal: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    """报告期长表 -> 宽表：每个交易日取已披露的最新报告期值。

    按实际披露日时，较早报告期可能晚于较新报告期披露（如一季报先于年报），
    这类"倒退"的记录丢弃，保证每天取到的是已披露的最新报告期。
    """
    df = df[df["qlib_code"].isin(codes)].dropna(subset=[field])
    df = df.sort_values(["qlib_code", "available_date", "period"])
    pnum = df["period"].astype(int)
    df = df[pnum >= pnum.groupby(df["qlib_code"]).cummax()]
    df = df.drop_duplicates(["qlib_code", "available_date"], keep="last")
    wide = df.pivot(index="available_date", columns="qlib_code", values=field).astype("float64")
    wide = wide.reindex(cal.union(wide.index)).sort_index().ffill().reindex(cal)
    return wide.reindex(columns=codes)


def _ttm(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """累计值（YTD）-> 单季 -> 连续 4 季之和；缺季则为 NaN（宁缺毋滥）。"""
    d = df[["qlib_code", "period", "available_date", field]].copy()
    d["year"] = d["period"].str[:4].astype(int)
    d["q"] = d["period"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    d = d.dropna(subset=["q", field]).sort_values(["qlib_code", "year", "q"])
    d["qi"] = d["year"] * 4 + d["q"].astype(int)
    g = d.groupby("qlib_code")
    prev_val, prev_qi = g[field].shift(1), g["qi"].shift(1)
    single = np.where(d["q"] == 1, d[field], np.where(prev_qi == d["qi"] - 1, d[field] - prev_val, np.nan))
    d["single"] = single
    g = d.groupby("qlib_code")  # 新增列后重建分组
    d["ttm"] = g["single"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    consecutive = g["qi"].shift(3) == d["qi"] - 3
    d.loc[~consecutive, "ttm"] = np.nan
    return d.dropna(subset=["ttm"]).rename(columns={"ttm": f"{field}_ttm"})


def _event_rolling(events: pd.DataFrame, value: str, cal: pd.DatetimeIndex, codes: list[str],
                   window: str) -> pd.DataFrame:
    """事件（code, date, value）在交易日历上的滚动窗口求和（事件日对齐到其后首个交易日）。"""
    ev = events[events["qlib_code"].isin(codes)].copy()
    pos = cal.searchsorted(pd.DatetimeIndex(ev["date"]))
    ev = ev[pos < len(cal)]
    ev["day"] = cal[pos[pos < len(cal)]]
    wide = ev.pivot_table(index="day", columns="qlib_code", values=value, aggfunc="sum")
    wide = wide.reindex(index=cal, columns=codes).fillna(0.0)
    return wide.rolling(window).sum()


def build_fundamental(codes: list[str], start: str, end: str, close: pd.DataFrame,
                        factor: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """返回 {特征名: 宽表(交易日 × 代码)}。close / factor 为 qlib $close / $factor 宽表，
    需从 START_YEAR 起（用于报告期末复权因子），行为交易日。"""
    cal = pd.DatetimeIndex(close.loc[start:end].index)
    close_out = close.reindex(index=cal, columns=codes)
    fac = factor.reindex(columns=codes).ffill()
    out: dict[str, pd.DataFrame] = {}

    yj = fd.load_yjbb_all(START_YEAR)
    yj = yj[yj["qlib_code"].isin(codes)].copy()
    for col in ("eps", "bps", "roe", "gross_margin", "revenue_yoy", "netprofit_yoy", "netprofit"):
        yj[col] = pd.to_numeric(yj[col], errors="coerce")
    yj["period_end"] = pd.to_datetime(yj["period"])
    f_at = _asof(fac, yj["period_end"], yj["qlib_code"])
    yj["adj_eps"] = yj["eps"] * f_at   # 复权口径每股收益（累计值）
    yj["adj_bps"] = yj["bps"] * f_at

    eps_ttm = _latest_visible(_ttm(yj, "adj_eps"), "adj_eps_ttm", cal, codes)
    out["ep_ttm"] = eps_ttm / close_out
    out["bp"] = _latest_visible(yj, "adj_bps", cal, codes) / close_out
    for f in ("roe", "gross_margin", "revenue_yoy", "netprofit_yoy"):
        out[f] = _latest_visible(yj, f, cal, codes)

    xj = fd.load_xjll_all(START_YEAR)
    xj = xj[xj["qlib_code"].isin(codes)].copy()
    ocf = _latest_visible(_ttm(xj, "ocf"), "ocf_ttm", cal, codes)
    np_ttm = _latest_visible(_ttm(yj, "netprofit"), "netprofit_ttm", cal, codes)
    out["ocf_to_netprofit"] = ocf / np_ttm.where(np_ttm > 0)

    try:
        fh = fd._load_periods("fhps", START_YEAR)
    except FileNotFoundError:
        fh = None
    if fh is not None:
        fh["dps"] = pd.to_numeric(fh["dps"], errors="coerce")
        fh = fh[fh["qlib_code"].isin(codes) & (fh["dps"] > 0) & fh["announce_date"].notna()].copy()
        fh = fh.drop_duplicates(["qlib_code", "period", "announce_date"])
        fh["date"] = pd.to_datetime(fh["announce_date"])
        fh["adj_dps"] = fh["dps"] * _asof(fac, fh["date"], fh["qlib_code"])
        fh["one"] = 1.0
        # 预热：窗口需要从 start 之前开始累计
        full_cal = pd.DatetimeIndex(close.index)
        dps = _event_rolling(fh, "adj_dps", full_cal, codes, "365D").reindex(cal)
        cnt = _event_rolling(fh, "one", full_cal, codes, "1826D").reindex(cal)
        out["dividend_yield"] = dps / close_out
        out["dividend_continuity"] = cnt
    else:
        out["dividend_yield"] = pd.DataFrame(0.0, index=cal, columns=codes)
        out["dividend_continuity"] = pd.DataFrame(0.0, index=cal, columns=codes)

    return {k: out[k].replace([np.inf, -np.inf], np.nan).astype("float32") for k in FEATURES}


def panel(ctx: dict) -> dict[str, pd.DataFrame]:
    """ctx 区间与股票上的基本面特征（FEATURES），首次调用时计算并缓存在 ctx["_fund"]。"""
    if "_fund" not in ctx:
        from ..data.market import load_wide

        dates = pd.DatetimeIndex(ctx["dates"])
        codes = list(ctx["codes"])
        wide = load_wide(codes, f"{START_YEAR}-01-01", str(dates[-1].date()), ("close", "factor"))
        close = wide["close"].reindex(columns=codes)
        factor = wide["factor"].reindex(columns=codes)
        feats = build_fundamental(codes, str(dates[0].date()), str(dates[-1].date()), close, factor)
        ctx["_fund"] = {k: v.reindex(index=dates, columns=codes) for k, v in feats.items()}
    return ctx["_fund"]


def industry_map() -> pd.Series:
    """qlib_code -> 行业（只用一套分类，避免混用口径）：业绩报表"所处行业"（东财行业）优先，
    无缓存时用 baostock 证监会行业快照。两者都是**当前**分类，用于历史回测有轻微前视，报告中注明。"""
    try:
        yj = fd.load_yjbb_all(START_YEAR).sort_values("period").dropna(subset=["industry"])
        em = yj.drop_duplicates("qlib_code", keep="last").set_index("qlib_code")["industry"]
        if len(em):
            return em
    except FileNotFoundError:
        pass
    from ..data.baostock_data import load_industry

    return load_industry()
