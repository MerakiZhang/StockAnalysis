"""基本面因子：质量/成长/估值/股息，及其时点面板构建。

核心原则（防未来函数）：报告期数据从 available_date（披露截止日）
起才向前填充到交易日，绝不用未来数据。

因子方向统一为"数值越大越好"（取反在各因子函数内完成）。
"""
from __future__ import annotations

import pandas as pd
from qlib.data import D

from ..data import fundamental as fd
from ..data.market import init_qlib


# ============================================================ 时点面板构建
def trade_calendar(start: str, end: str) -> pd.DatetimeIndex:
    init_qlib()
    return pd.DatetimeIndex(D.calendar(start_time=start, end_time=end))


def _report_wide(
    df: pd.DataFrame,
    field: str,
    codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """报告期长表字段 -> 宽表(date x code)，按披露可见日向前填充。"""
    df = df[df["qlib_code"].isin(codes)]
    df = df.sort_values("available_date")
    # 每个代码在每个可见日只保留最新报告期的值
    df = df.drop_duplicates(subset=["qlib_code", "available_date"], keep="last")
    wide = df.pivot(index="available_date", columns="qlib_code", values=field)
    wide = wide.apply(pd.to_numeric, errors="coerce").astype("float32")
    cal = trade_calendar(start, end)
    return wide.reindex(cal.union(wide.index)).sort_index().ffill().reindex(cal)


def yjbb_wide(
    field: str,
    codes: list[str],
    start: str,
    end: str,
    start_year: int = 2017,
) -> pd.DataFrame:
    """业绩报表字段 -> 宽表(date x code)，按披露可见日向前填充。

    field 如 roe / gross_margin / revenue_yoy / netprofit_yoy。
    """
    return _report_wide(fd.load_yjbb_all(start_year), field, codes, start, end)


def ttm_wide(
    source: str,
    field: str,
    codes: list[str],
    start: str,
    end: str,
    start_year: int = 2016,
) -> pd.DataFrame:
    """流量字段（净利润/经营现金流等）-> TTM 宽表(date x code)。

    东财报告期数据为年内累计值（YTD）：先按同年差分成单季，
    再以最近 4 个单季之和为 TTM，对齐到最新一期报告的可见日后向前填充。

    source: "yjbb"（业绩报表）| "xjll"（现金流量表）。
    """
    df = fd.load_yjbb_all(start_year) if source == "yjbb" else fd.load_xjll_all(start_year)
    df = df[df["qlib_code"].isin(codes)][["qlib_code", "period", "available_date", field]]
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df["year"] = df["period"].str[:4]
    df["q"] = df["period"].str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    df = df.dropna(subset=["q"]).sort_values(["qlib_code", "year", "q"])
    df["q"] = df["q"].astype(int)
    # 单季值 = 本年累计 - 上一期累计（一季报即单季）
    prev = df.groupby(["qlib_code", "year"])[field].shift(1)
    single = df[field].where(df["q"] == 1, df[field] - prev)
    df["single"] = single
    # TTM = 最近 4 个单季之和（不足 4 季为 NaN，宁缺毋滥）
    df["ttm"] = (
        df.groupby("qlib_code")["single"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    ttm = df.dropna(subset=["ttm"])[["qlib_code", "period", "available_date", "ttm"]]
    return _report_wide(ttm.rename(columns={"ttm": field}), field, codes, start, end)


def valuation_wide(
    indicator: str,
    codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """估值指标（pe_ttm/pb）-> 宽表(date x code)。

    逐股读缓存 parquet 后合并；缓存缺失的列为 NaN。
    """
    cal = trade_calendar(start, end)
    out = pd.DataFrame(index=cal)
    for code in codes:
        path = fd.FUND_DIR / "valuation" / f"{code.upper()}_{indicator}.parquet"
        if not path.exists():
            continue
        s = pd.read_parquet(path).set_index("date")["value"]
        out[code] = s.reindex(cal.union(s.index)).sort_index().ffill().reindex(cal)
    return out.astype("float32")


def dividend_yield_wide(
    codes: list[str],
    start: str,
    end: str,
    close: pd.DataFrame,
    start_year: int = 2017,
) -> pd.DataFrame:
    """股息率 TTM：过去 365 天内已披露分红方案的每股股利之和 / 收盘价。

    fhps 的预案公告日为真实披露日。无分红记录则为 0。
    """
    frames = []
    for p in sorted((fd.FUND_DIR / "fhps").glob("fhps_*.parquet")):
        if int(p.stem.split("_")[1][:4]) >= start_year:
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(0.0, index=close.index, columns=codes, dtype="float32")
    fh = pd.concat(frames, ignore_index=True)
    fh = fh[fh["qlib_code"].isin(codes) & fh["dps"].notna() & fh["announce_date"].notna()]
    fh["dps"] = pd.to_numeric(fh["dps"], errors="coerce")

    cal = close.index
    codes = [c for c in codes if c in close.columns]  # 窗口内无行情的代码跳过
    dps_ttm = pd.DataFrame(0.0, index=cal, columns=codes, dtype="float32")
    # 按披露日把每笔分红累加到其后 365 天的窗口
    for code, g in fh.groupby("qlib_code"):
        if code not in dps_ttm.columns:
            continue
        s = pd.Series(0.0, index=cal)
        for _, row in g.iterrows():
            window = (cal >= row["announce_date"]) & (cal <= row["announce_date"] + pd.Timedelta(days=365))
            s[window] += float(row["dps"])
        dps_ttm[code] = s
    return (dps_ttm / close.replace(0, pd.NA)).astype("float32")


# ============================================================ 质量类
def roe(codes: list[str], start: str, end: str) -> pd.DataFrame:
    return yjbb_wide("roe", codes, start, end)


def gross_margin(codes: list[str], start: str, end: str) -> pd.DataFrame:
    return yjbb_wide("gross_margin", codes, start, end)


def ocf_to_netprofit(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """盈利质量：经营现金流 TTM / 净利润 TTM。

    >1 表示利润有现金支撑；净利润 ≤0 时比值无意义，置 NaN。
    """
    ocf = ttm_wide("xjll", "ocf", codes, start, end)
    np_ = ttm_wide("yjbb", "netprofit", codes, start, end)
    ratio = ocf / np_.where(np_ > 0)
    return ratio.astype("float32")


# ============================================================ 成长类
def revenue_yoy(codes: list[str], start: str, end: str) -> pd.DataFrame:
    return yjbb_wide("revenue_yoy", codes, start, end)


def netprofit_yoy(codes: list[str], start: str, end: str) -> pd.DataFrame:
    return yjbb_wide("netprofit_yoy", codes, start, end)


# ============================================================ 估值/股息类
def neg_pe_ttm(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """负市盈率。亏损股 PE<0 无估值含义，置为 NaN 交给下游处理。"""
    pe = valuation_wide("pe_ttm", codes, start, end)
    return -pe.where(pe > 0)


def neg_pb(codes: list[str], start: str, end: str) -> pd.DataFrame:
    pb = valuation_wide("pb", codes, start, end)
    return -pb.where(pb > 0)


def dividend_yield(
    codes: list[str],
    start: str,
    end: str,
    close: pd.DataFrame,
) -> pd.DataFrame:
    return dividend_yield_wide(codes, start, end, close)


def rolling_percentile(df: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    """每列在自身过去 window 个交易日中的分位（0~1）。"""
    return df.rolling(window, min_periods=window // 2).apply(
        lambda w: (w <= w[-1]).mean(), raw=True
    )


def neg_pe_ttm_percentile(codes: list[str], start: str, end: str, window: int = 500) -> pd.DataFrame:
    """负 PE 自身历史分位：当前估值在自身过去 window 日中越低越好。"""
    pe = valuation_wide("pe_ttm", codes, start, end)
    return -rolling_percentile(pe.where(pe > 0), window)


def neg_pb_percentile(codes: list[str], start: str, end: str, window: int = 500) -> pd.DataFrame:
    pb = valuation_wide("pb", codes, start, end)
    return -rolling_percentile(pb.where(pb > 0), window)


def dividend_continuity(codes: list[str], start: str, end: str, years: int = 5) -> pd.DataFrame:
    """股息连续性：过去 years 年内有现金分红（dps>0）的记录次数（时点可见）。

    同一财政年度多次分红（中报+年报）会重复计数，作为"分红频率"代理。
    """
    cal = trade_calendar(start, end)
    frames = []
    for p in sorted((fd.FUND_DIR / "fhps").glob("fhps_*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(0.0, index=cal, columns=codes, dtype="float32")
    fh = pd.concat(frames, ignore_index=True)
    fh = fh[fh["qlib_code"].isin(codes) & (pd.to_numeric(fh["dps"], errors="coerce") > 0)
           & fh["announce_date"].notna()]

    out = pd.DataFrame(0.0, index=cal, columns=codes, dtype="float32")
    for code, g in fh.groupby("qlib_code"):
        if code not in out.columns:
            continue
        s = pd.Series(0.0, index=cal)
        for ann in g["announce_date"]:
            window = (cal >= ann) & (cal <= ann + pd.Timedelta(days=365 * years))
            s[window] += 1.0
        out[code] = s
    return out
