"""akshare 基本面数据抓取与本地缓存。

数据落盘布局（data/fundamental/）：
    yjbb/yjbb_<period>.parquet        业绩报表（全市场批量，按报告期）
    fhps/fhps_<period>.parquet        分红送配（全市场批量，按报告期）
    spot/spot_<yyyymmdd>.parquet      当日快照（名称/ST 状态）

防未来函数：`最新公告日期`列是公司最新公告的日期（不是该报告的披露日），
不能使用；按交易所披露截止规则推断可见日：
年报→次年4/30，一季报→当年4/30，半年报→当年8/31，三季报→当年10/31。
"""
from __future__ import annotations

import time

import pandas as pd

from ..paths import FUND_DIR

# 报告期 -> 披露截止日（该报告数据从这天起才可见）
DISCLOSURE_LAG = {"0331": ("0", "0430"), "0630": ("0", "0831"), "0930": ("0", "1031"), "1231": ("1", "0430")}


def available_date(period: str) -> pd.Timestamp:
    """报告期数据对外的可见日期（披露截止日）。"""
    y, md = period[:4], period[4:]
    year_shift, mmdd = DISCLOSURE_LAG[md]
    return pd.Timestamp(f"{int(y) + int(year_shift)}-{mmdd[:2]}-{mmdd[2:]}")


def normalize_code(code: str) -> str:
    """6 位数字代码 -> qlib 风格（SH600519/SZ000001/BJ920296）。"""
    c = code[-6:]
    if c.startswith(("60", "68", "11", "5")):
        return "SH" + c
    if c.startswith(("00", "30", "12", "15", "16", "18")):
        return "SZ" + c
    return "BJ" + c

YJBB_COLMAP = {
    "股票代码": "code",
    "股票简称": "name",
    "每股收益": "eps",
    "营业总收入-营业总收入": "revenue",
    "营业总收入-同比增长": "revenue_yoy",
    "净利润-净利润": "netprofit",
    "净利润-同比增长": "netprofit_yoy",
    "每股净资产": "bps",
    "净资产收益率": "roe",
    "销售毛利率": "gross_margin",
    "所处行业": "industry",
    "最新公告日期": "announce_date",
}

FHPS_COLMAP = {
    "代码": "code",
    "名称": "name",
    "现金分红-现金分红比例": "dps10",  # 每10股派现（税前），÷10 得每股股利
    "预案公告日": "announce_date",
}

XJLL_COLMAP = {
    "股票代码": "code",
    "股票简称": "name",
    "经营性现金流-现金流量净额": "ocf",
    "投资性现金流-现金流量净额": "icf",
    "融资性现金流-现金流量净额": "fcf",
    "净现金流-净现金流": "net_cf",
    "公告日期": "announce_date",
}



def _retry(fn, *args, retries: int = 3, sleep: float = 1.5, **kwargs):
    last_err: Exception | None = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - akshare 接口异常类型不统一
            last_err = e
            time.sleep(sleep * (i + 1))
    raise last_err


def report_periods(start_year: int = 2017, end_year: int | None = None) -> list[str]:
    """报告期列表：YYYY0331/0630/0930/1231。多取一年用于同比。"""
    end_year = end_year or pd.Timestamp.now().year
    return [
        f"{y}{md}"
        for y in range(start_year, end_year + 1)
        for md in ("0331", "0630", "0930", "1231")
    ]


def fetch_yjbb(period: str, use_cache: bool = True) -> pd.DataFrame:
    """业绩报表（全市场）。period 如 '20241231'。"""
    import akshare as ak

    path = FUND_DIR / "yjbb" / f"yjbb_{period}.parquet"
    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        if set(YJBB_COLMAP.values()) <= set(cached.columns):
            return cached
        # 旧缓存缺新列（如 bps）时重新抓取
    df = _retry(ak.stock_yjbb_em, date=period)
    df = df.rename(columns=YJBB_COLMAP)[list(YJBB_COLMAP.values())]
    df["qlib_code"] = df["code"].map(normalize_code)
    df["available_date"] = available_date(period)
    df["period"] = period
    # 同一报告期同一名称可能因北交所代码切换出现重复行，保留第一条
    df = df.drop_duplicates(subset=["period", "name"], keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def fetch_fhps(period: str, use_cache: bool = True) -> pd.DataFrame:
    """分红送配（全市场）。period 如 '20241231'。"""
    import akshare as ak

    path = FUND_DIR / "fhps" / f"fhps_{period}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    df = _retry(ak.stock_fhps_em, date=period)
    missing = [c for c in FHPS_COLMAP if c not in df.columns]
    if missing:
        raise KeyError(f"fhps {period} 缺列 {missing}，实际列 {list(df.columns)}")
    df = df.rename(columns=FHPS_COLMAP)[list(FHPS_COLMAP.values())]
    df["dps"] = pd.to_numeric(df["dps10"], errors="coerce") / 10  # 每股股利
    df["qlib_code"] = df["code"].map(normalize_code)
    # fhps 的预案公告日是真实披露日，可直接用于时点对齐
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    df["period"] = period
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def fetch_xjll(period: str, use_cache: bool = True) -> pd.DataFrame:
    """现金流量表（全市场）。period 如 '20241231'。

    时点对齐与 yjbb 一致用固定披露截止日（announce_date 仅供参考）：
    该接口的"公告日期"是真实披露日，但存在个别异常值，统一用截止规则更稳。
    """
    import akshare as ak

    path = FUND_DIR / "xjll" / f"xjll_{period}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    df = _retry(ak.stock_xjll_em, date=period)
    missing = [c for c in XJLL_COLMAP if c not in df.columns]
    if missing:
        raise KeyError(f"xjll {period} 缺列 {missing}，实际列 {list(df.columns)}")
    df = df.rename(columns=XJLL_COLMAP)[list(XJLL_COLMAP.values())]
    for col in ("ocf", "icf", "fcf", "net_cf"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["qlib_code"] = df["code"].map(normalize_code)
    df["available_date"] = available_date(period)
    df["period"] = period
    df = df.drop_duplicates(subset=["period", "qlib_code"], keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_xjll_all(start_year: int = 2012) -> pd.DataFrame:
    """加载所有已缓存报告期的现金流量表，合并为一张长表。"""
    frames = []
    for p in sorted((FUND_DIR / "xjll").glob("xjll_*.parquet")):
        period = p.stem.split("_")[1]
        if int(period[:4]) >= start_year:
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("无 xjll 缓存，请先运行 python -X utf8 -m stockanalysis update-data")
    return pd.concat(frames, ignore_index=True)


def fetch_spot(use_cache: bool = True) -> pd.DataFrame:
    """A股当日快照：code/name/is_st。当日缓存。"""
    import akshare as ak

    today = pd.Timestamp.now().strftime("%Y%m%d")
    path = FUND_DIR / "spot" / f"spot_{today}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    df = _retry(ak.stock_zh_a_spot_em)
    out = pd.DataFrame(
        {
            "code": df["代码"].str[-6:],
            "name": df["名称"],
        }
    )
    out["is_st"] = out["name"].str.contains("ST", case=False, na=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def load_yjbb_all(start_year: int = 2012) -> pd.DataFrame:
    """加载所有已缓存报告期的业绩报表，合并为一张长表。"""
    frames = []
    for p in sorted((FUND_DIR / "yjbb").glob("yjbb_*.parquet")):
        period = p.stem.split("_")[1]
        if int(period[:4]) >= start_year:
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("无 yjbb 缓存，请先运行 python -X utf8 -m stockanalysis update-data")
    return pd.concat(frames, ignore_index=True)


def update_all(start_year: int = 2012, log=print) -> None:
    """抓取 / 补齐 start_year 起各报告期的业绩报表、分红送配、现金流量表缓存（已缓存的跳过）。"""
    periods = report_periods(start_year)
    for label, fetch in (("业绩报表 yjbb", fetch_yjbb), ("分红送配 fhps", fetch_fhps), ("现金流量表 xjll", fetch_xjll)):
        log(f"[{label}] {len(periods)} 个报告期")
        for p in periods:
            try:
                log(f"  {p}: {len(fetch(p))} rows")
            except Exception as e:  # noqa: BLE001 - 未来报告期 / 无数据季度会失败
                log(f"  {p} FAILED: {e}")
