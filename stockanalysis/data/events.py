"""akshare 事件数据抓取与本地缓存：两融、龙虎榜、股东户数、限售解禁、股东增减持、回购、公告。

缓存布局（data/raw/akshare/<数据集>/<单元>.parquet），单元为日 / 月 / 年 / 报告期 / 快照（抓取日，只留最新一份）：

    margin/<yyyymmdd>      两融明细（沪 + 深，按交易日）       qlib_code, date, fin_balance, fin_buy
    lhb/<yyyymm>           龙虎榜明细（按月）                  qlib_code, date, net_buy, buy, sell, deal_amount, market_amount, reason
    lhb_jg/<yyyymm>        龙虎榜机构买卖统计（按月）          qlib_code, date, jg_buy_n, jg_sell_n, jg_net_buy, market_amount
    gdhs/<period>          股东户数（按报告期）                qlib_code, end_date, holders, holders_prev, change_pct, announce_date
    release/<yyyy>         限售解禁（按年，含未来日程）        qlib_code, date, shares, value, float_ratio, kind
    ggcg/<yyyymmdd>        股东增减持（全量快照）              qlib_code, holder, direction, float_ratio, start, end, announce_date
    repurchase/<yyyymmdd>  回购（全量快照）                    qlib_code, amount_low, amount_high, start, progress, announce_date
    notice/<yyyymmdd>      全部公告（按自然日）                qlib_code, date, title, kind

时点（防未来）：每个数据集给出 `visible`（该记录从哪天起可用于 T 日收盘信号）：
    两融：交易所 T+1 早上公布 T 日数据 → visible = 下一交易日
    龙虎榜：T 日盘后公布 → visible = T（与 T 日收盘价同时可得）
    股东户数 / 增减持 / 回购 / 公告：visible = 公告日次日（多为盘后发布）
    限售解禁：日程在上市 / 定增时即已公开 → 按解禁日前视使用（只用于"未来 N 日将解禁"）

缓存完整性：日 / 月 / 年单元结束 3 天后抓取的视为完整；之前抓取的超过 1 天即重抓。
快照类每天至多抓一次。两融任一交易所当日为空（深交所次日才公布）时不落盘，记为失败单元，下次重抓。
"""
from __future__ import annotations

import time

import pandas as pd

from ..paths import AKSHARE_DIR
from .cache import latest_snapshot, save_snapshot, snapshot_path
from .fundamental import _retry, normalize_code, report_periods
from .universe import is_stock

COMPLETE_AFTER_DAYS = 3
STALE_AFTER_HOURS = 24
EVENTS_START = "2017-01-01"


def _path(dataset: str, unit: str):
    return AKSHARE_DIR / dataset / f"{unit}.parquet"


def _valid(path, unit_end: pd.Timestamp, now: pd.Timestamp | None = None) -> bool:
    if not path.exists():
        return False
    now = now or pd.Timestamp.now()
    mtime = pd.Timestamp.fromtimestamp(path.stat().st_mtime)
    if mtime >= unit_end + pd.Timedelta(days=COMPLETE_AFTER_DAYS):
        return True
    return now - mtime < pd.Timedelta(hours=STALE_AFTER_HOURS)


def _save(df: pd.DataFrame, path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _codes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(6).map(normalize_code)


def _stocks_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["qlib_code"].map(is_stock)].reset_index(drop=True)


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class IncompleteError(RuntimeError):
    """数据源当日数据尚未完整公布：不落盘，下次重抓。"""


# ------------------------------------------------------------------ 两融（按交易日）
def fetch_margin(day: pd.Timestamp, use_cache: bool = True) -> pd.DataFrame:
    import akshare as ak

    d = pd.Timestamp(day).strftime("%Y%m%d")
    path = _path("margin", d)
    if use_cache and _valid(path, pd.Timestamp(day)):
        return pd.read_parquet(path)
    sh = _retry(ak.stock_margin_detail_sse, date=d)
    sz = _retry(ak.stock_margin_detail_szse, date=d)
    if not len(sh) or not len(sz):
        raise IncompleteError(f"两融 {d} 不完整（沪 {len(sh)} 行 / 深 {len(sz)} 行，可能尚未公布）")
    parts = []
    if len(sh):
        parts.append(pd.DataFrame({"qlib_code": _codes(sh["标的证券代码"]),
                                   "fin_balance": sh["融资余额"], "fin_buy": sh["融资买入额"]}))
    if len(sz):
        parts.append(pd.DataFrame({"qlib_code": _codes(sz["证券代码"]),
                                   "fin_balance": sz["融资余额"], "fin_buy": sz["融资买入额"]}))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["qlib_code", "fin_balance", "fin_buy"])
    df = _num(df, ["fin_balance", "fin_buy"])
    df["date"] = pd.Timestamp(day)
    return _save(_stocks_only(df), path)


# ------------------------------------------------------------------ 龙虎榜（按月）
def _month_range(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
    return s, s + pd.offsets.MonthEnd(0)


def fetch_lhb(month: str, use_cache: bool = True) -> pd.DataFrame:
    """龙虎榜明细（同一天同一股票可能因多个上榜原因出现多行）。month 如 '201806'。"""
    import akshare as ak

    s, e = _month_range(month)
    path = _path("lhb", month)
    if use_cache and _valid(path, e):
        return pd.read_parquet(path)
    raw = _retry(ak.stock_lhb_detail_em, start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d"))
    cols = ["qlib_code", "date", "net_buy", "buy", "sell", "deal_amount", "market_amount", "reason"]
    if raw is None or not len(raw) or "代码" not in raw.columns:
        return _save(pd.DataFrame(columns=cols), path)
    df = pd.DataFrame({"qlib_code": _codes(raw["代码"]), "date": pd.to_datetime(raw["上榜日"]),
                       "net_buy": raw["龙虎榜净买额"], "buy": raw["龙虎榜买入额"], "sell": raw["龙虎榜卖出额"],
                       "deal_amount": raw["龙虎榜成交额"], "market_amount": raw["市场总成交额"],
                       "reason": raw["上榜原因"]})
    df = _num(df, ["net_buy", "buy", "sell", "deal_amount", "market_amount"])
    return _save(_stocks_only(df), path)


def fetch_lhb_jg(month: str, use_cache: bool = True) -> pd.DataFrame:
    """龙虎榜机构买卖统计。"""
    import akshare as ak

    s, e = _month_range(month)
    path = _path("lhb_jg", month)
    if use_cache and _valid(path, e):
        return pd.read_parquet(path)
    raw = _retry(ak.stock_lhb_jgmmtj_em, start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d"))
    cols = ["qlib_code", "date", "jg_buy_n", "jg_sell_n", "jg_net_buy", "market_amount"]
    if raw is None or not len(raw):
        return _save(pd.DataFrame(columns=cols), path)
    df = pd.DataFrame({"qlib_code": _codes(raw["代码"]), "date": pd.to_datetime(raw["上榜日期"]),
                       "jg_buy_n": raw["买方机构数"], "jg_sell_n": raw["卖方机构数"],
                       "jg_net_buy": raw["机构买入净额"], "market_amount": raw["市场总成交额"]})
    df = _num(df, ["jg_buy_n", "jg_sell_n", "jg_net_buy", "market_amount"])
    return _save(_stocks_only(df), path)


# ------------------------------------------------------------------ 股东户数（按报告期）
def fetch_gdhs(period: str, use_cache: bool = True) -> pd.DataFrame:
    import akshare as ak

    from .fundamental import available_date

    path = _path("gdhs", period)
    if use_cache and _valid(path, available_date(period)):
        return pd.read_parquet(path)
    raw = _retry(ak.stock_zh_a_gdhs, symbol=period)
    df = pd.DataFrame({"qlib_code": _codes(raw["代码"]),
                       "end_date": pd.to_datetime(raw["股东户数统计截止日-本次"], errors="coerce"),
                       "holders": raw["股东户数-本次"], "holders_prev": raw["股东户数-上次"],
                       "change_pct": raw["股东户数-增减比例"],
                       "announce_date": pd.to_datetime(raw["公告日期"], errors="coerce")})
    df = _num(df, ["holders", "holders_prev", "change_pct"])
    df["period"] = period
    return _save(_stocks_only(df), path)


# ------------------------------------------------------------------ 限售解禁（按年）
def fetch_release(year: int, use_cache: bool = True) -> pd.DataFrame:
    import akshare as ak

    path = _path("release", str(year))
    if use_cache and _valid(path, pd.Timestamp(f"{year}-12-31")):
        return pd.read_parquet(path)
    raw = _retry(ak.stock_restricted_release_detail_em, start_date=f"{year}0101", end_date=f"{year}1231")
    cols = ["qlib_code", "date", "shares", "value", "float_ratio", "kind"]
    if raw is None or not len(raw):
        return _save(pd.DataFrame(columns=cols), path)
    df = pd.DataFrame({"qlib_code": _codes(raw["股票代码"]), "date": pd.to_datetime(raw["解禁时间"], errors="coerce"),
                       "shares": raw["实际解禁数量"], "value": raw["实际解禁市值"],
                       "float_ratio": raw["占解禁前流通市值比例"], "kind": raw["限售股类型"]})
    df = _num(df, ["shares", "value", "float_ratio"])
    return _save(_stocks_only(df), path)


# ------------------------------------------------------------------ 快照类（全量历史，每天至多一次，只留最新一份）


def fetch_ggcg(use_cache: bool = True) -> pd.DataFrame:
    """股东增减持（全量历史，约 3~4 分钟）。"""
    import akshare as ak

    path = snapshot_path(AKSHARE_DIR / "ggcg")
    if use_cache and path.exists():
        return pd.read_parquet(path)
    raw = _retry(ak.stock_ggcg_em, symbol="全部")
    df = pd.DataFrame({"qlib_code": _codes(raw["代码"]), "holder": raw["股东名称"],
                       "direction": raw["持股变动信息-增减"], "float_ratio": raw["持股变动信息-占流通股比例"],
                       "total_ratio": raw["持股变动信息-占总股本比例"],
                       "start": pd.to_datetime(raw["变动开始日"], errors="coerce"),
                       "end": pd.to_datetime(raw["变动截止日"], errors="coerce"),
                       "announce_date": pd.to_datetime(raw["公告日"], errors="coerce")})
    df = _num(df, ["float_ratio", "total_ratio"])
    return save_snapshot(_stocks_only(df), path.parent)


def fetch_repurchase(use_cache: bool = True) -> pd.DataFrame:
    """股票回购（全量历史）。回购起始时间 = 方案首次公告日。"""
    import akshare as ak

    path = snapshot_path(AKSHARE_DIR / "repurchase")
    if use_cache and path.exists():
        return pd.read_parquet(path)
    raw = _retry(ak.stock_repurchase_em)
    df = pd.DataFrame({"qlib_code": _codes(raw["股票代码"]),
                       "amount_low": raw["计划回购金额区间-下限"], "amount_high": raw["计划回购金额区间-上限"],
                       "ratio_low": raw["占公告前一日总股本比例-下限"],
                       "start": pd.to_datetime(raw["回购起始时间"], errors="coerce"),
                       "progress": raw["实施进度"],
                       "announce_date": pd.to_datetime(raw["最新公告日期"], errors="coerce")})
    df = _num(df, ["amount_low", "amount_high", "ratio_low"])
    return save_snapshot(_stocks_only(df), path.parent)


def load_snapshot(dataset: str) -> pd.DataFrame:
    path = latest_snapshot(AKSHARE_DIR / dataset)
    if path is None:
        raise FileNotFoundError(f"无 {dataset} 缓存，请先运行 python -X utf8 -m stockanalysis update-data --only events")
    return pd.read_parquet(path)


# ------------------------------------------------------------------ 公告（按自然日）
def fetch_notice(day: pd.Timestamp, use_cache: bool = True) -> pd.DataFrame:
    """当日全部公告（标题 + 类型）。立案调查等风险公告多归在"其他"类，需按标题关键词识别。"""
    import akshare as ak

    d = pd.Timestamp(day).strftime("%Y%m%d")
    path = _path("notice", d)
    if use_cache and _valid(path, pd.Timestamp(day)):
        return pd.read_parquet(path)
    cols = ["qlib_code", "date", "title", "kind"]
    try:
        raw = ak.stock_notice_report(symbol="全部", date=d)
    except KeyError:   # 节假日 akshare 内部取 "代码" 列报 KeyError：当作无公告
        return _save(pd.DataFrame(columns=cols), path)
    except Exception:  # noqa: BLE001 - 其余网络异常走重试
        raw = _retry(ak.stock_notice_report, symbol="全部", date=d)
    if raw is None or not len(raw) or "代码" not in raw.columns:   # 节假日返回无列空表
        return _save(pd.DataFrame(columns=cols), path)
    df = pd.DataFrame({"qlib_code": _codes(raw["代码"]), "date": pd.to_datetime(raw["公告日期"], errors="coerce"),
                       "title": raw["公告标题"], "kind": raw["公告类型"]})
    return _save(_stocks_only(df), path)


# ------------------------------------------------------------------ 读取
def load_units(dataset: str, start: str | None = None) -> pd.DataFrame:
    """合并某数据集全部单元缓存（start 按单元名粗筛：日 / 月 / 年 / 报告期均以年份开头）。"""
    files = sorted((AKSHARE_DIR / dataset).glob("*.parquet"))
    if start:
        y = pd.Timestamp(start).year
        files = [f for f in files if int(f.stem[:4]) >= y]
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise FileNotFoundError(f"无 {dataset} 缓存，请先运行 python -X utf8 -m stockanalysis update-data --only events")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ 批量更新
DATASETS = ("gdhs", "release", "ggcg", "repurchase", "lhb", "margin", "notice")


def _months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")]


def update_events(datasets=DATASETS, start: str = EVENTS_START, end: str | None = None, log=print) -> dict[str, list]:
    """按数据集补齐缓存；返回 {数据集: 失败单元}。单元失败只记录，不中断。"""
    from .market import trade_calendar

    end = end or str(pd.Timestamp.now().date())
    failed: dict[str, list] = {}

    def run(name, units, fetch):
        log(f"[{name}] {len(units)} 个单元")
        bad, t0 = [], time.time()
        for i, u in enumerate(units, 1):
            try:
                fetch(u)
            except IncompleteError as e:
                bad.append(u)
                log(f"  {name} {u} 不完整，下次重抓: {e}")
            except Exception as e:  # noqa: BLE001 - 接口异常类型不统一
                bad.append(u)
                log(f"  {name} {u} FAILED: {str(e)[:120]}")
            if i % 100 == 0:
                log(f"  {name} {i}/{len(units)} {time.time() - t0:.0f}s 失败 {len(bad)}")
        failed[name] = bad

    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    for name in datasets:
        if name == "gdhs":
            run(name, [p for p in report_periods(y0, y1) if pd.Timestamp(p) <= pd.Timestamp(end)], fetch_gdhs)
        elif name == "release":
            run(name, list(range(y0, y1 + 2)), fetch_release)   # 含下一年日程
        elif name == "ggcg":
            run(name, ["snapshot"], lambda _: fetch_ggcg())
        elif name == "repurchase":
            run(name, ["snapshot"], lambda _: fetch_repurchase())
        elif name == "lhb":
            run(name, _months(start, end), lambda m: (fetch_lhb(m), fetch_lhb_jg(m)))
        elif name == "margin":
            run(name, list(trade_calendar(start, end)), fetch_margin)
        elif name == "notice":
            run(name, list(pd.date_range(start, end, freq="D")), fetch_notice)
        else:
            raise ValueError(f"未知数据集 {name}，可选 {DATASETS}")
    return failed
