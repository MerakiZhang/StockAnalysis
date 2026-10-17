"""数据截止日：各数据源的最后日期对比信号日 T，滞后的标 ⚠（选股简报与 `update-data --status` 共用）。

应到日（数据最晚应覆盖到哪天）：
    qlib 行情 / baostock 逐日 / 公告 / 龙虎榜 / 市场温度   T
    两融                                                   T 的前一交易日（交易所 T+1 才公布 T 日数据，T+1 才使用）
    快照（增减持 / 回购 / 行业）                           抓取日 ≥ T
滞后不中断选股：baostock 历史 ST 沿用前一日、事件按已有数据计算，简报里标出即可。
"""
from __future__ import annotations

import pandas as pd

from ..paths import AKSHARE_DIR, BAOSTOCK_DIR
from .cache import latest_snapshot

OK = "正常"


def _stem_date(path) -> pd.Timestamp | None:
    return pd.Timestamp(path.stem) if path is not None else None


def _last_unit(dataset: str) -> pd.Timestamp | None:
    files = sorted((AKSHARE_DIR / dataset).glob("*.parquet"))
    return _stem_date(files[-1]) if files else None


def _max_date_in_last(folder, col: str = "date") -> pd.Timestamp | None:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return None
    d = pd.read_parquet(files[-1], columns=[col])[col]
    return pd.Timestamp(d.max()) if len(d) else None


def _baostock_last() -> pd.Timestamp | None:
    from .baostock_data import is_st_panel

    p = is_st_panel()
    return pd.Timestamp(p.index.max()) if len(p) else None


def last_dates() -> dict[str, pd.Timestamp | None]:
    from .longbridge_cli import TEMP_DIR
    from .market import latest_trade_date

    getters = {
        "qlib 行情": latest_trade_date,
        "baostock 逐日（历史 ST）": _baostock_last,
        "公告": lambda: _last_unit("notice"),
        "两融": lambda: _last_unit("margin"),
        "龙虎榜": lambda: _max_date_in_last(AKSHARE_DIR / "lhb"),
        "市场温度": lambda: _max_date_in_last(TEMP_DIR),
        "增减持快照": lambda: _stem_date(latest_snapshot(AKSHARE_DIR / "ggcg")),
        "回购快照": lambda: _stem_date(latest_snapshot(AKSHARE_DIR / "repurchase")),
        "行业快照": lambda: _stem_date(latest_snapshot(BAOSTOCK_DIR / "industry")),
    }
    out = {}
    for name, fn in getters.items():
        try:
            out[name] = fn()
        except Exception:  # noqa: BLE001 - 读不到即视为缺失
            out[name] = None
    return out


def freshness(t: pd.Timestamp | str | None = None, last: dict | None = None) -> pd.DataFrame:
    """各数据源：截止日、应到日、状态（正常 / ⚠ 滞后 N 个交易日 / ⚠ 无数据）。"""
    from .market import latest_trade_date, trade_calendar

    t = pd.Timestamp(t) if t is not None else latest_trade_date()
    cal = trade_calendar(end=str(t.date()))
    prev = cal[-2] if len(cal) >= 2 else t
    last = last if last is not None else last_dates()
    rows = []
    for name, d in last.items():
        need = prev if name == "两融" else t
        if d is None:
            state = "⚠ 无数据"
        elif d >= need:
            state = OK
        else:
            lag = int(((cal > d) & (cal <= need)).sum())
            state = f"⚠ 滞后 {lag} 个交易日" if lag else f"⚠ 早于 {need.date()}"
        rows.append({"数据": name, "截止": d.date() if d is not None else "", "应到": need.date(), "状态": state})
    return pd.DataFrame(rows)


def lagging(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["状态"] != OK]
