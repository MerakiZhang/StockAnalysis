"""数据层：缓存完整性判定、实际披露日可见性、最新报告期不倒退、代码 / 时间转换。"""
import os

import numpy as np
import pandas as pd
import pytest

from stockanalysis.data import events as E
from stockanalysis.data import fundamental as fd


def _touch(path, when: pd.Timestamp):
    path.write_bytes(b"x")
    ts = when.to_pydatetime().timestamp()   # naive datetime 按本地时间，与 fromtimestamp 一致
    os.utime(path, (ts, ts))


def test_cache_valid_complete_vs_stale(tmp_path):
    p = tmp_path / "yjbb_20260630.parquet"
    assert not fd.cache_valid(p, "20260630")
    # 截止日 8/31 + 7 天之后抓取：完整，永远有效
    _touch(p, pd.Timestamp("2026-09-10"))
    assert fd.cache_valid(p, "20260630", now=pd.Timestamp("2027-01-01"))
    # 截止日之前抓取：1 天内有效，之后需重抓
    _touch(p, pd.Timestamp("2026-08-20 10:00"))
    assert fd.cache_valid(p, "20260630", now=pd.Timestamp("2026-08-20 20:00"))
    assert not fd.cache_valid(p, "20260630", now=pd.Timestamp("2026-08-22"))


def test_events_unit_valid(tmp_path):
    p = tmp_path / "20181228.parquet"
    _touch(p, pd.Timestamp("2018-12-28 18:00"))
    assert not E._valid(p, pd.Timestamp("2018-12-28"), now=pd.Timestamp("2018-12-30"))
    _touch(p, pd.Timestamp("2019-01-05"))
    assert E._valid(p, pd.Timestamp("2018-12-28"), now=pd.Timestamp("2026-01-01"))


def test_apply_disclosure_uses_next_day_of_actual_date():
    df = pd.DataFrame({"qlib_code": ["A", "B"], "period": ["20231231", "20231231"],
                       "available_date": [fd.available_date("20231231")] * 2})
    dates = pd.DataFrame({"qlib_code": ["A"], "period": ["20231231"], "actual_date": [pd.Timestamp("2024-03-20")]})
    out = fd.apply_disclosure(df, dates).set_index("qlib_code")
    assert out.loc["A", "available_date"] == pd.Timestamp("2024-03-21")   # 实际披露日次日
    assert out.loc["B", "available_date"] == pd.Timestamp("2024-04-30")   # 无记录：披露截止日


def test_latest_visible_never_goes_back_to_older_period():
    from stockanalysis.features.fundamental import _latest_visible

    cal = pd.bdate_range("2024-04-01", "2024-05-10")
    # 一季报 4/16 先披露，年报 4/26 才披露：4/26 之后仍应取一季报
    df = pd.DataFrame({"qlib_code": ["A", "A"], "period": ["20240331", "20231231"],
                       "available_date": [pd.Timestamp("2024-04-16"), pd.Timestamp("2024-04-26")],
                       "roe": [3.0, 12.0]})
    w = _latest_visible(df, "roe", cal, ["A"])["A"]
    assert np.isnan(w.loc["2024-04-15"])
    assert w.loc["2024-04-16"] == 3.0
    assert w.loc["2024-05-06"] == 3.0


def test_code_and_time_conversions():
    from stockanalysis.data.baostock_data import to_bs_code
    from stockanalysis.data.longbridge_cli import _to_cn_date, to_lb_symbol

    assert to_bs_code("SH600519") == "sh.600519"
    assert to_lb_symbol("SZ000001") == "000001.SZ"
    assert _to_cn_date("2018-01-01T16:00:00Z") == pd.Timestamp("2018-01-02")
    assert E._codes(pd.Series([1, "600519"])).tolist() == ["SZ000001", "SH600519"]


def test_baostock_clean_daily_numeric():
    from stockanalysis.data.baostock_data import DAILY_FIELDS, _clean_daily

    raw = pd.DataFrame([["2018-01-02", "11.5", "1e8", "0.5", "1", "8.1", "1.2", "", "3.3", "0"]],
                       columns=list(DAILY_FIELDS))
    df = _clean_daily(raw)
    assert df["date"].iloc[0] == pd.Timestamp("2018-01-02")
    assert df["close_raw"].iloc[0] == pytest.approx(11.5)
    assert np.isnan(df["ps_ttm"].iloc[0]) and df["is_st"].iloc[0] == 0


def test_market_stocks_includes_delisted_and_filters(monkeypatch):
    from stockanalysis.data import universe as U

    spans = {"SH600000": (pd.Timestamp("2005-01-04"), pd.Timestamp("2026-10-15")),
             "SH600087": (pd.Timestamp("2005-01-04"), pd.Timestamp("2019-06-01")),   # 区间内退市：保留
             "SZ000002": (pd.Timestamp("2005-01-04"), pd.Timestamp("2016-01-01")),   # 区间前退市：剔除
             "SH000300": (pd.Timestamp("2005-01-04"), pd.Timestamp("2026-10-15")),   # 指数：剔除
             "BJ920001": (pd.Timestamp("2022-01-04"), pd.Timestamp("2026-10-15"))}
    monkeypatch.setattr(U, "listing_spans", lambda: spans)
    assert U.market_stocks("2018-01-01", "2026-10-15") == ["BJ920001", "SH600000", "SH600087"]
    assert U.market_stocks("2018-01-01", "2026-10-15", exchanges=("SH", "SZ")) == ["SH600000", "SH600087"]


def test_save_snapshot_keeps_only_latest(tmp_path):
    from stockanalysis.data import cache as C

    (tmp_path / "20200101.parquet").write_bytes(b"old")
    df = C.save_snapshot(pd.DataFrame({"a": [1]}), tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == [f"{C.today_tag()}.parquet"]
    assert C.latest_snapshot(tmp_path).name == files[0] and len(df) == 1


def test_margin_one_exchange_empty_not_saved(tmp_path, monkeypatch):
    import akshare as ak

    monkeypatch.setattr(E, "AKSHARE_DIR", tmp_path)
    sh = pd.DataFrame({"标的证券代码": ["600000"], "融资余额": [1.0], "融资买入额": [2.0]})
    monkeypatch.setattr(ak, "stock_margin_detail_sse", lambda date: sh)
    monkeypatch.setattr(ak, "stock_margin_detail_szse", lambda date: pd.DataFrame())
    with pytest.raises(E.IncompleteError):
        E.fetch_margin(pd.Timestamp("2026-10-16"))
    assert not list(tmp_path.rglob("*.parquet"))


def test_freshness_flags_lagging_and_margin_prev_day(monkeypatch):
    import stockanalysis.data.market as M
    from stockanalysis.data import freshness as Fr

    cal = pd.bdate_range("2026-10-12", "2026-10-16")
    monkeypatch.setattr(M, "trade_calendar", lambda start=None, end=None: cal[cal <= pd.Timestamp(end)] if end else cal)
    t = pd.Timestamp("2026-10-16")
    last = {"qlib 行情": t, "baostock 逐日（历史 ST）": pd.Timestamp("2026-10-14"),
            "两融": pd.Timestamp("2026-10-15"), "公告": pd.Timestamp("2026-10-17"), "回购快照": None}
    tab = Fr.freshness(t, last).set_index("数据")["状态"]
    assert tab["qlib 行情"] == Fr.OK and tab["公告"] == Fr.OK
    assert tab["两融"] == Fr.OK                         # 两融到 T-1 即正常
    assert tab["baostock 逐日（历史 ST）"] == "⚠ 滞后 2 个交易日"
    assert tab["回购快照"] == "⚠ 无数据"
    assert list(Fr.lagging(Fr.freshness(t, last))["数据"]) == ["baostock 逐日（历史 ST）", "回购快照"]
