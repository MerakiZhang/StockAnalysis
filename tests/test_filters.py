"""过滤规则面板、在市时点、exit / entry 合成。全部用合成小面板，不依赖 qlib 数据。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stockanalysis.selection import filters as F
from stockanalysis.selection.signal import combine_masks
from stockanalysis.data.universe import membership_panel

DATES = pd.bdate_range("2020-08-17", periods=10)
CODES = ["SH600000", "SZ300001", "SH688001", "SZ000001"]


def make_ctx(**over):
    close = pd.DataFrame(10.0, index=DATES, columns=CODES)
    ctx = {
        "close": close,
        "volume": pd.DataFrame(1e6, index=DATES, columns=CODES),
        "amount": pd.DataFrame(1e5, index=DATES, columns=CODES),  # 千元 → 1 亿元
        "membership": pd.DataFrame(True, index=DATES, columns=CODES),
    }
    ctx.update(over)
    return ctx


def test_membership_panel_respects_join_and_leave():
    spans = {"SH600000": [(DATES[3], DATES[6])],
             "SZ000001": [(DATES[0], DATES[1]), (DATES[8], pd.Timestamp("2099-12-31"))]}
    m = membership_panel(spans, DATES, ["SH600000", "SZ000001", "SH601000"])
    assert m["SH600000"].tolist() == [False] * 3 + [True] * 4 + [False] * 3
    assert m["SZ000001"].tolist() == [True, True] + [False] * 6 + [True, True]
    assert not m["SH601000"].any()


def test_in_universe_uses_membership():
    mem = pd.DataFrame(True, index=DATES, columns=CODES)
    mem.loc[DATES[:5], "SH600000"] = False
    ok = F.in_universe(make_ctx(membership=mem))
    assert not ok.loc[DATES[:5], "SH600000"].any()
    assert ok.loc[DATES[5:], "SH600000"].all()


def test_boards():
    ok = F.boards(make_ctx(), boards=["main"])
    assert ok["SH600000"].all() and ok["SZ000001"].all()
    assert not ok["SZ300001"].any() and not ok["SH688001"].any()


def test_min_list_days(monkeypatch):
    monkeypatch.setattr(F, "listing_dates", lambda: {"SH600000": DATES[0] - pd.Timedelta(days=3)})
    ok = F.min_list_days(make_ctx(), days=5)
    # 第 1 天上市 3 天，DATES[0]=周一 → 周三起满 5 天
    assert ok["SH600000"].tolist()[:3] == [False, False, True]
    assert ok["SZ000001"].all()  # 无上市日记录视为老股


def test_exclude_suspend_window():
    vol = pd.DataFrame(1e6, index=DATES, columns=CODES)
    vol.loc[DATES[2], "SH600000"] = np.nan
    vol.loc[DATES[4], "SZ000001"] = 0
    ok = F.exclude_suspend(make_ctx(volume=vol), days=3)
    assert ok["SH600000"].tolist() == [True, True, False, False, False, True, True, True, True, True]
    assert ok["SZ000001"].tolist()[4:7] == [False, False, False]


def test_exclude_limit_board_and_cyb_switch():
    close = pd.DataFrame(10.0, index=DATES, columns=CODES)
    # 2020-08-21（创业板改 20% 之前）创业板涨 10%：涨停
    close.loc[DATES[4]:, "SZ300001"] = 11.0
    # 2020-08-25（改 20% 之后）创业板涨 10%：未涨停
    close.loc[DATES[6]:, "SZ300001"] = 12.1
    # 主板涨 9.8%：涨停（容差 0.5%）
    close.loc[DATES[5]:, "SH600000"] = 10.98
    # 科创板涨 15%：未涨停
    close.loc[DATES[5]:, "SH688001"] = 11.5
    ok = F.exclude_limit(make_ctx(close=close))
    assert not ok.loc[DATES[4], "SZ300001"]
    assert ok.loc[DATES[6], "SZ300001"]
    assert not ok.loc[DATES[5], "SH600000"]
    assert ok.loc[DATES[5], "SH688001"]
    assert ok.loc[DATES[0]].all()


def test_min_amount_rolling_mean():
    amt = pd.DataFrame(1e5, index=DATES, columns=CODES)
    amt.loc[DATES[3], "SH600000"] = 0  # 单日缩量
    ok = F.min_amount(make_ctx(amount=amt), amount=5e7, window=5)
    assert ok["SH600000"].all()  # 5 日均值 8000 万 ≥ 5000 万，不抖动
    ok2 = F.min_amount(make_ctx(amount=amt), amount=9e7, window=5)
    assert not ok2.loc[DATES[3], "SH600000"]


def test_all_filters_have_scope():
    assert {F.scope_of(n) for n in F.FILTERS.names()} <= {"entry", "exit"}
    assert F.scope_of("in_universe") == "exit"
    assert F.scope_of("boards") == "entry"


def test_combine_masks_splits_by_scope():
    mem = pd.DataFrame(True, index=DATES, columns=CODES)
    mem.loc[:, "SZ000001"] = False
    pool = SimpleNamespace(filters={"in_universe": {}, "boards": {"boards": ["main"]}})
    exit_ok, entry_ok = combine_masks(pool, make_ctx(membership=mem))
    assert not exit_ok["SZ000001"].any() and exit_ok["SZ300001"].all()
    assert not entry_ok["SZ300001"].any() and entry_ok["SZ000001"].all()

