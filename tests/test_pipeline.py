"""信号折叠、分段、持有天数、选股 select()、真实池配置校验。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stockanalysis.backtest.engine import scores_on_rebalance_dates
from stockanalysis.backtest.metrics import segments
from stockanalysis.config import RebalanceRule, list_pools, load_pool
from stockanalysis.selection.selector import holding_days, select
from stockanalysis.selection.signal import Signal

CAL = pd.bdate_range("2024-01-01", periods=30)


def test_daily_fold_is_identity():
    df = pd.DataFrame(np.arange(6.0).reshape(3, 2), index=CAL[:3], columns=["A", "B"])
    assert scores_on_rebalance_dates(df, "daily") is df


def test_weekly_fold_keeps_last_day_of_prev_week():
    df = pd.DataFrame(1.0, index=CAL[:10], columns=["A"])
    out = scores_on_rebalance_dates(df, "weekly")
    assert out.dropna().index.tolist() == [CAL[0], CAL[4]]  # 首日 + 第一周周五


def test_segments_split_by_in_sample_end():
    segs = segments("2021-03-01", "2022-12-31", "2023-06-30")
    tags = [t for t, _, _ in segs]
    assert tags == ["2021", "2022", "2023", "研发区", "保留集", "全区间"]
    assert segs[3][2] == "2022-12-31" and segs[4][1] == "2023-01-01"


def test_segments_without_holdout():
    tags = [t for t, _, _ in segments("2020-01-01", "2022-12-31", "2021-06-30")]
    assert "保留集" not in tags and "研发区" in tags


def test_holding_days_counts_trading_days():
    prev = pd.DataFrame({"code": ["A", "B"], "entry_date": [CAL[2], CAL[5]]})
    assert holding_days(prev, CAL[7], CAL) == {"A": 5, "B": 2}
    assert holding_days(None, CAL[7], CAL) == {}


def fake_signal(day, scores: dict, entry: dict | None = None, topk=2):
    s = pd.DataFrame([scores], index=[day])
    e = pd.DataFrame([entry or {c: True for c in scores}], index=[day])
    return Signal(score=s, entry=e.astype(bool), topk=pd.Series([topk], index=[day]))


def test_select_keeps_entry_date_and_marks_status():
    pool = SimpleNamespace(rebalance=RebalanceRule(n_drop=1, hold_thresh=1))
    day = CAL[10]
    sig = fake_signal(day, {"A": 3.0, "B": 2.0, "C": 1.0, "D": np.nan})
    prev = pd.DataFrame({"code": ["B", "D"], "entry_date": [CAL[3], CAL[4]]})
    sel = select(pool, sig, day, prev, CAL)
    t = sel.table.set_index("code")
    assert sel.result.sell_exit == ["D"]
    assert list(t.index) == ["A", "B"]
    assert t.loc["B", "status"] == "留存" and t.loc["B", "entry_date"] == CAL[3]
    assert t.loc["A", "status"] == "新进" and t.loc["A", "entry_date"] == day
    assert t.loc["A", "rank"] == 1 and sel.reference == ["A", "B"]


def test_repo_pools_validate_against_registries():
    for name in list_pools():
        spec = load_pool(name)  # 校验注册名与参数名（含买卖点条件名与参数）
        assert spec.strategy["name"] in ("entry_aware_topk_dropout", "signal_points")
        assert (spec.points is not None) == (spec.strategy["name"] == "signal_points")


def test_override_replaces_whole_section():
    spec = load_pool("long", override={"scorer": {"name": "random", "seed": 1}})
    assert spec.scorer == {"name": "random", "seed": 1}


def test_topk_series_uses_sizing():
    from stockanalysis.config import Sizing
    from stockanalysis.selection.sizing import topk_series

    entry = pd.DataFrame({"A": [True] * 3, "B": [True, False, True]}, index=CAL[:3])
    assert topk_series(entry, Sizing(top_pct=0.5, min_topk=1)).tolist() == [1, 1, 1]
    assert topk_series(entry, Sizing(topk=7)).tolist() == [7, 7, 7]


def test_cli_parses_all_commands():
    import pytest

    from stockanalysis.cli import COMMANDS, main

    for cmd in COMMANDS:
        with pytest.raises(SystemExit) as e:
            main([cmd, "--help"])
        assert e.value.code == 0


def test_signal_mask_matches_period_ends():
    from stockanalysis.selection.schedule import signal_mask

    days = CAL[:10]                                  # 两周：周一 ~ 周五
    assert signal_mask(days, "daily").all()
    assert list(np.flatnonzero(signal_mask(days, "weekly"))) == [0, 4]   # 首日 + 第一周周五
    month = pd.bdate_range("2024-01-25", "2024-02-06")
    assert [d.date().isoformat() for d in month[signal_mask(month, "monthly")]] == ["2024-01-25", "2024-01-31"]


def test_weekly_fold_uses_same_schedule():
    from stockanalysis.selection.schedule import signal_mask

    df = pd.DataFrame(1.0, index=CAL[:10], columns=["A"])
    folded = scores_on_rebalance_dates(df, "weekly")
    assert list(folded["A"].notna().to_numpy()) == list(signal_mask(CAL[:10], "weekly"))


def test_select_carries_over_on_non_signal_day():
    pool = SimpleNamespace(rebalance=RebalanceRule(n_drop=1, hold_thresh=1))
    day = CAL[10]
    sig = fake_signal(day, {"A": 3.0, "B": 2.0, "C": 1.0, "D": np.nan})
    prev = pd.DataFrame({"code": ["C", "D"], "entry_date": [CAL[3], CAL[4]]})
    sel = select(pool, sig, day, prev, CAL, rebalance_today=False)
    assert not sel.rebalanced
    assert set(sel.table["code"]) == {"C", "D"} and not sel.result.buy and not sel.result.sell_exit
    assert (sel.table["status"] == "留存").all()
    first = select(pool, sig, day, None, CAL, rebalance_today=False)   # 首次建仓不受日程限制
    assert first.rebalanced and set(first.table["code"]) == {"A", "B"}
