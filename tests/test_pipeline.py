"""区间切分、检查日日程、真实池配置校验、命令行。"""
import numpy as np
import pandas as pd
import pytest

from stockanalysis.backtest.metrics import segments
from stockanalysis.config import list_pools, load_pool

CAL = pd.bdate_range("2024-01-01", periods=30)


def test_segments_split_by_in_sample_end():
    segs = segments("2021-03-01", "2022-12-31", "2023-06-30")
    tags = [t for t, _, _ in segs]
    assert tags == ["2021", "2022", "2023", "研发区", "保留集", "全区间"]
    assert segs[3][2] == "2022-12-31" and segs[4][1] == "2023-01-01"


def test_segments_without_holdout():
    tags = [t for t, _, _ in segments("2020-01-01", "2022-12-31", "2021-06-30")]
    assert "保留集" not in tags and "研发区" in tags


def test_repo_pools_validate_against_registries():
    for name in list_pools():
        spec = load_pool(name)  # 校验注册名与参数名（含买卖点条件名与参数）
        assert spec.points.max_positions > 0 and spec.points.groups()["buy"]


def test_override_replaces_whole_section():
    spec = load_pool("long", override={"scorer": {"name": "random", "seed": 1}})
    assert spec.scorer == {"name": "random", "seed": 1}


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


def test_bench_returns_ffill_missing_day(monkeypatch):
    import stockanalysis.data.market as M
    from stockanalysis.backtest import points_engine as PE

    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"X": [100.0, np.nan, 110.0, 110.0]}, index=dates)
    monkeypatch.setattr(M, "load_wide", lambda codes, s, e, fields: {"close": close})
    r = PE._bench_returns("X", dates)
    assert r.tolist() == [0.0, 0.0, pytest.approx(0.10), 0.0]   # 缺日记 0，次日拿到完整涨幅
