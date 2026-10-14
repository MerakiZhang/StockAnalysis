"""补充基本面特征：TTM 计算、复权口径（送转前后连续）、时点可见性、分红滚动窗口。"""
import numpy as np
import pandas as pd
import pytest

from src.factors import supplementary as S


def yjbb_rows(code, rows):
    """rows: [(period, eps_cum, bps, netprofit_cum)]"""
    from src.data.fundamental import available_date
    return pd.DataFrame([{"qlib_code": code, "period": p, "available_date": available_date(p), "eps": e,
                          "bps": b, "netprofit": n, "roe": 10.0, "gross_margin": 30.0, "revenue_yoy": 5.0,
                          "netprofit_yoy": 6.0, "name": code} for p, e, b, n in rows])


def test_ttm_sums_last_four_quarters_and_requires_consecutive():
    df = yjbb_rows("A", [("20200331", 1, 0, 0), ("20200630", 2, 0, 0), ("20200930", 3, 0, 0),
                         ("20201231", 4, 0, 0), ("20210331", 2, 0, 0), ("20210930", 6, 0, 0)])
    t = S._ttm(df, "eps").set_index("period")["eps_ttm"]
    assert t["20201231"] == 4                       # 1+1+1+1
    assert t["20210331"] == 5                       # 2(Q1'21)+1+1+1
    assert "20210930" not in t.index                # 缺 2021Q2，不能拼出连续 4 季


def test_ep_continuous_through_split_and_visible_after_disclosure(monkeypatch):
    cal = pd.bdate_range("2020-01-01", "2021-12-31")
    code = "SH600000"
    # 2021-06-01 起 10 送 10：真实价减半，复权因子翻倍，复权收盘价不变
    factor = pd.DataFrame(1.0, index=cal, columns=[code])
    factor.loc["2021-06-01":, code] = 2.0
    close = pd.DataFrame(20.0, index=cal, columns=[code])
    yj = yjbb_rows(code, [("20200331", 0.25, 5, 25), ("20200630", 0.5, 5, 50), ("20200930", 0.75, 5, 75),
                          ("20201231", 1.0, 5, 100), ("20210331", 0.25, 5, 25),
                          ("20210630", 0.25, 2.5, 50)])   # 送转后股本翻倍，每股值减半
    xj = yj.rename(columns={"netprofit": "ocf"})[["qlib_code", "period", "available_date", "ocf"]]
    monkeypatch.setattr(S.fd, "load_yjbb_all", lambda start_year=2012: yj)
    monkeypatch.setattr(S.fd, "load_xjll_all", lambda start_year=2012: xj)
    monkeypatch.setattr(S.fd, "FUND_DIR", S.fd.FUND_DIR.parent / "__nonexistent__")
    f = S.build_supplementary([code], "2020-01-01", "2021-12-31", close, factor)
    ep = f["ep_ttm"][code]
    assert np.isnan(ep.loc["2021-04-29"])                     # 2020 年报 4/30 才可见
    assert ep.loc["2021-05-06"] == pytest.approx(1.0 / 20)    # TTM EPS 1.0 / 价 20
    assert ep.loc["2021-06-15"] == pytest.approx(1.0 / 20)    # 送转后仍连续（未披露新报告）
    assert ep.loc["2021-09-01"] == pytest.approx(1.0 / 20)    # 半年报：每股 0.25 × 因子 2 = 0.5 → 单季口径一致
    assert f["bp"][code].loc["2021-09-01"] == pytest.approx(5.0 / 20)
    assert f["ocf_to_netprofit"][code].loc["2021-05-06"] == pytest.approx(1.0)
    assert (f["dividend_yield"][code] == 0).all()


def test_event_rolling_window():
    cal = pd.bdate_range("2020-01-01", "2021-06-30")
    ev = pd.DataFrame({"qlib_code": ["A", "A"], "date": pd.to_datetime(["2020-01-04", "2020-07-01"]),
                       "v": [1.0, 2.0]})
    w = S._event_rolling(ev, "v", cal, ["A", "B"], "365D")
    assert w.loc["2020-01-06", "A"] == 1                     # 周六公告 → 下周一生效
    assert w.loc["2020-07-01", "A"] == 3
    assert w.loc["2021-01-06", "A"] == 2                     # 一年后第一笔滚出
    assert (w["B"] == 0).all()
