"""烂股识别（features/risk.py）与对应过滤规则：按历史时点判定、缺数据不误杀。"""
import numpy as np
import pandas as pd

from stockanalysis.features import risk as R

DATES = pd.bdate_range("2020-01-01", "2021-12-31")


def make_ctx(codes, close=10.0, factor=1.0):
    frame = lambda v: pd.DataFrame(v, index=DATES, columns=codes, dtype=np.float32)  # noqa: E731
    c = frame(close)
    return {"close": c, "high": c.copy(), "low": c.copy(), "factor": frame(factor), "volume": frame(1e6),
            "codes": list(codes), "dates": DATES}


def yjbb(rows):
    """rows: [(code, period, available_date, eps, bps, netprofit, revenue)]"""
    return pd.DataFrame([{"qlib_code": c, "period": p, "available_date": pd.Timestamp(a), "eps": e, "bps": b,
                          "netprofit": n, "revenue": r} for c, p, a, e, b, n, r in rows])


def test_st_by_price_limit_main_board_only():
    ctx = make_ctx(["SH600001", "SZ300001", "SH600002"])
    for c in ctx["codes"]:
        if c != "SH600002":   # 600002 每天涨 6%：明显不是 ST
            ctx["close"].loc["2020-06-01", c] = ctx["close"].loc["2020-05-29", c] * 1.05   # 收盘涨 5% 且收在最高
    ctx["close"]["SH600002"] = 10.0 * (1.06 ** np.arange(len(DATES)))
    ctx["high"], ctx["low"] = ctx["close"].copy(), ctx["close"].copy()
    st = R.st_by_price_limit(ctx)
    assert st.loc["2020-06-01", "SH600001"] and st.loc["2020-08-20", "SH600001"]
    assert not st.loc["2020-12-31", "SH600001"]            # 超出 60 日窗口
    assert not st["SZ300001"].any()                         # 创业板无法用 5% 区分
    assert not st["SH600002"].any()


def test_consecutive_loss_and_revenue_floor_visible_after_disclosure(monkeypatch):
    rows = [("SH600001", "20191231", "2020-04-28", -0.1, 2.0, -1e7, 2e8),
            ("SH600001", "20201231", "2021-04-20", -0.2, 1.8, -2e7, 2e8),
            ("SH600002", "20191231", "2020-04-28", -0.1, 2.0, -1e7, 5e8),
            ("SH600002", "20201231", "2021-04-20", 0.3, 2.3, 3e7, 5e8)]
    monkeypatch.setattr(R.fd, "load_yjbb_all", lambda start_year=2012: yjbb(rows))
    ctx = make_ctx(["SH600001", "SH600002"])
    loss = R.consecutive_loss(ctx, 2)
    assert not loss.loc["2021-04-19", "SH600001"]          # 第二年年报未披露
    assert loss.loc["2021-04-20", "SH600001"]
    assert not loss["SH600002"].any()                       # 只亏一年
    small = R.small_revenue_loss(ctx, 3e8)
    assert small.loc["2020-05-06", "SH600001"] and not small["SH600002"].any()
    assert not R.negative_equity(ctx).any().any()


def test_market_cap_from_netprofit_over_eps(monkeypatch):
    rows = [("SH600001", "20191231", "2020-04-28", 0.5, 3.0, 5e8, 1e10)]   # 股本 10 亿
    monkeypatch.setattr(R.fd, "load_yjbb_all", lambda start_year=2012: yjbb(rows))
    ctx = make_ctx(["SH600001"], close=20.0, factor=2.0)                  # 不复权价 10 元
    cap = R.market_cap(ctx)["SH600001"]
    assert np.isnan(cap.loc["2020-04-27"])
    assert cap.loc["2020-04-28"] == 1e10


def test_risk_notice_window_and_exclusions(monkeypatch, tmp_path):
    nt = pd.DataFrame({"qlib_code": ["SH600001", "SH600002"],
                       "date": [pd.Timestamp("2020-03-02"), pd.Timestamp("2020-03-02")],
                       "title": ["关于收到中国证监会立案调查通知书的公告", "关于撤销退市风险警示的公告"],
                       "kind": ["其他", "其他"]})
    import stockanalysis.data.events as E
    import stockanalysis.data.panel as P
    (tmp_path / "notice").mkdir()
    nt.to_parquet(tmp_path / "notice" / "20200302.parquet", index=False)
    monkeypatch.setattr(E, "AKSHARE_DIR", tmp_path)
    monkeypatch.setattr(P, "PANEL_DIR", tmp_path / "panel")
    ctx = make_ctx(["SH600001", "SH600002"])
    flag = R.risk_notice(ctx, 365)
    assert not flag.loc["2020-03-02", "SH600001"]          # 公告日当天（盘后发布）不可用
    assert flag.loc["2020-03-03", "SH600001"] and flag.loc["2021-02-26", "SH600001"]
    assert not flag.loc["2021-03-03", "SH600001"]
    assert not flag["SH600002"].any()


def test_release_ahead_looks_forward_only(monkeypatch):
    rl = pd.DataFrame({"qlib_code": ["SH600001", "SH600001"],
                       "date": [pd.Timestamp("2020-06-15"), pd.Timestamp("2022-01-10")],   # 后者在日历之外
                       "float_ratio": [0.12, 0.30], "shares": [1, 1], "value": [1, 1], "kind": ["", ""]})
    import stockanalysis.data.events as E
    monkeypatch.setattr(E, "load_units", lambda dataset, start=None: rl)
    flag = R.release_ahead(make_ctx(["SH600001"]), days=20, ratio=0.05)["SH600001"]
    assert not flag.loc["2020-05-01"]
    assert flag.loc["2020-06-01"] and flag.loc["2020-06-12"]
    assert not flag.loc["2020-06-15"]                       # 解禁日当天及之后不再限制
    assert flag.loc["2021-12-31"]                           # 日历之外的未来解禁也能看到


def test_account_boards_injected_into_boards_filter():
    from stockanalysis.config.loader import _resolve_filters

    out = _resolve_filters("x", {"boards": {}}, ["main", "cyb"])
    assert out["boards"] == {"boards": ["main", "cyb"]}
    out = _resolve_filters("x", {"boards": {"boards": ["main"]}}, ["main", "cyb"])   # 池内显式指定优先
    assert out["boards"] == {"boards": ["main"]}


def test_build_if_stale_rebuilds_on_source_or_extra_change(tmp_path, monkeypatch):
    import os

    import stockanalysis.data.panel as P
    monkeypatch.setattr(P, "PANEL_DIR", tmp_path / "panel")
    src = tmp_path / "a.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(src)
    calls = []

    def build():
        calls.append(1)
        return pd.read_parquet(src)

    P.build_if_stale("t", [src], build, extra={"k": "1"})
    P.build_if_stale("t", [src], build, extra={"k": "1"})
    assert len(calls) == 1                                  # 未变化：直接读
    P.build_if_stale("t", [src], build, extra={"k": "2"})
    assert len(calls) == 2                                  # 额外键变化
    pd.DataFrame({"x": [2]}).to_parquet(src)
    st = os.stat(src)
    os.utime(src, (st.st_atime, st.st_mtime + 10))
    assert P.build_if_stale("t", [src], build, extra={"k": "2"})["x"].tolist() == [2]
    assert len(calls) == 3                                  # 源文件变化


def test_is_st_panel_matches_per_file_read(tmp_path, monkeypatch):
    import stockanalysis.data.baostock_data as B
    import stockanalysis.data.panel as P
    monkeypatch.setattr(P, "PANEL_DIR", tmp_path / "panel")
    monkeypatch.setattr(B, "DAILY_DIR", tmp_path)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "is_st": [0, 1]}).to_parquet(tmp_path / "SH600001.parquet")
    pd.DataFrame({"date": pd.to_datetime(["2020-01-03"]), "is_st": [1]}).to_parquet(tmp_path / "SH600002.parquet")
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    w = B.load_daily("is_st", ["SH600001", "SH600002", "SH600003"], dates)
    assert w["SH600001"].tolist()[:2] == [0, 1] and np.isnan(w.loc["2020-01-06", "SH600001"])
    assert np.isnan(w.loc["2020-01-02", "SH600002"]) and w.loc["2020-01-03", "SH600002"] == 1
    assert w["SH600003"].isna().all() and w.dtypes.eq(np.float32).all()
