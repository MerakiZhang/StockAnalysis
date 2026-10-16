"""技术 / 资金面指标与条件库：手算对数、停牌缺口处理、无前视（突破用前一日为止的高点）。"""
import numpy as np
import pandas as pd
import pytest

from stockanalysis.features import conditions as C
from stockanalysis.features import technical as T

DATES = pd.bdate_range("2020-01-01", periods=120)


def make_ctx(close: np.ndarray, high=None, low=None, amount=None, codes=("SH600001",)):
    frame = lambda v: pd.DataFrame(np.asarray(v, dtype=np.float64).reshape(len(DATES), -1),  # noqa: E731
                                   index=DATES, columns=list(codes))
    c = frame(close)
    return {"close": c, "high": frame(high if high is not None else close),
            "low": frame(low if low is not None else close),
            "amount": frame(amount if amount is not None else np.full(len(DATES), 1000.0)),
            "vwap": c.copy(), "codes": list(codes), "dates": DATES}


def test_ma_ret_and_cross():
    close = np.arange(1, 121, dtype=float)
    ctx = make_ctx(close)
    assert T.get(ctx, "ma5").iloc[4, 0] == pytest.approx(3.0)
    assert np.isnan(T.get(ctx, "ma5").iloc[3, 0])
    assert T.get(ctx, "ret5").iloc[10, 0] == pytest.approx(11 / 6 - 1)
    a = pd.DataFrame({"x": [1, 2, 3, 2, 4]})
    assert T.cross_up(a, 2.5)["x"].tolist() == [False, False, True, False, True]


def test_clv_and_cmf_by_hand():
    close = np.full(len(DATES), 10.0)
    high, low = close + 1, close - 1
    close_at_high = close + 1                          # 收在最高 → clv = 1
    ctx = make_ctx(close_at_high, high, low, amount=np.full(len(DATES), 2.0))
    assert T.get(ctx, "clv").iloc[5, 0] == pytest.approx(1.0)
    assert T.get(ctx, "cmf20").iloc[30, 0] == pytest.approx(1.0)
    ctx2 = make_ctx(close, high, low)                  # 收在中间 → clv = 0
    assert T.get(ctx2, "clv").iloc[5, 0] == pytest.approx(0.0)


def test_talib_per_column_skips_suspension_gap():
    rng = np.random.default_rng(0)
    close = 10 + np.cumsum(rng.normal(0, 0.1, len(DATES)))
    close[50:55] = np.nan                               # 停牌 5 天
    ctx = make_ctx(close, close + 0.2, close - 0.2)
    rsi = T.get(ctx, "rsi14").iloc[:, 0]
    assert rsi.iloc[50:55].isna().all()
    assert rsi.iloc[60:].notna().all()                  # 复牌后恢复计算，不会从此全是 NaN
    dif = T.get(ctx, "macd_dif").iloc[:, 0]
    assert dif.iloc[-1] == dif.iloc[-1]


def test_breakout_uses_prior_high_only():
    close = np.full(len(DATES), 10.0)
    close[100] = 11.0                                   # 第 100 天突破
    amount = np.full(len(DATES), 1000.0)
    amount[100] = 3000.0
    ctx = make_ctx(close, close, close, amount)
    flag = C.breakout(ctx, n=20, amt=1.5).iloc[:, 0]
    assert flag.iloc[100] and not flag.iloc[99] and not flag.iloc[101]


def test_pullback_shrink_condition():
    close = np.linspace(10, 20, len(DATES))
    close[-5:] = close[-6] * np.array([0.99, 0.98, 0.97, 0.96, 0.95])   # 近 5 日回调约 5%
    amount = np.full(len(DATES), 1000.0)
    amount[-5:] = 400.0                                                   # 缩量
    ctx = make_ctx(close, close, close, amount)
    assert C.pullback_shrink(ctx, amt=0.7).iloc[-1, 0]
    assert not C.pullback_shrink(ctx, amt=0.7).iloc[-20, 0]


def test_all_conditions_registered_with_metadata():
    for name in C.CONDITIONS.names():
        meta = C.CONDITIONS.meta(name)
        assert meta["source"] and meta["kind"] in ("state", "trigger")
        C.CONDITIONS.check_params(name, {k: v[0] for k, v in meta["grid"].items()})   # 网格参数名与签名一致
