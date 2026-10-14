"""原地融合处理器与 qlib 原版 CSZScoreNorm(robust) + Fillna 等价，且不复制整表。"""
import numpy as np
import pandas as pd
from qlib.data.dataset.processor import CSZScoreNorm, Fillna

from src.factors.alpha158 import infer_processors


def make_df(seed=0, n_dates=6, n_inst=40):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=n_dates), [f"S{i:03d}" for i in range(n_inst)]],
                                     names=["datetime", "instrument"])
    cols = pd.MultiIndex.from_tuples([("feature", f"f{j}") for j in range(5)] + [("label", "LABEL0")])
    x = rng.standard_t(3, size=(len(idx), 6)).astype("float32")
    x[rng.random(x.shape) < 0.05] = np.nan
    return pd.DataFrame(x, index=idx, columns=cols)


def test_matches_qlib_robust_zscore_and_fillna():
    df = make_df()
    want = Fillna(fields_group="feature")(CSZScoreNorm(fields_group="feature", method="robust")(df.copy()))
    got = infer_processors()[0](df.copy())
    np.testing.assert_allclose(got["feature"].to_numpy(), want["feature"].to_numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(np.isnan(got["label"].to_numpy()), np.isnan(df["label"].to_numpy()))


def test_label_untouched_and_inplace():
    df = make_df(1)
    label = df["label"].copy()
    df._consolidate_inplace()
    before = df._mgr.blocks[0].values
    out = infer_processors()[0](df)
    assert out is df and np.shares_memory(out._mgr.blocks[0].values, before)
    pd.testing.assert_frame_equal(out["label"], label)


def test_inf_becomes_zero_and_bounded():
    df = make_df(2)
    df.iloc[3, 0] = np.inf
    out = infer_processors()[0](df)
    f = out["feature"].to_numpy()
    assert np.isfinite(f).all() and f.max() <= 3 and f.min() >= -3
    assert out.iloc[3, 0] == 0.0
