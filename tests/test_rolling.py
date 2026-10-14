"""滚动窗口切分：purge gap、无重叠、覆盖测试区间。"""
import pandas as pd

from src.models.rolling import rolling_windows

CAL = pd.bdate_range("2012-01-02", "2026-10-14")


def test_windows_cover_test_years_without_overlap():
    ws = rolling_windows(CAL, "2018-01-01", "2026-10-14", train_years=4, valid_months=6, gap=6)
    assert [w.test[0][:4] for w in ws] == [str(y) for y in range(2018, 2027)]
    assert ws[-1].test[1] == "2026-10-14"
    for a, b in zip(ws, ws[1:]):
        assert pd.Timestamp(a.test[1]) < pd.Timestamp(b.test[0])


def test_purge_gap_between_segments():
    w = rolling_windows(CAL, "2018-01-01", "2018-12-31", gap=6)[0]
    pos = {d: i for i, d in enumerate(CAL)}
    assert pos[pd.Timestamp(w.test[0])] - pos[pd.Timestamp(w.valid[1])] == 7   # 中间空 6 个交易日
    assert pos[pd.Timestamp(w.valid[0])] - pos[pd.Timestamp(w.train[1])] == 7


def test_train_and_valid_lengths():
    w = rolling_windows(CAL, "2020-01-01", "2020-12-31", train_years=4, valid_months=6, gap=6)[0]
    assert 5.5 <= (pd.Timestamp(w.valid[1]) - pd.Timestamp(w.valid[0])).days / 30.4 <= 6.1
    assert 3.9 <= (pd.Timestamp(w.train[1]) - pd.Timestamp(w.train[0])).days / 365.25 <= 4.01


def test_quarterly_retrain():
    ws = rolling_windows(CAL, "2024-01-01", "2024-12-31", retrain="quarterly")
    assert [w.test[0][:7] for w in ws] == ["2024-01", "2024-04", "2024-07", "2024-10"]


def test_model_params_scorer_section_only_for_own_kind():
    from src.config import load_pool
    from src.models.rolling import model_params, train_signature

    lgb = load_pool("hs300", override={"scorer": {"name": "lgbm", "learning_rate": 0.1}})
    assert model_params(lgb, "lgbm")["learning_rate"] == 0.1
    assert "learning_rate" not in model_params(lgb, "linear")
    base = load_pool("hs300", override={"scorer": {"name": "lgbm"}})
    assert train_signature(lgb, "lgbm") != train_signature(base, "lgbm")
    assert train_signature(lgb, "linear") == train_signature(base, "linear")


def test_linear_signature_independent_of_lgbm_settings():
    from src.config import load_pool
    from src.models.rolling import train_signature

    a = load_pool("hs300")
    b = load_pool("hs300", sets=["model.lgbm.learning_rate=0.3", "model.lgbm.num_leaves=31"])
    assert train_signature(a, "linear") == train_signature(b, "linear")
    assert train_signature(a, "lgbm") != train_signature(b, "lgbm")
