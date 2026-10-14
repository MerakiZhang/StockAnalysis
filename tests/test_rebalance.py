"""换仓规则：首次建仓、n_drop 上限、hold_thresh、exit 强制调出、entry 不触发卖出、topk 变化。"""
import numpy as np
import pandas as pd

from src.selection.rebalance import rank_codes, rebalance

CODES = [f"S{i:02d}" for i in range(10)]


def sc(**vals):
    """分数：默认 S00 最高、S09 最低；可覆盖单只。"""
    s = pd.Series({c: 10.0 - i for i, c in enumerate(CODES)})
    for k, v in vals.items():
        s[k] = v
    return s


def all_entry(**over):
    e = pd.Series(True, index=CODES)
    for k, v in over.items():
        e[k] = v
    return e


def test_first_build_takes_top():
    r = rebalance({}, sc(), all_entry(), topk=3, n_drop=1)
    assert r.holdings == ["S00", "S01", "S02"] and r.buy == r.holdings
    assert not r.sell_rank and not r.sell_exit


def test_first_build_skips_non_entry():
    r = rebalance({}, sc(), all_entry(S01=False), topk=3, n_drop=1)
    assert r.holdings == ["S00", "S02", "S03"]


def test_unchanged_ranking_no_trade():
    h = {"S00": 5, "S01": 5, "S02": 5}
    r = rebalance(h, sc(), all_entry(), topk=3, n_drop=1)
    assert r.holdings == ["S00", "S01", "S02"] and not r.buy and not r.sell_rank


def test_n_drop_caps_turnover():
    h = {"S07": 5, "S08": 5, "S09": 5}  # 全部排名靠后
    r = rebalance(h, sc(), all_entry(), topk=3, n_drop=1)
    assert r.sell_rank == ["S09"] and r.buy == ["S00"]
    assert set(r.holdings) == {"S00", "S07", "S08"}


def test_hold_thresh_blocks_rank_sell_and_its_replacement():
    h = {"S07": 5, "S08": 5, "S09": 1}
    r = rebalance(h, sc(), all_entry(), topk=3, n_drop=1, hold_thresh=3)
    assert r.sell_rank == [] and r.buy == []
    assert set(r.holdings) == {"S07", "S08", "S09"}


def test_exit_forced_out_not_counted_in_n_drop():
    h = {"S00": 5, "S08": 5, "S09": 5}
    score = sc(S00=np.nan)  # S00 不再满足持有规则（如被调出指数）
    r = rebalance(h, score, all_entry(), topk=3, n_drop=1, hold_thresh=3)
    assert r.sell_exit == ["S00"]
    assert r.sell_rank == ["S09"]
    assert r.buy == ["S01", "S02"]
    assert r.holdings == ["S01", "S02", "S08"]


def test_exit_ignores_hold_thresh():
    r = rebalance({"S00": 0}, sc(S00=np.nan), all_entry(), topk=1, n_drop=1, hold_thresh=5)
    assert r.sell_exit == ["S00"] and r.holdings == ["S01"]


def test_entry_failure_does_not_sell_holding():
    h = {"S00": 5, "S01": 5, "S02": 5}
    r = rebalance(h, sc(), all_entry(S00=False, S01=False, S02=False), topk=3, n_drop=1)
    assert r.holdings == ["S00", "S01", "S02"] and not r.sell_rank and not r.sell_exit


def test_missing_holding_in_score_is_exit():
    r = rebalance({"XX": 5}, sc(), all_entry(), topk=1, n_drop=1)
    assert r.sell_exit == ["XX"] and r.holdings == ["S00"]


def test_topk_grows_buys_more():
    h = {"S00": 5, "S01": 5}
    r = rebalance(h, sc(), all_entry(), topk=4, n_drop=1)
    assert r.buy == ["S02", "S03"] and len(r.holdings) == 4


def test_topk_shrinks_converges_via_n_drop():
    h = {c: 5 for c in ["S00", "S01", "S02", "S03"]}
    r = rebalance(h, sc(), all_entry(), topk=2, n_drop=1)
    assert r.sell_rank == ["S03"] and r.buy == []
    assert len(r.holdings) == 3


def test_new_candidate_must_beat_holding():
    h = {"S01": 5, "S02": 5}
    r = rebalance(h, sc(), all_entry(S00=False), topk=2, n_drop=1)
    assert r.holdings == ["S01", "S02"] and not r.buy  # 唯一更强的 S00 不准入


def test_rank_codes_ties_deterministic():
    s = pd.Series({"B": 1.0, "A": 1.0, "C": np.nan, "D": 2.0})
    assert rank_codes(s) == ["D", "A", "B", "C"]
