"""配置层：深合并、池发现、注册名与参数名校验。"""
import shutil

import pytest
import yaml

from src.config import POOLS_DIR, build_pool_spec, deep_merge, list_pools, load_pool
from src.registry import Registry


def fake_registries():
    filters = Registry("filter", skip=1)

    @filters.register("in_universe", scope="exit")
    def in_universe(ctx):
        ...

    @filters.register("boards", scope="entry")
    def boards(ctx, boards):
        ...

    @filters.register("min_amount", scope="entry")
    def min_amount(ctx, amount, window=20):
        ...

    scorers = Registry("scorer")

    @scorers.register("momentum")
    class Momentum:
        def __init__(self, window=20):
            ...

    strategies = Registry("strategy")

    @strategies.register("topk", params=("n_drop", "hold_thresh"))
    class Topk:
        def __init__(self, **kwargs):
            ...

    return {"filters": filters, "scorers": scorers, "strategies": strategies}


GLOBAL = {"account": 1e6, "cost": {"commission": 0.00025, "slippage": 0.001},
          "backtest": {"start": "2018-01-01", "in_sample_end": "2022-12-31"}}
POOL = {"name": "p1", "universe": "csi300", "benchmark": "SH000300",
        "strategy": {"name": "topk", "n_drop": 1},
        "scorer": {"name": "momentum"},
        "filters": {"boards": {"boards": ["main"]}},
        "cost": {"slippage": 0.002}}


def test_deep_merge_overrides_nested_only():
    out = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
    assert out == {"a": {"x": 1, "y": 3}, "b": 1}


def test_pool_overrides_global_cost():
    spec = build_pool_spec("p1", POOL, GLOBAL, fake_registries())
    assert spec.cost == {"commission": 0.00025, "slippage": 0.002}
    assert spec.backtest["start"] == "2018-01-01"
    assert spec.account == 1e6


def test_mandatory_filter_added_first():
    spec = build_pool_spec("p1", POOL, GLOBAL, fake_registries())
    assert list(spec.filters) == ["in_universe", "boards"]


def test_mandatory_filter_cannot_be_disabled():
    pool = {**POOL, "filters": {"in_universe": False}}
    with pytest.raises(ValueError):
        build_pool_spec("p1", pool, GLOBAL, fake_registries())


def test_unknown_filter_name():
    pool = {**POOL, "filters": {"no_such_rule": {}}}
    with pytest.raises(KeyError, match="no_such_rule"):
        build_pool_spec("p1", pool, GLOBAL, fake_registries())


def test_unknown_filter_param():
    pool = {**POOL, "filters": {"min_amount": {"amount": 1, "windw": 5}}}
    with pytest.raises(KeyError, match="windw"):
        build_pool_spec("p1", pool, GLOBAL, fake_registries())


def test_missing_required_param():
    pool = {**POOL, "filters": {"boards": {}}}
    with pytest.raises(KeyError, match="boards"):
        build_pool_spec("p1", pool, GLOBAL, fake_registries())


def test_explicit_strategy_params():
    pool = {**POOL, "strategy": {"name": "topk", "n_drop": 1, "foo": 2}}
    with pytest.raises(KeyError, match="foo"):
        build_pool_spec("p1", pool, GLOBAL, fake_registries())


def test_unknown_scorer():
    pool = {**POOL, "scorer": {"name": "nope"}}
    with pytest.raises(KeyError, match="nope"):
        build_pool_spec("p1", pool, GLOBAL, fake_registries())


def test_name_must_match_file():
    with pytest.raises(ValueError):
        build_pool_spec("other", POOL, GLOBAL, fake_registries())


def test_config_hash_changes_with_params():
    a = build_pool_spec("p1", POOL, GLOBAL, fake_registries())
    b = build_pool_spec("p1", {**POOL, "min_topk": 6}, GLOBAL, fake_registries())
    assert a.config_hash != b.config_hash


def test_list_pools_discovers_new_yaml(tmp_path):
    for f in POOLS_DIR.glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    shutil.copy(POOLS_DIR / "hs300.yaml", tmp_path / "test_pool.yaml")
    (tmp_path / "_template.yaml").write_text("x: 1", encoding="utf-8")
    assert "test_pool" in list_pools(tmp_path)
    assert "_template" not in list_pools(tmp_path)


def test_repo_pools_parse_without_validation():
    for name in list_pools():
        spec = load_pool(name, validate=False)
        assert spec.universe.startswith("csi")
        assert spec.filters and next(iter(spec.filters)) == "in_universe"
        raw = yaml.safe_load((POOLS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        assert raw["name"] == name
