"""配置层：合并顺序（global → pool_defaults → pool → override → sets）、分段校验、注册名与参数名校验、池发现。"""
import pytest
import yaml

from stockanalysis.config import deep_merge, list_pools, load_pool, parse_sets
from stockanalysis.registry import Registry


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

    conditions = Registry("condition", skip=1)

    @conditions.register("cheap")
    def cheap(ctx, pct=0.7):
        ...

    return {"filters": filters, "scorers": scorers, "conditions": conditions}


GLOBAL = {"account": 1e6, "cost": {"commission": 0.00025, "slippage": 0.001},
          "backtest": {"start": "2018-01-01", "in_sample_end": "2022-12-31"}}
DEFAULTS = {"scorer": {"name": "momentum"}, "filters": {"in_universe": {}, "boards": {"boards": ["main"]}}}
POOL = {"name": "p1", "universe": "all", "benchmark": "SH000906", "cost": {"slippage": 0.002},
        "points": {"max_positions": 10, "buy": {"cheap": {}}, "stop_loss": 0.08}}


@pytest.fixture
def cfg_dir(tmp_path):
    (tmp_path / "pools").mkdir()
    for name, data in (("global.yaml", GLOBAL), ("pool_defaults.yaml", DEFAULTS)):
        (tmp_path / name).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def write_pool(name, data):
        (tmp_path / "pools" / f"{name}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    write_pool("p1", POOL)
    return tmp_path, write_pool


def load(cfg_dir, name="p1", **kw):
    d, _ = cfg_dir
    return load_pool(name, pools_dir=d / "pools", global_path=d / "global.yaml",
                     defaults_path=d / "pool_defaults.yaml", registries=fake_registries(), **kw)


def test_deep_merge_overrides_nested_only():
    assert deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}, "b": 1}


def test_merge_chain_global_defaults_pool(cfg_dir):
    spec = load(cfg_dir)
    assert spec.cost == {"commission": 0.00025, "slippage": 0.002}      # 池覆盖 global
    assert spec.scorer == {"name": "momentum"}                             # 来自 pool_defaults
    assert spec.points.max_positions == 10 and spec.points.freq == "daily" and spec.points.max_hold is None
    assert spec.enabled and spec.status == "observe"
    assert list(spec.filters) == ["in_universe", "boards"]


def test_pool_overrides_one_rule_and_adds_filter(cfg_dir):
    _, write_pool = cfg_dir
    write_pool("p2", {**POOL, "name": "p2", "cost": {"commission": 0.0003},
                      "filters": {"min_amount": {"amount": 1e8}}})
    spec = load(cfg_dir, "p2")
    assert spec.cost == {"commission": 0.0003, "slippage": 0.001}       # 只改一项，其余继承
    assert list(spec.filters) == ["in_universe", "boards", "min_amount"]


def test_disable_default_filter_with_false(cfg_dir):
    _, write_pool = cfg_dir
    write_pool("p2", {**POOL, "name": "p2", "filters": {"boards": False}})
    assert list(load(cfg_dir, "p2").filters) == ["in_universe"]


def test_mandatory_filter_cannot_be_disabled(cfg_dir):
    _, write_pool = cfg_dir
    write_pool("p2", {**POOL, "name": "p2", "filters": {"in_universe": False}})
    with pytest.raises(ValueError):
        load(cfg_dir, "p2")


def test_sets_and_override(cfg_dir):
    spec = load(cfg_dir, sets=["points.max_hold=3", "points.buy={cheap: {pct: 0.8}}"],
                override={"scorer": {"name": "momentum", "window": 5}})
    assert spec.points.max_hold == 3 and spec.points.buy == {"cheap": {"pct": 0.8}}
    assert spec.points.stop_loss == 0.08                                 # 其余 points 项保留
    assert spec.scorer == {"name": "momentum", "window": 5}


@pytest.mark.parametrize("extra, match", [
    ({"top_pct": 0.02}, "未知配置项"),
    ({"rebalance": {"hold_thresh": 5}}, "未知配置项"),             # 旧版 topk 换仓写法已删除
    ({"points": {"buy": {"cheap": {}}, "stop_los": 0.05}}, "stop_los"),
    ({"points": {"buy": {"cheap": {}}, "freq": "hourly"}}, "freq"),
    ({"points": {}}, "买点"),
    ({"status": "live"}, "status"),
])
def test_invalid_config_rejected(cfg_dir, extra, match):
    _, write_pool = cfg_dir
    write_pool("p2", {**POOL, "name": "p2", **extra})
    with pytest.raises((KeyError, ValueError), match=match):
        load(cfg_dir, "p2")


@pytest.mark.parametrize("extra, match", [
    ({"filters": {"no_such_rule": {}}}, "no_such_rule"),
    ({"filters": {"min_amount": {"amount": 1, "windw": 5}}}, "windw"),
    ({"filters": {"min_amount": {"window": 5}}}, "amount"),   # 缺必填参数（boards: {} 会继承默认参数，不算缺）
    ({"scorer": {"name": "nope"}}, "nope"),
    ({"points": {"buy": {"nope_cond": {}}}}, "nope_cond"),
    ({"points": {"buy": {"cheap": {"pctt": 1}}}}, "pctt"),
])
def test_registry_validation(cfg_dir, extra, match):
    _, write_pool = cfg_dir
    write_pool("p2", {**POOL, "name": "p2", **extra})
    with pytest.raises(KeyError, match=match):
        load(cfg_dir, "p2")


def test_name_must_match_file(cfg_dir):
    _, write_pool = cfg_dir
    write_pool("p2", POOL)  # name: p1
    with pytest.raises(ValueError):
        load(cfg_dir, "p2")


def test_config_hash_changes_with_params(cfg_dir):
    assert load(cfg_dir).config_hash != load(cfg_dir, sets=["points.stop_loss=0.05"]).config_hash


def test_list_pools_enabled_and_templates(cfg_dir):
    d, write_pool = cfg_dir
    write_pool("p2", {**POOL, "name": "p2", "enabled": False})
    write_pool("_template", {"name": "x"})
    assert list_pools(d / "pools") == ["p1", "p2"]
    assert list_pools(d / "pools", enabled_only=True) == ["p1"]


def test_parse_sets_nested_and_typed():
    out = parse_sets(["points.atr_stop.k=2.5", "points.max_hold=10", "points.take_profit=null"])
    assert out == {"points": {"atr_stop": {"k": 2.5}, "max_hold": 10, "take_profit": None}}


# ---------------------------------------------------------------- 仓库内的真实配置
RULE_POOLS = ("long", "short_reversal", "short_breakout")               # 全市场规则池


def test_repo_pools_load_and_validate():
    for name in list_pools():
        spec = load_pool(name)
        assert spec.status in ("production", "observe")
        assert next(iter(spec.filters)) == "in_universe"
        assert spec.filters["boards"] == {"boards": spec.raw["account_boards"]}   # 板块权限取全局设置
    for name in RULE_POOLS:
        spec = load_pool(name)
        assert spec.universe == "all" and "not_st" in spec.filters and "min_amount" in spec.filters
        assert spec.points is not None and spec.points.max_positions == 10


def test_repo_pool_yaml_only_has_overrides():
    """池 yaml 不重复 pool_defaults 的规则：保证改默认规则一处生效。"""
    from stockanalysis.paths import POOLS_DIR

    for name in list_pools():
        raw = yaml.safe_load((POOLS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        assert not set(raw.get("filters") or {}) & {"in_universe", "not_st", "boards"}, name


def test_apply_sets_replaces_leaf_dicts():
    from stockanalysis.config.loader import apply_sets

    cfg = {"points": {"buy": {"a": {}, "b": {"x": 1}}, "stop_loss": 0.05}, "cost": {"fee": {"lr": 0.1, "n": 5}}}
    out = apply_sets(cfg, ["points.buy={a: {}}", "points.buy_any={}", "cost.fee.lr=0.05"])
    assert out["points"]["buy"] == {"a": {}}           # 字典整体替换，不合并
    assert out["points"]["buy_any"] == {} and out["points"]["stop_loss"] == 0.05
    assert out["cost"]["fee"] == {"lr": 0.05, "n": 5}   # 中间层保留
    assert cfg["points"]["buy"] == {"a": {}, "b": {"x": 1}}   # 不改原对象
