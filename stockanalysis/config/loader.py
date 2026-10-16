"""配置加载：发现股池、按层深合并、校验、计算配置 hash。

合并顺序（后者覆盖前者，dict 递归合并）：
    config/global.yaml → config/pool_defaults.yaml → config/pools/<name>.yaml → override（整段替换）→ sets（点路径）

- 加股池 = 在 config/pools/ 加一个 yaml（只写与默认不同的项）；`enabled: false` 暂停而不删除。
- 改所有池的规则 = 改 pool_defaults.yaml；只改某池 = 在该池 yaml 覆盖。
- 关闭某池的默认过滤规则 = 该池 yaml 写 `filters: {规则名: false}`（in_universe 不可关闭）。
加载即校验：未知顶层键 / 分段键、未注册的规则 / 打分器 / 策略、与签名不符的参数名，都会直接报错。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from .. import paths
from .schema import (MANDATORY_FILTERS, REBALANCE_FREQS, STATUSES, TOP_LEVEL_KEYS, PoolSpec, RebalanceRule,
                     Sizing, section)


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 中的 dict 与 base 同名 dict 合并，其余值直接覆盖。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_yaml(path: Path) -> dict:
    """读 yaml；文件不存在返回 {}（pool_defaults.yaml 可省略）。"""
    path = Path(path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_global(path: Path | None = None) -> dict:
    return read_yaml(path or paths.GLOBAL_CONFIG)


def parse_sets(sets: list[str] | None) -> dict:
    """["model.lgbm.learning_rate=0.05", "rebalance.hold_thresh=10"] -> 嵌套 dict（值按 yaml 解析）。"""
    out: dict = {}
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set 需为 key.path=value 形式: {item!r}")
        key_path, raw = item.split("=", 1)
        node = out
        keys = key_path.strip().split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def default_registries() -> dict:
    # 延迟导入：注册表模块依赖 qlib/pandas，仅在校验时加载
    from ..backtest.strategies import STRATEGIES
    from ..models.scorers import SCORERS
    from ..selection.filters import FILTERS

    return {"filters": FILTERS, "scorers": SCORERS, "strategies": STRATEGIES}


def _resolve_filters(name: str, filters: dict | None, account_boards: list[str] | None = None) -> dict[str, dict]:
    filters = dict(filters or {})
    if isinstance(filters.get("boards"), dict) and "boards" not in filters["boards"]:
        if not account_boards:
            raise ValueError(f"池 '{name}'：boards 规则未指定板块，且 global.yaml 未设置 account_boards")
        filters["boards"] = {**filters["boards"], "boards": list(account_boards)}
    for f in MANDATORY_FILTERS:
        if filters.get(f) is False:
            raise ValueError(f"池 '{name}'：过滤规则 {f} 不可关闭")
        if f not in filters:
            filters = {f: {}, **filters}
    return {k: (v or {}) for k, v in filters.items() if v is not False}


def build_pool_spec(name: str, cfg: dict, registries: dict | None = None, validate: bool = True) -> PoolSpec:
    """cfg 为已合并好的完整配置。"""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("name", name)
    if cfg["name"] != name:
        raise ValueError(f"池 '{name}' 的 yaml 中 name={cfg['name']!r} 与文件名不一致")
    unknown = set(cfg) - set(TOP_LEVEL_KEYS)
    if unknown:
        raise KeyError(f"池 '{name}' 含未知配置项 {sorted(unknown)}，可用: {list(TOP_LEVEL_KEYS)}")
    for key in ("universe", "benchmark", "scorer", "strategy"):
        if key not in cfg:
            raise KeyError(f"池 '{name}' 缺少必填项 {key}")
    if cfg.get("status", "observe") not in STATUSES:
        raise ValueError(f"池 '{name}'：status 须为 {STATUSES}")

    sizing = section(Sizing, cfg.get("sizing"), f"池 {name}.sizing")
    sizing = Sizing(top_pct=float(sizing.top_pct), min_topk=int(sizing.min_topk),
                    topk=None if sizing.topk is None else int(sizing.topk))
    rebalance = section(RebalanceRule, cfg.get("rebalance"), f"池 {name}.rebalance")
    if rebalance.freq not in REBALANCE_FREQS:
        raise ValueError(f"池 '{name}'：rebalance.freq 须为 {REBALANCE_FREQS}")
    rebalance = RebalanceRule(freq=rebalance.freq, n_drop=int(rebalance.n_drop), hold_thresh=int(rebalance.hold_thresh))
    signal = dict(cfg.get("signal") or {})
    if set(signal) - {"score_ema"}:
        raise KeyError(f"池 {name}.signal 含未知项 {sorted(set(signal) - {'score_ema'})}")
    cfg["filters"] = _resolve_filters(name, cfg.get("filters"), cfg.get("account_boards"))

    spec = PoolSpec(
        name=name,
        enabled=bool(cfg.get("enabled", True)),
        status=cfg.get("status", "observe"),
        universe=cfg["universe"],
        benchmark=cfg["benchmark"],
        sizing=sizing,
        rebalance=rebalance,
        score_ema=None if signal.get("score_ema") in (None, 0) else int(signal["score_ema"]),
        scorer=dict(cfg["scorer"]),
        strategy=dict(cfg["strategy"]),
        filters=cfg["filters"],
        backtest=dict(cfg.get("backtest") or {}),
        cost=dict(cfg.get("cost") or {}),
        account=float(cfg.get("account", 1e6)),
        exchange=dict(cfg.get("exchange") or {}),
        model=dict(cfg.get("model") or {}),
        data=dict(cfg.get("data") or {}),
        raw=cfg,
        config_hash=config_hash(cfg),
    )
    if validate:
        validate_spec(spec, registries or default_registries())
    return spec


def validate_spec(spec: PoolSpec, registries: dict) -> None:
    where = f"池 {spec.name}"
    for fname, params in spec.filters.items():
        registries["filters"].check_params(fname, params, f"{where}.filters.{fname}")
    registries["scorers"].check_params(spec.scorer["name"], spec.scorer_params, f"{where}.scorer")
    registries["strategies"].check_params(spec.strategy["name"], spec.strategy_params, f"{where}.strategy")


def list_pools(pools_dir: Path | None = None, enabled_only: bool = False) -> list[str]:
    """config/pools/ 下的池名（忽略 _ 开头的文件）；enabled_only 时跳过 enabled: false 的池。"""
    d = pools_dir or paths.POOLS_DIR
    names = sorted(p.stem for p in d.glob("*.yaml") if not p.stem.startswith("_"))
    if enabled_only:
        names = [n for n in names if read_yaml(d / f"{n}.yaml").get("enabled", True) is not False]
    return names


def merged_config(name: str, pools_dir: Path | None = None, global_path: Path | None = None,
                  defaults_path: Path | None = None, override: dict | None = None,
                  sets: list[str] | None = None) -> dict:
    d = pools_dir or paths.POOLS_DIR
    path = d / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"未找到股池配置 {path}，可选: {list_pools(d)}")
    cfg = deep_merge(load_global(global_path), read_yaml(defaults_path or paths.POOL_DEFAULTS))
    cfg = deep_merge(cfg, read_yaml(path))
    cfg = {**cfg, **(override or {})}
    return deep_merge(cfg, parse_sets(sets))


def load_pool(name: str, pools_dir: Path | None = None, global_path: Path | None = None,
              defaults_path: Path | None = None, registries: dict | None = None, validate: bool = True,
              override: dict | None = None, sets: list[str] | None = None) -> PoolSpec:
    """override：按顶层键**整段替换**（如 {"scorer": {"name": "momentum"}}）；
    sets：点路径**深合并**覆盖（如 ["rebalance.hold_thresh=5"]）。用于命令行覆盖与参数网格。"""
    cfg = merged_config(name, pools_dir, global_path, defaults_path, override, sets)
    return build_pool_spec(name, cfg, registries, validate)
