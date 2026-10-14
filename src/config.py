"""配置加载与校验。

布局：
    config/global.yaml        全局默认（account/cost/backtest/exchange/model/data）
    config/pools/<name>.yaml  每个股池一个文件；与 global 深合并，池内同名项覆盖全局

加股池 = 复制一份 pools/*.yaml 改参数，不改代码（list_pools() 自动发现）。
加载即校验：过滤规则 / 打分器 / 回测策略的名字必须已注册，参数名必须与其签名匹配。

旧接口 load_config()（读 config/selector.yaml）保留给尚未迁移的脚本，重构收尾时删除。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
POOLS_DIR = CONFIG_DIR / "pools"
GLOBAL_PATH = CONFIG_DIR / "global.yaml"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

MANDATORY_FILTERS = ("in_universe",)  # 不可在 yaml 中关闭
REBALANCE_FREQS = ("daily", "weekly", "monthly")


@dataclass
class PoolSpec:
    name: str
    universe: str
    benchmark: str
    top_pct: float
    min_topk: int
    topk: int | None
    rebalance: str
    strategy: dict            # {"name": ..., 其余为策略参数}
    scorer: dict              # {"name": ..., 其余为打分器参数}
    filters: dict[str, dict]  # 规则名 -> 参数（按 yaml 顺序）
    backtest: dict
    cost: dict
    account: float
    exchange: dict
    model: dict
    data: dict
    raw: dict = field(repr=False)  # 合并后的完整配置
    config_hash: str = ""

    @property
    def strategy_params(self) -> dict:
        return {k: v for k, v in self.strategy.items() if k != "name"}

    @property
    def scorer_params(self) -> dict:
        return {k: v for k, v in self.scorer.items() if k != "name"}


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 中的 dict 与 base 同名 dict 合并，其余值直接覆盖。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_global(path: Path | None = None) -> dict:
    return _read_yaml(path or GLOBAL_PATH)


def list_pools(pools_dir: Path | None = None) -> list[str]:
    """config/pools/ 下全部池名（文件名去扩展名，忽略 _ 开头的模板）。"""
    d = pools_dir or POOLS_DIR
    return sorted(p.stem for p in d.glob("*.yaml") if not p.stem.startswith("_"))


def _default_registries() -> dict:
    # 延迟导入：注册表模块依赖 qlib/pandas，仅在校验时加载
    from .backtest.strategies import STRATEGIES
    from .models.scorers import SCORERS
    from .selection.filters import FILTERS

    return {"filters": FILTERS, "scorers": SCORERS, "strategies": STRATEGIES}


def _config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def build_pool_spec(name: str, pool_cfg: dict, global_cfg: dict,
                    registries: dict | None = None, validate: bool = True) -> PoolSpec:
    cfg = deep_merge(global_cfg, pool_cfg)
    cfg.setdefault("name", name)
    if cfg["name"] != name:
        raise ValueError(f"池 '{name}' 的 yaml 中 name={cfg['name']!r} 与文件名不一致")

    filters = dict(cfg.get("filters") or {})
    for f in MANDATORY_FILTERS:  # 强制规则放最前
        if f not in filters:
            filters = {f: {}, **filters}
        elif filters[f] is False:
            raise ValueError(f"池 '{name}'：过滤规则 {f} 不可关闭")
    filters = {k: (v or {}) for k, v in filters.items() if v is not False}
    cfg["filters"] = filters

    for key in ("universe", "benchmark", "strategy", "scorer"):
        if key not in cfg:
            raise KeyError(f"池 '{name}' 缺少必填项 {key}")
    if cfg.get("rebalance", "daily") not in REBALANCE_FREQS:
        raise ValueError(f"池 '{name}'：rebalance 须为 {REBALANCE_FREQS}")

    spec = PoolSpec(
        name=name,
        universe=cfg["universe"],
        benchmark=cfg["benchmark"],
        top_pct=float(cfg.get("top_pct", 0.01)),
        min_topk=int(cfg.get("min_topk", 5)),
        topk=None if cfg.get("topk") is None else int(cfg["topk"]),
        rebalance=cfg.get("rebalance", "daily"),
        strategy=dict(cfg["strategy"]),
        scorer=dict(cfg["scorer"]),
        filters=filters,
        backtest=dict(cfg.get("backtest") or {}),
        cost=dict(cfg.get("cost") or {}),
        account=float(cfg.get("account", 1e6)),
        exchange=dict(cfg.get("exchange") or {}),
        model=dict(cfg.get("model") or {}),
        data=dict(cfg.get("data") or {}),
        raw=cfg,
        config_hash=_config_hash(cfg),
    )
    if validate:
        validate_spec(spec, registries or _default_registries())
    return spec


def validate_spec(spec: PoolSpec, registries: dict) -> None:
    where = f"池 {spec.name}"
    for fname, params in spec.filters.items():
        registries["filters"].get(fname)
        registries["filters"].check_params(fname, params, f"{where}.filters.{fname}")
    registries["scorers"].check_params(spec.scorer["name"], spec.scorer_params, f"{where}.scorer")
    registries["strategies"].check_params(spec.strategy["name"], spec.strategy_params, f"{where}.strategy")


def load_pool(name: str, pools_dir: Path | None = None, global_path: Path | None = None,
              registries: dict | None = None, validate: bool = True) -> PoolSpec:
    d = pools_dir or POOLS_DIR
    path = d / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"未找到股池配置 {path}，可选: {list_pools(d)}")
    return build_pool_spec(name, _read_yaml(path), load_global(global_path), registries, validate)


# ---------------------------------------------------------------- 旧接口（待删除）
CONFIG_PATH = CONFIG_DIR / "selector.yaml"


def load_config(path: str | Path | None = None, validate: bool = True) -> dict:
    """读取旧 selector.yaml；validate=True 时校验因子名已注册。"""
    p = Path(path) if path else CONFIG_PATH
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if validate:
        _validate_factors(cfg)
    return cfg


def _validate_factors(cfg: dict) -> None:
    from .factors.registry import FACTORS  # 延迟导入避免循环依赖

    for pool in ("stable", "short"):
        weights = cfg.get(pool, {}).get("factor_weights", {})
        unknown = [k for k in weights if k not in FACTORS]
        if unknown:
            raise KeyError(
                f"config {pool}.factor_weights 含未注册因子: {unknown}"
                f"（请在 src/factors/registry.py 的 FACTORS 中注册）"
            )
