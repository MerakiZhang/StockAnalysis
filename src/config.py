"""配置加载与校验。

布局：
    config/global.yaml        全局默认（account/cost/backtest/exchange/model/data）
    config/pools/<name>.yaml  每个股池一个文件；与 global 深合并，池内同名项覆盖全局

加股池 = 复制一份 pools/*.yaml 改参数，不改代码（list_pools() 自动发现）。
加载即校验：过滤规则 / 打分器 / 回测策略的名字必须已注册，参数名必须与其签名匹配。
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
STATUSES = ("production", "observe")  # production = 可用于实盘参考；observe = 只生成股池跟踪，不建议交易


@dataclass
class PoolSpec:
    name: str
    status: str
    universe: str
    benchmark: str
    top_pct: float
    min_topk: int
    topk: int | None
    rebalance: str
    score_ema: int | None     # 分数 EMA 平滑跨度（交易日），None = 不平滑
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
    if cfg.get("status", "observe") not in STATUSES:
        raise ValueError(f"池 '{name}'：status 须为 {STATUSES}")
    if cfg.get("rebalance", "daily") not in REBALANCE_FREQS:
        raise ValueError(f"池 '{name}'：rebalance 须为 {REBALANCE_FREQS}")

    spec = PoolSpec(
        name=name,
        status=cfg.get("status", "observe"),
        universe=cfg["universe"],
        benchmark=cfg["benchmark"],
        top_pct=float(cfg.get("top_pct", 0.01)),
        min_topk=int(cfg.get("min_topk", 5)),
        topk=None if cfg.get("topk") is None else int(cfg["topk"]),
        rebalance=cfg.get("rebalance", "daily"),
        score_ema=None if cfg.get("score_ema") is None else int(cfg["score_ema"]),
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


def parse_sets(sets: list[str] | None) -> dict:
    """["model.lgbm.learning_rate=0.05", "strategy.hold_thresh=10"] -> 嵌套 dict（值按 yaml 解析）。"""
    out: dict = {}
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set 需为 key.path=value 形式: {item!r}")
        path, raw = item.split("=", 1)
        node = out
        keys = path.strip().split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def load_pool(name: str, pools_dir: Path | None = None, global_path: Path | None = None,
              registries: dict | None = None, validate: bool = True,
              override: dict | None = None, sets: list[str] | None = None) -> PoolSpec:
    """override：按顶层键**整段替换**池 yaml（如 {"scorer": {"name": "momentum"}}）；
    sets：点路径**深合并**覆盖（如 ["model.lgbm.learning_rate=0.05"]）。
    用于命令行覆盖与参数网格；之后照常与 global 深合并并校验。"""
    d = pools_dir or POOLS_DIR
    path = d / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"未找到股池配置 {path}，可选: {list_pools(d)}")
    pool_cfg = deep_merge({**_read_yaml(path), **(override or {})}, parse_sets(sets))
    return build_pool_spec(name, pool_cfg, load_global(global_path), registries, validate)
