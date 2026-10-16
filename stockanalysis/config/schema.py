"""股池配置结构：PoolSpec 及其分段（池容 / 换仓 / 信号）。

池 yaml（与 config/pool_defaults.yaml、config/global.yaml 深合并后）的顶层键：

    name / enabled / status / universe / benchmark      池身份
    sizing:    {top_pct, min_topk, topk}                池容规则
    rebalance: {freq, n_drop, hold_thresh}              换仓规则（选股与回测共用）
    signal:    {score_ema}                              分数后处理
    scorer:    {name, ...参数}                          打分器（SCORERS 注册表）
    strategy:  {name, ...参数}                          回测策略（STRATEGIES 注册表）
    filters:   {规则名: 参数 | false}                   过滤规则（FILTERS 注册表），false = 关闭默认规则
    account_boards: [main, ...]                         账户可交易板块（global.yaml）；boards 规则未写参数时用它
    backtest / cost / account / exchange / model / data   系统级设置（通常只在 global.yaml）
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

STATUSES = ("production", "observe")   # production = 可作实盘参考；observe = 只生成股池跟踪，不建议交易
REBALANCE_FREQS = ("daily", "weekly", "monthly")
MANDATORY_FILTERS = ("in_universe",)   # 不可关闭
TOP_LEVEL_KEYS = ("name", "enabled", "status", "universe", "benchmark", "sizing", "rebalance", "signal",
                  "scorer", "strategy", "filters", "backtest", "cost", "account", "account_boards", "exchange", "model",
                  "data")


@dataclass(frozen=True)
class Sizing:
    """池容 = topk（若指定）否则 max(min_topk, ceil(top_pct × 当日准入通过只数))。"""
    top_pct: float = 0.01
    min_topk: int = 5
    topk: int | None = None


@dataclass(frozen=True)
class RebalanceRule:
    """换仓规则：freq 调仓频率；n_drop 每次最多因排名调出只数；hold_thresh 持有不足 N 日不因排名调出。"""
    freq: str = "daily"
    n_drop: int = 1
    hold_thresh: int = 1


@dataclass
class PoolSpec:
    name: str
    enabled: bool
    status: str
    universe: str
    benchmark: str
    sizing: Sizing
    rebalance: RebalanceRule
    score_ema: int | None          # 分数 EMA 平滑跨度（交易日），None = 不平滑
    scorer: dict                   # {"name": ..., 其余为打分器参数}
    strategy: dict                 # {"name": ..., 其余为策略参数}
    filters: dict[str, dict]       # 规则名 -> 参数（按合并后顺序）
    backtest: dict
    cost: dict
    account: float
    exchange: dict
    model: dict
    data: dict
    raw: dict = field(repr=False)  # 合并后的完整配置
    config_hash: str = ""

    @property
    def scorer_params(self) -> dict:
        return {k: v for k, v in self.scorer.items() if k != "name"}

    @property
    def strategy_params(self) -> dict:
        return {k: v for k, v in self.strategy.items() if k != "name"}


def section(cls, raw: dict | None, where: str):
    """dict -> 分段 dataclass；拒绝未知键（拼写错误在加载时就报出来）。"""
    raw = dict(raw or {})
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise KeyError(f"{where} 含未知项 {sorted(unknown)}，可用: {sorted(allowed)}")
    return cls(**raw)
