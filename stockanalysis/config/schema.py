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
    points:    {max_positions, buy, sell, ...}          买卖点池（strategy.name = signal_points）的规则，见 PointsSpec
    backtest / cost / account / exchange / model / data   系统级设置（通常只在 global.yaml）
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

STATUSES = ("production", "observe")   # production = 可作实盘参考；observe = 只生成股池跟踪，不建议交易
REBALANCE_FREQS = ("daily", "weekly", "monthly")
MANDATORY_FILTERS = ("in_universe",)   # 不可关闭
TOP_LEVEL_KEYS = ("name", "enabled", "status", "universe", "benchmark", "sizing", "rebalance", "signal",
                  "scorer", "strategy", "filters", "backtest", "cost", "account", "account_boards", "exchange", "model",
                  "data", "points")
POINTS_STRATEGY = "signal_points"      # 买卖点池的策略名：走 selection/positions.py 模拟器，不走 qlib TopkDropout
FILL_MODES = ("strict", "always")
DEAL_PRICES = ("close", "vwap")


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


@dataclass(frozen=True)
class PointsSpec:
    """买卖点规则：买点入池、卖点出池（selection/points.py 构造信号，selection/positions.py 逐日模拟）。

    条件组（值为 {条件名: 参数}，条件名见 features/conditions.py 的 CONDITIONS）：
        buy      全部满足（AND）
        buy_any  至少满足其一（OR）；为空则不要求
        avoid    满足任一则不买（如追高、飞刀）
        sell     满足任一即卖（静态条件，如跌破年线）
    路径规则（依赖持仓状态，None = 不启用）：
        stop_loss      收盘价较买入价下跌 ≥ 该比例
        trailing_stop  收盘价较持有期最高收盘回撤 ≥ 该比例
        atr_stop       收盘价低于 买入价 - N × 买入时 ATR14
        take_profit    收盘价较买入价上涨 ≥ 该比例
        max_hold       持有满 N 个交易日
    其他：
        max_positions  持仓上限（只数，等权：每只 = 买入时净值 / max_positions）
        freq           新买入与条件卖出的检查频率（daily / weekly / monthly）；止损类路径规则每日检查
        fill           strict = 只买满足买点的（不足留现金）；always = 不足时用优先分最高的准入股补满
        rank_exit      优先分在可买股票中的分位跌破该值即卖（如 0.4 = 跌出前 60%）
        industry_max   单一行业最多持有只数
        regime         择时参数（features/regime.py state()）；不满足时不开新仓
        regime_exit    择时不满足时是否清仓
        live_start     实盘跟踪起始信号日：每日选股从这天起用同一模拟器推演到当日
        deal_price     成交价：close（T+1 收盘）/ vwap（T+1 全天均价，更接近盘中分批成交）
    """
    max_positions: int = 10
    freq: str = "daily"
    fill: str = "strict"
    buy: dict = field(default_factory=dict)
    buy_any: dict = field(default_factory=dict)
    avoid: dict = field(default_factory=dict)
    sell: dict = field(default_factory=dict)
    stop_loss: float | None = None
    trailing_stop: float | None = None
    atr_stop: float | None = None
    take_profit: float | None = None
    max_hold: int | None = None
    rank_exit: float | None = None
    industry_max: int | None = None
    regime: dict = field(default_factory=dict)
    regime_exit: bool = False
    live_start: str | None = None
    deal_price: str = "close"

    def groups(self) -> dict[str, dict]:
        return {"buy": self.buy, "buy_any": self.buy_any, "avoid": self.avoid, "sell": self.sell}


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
    points: PointsSpec | None = None   # 买卖点池（strategy.name = signal_points）才有

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
