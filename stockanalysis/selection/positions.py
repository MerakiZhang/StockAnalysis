"""买卖点持仓状态机（纯函数，选股与回测共用同一实现，保证同口径）。

时序（T = 信号日，T+1 = 执行日，与原项目一致：T 日收盘出信号、T+1 收盘成交）：
    第 i 天收盘：
      1. 执行第 i-1 天的计划：先卖（可卖：有成交且未跌停；卖不出则继续持有、当天重新判断），
         再买（可买：有成交且未涨停；买不进则放弃，次日按新信号重新排队）
      2. 更新持仓：持有天数 +1（当天买入的为 0）、持有期最高收盘
      3. 判断：卖点（持有规则 / 静态卖出条件 / 优先分跌出 / 止损 / 移动止盈 / ATR 止损 / 止盈 / 最长持有 / 择时清仓）
         → 空位按优先分从满足买点的股票里补（行业上限、择时、检查频率）→ 作为第 i+1 天的计划
止损类规则用收盘价判断、次日收盘成交，不是盘中触价成交；跳空低开时实际亏损可能大于止损线。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Position:
    code: str
    entry_date: pd.Timestamp     # 成交日（T+1）
    entry_price: float           # 成交价（复权收盘）
    entry_atr: float
    peak: float                  # 持有期最高收盘（复权）
    days: int                    # 已持有交易日（成交当天为 0）
    reason: str                  # 买入原因


@dataclass(frozen=True)
class Trade:
    date: pd.Timestamp           # 成交日
    signal_date: pd.Timestamp    # 决策（信号）日
    code: str
    side: str                    # buy / sell
    price: float
    reason: str


@dataclass
class SimResult:
    dates: pd.DatetimeIndex
    holdings: list[dict[str, Position]]          # 第 i 天收盘、执行完计划后的持仓
    trades: list[Trade]
    pending_buy: list[tuple[str, str]] = field(default_factory=list)    # 最后一天收盘后的计划（次日执行）
    pending_sell: list[tuple[str, str]] = field(default_factory=list)
    unfilled: list[tuple[pd.Timestamp, str, str]] = field(default_factory=list)   # (日期, 代码, buy/sell) 未成交


def sell_reasons(p: Position, i: int, j: int, ps, spec) -> list[str]:
    """第 i 天收盘，持仓 p（列 j）命中的卖点。"""
    out = []
    if not ps.exit_ok[i, j]:
        out.append("持有规则不通过")
    if ps.signal_day[i]:
        out += [name for name, hit in ps.sell_hits.items() if hit[i, j]]
        if spec.rank_exit is not None and np.isfinite(ps.score_pct[i, j]) and ps.score_pct[i, j] < spec.rank_exit:
            out.append(f"优先分跌出前 {1 - spec.rank_exit:.0%}")
    c = ps.close[i, j]
    if np.isfinite(c):
        if spec.stop_loss is not None and c <= p.entry_price * (1 - spec.stop_loss):
            out.append(f"止损 -{spec.stop_loss:.0%}")
        if spec.trailing_stop is not None and c <= p.peak * (1 - spec.trailing_stop):
            out.append(f"移动止盈（高点回撤 {spec.trailing_stop:.0%}）")
        if spec.atr_stop is not None and np.isfinite(p.entry_atr) and c <= p.entry_price - spec.atr_stop * p.entry_atr:
            out.append(f"ATR 止损 {spec.atr_stop:g}×")
        if spec.take_profit is not None and c >= p.entry_price * (1 + spec.take_profit):
            out.append(f"止盈 +{spec.take_profit:.0%}")
    if spec.max_hold is not None and p.days >= spec.max_hold:
        out.append(f"持有满 {spec.max_hold} 天")
    if spec.regime_exit and not ps.regime_ok[i]:
        out.append("择时清仓")
    return out


def buy_reason(i: int, j: int, ps, spec, filler: bool = False) -> str:
    if filler:
        return "补位（优先分）"
    parts = list(spec.buy) + [n for n, hit in ps.buy_hits.items() if hit[i, j]]
    return "买点：" + "、".join(parts) if parts else "买点"


def decide(positions: dict[str, Position], i: int, ps, spec) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """第 i 天收盘的决策：(卖出 [(代码, 原因)], 买入 [(代码, 原因)])，次日执行。"""
    col = ps.colmap
    sells = []
    for code, p in positions.items():
        r = sell_reasons(p, i, col[code], ps, spec)
        if r:
            sells.append((code, "；".join(r)))
    sold = {c for c, _ in sells}
    remain = [c for c in positions if c not in sold]
    slots = spec.max_positions - len(remain)
    buys: list[tuple[str, str]] = []
    if slots <= 0 or not ps.signal_day[i] or not ps.regime_ok[i]:
        return sells, buys

    ind_count: dict[str, int] = {}
    for c in remain:
        k = ps.industry[col[c]]
        if k is not None:
            ind_count[k] = ind_count.get(k, 0) + 1
    held = set(positions)
    chosen: set[str] = set()

    def take(mask: np.ndarray, filler: bool) -> None:
        nonlocal slots
        js = np.nonzero(mask)[0]
        if not len(js):
            return
        sc = ps.score[i, js]
        order = js[np.lexsort((js, -np.nan_to_num(sc, nan=-np.inf)))]   # 分数降序，同分按代码顺序
        for j in order:
            if slots <= 0:
                break
            code = ps.codes[j]
            if code in held or code in chosen:
                continue
            k = ps.industry[j]
            if spec.industry_max is not None and k is not None and ind_count.get(k, 0) >= spec.industry_max:
                continue
            buys.append((code, buy_reason(i, j, ps, spec, filler)))
            chosen.add(code)
            if k is not None:
                ind_count[k] = ind_count.get(k, 0) + 1
            slots -= 1

    take(ps.buy[i], filler=False)
    if spec.fill == "always" and slots > 0:
        take(ps.eligible[i], filler=True)
    return sells, buys


def simulate(ps, spec, initial: dict[str, Position] | None = None) -> SimResult:
    """从 ps.dates[0] 起逐日推演。initial 为起始持仓（默认空仓）。"""
    col = ps.colmap
    positions: dict[str, Position] = dict(initial or {})
    pend_sell: list[tuple[str, str]] = []
    pend_buy: list[tuple[str, str]] = []
    holdings, trades, unfilled = [], [], []
    for i, d in enumerate(ps.dates):
        prev_d = ps.dates[i - 1] if i else d
        # 1. 执行昨日计划
        for code, reason in pend_sell:
            j = col[code]
            if code in positions and ps.can_sell[i, j]:
                trades.append(Trade(d, prev_d, code, "sell", float(ps.fill[i, j]), reason))
                del positions[code]
            elif code in positions:
                unfilled.append((d, code, "sell"))
        bought = set()
        for code, reason in pend_buy:
            j = col[code]
            if len(positions) >= spec.max_positions or code in positions:
                continue
            if ps.can_buy[i, j]:
                px = float(ps.fill[i, j])
                positions[code] = Position(code, d, px, float(ps.atr[i, j]), px, 0, reason)
                trades.append(Trade(d, prev_d, code, "buy", px, reason))
                bought.add(code)
            else:
                unfilled.append((d, code, "buy"))
        # 2. 更新持仓
        for code, p in list(positions.items()):
            if code in bought:
                continue
            c = ps.close[i, col[code]]
            peak = max(p.peak, float(c)) if np.isfinite(c) else p.peak
            positions[code] = replace(p, days=p.days + 1, peak=peak)
        # 3. 决策
        pend_sell, pend_buy = decide(positions, i, ps, spec)
        holdings.append(dict(positions))
    return SimResult(ps.dates, holdings, trades, pend_buy, pend_sell, unfilled)


def round_trips(sim: SimResult) -> pd.DataFrame:
    """配对买卖为完整交易：code, 买入日, 卖出日, 持有天数, 收益（复权价，未扣成本）, 买入原因, 卖出原因。
    期末仍持有的按最后一天收盘计（卖出原因记为"期末持有"）。"""
    open_: dict[str, Trade] = {}
    rows = []
    for t in sim.trades:
        if t.side == "buy":
            open_[t.code] = t
        elif t.code in open_:
            b = open_.pop(t.code)
            rows.append({"code": t.code, "买入日": b.date, "卖出日": t.date, "收益": t.price / b.price - 1,
                         "买入原因": b.reason, "卖出原因": t.reason})
    last = sim.holdings[-1] if sim.holdings else {}
    for code, b in open_.items():
        p = last.get(code)
        if p is None:
            continue
        rows.append({"code": code, "买入日": b.date, "卖出日": pd.NaT, "收益": np.nan,
                     "买入原因": b.reason, "卖出原因": "期末持有"})
    df = pd.DataFrame(rows, columns=["code", "买入日", "卖出日", "收益", "买入原因", "卖出原因"])
    if len(df):
        pos = {d: k for k, d in enumerate(sim.dates)}
        df["持有天数"] = [pos.get(s, len(sim.dates) - 1) - pos[b] if pd.notna(s) else len(sim.dates) - 1 - pos[b]
                       for b, s in zip(df["买入日"], df["卖出日"])]
    return df
