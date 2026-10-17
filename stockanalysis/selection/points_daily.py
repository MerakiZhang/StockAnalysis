"""买卖点池每日选股：按跟踪段（selection/segments.py，第一段起点 = points.live_start，未设置则当日首次建仓）
用同一模拟器推演到信号日 T，输出：

    持仓    T 日收盘后的持仓：买入日 / 买入价 / 最新价 / 收益 / 持有天数 / 当前权重 / 当前卖出线（止损价、移动止盈价、均线价、剩余天数）
    数据    各数据源截止日，未覆盖信号日的标 ⚠（data/freshness.py）
    计划    T+1 收盘执行的计划买入（含买入原因、参考止损价）与计划卖出（含卖出原因）
    观察    接近买点：只差一个买点条件的股票（按优先分取前 10）
    体检    长桥 CLI：入池 / 计划买入股票的当日资金流向、机构评级、近期新闻与公告（只作人工参考）

股价一律显示**不复权价**（= 复权价 / 当日复权因子），与行情软件一致；卖出线按 T 日复权因子折算。
与回测同一实现（selection/positions.simulate），选股结果 = 回测在 T 日的持仓与计划。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store import pool_dir


def _raw(ps, i: int, j: int, adj_price: float) -> float:
    f = ps.extra["factor"][i, j]
    return float(adj_price / f) if np.isfinite(f) and f > 0 else np.nan


def _sell_lines(p, i: int, j: int, ps, spec) -> dict:
    """持仓当前的卖出线（不复权价，按 T 日因子折算）。"""
    out = {}
    if spec.stop_loss is not None:
        out["止损价"] = _raw(ps, i, j, p.entry_price * (1 - spec.stop_loss))
    if spec.trailing_stop is not None:
        out["移动止盈价"] = _raw(ps, i, j, p.peak * (1 - spec.trailing_stop))
    if spec.atr_stop is not None and np.isfinite(p.entry_atr):
        out["ATR止损价"] = _raw(ps, i, j, p.entry_price - spec.atr_stop * p.entry_atr)
    if spec.take_profit is not None:
        out["止盈价"] = _raw(ps, i, j, p.entry_price * (1 + spec.take_profit))
    for name, params in spec.sell.items():
        n = (params or {}).get("n")
        if name == "below_ma" and n and f"ma{n}" in ps.extra:
            out[f"MA{n}"] = _raw(ps, i, j, ps.extra[f"ma{n}"][i, j])
    if spec.max_hold is not None:
        out["剩余天数"] = max(spec.max_hold - p.days, 0)
    return out


def near_buy(ps, spec, i: int, held: set[str], limit: int = 10) -> pd.DataFrame:
    """只差一个条件就满足买点的股票（可买、未持有、不命中 avoid）。"""
    parts = dict(ps.extra["buy_parts"])
    if ps.extra["buy_any_parts"]:
        any_hit = np.zeros(len(ps.codes), bool)
        for h in ps.extra["buy_any_parts"].values():
            any_hit |= h[i]
        parts["buy_any 任一"] = None
    names = list(parts)
    if not names:
        return pd.DataFrame()
    ok = np.stack([(any_hit if parts[n] is None else parts[n][i]) for n in names])   # 条件 × 股票
    miss = (~ok).sum(axis=0)
    cand = ps.eligible[i] & (miss == 1) & ~ps.buy[i]
    for h in ps.extra["avoid_parts"].values():
        cand &= ~h[i]
    js = [j for j in np.nonzero(cand)[0] if ps.codes[j] not in held]
    js = sorted(js, key=lambda j: -np.nan_to_num(ps.score[i, j], nan=-np.inf))[:limit]
    return pd.DataFrame([{"code": ps.codes[j], "缺少条件": names[int(np.argmin(ok[:, j]))],
                          "优先分": ps.score[i, j], "最新价": _raw(ps, i, j, ps.close[i, j])} for j in js])


def _weights(legs) -> dict[str, float]:
    """按各段记账（与回测同口径，段间衔接）得到 T 日收盘各持仓占净值的比例：买入时等权，之后随涨跌漂移。"""
    from ..backtest.points_engine import account

    state = None
    for leg in legs:
        state = account(leg.sim, leg.ps, leg.pool, state=state).attrs["state"]
    nav = state["cash"] + sum(state["value"].values())
    return {c: v / nav for c, v in state["value"].items()} if nav > 0 else {}


def run_points_selection(pool, t: pd.Timestamp, names: dict[str, str], health: bool = True, log=print,
                         tracked: bool = True):
    """tracked=False（命令行 --set 临时改参数）：不读写跟踪段，按当前配置从 live_start 推演，简报注明。"""
    from ..data.sync import data_version
    from .segments import live_chain, rules_hash

    segs = None if tracked else [{"start": str(pool.points.live_start or t.date()), "hash": rules_hash(pool.raw),
                                  "config": pool.raw}]
    legs = live_chain(pool, t, segs)
    start = legs[0].start
    log(f"[{pool.name}] 模拟 {start.date()} ~ {t.date()}，{len(legs)} 个跟踪段（与回测同一实现）")
    pool, ps, sim = legs[-1].pool, legs[-1].ps, legs[-1].sim
    spec = pool.points
    weight = _weights(legs)
    i = len(ps.dates) - 1
    held = sim.holdings[-1]
    pend_sell = dict(sim.pending_sell)
    col = ps.colmap

    rows = []
    for code, p in held.items():
        j = col[code]
        px = _raw(ps, i, j, ps.close[i, j])
        rows.append({"code": code, "name": names.get(code, ""), "状态": "计划卖出" if code in pend_sell else "持有",
                     "买入日": p.entry_date.date(), "买入价": _raw(ps, i, j, p.entry_price), "最新价": px,
                     "收益%": (ps.close[i, j] / p.entry_price - 1) * 100 if np.isfinite(ps.close[i, j]) else np.nan,
                     "持有天数": p.days, "权重%": weight.get(code, np.nan) * 100, **_sell_lines(p, i, j, ps, spec),
                     "买入原因": p.reason, "卖出原因": pend_sell.get(code, "")})
    for code, reason in sim.pending_buy:
        j = col[code]
        px = _raw(ps, i, j, ps.close[i, j])
        row = {"code": code, "name": names.get(code, ""), "状态": "计划买入", "最新价": px,
               "优先分": ps.score[i, j], "买入原因": reason}
        if spec.stop_loss is not None and np.isfinite(px):
            row["止损价"] = px * (1 - spec.stop_loss)   # 以 T 日收盘近似买入价
        rows.append(row)
    table = pd.DataFrame(rows)
    near = near_buy(ps, spec, i, set(held))
    if len(near):
        near.insert(1, "name", near["code"].map(lambda c: names.get(c, "")))

    check = pd.DataFrame()
    if health and len(table):
        from ..data.longbridge_cli import health_check

        codes = [c for c in table["code"] if c in held or c in dict(sim.pending_buy)]
        check = health_check(codes, log=log)
        if len(check):
            check.insert(1, "name", check["code"].map(lambda c: names.get(c, "")))

    from ..data.freshness import freshness, lagging

    recent = [tr for tr in sim.trades if tr.date >= ps.dates[max(0, i - 4)]]
    fresh = lagging(freshness(t))
    if len(fresh):
        log("[warn] 数据滞后：\n" + fresh.to_string(index=False))
    meta = {"data_version": data_version(), "fresh": fresh, "start": start.date(), "tracked": tracked,
            "n_legs": len(legs), "leg_start": legs[-1].start.date(),
            "lag_days": (pd.Timestamp.now().normalize() - t).days,
            "regime_ok": bool(ps.regime_ok[i]), "signal_day": bool(ps.signal_day[i]),
            "n_eligible": int(ps.eligible[i].sum()), "n_buy": int(ps.buy[i].sum())}
    paths = save_points_pool(pool, t, table, near, check, recent, names, meta)
    log(table.to_string(index=False) if len(table) else "（空仓，无计划）")
    log(f"股池: {paths[0]}\n简报: {paths[1]}")
    return table


def _fresh_lines(fresh: pd.DataFrame | None) -> list[str]:
    """简报"数据截止"：全部到位一行带过；有滞后时列出滞后项。"""
    if fresh is None or not len(fresh):
        return ["> 数据截止：各数据源均已覆盖信号日"]
    return ["> ⚠ 数据截止：以下数据未覆盖信号日（滞后部分沿用最近已有数据）", "", _md(fresh)]


def _md(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "（无）"
    return df.to_markdown(index=False, floatfmt=".2f")


def save_points_pool(pool, t: pd.Timestamp, table, near, check, recent, names, meta):
    spec = pool.points
    d = pool_dir(pool.name)
    d.mkdir(parents=True, exist_ok=True)
    tag = t.strftime("%Y-%m-%d")
    csv_path, md_path = d / f"{tag}.csv", d / f"{tag}_brief.md"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    status_line = ("> **状态：production**（可作实盘参考）" if pool.status == "production" else
                   "> **状态：observe — 仅观察跟踪，不建议据此交易**（规则尚在研发区验证，见 RULES.md）")
    hold = table[table["状态"] != "计划买入"] if len(table) else table
    plan = table[table["状态"].isin(["计划买入", "计划卖出"])] if len(table) else table
    trades = pd.DataFrame([{"成交日": tr.date.date(), "code": tr.code, "name": names.get(tr.code, ""),
                            "方向": "买入" if tr.side == "buy" else "卖出", "原因": tr.reason} for tr in recent])
    lines = [
        f"# {pool.name} 股池 {tag}", "", status_line, "",
        f"> 基于 {tag} 收盘数据，计划于其后首个交易日收盘执行（数据滞后 {meta['lag_days']} 个自然日）。",
        f"> 数据版本 {meta['data_version']} · 配置 {pool.config_hash} · 模拟起点 {meta['start']}"
        + (f"（第 {meta['n_legs']} 个跟踪段，自 {meta['leg_start']} 起）" if meta["n_legs"] > 1 else "")
        + f" · 持仓上限 {spec.max_positions} 只（买入时等权"
        + (f"，单票上限 {spec.max_weight:.0%}" if spec.max_weight else "") + "）",
        *([] if meta["tracked"] else ["> ⚠ 临时配置（命令行 --set）：不读写跟踪段，不是实盘记录"]),
        *_fresh_lines(meta.get("fresh")), "",
        f"- 今日{'可以' if meta['regime_ok'] else '不'}开新仓（择时）"
        + ("" if meta["signal_day"] else f"；非 {spec.freq} 检查日，只执行止损类卖点"),
        f"- 可买股票 {meta['n_eligible']} 只，其中满足买点 {meta['n_buy']} 只",
        "- 价格为不复权价；止损 / 止盈线按收盘价判断、次日收盘执行，不是盘中触价；涨停买不进、跌停卖不出则顺延。", "",
        "## 计划（次一交易日执行）", "", _md(plan), "",
        "## 当前持仓与卖出线", "", _md(hold), "",
        "## 近 5 个交易日成交", "", _md(trades), "",
        "## 接近买点（只差一个条件）", "", _md(near), "",
        "## 长桥体检（人工参考，不参与选股）", "",
        "> 资金流向为运行本命令时的长桥实时快照（不一定是信号日当天），新闻 / 公告取最近 3 条。", "", _md(check), "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
