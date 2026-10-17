"""实盘跟踪报告：按跟踪段（selection/segments.py）从第一段起点推演到最新交易日并记账，得到实盘净值；
逐日核对已入库的股池 CSV；与回测中同样长度区间的收益分布比较。

实盘净值 = 各段用**当时存档的配置**推演 + 记账（与每日选股同一实现），所以改规则不会改写旧段；
已入库 CSV 与推演不一致只可能来自数据修订（行情 / 财报 / 公告被补录或更正），逐日列出。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HELD = ("持有", "计划卖出")   # CSV 中 T 日收盘时已持有的状态
MIN_DAYS_FOR_ADVICE = 20      # 实盘不足 20 个交易日不给降级建议


def live_report(pool, t) -> tuple[pd.DataFrame, list]:
    """各段推演 + 记账（段间衔接现金与持仓），返回 (逐日 report, legs)。"""
    from ..backtest.points_engine import _bench_returns, account
    from ..selection.segments import live_chain

    legs = live_chain(pool, t)
    parts, state = [], None
    for leg in legs:
        rep = account(leg.sim, leg.ps, leg.pool, _bench_returns(leg.pool.benchmark, leg.ps.dates), state)
        state = rep.attrs["state"]
        parts.append(rep)
    return pd.concat(parts), legs


def check_published(name: str, legs) -> list[str]:
    """已入库股池（T 日持仓）与推演的 T 日持仓逐日比较，返回差异行。"""
    from ..selection.store import pool_dir

    held_at = {}
    for leg in legs:
        for d, h in zip(leg.sim.dates, leg.sim.holdings):
            held_at[d] = set(h)
    out = []
    for f in sorted(pool_dir(name).glob("????-??-??.csv")):
        d = pd.Timestamp(f.stem)
        if d not in held_at:
            continue
        df = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str})
        pub = set(df.loc[df["状态"].isin(HELD), "code"]) if len(df) else set()
        sim = held_at[d]
        if pub != sim:
            out.append(f"| {d.date()} | 已发布独有 {sorted(pub - sim)} | 重算独有 {sorted(sim - pub)} |")
    return out


def window_distribution(report: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    """回测逐日 report 中所有长度 n 的区间：扣成本收益、超额（相对基准）。"""
    net = np.log1p((report["return"] - report["cost"]).to_numpy())
    bench = np.log1p(report["bench"].to_numpy())
    if len(net) < n or n <= 0:
        return np.array([]), np.array([])
    cs, cb = np.concatenate([[0], np.cumsum(net)]), np.concatenate([[0], np.cumsum(bench)])
    r = np.expm1(cs[n:] - cs[:-n])
    b = np.expm1(cb[n:] - cb[:-n])
    return r, r - b


def track_pool(name: str, t: str | None = None, with_backtest: bool = True) -> list[str]:
    from ..backtest.points_engine import run_points_backtest
    from ..config import load_pool
    from ..selection.daily import signal_date

    pool = load_pool(name)
    t = signal_date(t)
    report, legs = live_report(pool, t)
    start, n = legs[0].start, len(report)
    net = (1 + report["return"] - report["cost"]).prod() - 1
    bench = (1 + report["bench"]).prod() - 1
    lines = [f"# {name} 实盘跟踪 {start.date()} ~ {t.date()}", "",
             f"> 状态 {pool.status} · {n} 个交易日 · {len(legs)} 个跟踪段 · 基准 {pool.benchmark}", "",
             "| 项目 | 数值 |", "|---|---|",
             f"| 实盘收益（扣成本） | {net:+.2%} |", f"| 基准收益 | {bench:+.2%} |", f"| 超额 | {net - bench:+.2%} |",
             f"| 当前持仓 | {int(report['n_pos'].iloc[-1])} 只，现金 {report['cash'].iloc[-1]:.0%} |",
             f"| 累计成本 | {report['cost'].sum():.2%} |", ""]
    lines += ["## 跟踪段", "", "| 起点 | 配置 |", "|---|---|"]
    lines += [f"| {leg.start.date()} | {leg.pool.config_hash} |" for leg in legs]

    trades = [tr for leg in legs for tr in leg.sim.trades]
    lines += ["", "## 成交", "", "| 成交日 | code | 方向 | 原因 |", "|---|---|---|---|"]
    lines += [f"| {tr.date.date()} | {tr.code} | {'买入' if tr.side == 'buy' else '卖出'} | {tr.reason} |" for tr in trades]

    diff = check_published(name, legs)
    lines += ["", "## 已发布股池核对", ""]
    lines += (["一致：已入库的每日持仓与重算相同。"] if not diff else
              ["以下日期已发布持仓与重算不同（来自数据修订）：", "", "| 日期 | 已发布 | 重算 |", "|---|---|---|", *diff])

    if with_backtest:
        bs = str(pool.backtest.get("start", "2018-01-01"))
        be = str((start - pd.Timedelta(days=1)).date())
        bt = run_points_backtest(pool, bs, be)
        r, ex = window_distribution(bt["report"], n)
        lines += ["", f"## 与回测比较（{bs} ~ {be} 中所有 {n} 个交易日的区间，含已看过的保留集，仅作参考）", ""]
        if len(r):
            pr, pe = (r <= net).mean(), (ex <= net - bench).mean()
            lines += [f"- 实盘收益处于回测同长度区间的第 {pr:.0%} 分位（中位 {np.median(r):+.2%}，"
                      f"10%~90% 区间 {np.quantile(r, 0.1):+.2%} ~ {np.quantile(r, 0.9):+.2%}）",
                      f"- 实盘超额处于第 {pe:.0%} 分位（中位 {np.median(ex):+.2%}，"
                      f"10%~90% 区间 {np.quantile(ex, 0.1):+.2%} ~ {np.quantile(ex, 0.9):+.2%}）"]
            if n < MIN_DAYS_FOR_ADVICE:
                lines.append(f"- 实盘仅 {n} 个交易日，样本太短，不作判断。")
            elif pe < 0.1 and pool.status == "production":
                lines.append("- ⚠ 实盘超额低于回测同长度区间的 10% 分位：建议评估是否降为 observe。")
    lines.append("")
    return lines
