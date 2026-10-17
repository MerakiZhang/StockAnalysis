"""实盘跟踪段：规则改动后不改写已发布的实盘记录。

data/pool/<name>/segments.json（入库，与每日股池同目录）：
    [{"start": "2026-10-15", "hash": "<规则哈希>", "config": {合并后的完整池配置}}, ...]

- 首次选股时自动建立第一段：start = points.live_start，config = 当时的配置。
- 之后改了影响选股的配置（规则哈希变化），选股会报错，须显式开新段：
  `track --pool <名> --new-segment <信号日>`，新配置从该信号日起生效。
- 推演（每日选股与实盘跟踪共用 live_chain）：逐段用该段存档的配置模拟；后一段以前一段最后一天收盘后的持仓
  为起始持仓（换规则不清仓；前一段最后一天定下的次日计划作废，由新规则重新决策）。
规则哈希不含 enabled / status / data（数据同步设置）与 points.live_start：这些不改变选股结果。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pandas as pd

from .store import pool_dir

RULE_EXCLUDE = ("enabled", "status", "data")


@dataclass
class Leg:
    """一段推演：该段配置、信号、模拟结果。"""
    start: pd.Timestamp
    pool: object
    ps: object
    sim: object


def rules_hash(cfg: dict) -> str:
    from ..config.loader import config_hash

    c = copy.deepcopy(cfg)
    for k in RULE_EXCLUDE:
        c.pop(k, None)
    (c.get("points") or {}).pop("live_start", None)
    return config_hash(c)


def seg_path(name: str):
    return pool_dir(name) / "segments.json"


def load_segments(name: str) -> list[dict]:
    p = seg_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_segments(name: str, segs: list[dict]) -> None:
    p = seg_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(segs, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8", newline="")


def ensure_segments(pool) -> list[dict]:
    """读取跟踪段；没有则按 live_start 建第一段；当前规则与最后一段不同则报错。"""
    segs = load_segments(pool.name)
    h = rules_hash(pool.raw)
    if not segs:
        if not pool.points.live_start:
            return []
        segs = [{"start": str(pool.points.live_start), "hash": h, "config": pool.raw}]
        save_segments(pool.name, segs)
    elif segs[-1]["hash"] != h:
        raise RuntimeError(
            f"池 {pool.name} 的规则与当前跟踪段（{segs[-1]['start']} 起）不同。确认启用新规则：运行 "
            f"`python -X utf8 -m stockanalysis track --pool {pool.name} --new-segment <信号日>`；否则还原配置。")
    return segs


def new_segment(pool, start: str) -> list[dict]:
    """当前配置从信号日 start 起生效（追加一段）。"""
    segs = load_segments(pool.name)
    if not segs:
        raise ValueError(f"池 {pool.name} 尚无跟踪段：先运行 select（按 points.live_start 建立第一段）")
    h = rules_hash(pool.raw)
    if segs[-1]["hash"] == h:
        raise ValueError(f"池 {pool.name} 的规则与当前跟踪段相同，无需开新段")
    if pd.Timestamp(start) <= pd.Timestamp(segs[-1]["start"]):
        raise ValueError(f"新段起点 {start} 须晚于当前段起点 {segs[-1]['start']}")
    segs.append({"start": str(pd.Timestamp(start).date()), "hash": h, "config": pool.raw})
    save_segments(pool.name, segs)
    return segs


def segment_pools(pool, segs: list[dict]) -> list:
    """各段存档配置 → PoolSpec（最后一段与当前配置规则相同，直接用当前 pool 以保留 status 等显示项）。"""
    from ..config.loader import build_pool_spec

    out = []
    for k, seg in enumerate(segs):
        same = k == len(segs) - 1 and seg["hash"] == rules_hash(pool.raw)
        out.append(pool if same else build_pool_spec(pool.name, seg["config"]))
    return out


def live_chain(pool, t: pd.Timestamp, segs: list[dict] | None = None) -> list[Leg]:
    """从第一段起点推演到信号日 t，返回各段的推演结果（最后一个 Leg 即 t 日状态）。

    segs 为 None 时读取 / 建立跟踪段；没有跟踪段（未设 live_start）时从 t 当日建仓。"""
    from ..data.market import trade_calendar
    from .points import build_point_signal
    from .positions import simulate

    t = pd.Timestamp(t)
    segs = ensure_segments(pool) if segs is None else segs
    if not segs:
        segs = [{"start": str(t.date()), "hash": rules_hash(pool.raw), "config": pool.raw}]
    starts = [pd.Timestamp(s["start"]) for s in segs]
    if starts[0] > t:
        raise ValueError(f"跟踪起点 {starts[0].date()} 晚于信号日 {t.date()}")
    pools = segment_pools(pool, segs)
    legs: list[Leg] = []
    carry = None
    for k, (s, p) in enumerate(zip(starts, pools)):
        if s > t:
            break
        nxt = starts[k + 1] if k + 1 < len(starts) else None
        if nxt is not None and nxt <= t:
            cal = trade_calendar(str(s.date()), str(t.date()))
            e = cal[cal < nxt][-1]
        else:
            e = t
        ps = build_point_signal(p, str(s.date()), str(e.date()))
        init = {c: pos for c, pos in (carry or {}).items() if c in ps.colmap}
        sim = simulate(ps, p.points, initial=init)
        legs.append(Leg(s, p, ps, sim))
        carry = sim.holdings[-1]
    return legs
