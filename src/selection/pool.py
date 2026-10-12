"""股池生成、对比与简报输出。

因子按 config 的 factor_weights 从注册表（factors.registry）取，
权重为 0 的因子不计算；行情宽表经 market_ctx 只加载一次，
同时供因子计算与过滤器使用。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT
from ..data import fundamental as fd
from ..data.market import list_instruments
from ..factors.registry import market_ctx, weighted_factor_tables
from .scoring import combine
from .universe import build_filter_mask, is_stock

POOL_DIR = PROJECT_ROOT / "data" / "pool"


def _name_map() -> dict[str, str]:
    """qlib 代码 -> 股票名称（用最新 yjbb + 当日快照合并）。"""
    names: dict[str, str] = {}
    try:
        yj = fd.load_yjbb_all().sort_values("period")
        for code, name in zip(yj["qlib_code"], yj["name"]):
            names[code] = name
    except FileNotFoundError:
        pass
    try:
        spot = fd.fetch_spot()
        for code, name in zip(spot["code"], spot["name"]):
            names.setdefault("SH" + code if code.startswith(("60", "68")) else
                             "SZ" + code if code.startswith(("00", "30")) else
                             "BJ" + code, name)
    except Exception:  # noqa: BLE001
        pass
    return names


def _st_codes() -> set[str]:
    """当前 ST 股票集合。快照接口失败时回退到 yjbb 最新名称判断。"""
    try:
        spot = fd.fetch_spot()
        st = spot[spot["is_st"]]["code"]
        return {fd.normalize_code(c) for c in st}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] spot 快照失败，ST 名单回退到 yjbb 名称: {e}")
    try:
        yj = fd.load_yjbb_all().sort_values("period").drop_duplicates("qlib_code", keep="last")
        return set(yj[yj["name"].str.contains("ST", case=False, na=False)]["qlib_code"])
    except Exception:  # noqa: BLE001
        return set()


def stable_scores(date: str | pd.Timestamp, cfg: dict) -> pd.Series:
    """稳健池：基本面 + 技术面多因子总分（date 当日截面）。"""
    date = pd.Timestamp(date)
    scfg = cfg["stable"]
    codes = list(list_instruments(scfg["universe"], start="2019-01-01", end=str(date.date())))
    win_start = str((date - pd.Timedelta(days=450)).date())
    end = str(date.date())

    ctx = market_ctx(codes, win_start, end)
    factors = weighted_factor_tables(scfg["factor_weights"], codes, win_start, end, ctx)
    total = combine(factors, scfg["factor_weights"])

    mask = build_filter_mask(
        date, codes, ctx["close"], ctx["volume"], ctx["amount"],
        st_names=_st_codes(),
        cfg={
            "exclude_st": cfg["filters"]["exclude_st"],
            "min_list_days": scfg["min_list_days"],
            "exclude_suspend_days": cfg["filters"]["exclude_suspend_days"],
            "exclude_limit_on_rebalance": cfg["filters"]["exclude_limit_on_rebalance"],
            "boards": scfg.get("boards"),
        },
    )
    score = total.loc[:date].iloc[-1]
    return score[score.index.isin(mask[mask].index)].dropna().sort_values(ascending=False)


def short_scores(date: str | pd.Timestamp, cfg: dict) -> pd.Series:
    """短线池：技术面/情绪因子总分（date 当日截面）。

    IC 检验结论：追涨方向全为负 IC，短线只做超跌反转/缩量低吸。
    """
    date = pd.Timestamp(date)
    scfg = cfg["short"]
    codes = list(list_instruments(scfg["universe"], start=str((date - pd.Timedelta(days=30)).date()), end=str(date.date())))
    codes = [c for c in codes if is_stock(c)]
    win_start = str((date - pd.Timedelta(days=200)).date())
    end = str(date.date())

    ctx = market_ctx(codes, win_start, end)
    factors = weighted_factor_tables(scfg["factor_weights"], codes, win_start, end, ctx)
    total = combine(factors, scfg["factor_weights"])

    mask = build_filter_mask(
        date, codes, ctx["close"], ctx["volume"], ctx["amount"],
        st_names=_st_codes(),
        cfg={
            "exclude_st": cfg["filters"]["exclude_st"],
            "min_list_days": scfg["min_list_days"],
            "exclude_suspend_days": cfg["filters"]["exclude_suspend_days"],
            "exclude_limit_on_rebalance": cfg["filters"]["exclude_limit_on_rebalance"],
            "min_amount": scfg["min_amount"],
            "chase_limit": scfg["chase_limit"],
            "falling_knife": scfg.get("falling_knife"),
            "boards": scfg.get("boards"),
        },
    )
    score = total.loc[:date].iloc[-1]
    return score[score.index.isin(mask[mask].index)].dropna().sort_values(ascending=False)


def build_pool(score: pd.Series, topk: int, names: dict[str, str]) -> pd.DataFrame:
    pool = score.head(topk).rename("score").to_frame()
    pool.index.name = "code"
    pool.insert(0, "name", pool.index.map(lambda c: names.get(c, "")))
    return pool.reset_index()


def diff_pool(prev: pd.DataFrame | None, cur: pd.DataFrame) -> dict[str, list[str]]:
    if prev is None or prev.empty:
        return {"new": list(cur["code"]), "out": [], "kept": []}
    p, c = set(prev["code"]), set(cur["code"])
    return {
        "new": sorted(c - p),
        "out": sorted(p - c),
        "kept": sorted(p & c),
    }


def save_pool(pool: pd.DataFrame, pool_type: str, date: pd.Timestamp,
              diff: dict[str, list[str]], names: dict[str, str]) -> Path:
    """写股池 csv + 追加简报 md。返回简报路径。"""
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    tag = date.strftime("%Y-%m") if pool_type == "stable" else date.strftime("%Y-%m-%d")
    csv_path = POOL_DIR / f"{pool_type}_{tag}.csv"
    pool.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = POOL_DIR / f"{pool_type}_{tag}_brief.md"
    def fmt(codes: list[str]) -> str:
        return "、".join(f"{names.get(c, '')}({c})" for c in codes) or "无"
    lines = [
        f"# {'稳健池' if pool_type == 'stable' else '短线池'} {tag} 调仓简报",
        "",
        f"- 池内数量: {len(pool)}",
        f"- 新进: {fmt(diff['new'])}",
        f"- 剔除: {fmt(diff['out'])}",
        f"- 留存: {len(diff['kept'])} 只",
        "",
        "## 本期清单",
        "",
        pool.to_markdown(index=False),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def latest_pool(pool_type: str) -> pd.DataFrame | None:
    files = sorted(POOL_DIR.glob(f"{pool_type}_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1], dtype={"code": str})
