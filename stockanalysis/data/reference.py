"""参考数据：股票名称、当前 ST 名单（akshare 当日快照，失败时回退到业绩报表缓存）。"""
from __future__ import annotations

from . import fundamental as fd


def name_map() -> dict[str, str]:
    """qlib 代码 -> 股票名称（业绩报表历史名称 + 当日快照，快照优先）。"""
    names: dict[str, str] = {}
    try:
        yj = fd.load_yjbb_all().sort_values("period")
        names.update(zip(yj["qlib_code"], yj["name"]))
    except FileNotFoundError:
        pass
    try:
        spot = fd.fetch_spot()
        names.update({fd.normalize_code(c): n for c, n in zip(spot["code"], spot["name"])})
    except Exception:  # noqa: BLE001 - 快照失败时只缺名称，不影响选股
        pass
    return names


def current_st_codes() -> set[str]:
    """当前 ST 股票集合。只有当前快照、没有历史 ST 数据源，因此只能用于"最新一天"的过滤。"""
    try:
        spot = fd.fetch_spot()
        return {fd.normalize_code(c) for c in spot[spot["is_st"]]["code"]}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] spot 快照失败，ST 名单回退到业绩报表名称: {e}")
    try:
        yj = fd.load_yjbb_all().sort_values("period").drop_duplicates("qlib_code", keep="last")
        return set(yj[yj["name"].str.contains("ST", case=False, na=False)]["qlib_code"])
    except Exception:  # noqa: BLE001
        return set()
