"""参考数据：股票名称（akshare 当日快照优先，失败时回退到业绩报表缓存里的名称）。

ST 判定不用这里：历史与当日 ST 都用 baostock 逐日 isST（features/risk.st）。
"""
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

