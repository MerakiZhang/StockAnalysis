"""回测区间指标（含成本、复利口径）与区间切分（逐年 / 研发区 / 保留集 / 全区间）。"""
from __future__ import annotations

import pandas as pd


def segments(bt_start: str, in_sample_end: str, bt_end: str) -> list[tuple[str, str, str]]:
    """逐年 + 研发区 / 保留集 + 全区间。返回 (label, start, end) 列表。"""
    b0, b1, ins = pd.Timestamp(bt_start), pd.Timestamp(bt_end), pd.Timestamp(in_sample_end)
    segs: list[tuple[str, str, str]] = []
    for y in range(b0.year, b1.year + 1):
        s, e = max(b0, pd.Timestamp(f"{y}-01-01")), min(b1, pd.Timestamp(f"{y}-12-31"))
        if s <= e:
            segs.append((str(y), str(s.date()), str(e.date())))
    if b0 <= ins:
        segs.append(("研发区", str(b0.date()), str(min(ins, b1).date())))
    if ins < b1:
        segs.append(("保留集", str((ins + pd.Timedelta(days=1)).date()), str(b1.date())))
    segs.append(("全区间", str(b0.date()), str(b1.date())))
    return segs


def segment_metrics(report: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """从 qlib 日频 report 切片算区间指标（含成本，复利口径）。"""
    from qlib.contrib.evaluate import risk_analysis

    seg = report.loc[start:end]
    if len(seg) < 2:
        return {}
    net = seg["return"] - seg["cost"]
    excess = net - seg["bench"]
    # product 模式：复利口径（年化为几何年化、回撤不会超过 -100%）；qlib 默认 sum 为简单累加
    ana = risk_analysis(excess, mode="product")["risk"]
    ana_net = risk_analysis(net, mode="product")["risk"]
    return {
        "ann": float(ana_net["annualized_return"]),
        "excess_ann": float(ana["annualized_return"]),
        "mdd": float(ana["max_drawdown"]),
        "ir": float(ana["information_ratio"]),
        "win_rate": float((excess > 0).mean()),
        "turnover": float(seg["turnover"].mean()),
        "cost": float(seg["cost"].sum() / max(len(seg) / 238, 1e-9)),  # 年化成本占比
    }
