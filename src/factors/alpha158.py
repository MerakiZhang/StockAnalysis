"""Alpha158 特征 + 标签的 qlib DataHandler 构造。

- 特征：qlib Alpha158DL（158 个价量因子，表达式引擎计算，纯本地行情）；
  补充基本面因子在 P6 经 StaticDataLoader 并入（NestedDataLoader），不写入 cn_data。
- 标签（与成交对齐）：T 日收盘打分、T+1 收盘成交 → h 日标签 = Ref($close,-(h+1))/Ref($close,-1)-1。
- 处理器均为截面处理（无需 fit）：同一 handler 可被所有滚动窗口复用，也不存在 fit 窗口泄漏。
    infer : ProcessInf → CSZScoreNorm(robust) → Fillna
    learn : DropnaLabel → CSRankNorm(label)
- instruments 传指数名（如 csi300），qlib 按成分文件的进出区间取样，训练样本只含当日成分。
"""
from __future__ import annotations

ALPHA158_CONF = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {},
}
FEATURE_WARMUP_DAYS = 120  # 自然日：Alpha158 最长 60 日窗口的预热


def label_expr(horizon: int) -> str:
    return f"Ref($close, -{horizon + 1}) / Ref($close, -1) - 1"


def feature_config() -> tuple[list[str], list[str]]:
    from qlib.contrib.data.loader import Alpha158DL

    return Alpha158DL.get_feature_config(ALPHA158_CONF)


def loader_config(horizon: int | None) -> dict:
    fields, names = feature_config()
    config = {"feature": (fields, names)}
    if horizon is not None:
        config["label"] = ([label_expr(horizon)], ["LABEL0"])
    return {"class": "QlibDataLoader", "module_path": "qlib.data.dataset.loader",
            "kwargs": {"config": config, "freq": "day"}}


INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature", "method": "robust"}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]
LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]


def build_handler(universe: str | list[str], start: str, end: str, horizon: int | None):
    """horizon=None 时只取特征（推理用）。"""
    from qlib.data.dataset.handler import DataHandlerLP

    from ..data.market import init_qlib

    init_qlib()
    return DataHandlerLP(
        instruments=universe,
        start_time=start,
        end_time=end,
        data_loader=loader_config(horizon),
        infer_processors=INFER_PROCESSORS,
        learn_processors=LEARN_PROCESSORS if horizon is not None else [],
        drop_raw=True,  # 处理后丢弃原始数据副本，省内存（infer 处理器不动标签，原始标签可从 DK_I 取）
    )
