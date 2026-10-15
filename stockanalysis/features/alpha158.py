"""特征集与 qlib DataHandler 构造。

- alpha158：qlib Alpha158DL（158 个价量因子，表达式引擎计算，纯本地行情）；
- fundamental（可选，model.features 开关）：补充基本面特征（features/fundamental.py），
  经 StaticDataLoader + NestedDataLoader 左连接到 Alpha158 索引，不写入 cn_data；
- 标签见 features/labels.py；处理器见 features/processors.py。
instruments 传指数名（如 csi300），qlib 按成分文件的进出区间取样，训练样本只含当日成分。
"""
from __future__ import annotations

from .labels import label_expr
from .processors import LEARN_PROCESSORS, infer_processors

ALPHA158_CONF = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {},
}
FEATURE_WARMUP_DAYS = 120  # 自然日：Alpha158 最长 60 日窗口的预热
FEATURE_SETS = ("alpha158", "fundamental")


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


def build_handler(universe: str | list[str], start: str, end: str, horizon: int | None,
                  features: tuple[str, ...] | list[str] = ("alpha158",)):
    """horizon=None 时只取特征（推理用）。"""
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import NestedDataLoader, StaticDataLoader

    from ..data.market import init_qlib

    unknown = set(features) - set(FEATURE_SETS)
    if unknown or "alpha158" not in features:
        raise ValueError(f"features 须包含 alpha158，可选 {FEATURE_SETS}，收到 {list(features)}")
    init_qlib()
    loader = loader_config(horizon)
    if "fundamental" in features:
        from .fundamental import fundamental_panel

        loader = NestedDataLoader([loader, StaticDataLoader(config=fundamental_panel(universe, start, end))],
                                  join="left")
    return DataHandlerLP(
        instruments=universe,
        start_time=start,
        end_time=end,
        data_loader=loader,
        infer_processors=infer_processors(),
        learn_processors=LEARN_PROCESSORS if horizon is not None else [],
        drop_raw=True,  # 处理后丢弃原始数据副本，省内存（infer 处理器不动标签，原始标签可从 DK_I 取）
    )
