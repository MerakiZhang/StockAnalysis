"""Alpha158 特征 + 标签的 qlib DataHandler 构造。

- 特征：qlib Alpha158DL（158 个价量因子，表达式引擎计算，纯本地行情）；
  补充基本面因子在 P6 经 StaticDataLoader 并入（NestedDataLoader），不写入 cn_data。
- 标签（与成交对齐）：T 日收盘打分、T+1 收盘成交 → h 日标签 = Ref($close,-(h+1))/Ref($close,-1)-1。
- 处理器均为截面处理（无需 fit）：同一 handler 可被所有滚动窗口复用，也不存在 fit 窗口泄漏。
    infer : CSRobustZScoreInplace（= ProcessInf + CSZScoreNorm(robust) + Fillna 的原地融合版）
    learn : DropnaLabel → CSRankNorm(label)
  qlib 原版三个处理器各自按日 groupby 并整表复制，中证1000（276 万行 × 159 列）峰值超出
  本机提交内存上限；融合版直接在 numpy 视图上逐日处理，不复制整表。
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


def _make_inplace_processor():
    import numpy as np
    from qlib.data.dataset.processor import Processor

    class CSRobustZScoreInplace(Processor):
        """逐交易日截面 robust z-score（中位数 / MAD×1.4826，截断 ±3），inf 视为缺失，缺失填 0。

        与 qlib ProcessInf + CSZScoreNorm(method="robust") + Fillna 等价（差异：inf 先置缺失再填 0，
        qlib 是填当日均值后再标准化），但**原地修改**输入表，不做整表复制。
        readonly() 返回 True 让 qlib 跳过防御性复制——必须与 drop_raw=True 搭配（原始数据随即丢弃）。
        """

        def __init__(self, fields_group: str = "feature"):
            self.fields_group = fields_group

        def readonly(self) -> bool:
            return True

        def __call__(self, df):
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            df._consolidate_inplace()
            arr = df.values  # 单一 float32 块时为可写视图
            if not np.shares_memory(arr, df._mgr.blocks[0].values):
                raise RuntimeError("CSRobustZScoreInplace 需要单一数据块（全部列同为 float32）")
            pos = np.flatnonzero(df.columns.get_level_values(0) == self.fields_group)
            cols = slice(pos[0], pos[-1] + 1) if len(pos) and (np.diff(pos) == 1).all() else pos
            codes = df.index.codes[0]
            bounds = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                for s, e in zip(bounds[:-1], bounds[1:]):
                    sub = arr[s:e, cols]
                    sub[~np.isfinite(sub)] = np.nan
                    sub -= np.nanmedian(sub, axis=0)
                    sub /= np.nanmedian(np.abs(sub), axis=0) * 1.4826
                    np.clip(sub, -3, 3, out=sub)
                    if not isinstance(cols, slice):
                        arr[s:e, cols] = sub
            feat = arr[:, cols]
            feat[np.isnan(feat)] = 0.0
            if not isinstance(cols, slice):
                arr[:, cols] = feat
            return df

    return CSRobustZScoreInplace


def infer_processors() -> list:
    return [_make_inplace_processor()(fields_group="feature")]

LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]


FEATURE_SETS = ("alpha158", "fundamental")


def build_handler(universe: str | list[str], start: str, end: str, horizon: int | None,
                  features: tuple[str, ...] | list[str] = ("alpha158",)):
    """horizon=None 时只取特征（推理用）。features 含 "fundamental" 时并入补充基本面特征
    （src/factors/supplementary.py，经 StaticDataLoader + NestedDataLoader 左连接到 Alpha158 索引）。"""
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import NestedDataLoader, StaticDataLoader

    from ..data.market import init_qlib

    unknown = set(features) - set(FEATURE_SETS)
    if unknown or "alpha158" not in features:
        raise ValueError(f"features 须包含 alpha158，可选 {FEATURE_SETS}，收到 {list(features)}")
    init_qlib()
    loader = loader_config(horizon)
    if "fundamental" in features:
        from .supplementary import supplementary_long

        static = StaticDataLoader(config=supplementary_long(universe, start, end))
        loader = NestedDataLoader([loader, static], join="left")
    return DataHandlerLP(
        instruments=universe,
        start_time=start,
        end_time=end,
        data_loader=loader,
        infer_processors=infer_processors(),
        learn_processors=LEARN_PROCESSORS if horizon is not None else [],
        drop_raw=True,  # 处理后丢弃原始数据副本，省内存（infer 处理器不动标签，原始标签可从 DK_I 取）
    )
