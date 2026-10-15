"""qlib 数据处理器（截面处理，无需 fit：同一 handler 可被所有滚动窗口复用，也不存在 fit 窗口泄漏）。"""
from __future__ import annotations

import warnings

import numpy as np
from qlib.data.dataset.processor import Processor


class CSRobustZScoreInplace(Processor):
    """逐交易日截面 robust z-score（中位数 / MAD×1.4826，截断 ±3），inf 视为缺失，缺失填 0。

    与 qlib ProcessInf + CSZScoreNorm(method="robust") + Fillna 等价（差异：inf 先置缺失再填 0，
    qlib 是填当日均值后再标准化），但**原地修改**输入表，不做整表复制：qlib 原版三个处理器各自按日
    groupby 并整表复制，中证1000（276 万行 × 159 列）峰值超出本机提交内存上限。
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
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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


LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]


def infer_processors() -> list:
    return [CSRobustZScoreInplace(fields_group="feature")]
