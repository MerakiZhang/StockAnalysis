"""打分器注册表 SCORERS。

接口：类，构造参数来自池 yaml 的 scorer 段（name 以外的键），实现
    predict(pool: PoolSpec, ctx: dict) -> pd.DataFrame
返回日频分数宽表（index=交易日, columns=代码，越大越好），覆盖 ctx["dates"] 区间；
ctx 见 selection.signal.build_ctx（含行情宽表、codes、start/end）。

换模型 = 改池 yaml 的 scorer.name；新增模型 = 写一个 @SCORERS.register 类。
momentum / random 为合成打分，仅用于冒烟与回归对照。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import Registry

SCORERS = Registry("scorer")


@SCORERS.register("momentum")
class MomentumScorer:
    """合成打分：N 日涨幅（冒烟用）。"""

    warmup_days = 60  # 自然日

    def __init__(self, window: int = 20):
        self.window = int(window)

    def predict(self, pool, ctx: dict) -> pd.DataFrame:
        return ctx["close"].pct_change(self.window, fill_method=None)


@SCORERS.register("random")
class RandomScorer:
    """合成打分：固定种子随机数（冒烟用）。"""

    warmup_days = 0

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def predict(self, pool, ctx: dict) -> pd.DataFrame:
        close = ctx["close"]
        rng = np.random.default_rng(self.seed)
        return pd.DataFrame(rng.random(close.shape), index=close.index, columns=close.columns)


def make_scorer(pool):
    return SCORERS.get(pool.scorer["name"])(**pool.scorer_params)


class _RollingModelScorer:
    """滚动训练模型打分：读取 run_train.py 产出的样本外预测，之后的日期用最近一期模型外推。

    构造参数为模型超参（覆盖 global.yaml model.<kind> 与内置默认值），会计入训练签名。
    """

    kind = ""
    warmup_days = 0  # 特征预热在模型推理内部处理

    def __init__(self, **params):
        self.params = params

    def predict(self, pool, ctx: dict) -> pd.DataFrame:
        from .rolling import predict

        return predict(pool, self.kind, ctx["codes"], ctx["dates"])


@SCORERS.register("lgbm")
class LGBScorer(_RollingModelScorer):
    """qlib LGBModel（LightGBM）+ Alpha158，walk-forward。"""

    kind = "lgbm"


@SCORERS.register("linear")
class LinearScorer(_RollingModelScorer):
    """qlib LinearModel（默认 ridge）+ Alpha158，walk-forward；作为 lgbm 的对照基线。"""

    kind = "linear"
