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
    """滚动训练模型打分：读取 train 命令产出的样本外预测，之后的日期用最近一期模型外推。

    构造参数为模型超参（覆盖 global.yaml model.<kind> 与内置默认值），会计入训练签名。
    """

    kind = ""
    warmup_days = 0  # 特征预热在模型推理内部处理

    def __init__(self, **params):
        self.params = params

    def predict(self, pool, ctx: dict) -> pd.DataFrame:
        from .inference import predict

        return predict(pool, self.kind, ctx["codes"], ctx["dates"])


@SCORERS.register("lgbm")
class LGBScorer(_RollingModelScorer):
    """qlib LGBModel（LightGBM）+ Alpha158，walk-forward。"""

    kind = "lgbm"


@SCORERS.register("linear")
class LinearScorer(_RollingModelScorer):
    """qlib LinearModel（默认 ridge）+ Alpha158，walk-forward；作为 lgbm 的对照基线。"""

    kind = "linear"


@SCORERS.register("multifactor")
class MultiFactorScorer:
    """多因子优先分：Σ 权重 × 当日横截面百分位排名（因子名见 features/conditions.py 的 CONTINUOUS）。

    权重为负表示越小越好（如 vol20: -1 = 低波动优先）。缺失值按中位（0.5）处理，不因缺数据被排到最前或最后。
    用途：买卖点池里给满足买点的候选排队（空位不够时先买谁）、以及 rank_exit 卖出。
    """

    warmup_days = 420  # 自然日：覆盖 250 日均线 / 年度动量

    def __init__(self, factors: dict[str, float]):
        if not factors:
            raise ValueError("multifactor 需要至少一个因子")
        self.factors = {k: float(v) for k, v in factors.items()}

    def predict(self, pool, ctx: dict) -> pd.DataFrame:
        from ..features.conditions import CONTINUOUS

        unknown = set(self.factors) - set(CONTINUOUS)
        if unknown:
            raise KeyError(f"multifactor 未知因子 {sorted(unknown)}，可选: {sorted(CONTINUOUS)}")
        total = None
        for name, w in self.factors.items():
            x = CONTINUOUS[name](ctx).reindex_like(ctx["close"])
            r = x.rank(axis=1, pct=True).fillna(0.5) * w
            total = r if total is None else total + r
        return total / sum(abs(w) for w in self.factors.values())
