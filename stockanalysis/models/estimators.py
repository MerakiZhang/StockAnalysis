"""模型构造：qlib LGBModel / LinearModel 及参数合成。"""
from __future__ import annotations

import lightgbm as lgb
from qlib.contrib.model.gbdt import LGBModel

KINDS = ("lgbm", "linear")

# qlib benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml 的参数（global.yaml model.lgbm 可覆盖）
LGBM_DEFAULTS = {
    "loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210,
    "num_threads": 16, "early_stopping_rounds": 50, "num_boost_round": 1000,
}
LINEAR_DEFAULTS = {"estimator": "ridge", "alpha": 1.0}


def model_params(pool, kind: str) -> dict:
    """默认参数 ← global/池 yaml 的 model.<kind> ← 池 scorer 段（仅当 scorer 就是该模型时）。"""
    base = LGBM_DEFAULTS if kind == "lgbm" else LINEAR_DEFAULTS
    own = pool.scorer_params if pool.scorer.get("name") == kind else {}
    params = {**base, **pool.model.get(kind, {}), **own}
    if kind == "lgbm":
        params.setdefault("seed", int(pool.model.get("seed", 42)))
    return params


class FixedRoundLGBModel(LGBModel):
    """固定树数的 LGBModel（不早停）。

    qlib 原版总是挂 early_stopping 回调：验证集 l2 在噪声大的排序标签上常常第 1 棵树后就不再下降，
    模型退化为 1 棵树 → 分数只有少数几个取值、头部大量并列。固定树数（在研发区选定）保证模型不会塌缩。
    """

    def fit(self, dataset, num_boost_round=None, verbose_eval=50, evals_result=None, reweighter=None, **kwargs):
        evals_result = {} if evals_result is None else evals_result
        ds, names = list(zip(*self._prepare_data(dataset, reweighter)))
        self.model = lgb.train(
            self.params, ds[0],
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,
            valid_sets=ds, valid_names=names,
            callbacks=[lgb.log_evaluation(period=verbose_eval), lgb.record_evaluation(evals_result)],
            **kwargs,
        )


def make_model(kind: str, params: dict):
    if kind == "lgbm":
        if int(params.get("early_stopping_rounds", 50) or 0) <= 0:
            return FixedRoundLGBModel(**params)  # early_stopping_rounds ≤ 0：固定树数
        return LGBModel(**params)
    if kind == "linear":
        from qlib.contrib.model.linear import LinearModel
        return LinearModel(**params)
    raise KeyError(f"unknown model kind {kind}")
