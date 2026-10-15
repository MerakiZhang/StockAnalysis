"""推理：训练期内取滚动样本外预测；之后的日期用最近一期模型对新数据打分（外推）。"""
from __future__ import annotations

import pandas as pd

from .store import feature_sets, load_latest_model, load_manifest, load_predictions

MAX_EXTRAPOLATE_DAYS = 250  # 外推超过一年应重新训练


def predict(pool, kind: str, codes: list[str], dates: pd.DatetimeIndex, log=print) -> pd.DataFrame:
    """日频预测宽表（index=交易日, columns=代码）。"""
    from qlib.data.dataset import DatasetH

    from ..features.alpha158 import FEATURE_WARMUP_DAYS, build_handler

    manifest, d = load_manifest(pool, kind)
    pred = load_predictions(d)
    dates = pd.DatetimeIndex(dates)
    missing = dates[dates > pred.index.get_level_values(0).max()]
    if len(missing):
        model = load_latest_model(d)
        log(f"[{pool.name}/{kind}] 用 {manifest['windows'][-1]['window']} 模型外推 {len(missing)} 个交易日"
            f"（{missing[0].date()} ~ {missing[-1].date()}）")
        if len(missing) > MAX_EXTRAPOLATE_DAYS:
            log("[warn] 外推超过一年，请重新训练：python -X utf8 -m stockanalysis train --pool " + pool.name)
        start = str((missing[0] - pd.Timedelta(days=FEATURE_WARMUP_DAYS)).date())
        handler = build_handler(pool.universe, start, str(missing[-1].date()), None, feature_sets(pool))
        ds = DatasetH(handler, segments={"test": (str(missing[0].date()), str(missing[-1].date()))})
        pred = pd.concat([pred, model.predict(ds, segment="test")])
    wide = pred.unstack(level=-1)
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.reindex(index=dates, columns=codes)
