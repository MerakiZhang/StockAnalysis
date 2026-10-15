"""滚动训练：qlib DatasetH + 模型 + Recorder。

handler 只构建一次（截面处理器无需 fit），所有窗口、所有模型复用。
每个窗口在 qlib Recorder（data/mlruns/）留存参数、模型、预测；拼接后的样本外预测与最近模型见 store.py。
"""
from __future__ import annotations

import os
from dataclasses import asdict

import pandas as pd

from ..paths import MLRUNS_DIR
from .estimators import KINDS, make_model, model_params
from .store import feature_sets, model_dir, save_artifacts, train_signature
from .windows import pool_windows

# 新版 mlflow 默认拒绝文件存储后端；本项目单机使用，文件存储足够且产物集中在 data/mlruns/
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def daily_ic(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    """逐日 IC / RankIC（qlib.contrib.eva.alpha.calc_ic）。"""
    from qlib.contrib.eva.alpha import calc_ic

    df = pd.concat({"pred": pred, "label": label}, axis=1).dropna()
    ic, ric = calc_ic(df["pred"], df["label"])
    return pd.DataFrame({"ic": ic, "rank_ic": ric})


def ic_summary(ic: pd.DataFrame) -> dict:
    out = {}
    for col in ("ic", "rank_ic"):
        s = ic[col].dropna()
        out[col] = float(s.mean())
        out[col + "_ir"] = float(s.mean() / s.std()) if s.std() > 0 else float("nan")
    return out


def _feature_importance(model, handler, top: int = 30) -> dict[str, float]:
    """LightGBM 拿到的是无列名矩阵（Column_i），按特征顺序映射回特征名。"""
    from qlib.data.dataset.handler import DataHandlerLP

    names = list(handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).columns.get_level_values(-1))
    imp = model.get_feature_importance()
    imp.index = [names[int(str(i).split("_")[-1])] if str(i).startswith("Column_") else i for i in imp.index]
    return {k: float(v) for k, v in imp.sort_values(ascending=False).head(top).items()}


def train_rolling(pool, kinds=KINDS, first_test: str | None = None, end: str | None = None,
                  log=print) -> dict[str, dict]:
    """对一个池滚动训练各模型，返回 {kind: manifest}。"""
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    from ..data.market import trade_calendar
    from ..data.sync import data_version
    from ..features.alpha158 import build_handler

    horizon = int(pool.model.get("label_horizon", 5))
    first_test = first_test or pool.backtest.get("start", "2018-01-01")
    cal = trade_calendar()
    end = end or str(cal[-1].date())
    windows = pool_windows(pool, cal, first_test, end)
    log(f"[{pool.name}] 特征集 {feature_sets(pool)}；{len(windows)} 个窗口：首个 train {windows[0].train} 末个 test {windows[-1].test}")

    handler = build_handler(pool.universe, windows[0].train[0], end, horizon, feature_sets(pool))
    label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]  # 原始标签（infer 不处理标签）
    log(f"[{pool.name}] 特征 {handler.fetch(col_set='feature', data_key=DataHandlerLP.DK_I).shape}")

    manifests = {}
    for kind in kinds:
        params = model_params(pool, kind)
        preds, rows, model = [], [], None
        for w in windows:
            ds = DatasetH(handler, segments={"train": w.train, "valid": w.valid, "test": w.test})
            model = make_model(kind, params)
            with R.start(experiment_name=f"{pool.name}_{kind}", recorder_name=w.name,
                         uri=MLRUNS_DIR.resolve().as_uri()):
                R.log_params(**{k: str(v) for k, v in {**params, **asdict(w)}.items()})
                model.fit(ds)
                pred = model.predict(ds, segment="test")
                R.save_objects(**{"model.pkl": model, "pred.pkl": pred})
                rid = R.get_recorder().id
            preds.append(pred)
            s = ic_summary(daily_ic(pred, label))
            rows.append({"window": w.name, **asdict(w), "recorder_id": rid, **s})
            log(f"[{pool.name}/{kind}] {w.test[0]}~{w.test[1]} RankIC {s['rank_ic']:.4f} ICIR {s['rank_ic_ir']:.2f}")

        pred_all = pd.concat(preds).sort_index()
        pred_all = pred_all[~pred_all.index.duplicated(keep="last")]
        ic = daily_ic(pred_all, label)
        sig = train_signature(pool, kind)
        manifest = {
            "pool": pool.name, "kind": kind, "signature": sig,
            "data_version": data_version(), "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "label_horizon": horizon, "params": params, "windows": rows,
            "pred_range": [str(pred_all.index.get_level_values(0).min().date()),
                           str(pred_all.index.get_level_values(0).max().date())],
            "ic": ic_summary(ic),
            "ic_dev": ic_summary(ic.loc[:pool.backtest.get("in_sample_end", "2022-12-31")]),
            "ic_by_year": {str(y): ic_summary(g) for y, g in ic.groupby(ic.index.year)},
        }
        if kind == "lgbm":
            manifest["feature_importance_top30"] = _feature_importance(model, handler)
        save_artifacts(model_dir(pool.name, kind, sig), pred_all, model, manifest)
        manifests[kind] = manifest
    return manifests
