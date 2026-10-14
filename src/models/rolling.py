"""滚动训练（walk-forward）：qlib LGBModel / LinearModel + DatasetH + Recorder。

切分（每个测试区间一个窗口，默认按年）：
    train（train_years 年）→ gap → valid（valid_months 个月，LGBM 早停）→ gap → test
gap = label_horizon + 1 个交易日：标签用到未来 h+1 天的收盘价，不留空档会把下一段价格泄漏进训练。
qlib RollingGen 不支持空档，因此窗口在此生成；训练/预测/记录仍用 qlib 组件。

handler 只构建一次（截面处理器无需 fit），所有窗口、所有模型复用。
产物：
    data/mlruns/                       qlib Recorder（mlflow）：每窗口的参数、模型、预测
    data/models/{pool}/{kind}/pred.parquet     各窗口样本外预测拼接（回测直接读取）
    data/models/{pool}/{kind}/latest_model.pkl 最近一窗口模型（选股推理用）
    data/models/{pool}/{kind}/manifest.json    窗口、recorder id、数据版本、参数 hash、IC 摘要
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "models"
MLRUNS_DIR = PROJECT_ROOT / "data" / "mlruns"
# 新版 mlflow 默认拒绝文件存储后端；本项目单机使用，文件存储足够且产物集中在 data/mlruns/
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
KINDS = ("lgbm", "linear")

# qlib benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml 的参数
LGBM_DEFAULTS = {
    "loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210,
    "num_threads": 16, "early_stopping_rounds": 50, "num_boost_round": 1000,
}
LINEAR_DEFAULTS = {"estimator": "ridge", "alpha": 1.0}


@dataclass
class Window:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]

    @property
    def name(self) -> str:
        return f"test_{self.test[0]}"


def rolling_windows(calendar: pd.DatetimeIndex, first_test: str, end: str, train_years: int = 4,
                    valid_months: int = 6, gap: int = 6, retrain: str = "yearly") -> list[Window]:
    cal = pd.DatetimeIndex(calendar)
    end_ts = min(pd.Timestamp(end), cal[-1])
    freq = {"yearly": "YS", "quarterly": "QS"}[retrain]
    starts = pd.date_range(pd.Timestamp(first_test), end_ts, freq=freq)
    if len(starts) == 0 or starts[0] > pd.Timestamp(first_test):
        starts = pd.DatetimeIndex([pd.Timestamp(first_test)]).append(starts)
    offset = pd.DateOffset(years=1) if retrain == "yearly" else pd.DateOffset(months=3)

    def day(i: int) -> str:
        return str(cal[max(i, 0)].date())

    windows = []
    for ps in starts:
        i = cal.searchsorted(ps)
        if i >= len(cal) or cal[i] > end_ts:
            break
        period_end = (pd.Timestamp(ps.year, ps.month, 1) + offset) - pd.Timedelta(days=1)
        te = cal[cal.searchsorted(min(period_end, end_ts), side="right") - 1]
        ve_i = i - gap - 1
        vs_i = cal.searchsorted(cal[ve_i] - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1))
        tre_i = vs_i - gap - 1
        trs_i = cal.searchsorted(cal[tre_i] - pd.DateOffset(years=train_years) + pd.Timedelta(days=1))
        windows.append(Window(train=(day(trs_i), day(tre_i)), valid=(day(vs_i), day(ve_i)),
                              test=(day(i), str(te.date()))))
    return windows


def model_params(pool, kind: str) -> dict:
    base = LGBM_DEFAULTS if kind == "lgbm" else LINEAR_DEFAULTS
    own = pool.scorer_params if pool.scorer.get("name") == kind else {}
    params = {**base, **pool.model.get(kind, {}), **own}
    if kind == "lgbm":
        params.setdefault("seed", int(pool.model.get("seed", 42)))
    return params


def make_model(kind: str, params: dict):
    if kind == "lgbm":
        from qlib.contrib.model.gbdt import LGBModel
        return LGBModel(**params)
    if kind == "linear":
        from qlib.contrib.model.linear import LinearModel
        return LinearModel(**params)
    raise KeyError(f"unknown model kind {kind}")


def feature_sets(pool) -> list[str]:
    """特征集（global/池 yaml 的 model.features），默认只用 Alpha158。"""
    return list(pool.model.get("features", ["alpha158"]))


def train_signature(pool, kind: str) -> str:
    """影响预测结果的配置：universe / 模型公共参数 / 模型参数 / 特征集。"""
    feats = feature_sets(pool)
    model = {k: v for k, v in pool.model.items() if k != "features"}
    blob = json.dumps({"universe": pool.universe, "model": model, "params": model_params(pool, kind),
                       "features": "alpha158" if feats == ["alpha158"] else "+".join(feats)},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def out_dir(pool_name: str, kind: str, signature: str) -> Path:
    """按训练签名分目录：改模型参数不会覆盖旧版本，便于 A/B。"""
    return MODEL_DIR / pool_name / f"{kind}_{signature}"


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


def train_rolling(pool, kinds=KINDS, first_test: str | None = None, end: str | None = None,
                  log=print) -> dict[str, dict]:
    """对一个池滚动训练各模型，返回 {kind: manifest}。"""
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    from ..data.market import init_qlib
    from ..data.sync import data_version
    from ..factors.alpha158 import build_handler

    init_qlib()
    m = pool.model
    horizon = int(m.get("label_horizon", 5))
    first_test = first_test or pool.backtest.get("start", "2018-01-01")
    cal = pd.DatetimeIndex(D.calendar())
    end = end or str(cal[-1].date())
    windows = rolling_windows(cal, first_test, end, int(m.get("train_years", 4)),
                              int(m.get("valid_months", 6)), horizon + 1, m.get("retrain", "yearly"))
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
        sig = train_signature(pool, kind)
        d = out_dir(pool.name, kind, sig)
        d.mkdir(parents=True, exist_ok=True)
        pred_all.rename("score").to_frame().to_parquet(d / "pred.parquet")
        with open(d / "latest_model.pkl", "wb") as f:
            pickle.dump(model, f)
        ic = daily_ic(pred_all, label)
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
            # LightGBM 拿到的是无列名矩阵（Column_i），按特征顺序映射回 Alpha158 名称
            names = list(handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).columns.get_level_values(-1))
            imp = model.get_feature_importance()
            imp.index = [names[int(str(i).split("_")[-1])] if str(i).startswith("Column_") else i for i in imp.index]
            imp = imp.sort_values(ascending=False)
            manifest["feature_importance_top30"] = {k: float(v) for k, v in imp.head(30).items()}
        (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
        manifests[kind] = manifest
    return manifests


# ------------------------------------------------------------------ 推理
def load_manifest(pool, kind: str) -> tuple[dict, Path]:
    """当前配置（训练签名）对应的训练产物。"""
    sig = train_signature(pool, kind)
    d = out_dir(pool.name, kind, sig)
    p = d / "manifest.json"
    if not p.exists():
        have = sorted(x.name for x in (MODEL_DIR / pool.name).glob(f"{kind}_*")) if (MODEL_DIR / pool.name).exists() else []
        raise FileNotFoundError(f"{pool.name}/{kind} 当前配置（签名 {sig}）尚未训练，已有: {have or '无'}；"
                                f"请先运行 scripts/run_train.py --pool {pool.name}（参数需与选股/回测一致）")
    return json.loads(p.read_text(encoding="utf-8")), d


def predict(pool, kind: str, codes: list[str], dates: pd.DatetimeIndex, log=print) -> pd.DataFrame:
    """日频预测宽表。训练期内取滚动样本外预测；之后的日期用最近一期模型外推。"""
    from qlib.data.dataset import DatasetH

    from ..factors.alpha158 import FEATURE_WARMUP_DAYS, build_handler

    manifest, d = load_manifest(pool, kind)
    pred = pd.read_parquet(d / "pred.parquet")["score"]
    dates = pd.DatetimeIndex(dates)
    last = pred.index.get_level_values(0).max()
    missing = dates[dates > last]
    if len(missing):
        with open(d / "latest_model.pkl", "rb") as f:
            model = pickle.load(f)
        log(f"[{pool.name}/{kind}] 用 {manifest['windows'][-1]['window']} 模型外推 {len(missing)} 个交易日"
            f"（{missing[0].date()} ~ {missing[-1].date()}）")
        if len(missing) > 250:
            log(f"[warn] 外推超过一年，请重新运行 run_train.py")
        start = str((missing[0] - pd.Timedelta(days=FEATURE_WARMUP_DAYS)).date())
        handler = build_handler(pool.universe, start, str(missing[-1].date()), None, feature_sets(pool))
        ds = DatasetH(handler, segments={"test": (str(missing[0].date()), str(missing[-1].date()))})
        pred = pd.concat([pred, model.predict(ds, segment="test")])
    wide = pred.unstack(level=-1)
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.reindex(index=dates, columns=codes)
