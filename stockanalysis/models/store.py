"""模型产物存取：按"训练签名"分目录，改参数不覆盖旧版本。

    data/models/{pool}/{kind}_{signature}/
        pred.parquet        各窗口样本外预测拼接（回测直接读取）
        latest_model.pkl    最近一窗口模型（选股推理用）
        manifest.json       窗口、recorder id、数据版本、参数、IC 摘要
训练签名 = universe + 模型公共参数 + 本模型参数 + 特征集；选股/回测按当前配置的签名找模型。
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from ..paths import MODEL_DIR
from .estimators import KINDS, model_params


def feature_sets(pool) -> list[str]:
    """特征集（model.features），默认只用 Alpha158。"""
    return list(pool.model.get("features", ["alpha158"]))


def train_signature(pool, kind: str) -> str:
    feats = feature_sets(pool)
    # 只取公共参数；各模型自己的子段（model.lgbm / model.linear）已体现在 params 里，
    # 不能整段计入，否则改 lgbm 参数会连带改变 linear 的签名
    common = {k: v for k, v in pool.model.items() if k != "features" and k not in KINDS}
    blob = json.dumps({"universe": pool.universe, "model": common, "params": model_params(pool, kind),
                       "features": "alpha158" if feats == ["alpha158"] else "+".join(feats)},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def model_dir(pool_name: str, kind: str, signature: str) -> Path:
    return MODEL_DIR / pool_name / f"{kind}_{signature}"


def save_artifacts(d: Path, pred: pd.Series, model, manifest: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    pred.rename("score").to_frame().to_parquet(d / "pred.parquet")
    with open(d / "latest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_manifest(pool, kind: str) -> tuple[dict, Path]:
    """当前配置（训练签名）对应的训练产物。"""
    sig = train_signature(pool, kind)
    d = model_dir(pool.name, kind, sig)
    p = d / "manifest.json"
    if not p.exists():
        root = MODEL_DIR / pool.name
        have = sorted(x.name for x in root.glob(f"{kind}_*")) if root.exists() else []
        raise FileNotFoundError(f"{pool.name}/{kind} 当前配置（签名 {sig}）尚未训练，已有: {have or '无'}；"
                                f"请先运行 python -X utf8 -m stockanalysis train --pool {pool.name}")
    return json.loads(p.read_text(encoding="utf-8")), d


def load_predictions(d: Path) -> pd.Series:
    return pd.read_parquet(d / "pred.parquet")["score"]


def load_latest_model(d: Path):
    with open(d / "latest_model.pkl", "rb") as f:
        return pickle.load(f)
