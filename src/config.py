"""统一配置加载与校验。

config/selector.yaml 是唯一规则来源；factor_weights 的键必须已在
src/factors/registry.py 的 FACTORS 中注册，否则加载即报错。
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "selector.yaml"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"


def load_config(path: str | Path | None = None, validate: bool = True) -> dict:
    """读取 selector.yaml；validate=True 时校验因子名已注册。"""
    p = Path(path) if path else CONFIG_PATH
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if validate:
        _validate_factors(cfg)
    return cfg


def _validate_factors(cfg: dict) -> None:
    from .factors.registry import FACTORS  # 延迟导入避免循环依赖

    for pool in ("stable", "short"):
        weights = cfg.get(pool, {}).get("factor_weights", {})
        unknown = [k for k in weights if k not in FACTORS]
        if unknown:
            raise KeyError(
                f"config {pool}.factor_weights 含未注册因子: {unknown}"
                f"（请在 src/factors/registry.py 的 FACTORS 中注册）"
            )
