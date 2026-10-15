"""项目内所有目录与文件位置（唯一定义处）。

    config/global.yaml          系统级设置：资金、成本、撮合、回测窗口、模型公共参数、数据同步
    config/pool_defaults.yaml   所有股池共用的选股规则（池容、换仓、打分器、过滤规则）
    config/pools/<name>.yaml    单个股池：只写与默认不同的项
    data/pool/<name>/           股池输出（入库，观察跟踪记录）
    data/models/<name>/         模型产物（不入库）
    data/mlruns/                qlib Recorder（不入库）
    data/reports/               研究报告（不入库）
    data/fundamental/           akshare 基本面缓存（不入库）
qlib 行情数据目录见 data/market.py（默认 ~/.qlib/qlib_data/cn_data，可用 QLIB_DATA 覆盖）。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
GLOBAL_CONFIG = CONFIG_DIR / "global.yaml"
POOL_DEFAULTS = CONFIG_DIR / "pool_defaults.yaml"
POOLS_DIR = CONFIG_DIR / "pools"

DATA_DIR = PROJECT_ROOT / "data"
POOL_OUTPUT_DIR = DATA_DIR / "pool"
MODEL_DIR = DATA_DIR / "models"
MLRUNS_DIR = DATA_DIR / "mlruns"
REPORT_DIR = DATA_DIR / "reports"
FUND_DIR = DATA_DIR / "fundamental"
