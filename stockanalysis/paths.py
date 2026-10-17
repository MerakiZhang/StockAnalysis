"""项目内所有目录与文件位置（唯一定义处）。

    config/global.yaml          系统级设置：资金、板块权限、成本、回测窗口、数据同步
    config/pool_defaults.yaml   所有股池共用的选股规则（打分器占位、过滤规则）
    config/pools/<name>.yaml    单个股池：只写与默认不同的项
    data/pool/<name>/           股池输出（入库，实盘跟踪记录）
    data/reports/               研究报告（不入库）
    data/raw/                   可重新下载的原始缓存（不入库），统一 <数据集>/<单元>.parquet
        akshare/                基本面（按报告期）、事件（两融 / 龙虎榜 / 股东户数 / 解禁 / 增减持 / 回购 / 公告）、当日快照
        baostock/               逐股逐日 ST / 停牌 / 估值 / 换手；行业快照
        longbridge/             长桥市场温度
    data/panel/                 由 raw 生成的紧凑表（不入库，可随时重建，见 data/panel.py）
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
REPORT_DIR = DATA_DIR / "reports"
RAW_DIR = DATA_DIR / "raw"
AKSHARE_DIR = RAW_DIR / "akshare"
BAOSTOCK_DIR = RAW_DIR / "baostock"
LONGBRIDGE_DIR = RAW_DIR / "longbridge"
PANEL_DIR = DATA_DIR / "panel"
