"""数据更新入口：akshare 基本面缓存（行情由 src/data/sync.py 按 GitHub 版本同步）。

用法：
    python -X utf8 scripts/update_data.py                   # 业绩报表/分红/现金流（2012 起）

估值类特征（EP/BP）由业绩报表 EPS/每股净资产 + 复权价推导（src/factors/supplementary.py）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fundamental as fd
from src.data.market import latest_trade_date
from src.data.sync import data_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2012)
    args = parser.parse_args()

    # 仅提示：行情以 GitHub release 版本为准，不要求对齐最新交易日
    print(f"[1/4] 行情数据版本 {data_version()}，最新交易日 {latest_trade_date().date()}（提示）")

    periods = fd.report_periods(args.start_year)
    print(f"[2/4] 业绩报表 yjbb: {len(periods)} 个报告期")
    for p in periods:
        try:
            df = fd.fetch_yjbb(p)
            print(f"  yjbb {p}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  yjbb {p} FAILED: {e}")

    print("[3/4] 分红送配 fhps")
    for p in periods:
        try:
            df = fd.fetch_fhps(p)
            print(f"  fhps {p}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  fhps {p} FAILED: {e}")

    print("[4/4] 现金流量表 xjll")
    for p in periods:
        try:
            df = fd.fetch_xjll(p)
            print(f"  xjll {p}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  xjll {p} FAILED: {e}")

    print("done")


if __name__ == "__main__":
    main()
