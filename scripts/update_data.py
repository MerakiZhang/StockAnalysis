"""数据更新入口：基本面缓存 + 数据新鲜度校验。

用法：
    python scripts/update_data.py                # 全部更新
    python scripts/update_data.py --skip-valuation   # 跳过逐股估值（最慢的部分）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fundamental as fd
from src.data.market import latest_trade_date, list_instruments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-valuation", action="store_true", help="跳过逐股估值抓取")
    parser.add_argument("--start-year", type=int, default=2017)
    args = parser.parse_args()

    print(f"[1/5] 行情数据最新交易日: {latest_trade_date()}")

    periods = fd.report_periods(args.start_year)
    print(f"[2/5] 业绩报表 yjbb: {len(periods)} 个报告期")
    for p in periods:
        try:
            df = fd.fetch_yjbb(p)
            print(f"  yjbb {p}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  yjbb {p} FAILED: {e}")

    print("[3/5] 分红送配 fhps")
    for p in periods:
        try:
            df = fd.fetch_fhps(p)
            print(f"  fhps {p}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  fhps {p} FAILED: {e}")

    print("[4/5] 现金流量表 xjll")
    for p in periods:
        try:
            df = fd.fetch_xjll(p)
            print(f"  xjll {p}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  xjll {p} FAILED: {e}")

    if not args.skip_valuation:
        codes = list(list_instruments("csi800", start="2020-01-01"))
        print(f"[5/5] csi800 估值序列: {len(codes)} 只 x 2 指标（最慢，约半小时）")
        failed = fd.fetch_valuation_batch(codes)
        for ind, codes_failed in failed.items():
            print(f"  {ind}: 失败 {len(codes_failed)} 只 -> {codes_failed[:10]}")
    else:
        print("[5/5] 跳过估值抓取")

    print("done")


if __name__ == "__main__":
    main()
