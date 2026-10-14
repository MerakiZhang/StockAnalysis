"""数据更新入口：akshare 基本面缓存（行情由 src/data/sync.py 按 GitHub 版本同步）。

用法：
    python -X utf8 scripts/update_data.py                   # 业绩报表/分红/现金流（2012 起）
    python -X utf8 scripts/update_data.py --with-valuation  # 另抓百度逐股估值（仅抽样核对用，很慢）

估值类特征（EP/BP）由业绩报表 EPS/每股净资产 + 复权价推导（src/factors/supplementary.py），
不再依赖逐股抓取；百度估值只用于抽样核对。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fundamental as fd
from src.data.market import latest_trade_date, list_instruments
from src.data.sync import data_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-valuation", action="store_true",
                        help="另抓百度逐股估值（所有池 universe 并集，仅抽样核对用，耗时数小时）")
    parser.add_argument("--start-year", type=int, default=2012)
    args = parser.parse_args()

    # 仅提示：行情以 GitHub release 版本为准，不要求对齐最新交易日
    print(f"[1/5] 行情数据版本 {data_version()}，最新交易日 {latest_trade_date().date()}（提示）")

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

    if args.with_valuation:
        from src.config import list_pools, load_pool

        codes = sorted({c for n in list_pools()
                        for c in list_instruments(load_pool(n, validate=False).universe, start=f"{args.start_year}-01-01")})
        print(f"[5/5] 各池 universe 并集估值序列: {len(codes)} 只 x 2 指标（很慢）")
        failed = fd.fetch_valuation_batch(codes)
        for ind, codes_failed in failed.items():
            print(f"  {ind}: 失败 {len(codes_failed)} 只 -> {codes_failed[:10]}")
    else:
        print("[5/5] 跳过百度逐股估值（EP/BP 由业绩报表推导；需要核对时加 --with-valuation）")

    print("done")


if __name__ == "__main__":
    main()
