"""update-data：抓取 / 补齐本地补充数据缓存（行情由 sync 命令管理）。

    fundamental   akshare 业绩报表 / 分红 / 现金流 / 实际披露日 / 业绩预告（按报告期）
    events        akshare 两融 / 龙虎榜 / 股东户数 / 解禁 / 增减持 / 回购 / 公告（--events 选子集）
    baostock      baostock 逐日 ST / 停牌 / 估值 / 换手率（逐股，首次全量约数小时）+ 行业快照
    market_temp   长桥 CLI 市场温度历史（2018 起）

均为断点续传：已完整的缓存跳过，中断后重跑即可。缓存位于 data/raw/，结束时重建 data/panel/ 派生表。
"""
HELP = "更新补充数据缓存（akshare / baostock / 长桥 CLI）"

SOURCES = ("fundamental", "events", "baostock", "market_temp")


def add_arguments(p) -> None:
    from ..cli.common import str_list
    from ..data.events import DATASETS, EVENTS_START

    p.add_argument("--only", type=str_list, default=list(SOURCES),
                   help=f"数据源，逗号分隔，默认全部: {','.join(SOURCES)}")
    p.add_argument("--events", type=str_list, default=list(DATASETS),
                   help=f"events 的数据集子集，默认全部: {','.join(DATASETS)}")
    p.add_argument("--start-year", type=int, default=2012, help="fundamental 起始报告期年份")
    p.add_argument("--events-start", default=EVENTS_START, help="events 起始日期")
    p.add_argument("--workers", type=int, default=4, help="baostock 并行下载进程数")
    p.add_argument("--status", action="store_true", help="只显示各数据源截止日（对比最新交易日），不下载")


def run(args) -> None:
    import pandas as pd

    from ..data.market import latest_trade_date
    from ..data.sync import data_version

    if args.status:
        print_status()
        return
    unknown = set(args.only) - set(SOURCES)
    if unknown:
        raise SystemExit(f"未知数据源 {sorted(unknown)}，可选 {SOURCES}")
    end = str(latest_trade_date().date())
    print(f"行情数据版本 {data_version()}，最新交易日 {end}（行情由 sync 命令管理）", flush=True)

    def log(msg):
        print(msg, flush=True)

    if "fundamental" in args.only:
        from ..data.fundamental import update_all

        update_all(args.start_year, log=log)
    if "events" in args.only:
        from ..data.events import update_events

        failed = update_events(args.events, start=args.events_start, log=log)
        for name, bad in failed.items():
            if bad:
                log(f"[events] {name} 失败 {len(bad)} 个单元（重跑即补）: {[str(b)[:10] for b in bad[:10]]}")
    if "baostock" in args.only:
        from ..data.baostock_data import DAILY_START, fetch_industry, update_daily_parallel
        from ..data.universe import market_stocks

        codes = market_stocks(DAILY_START, end, exchanges=("SH", "SZ"))
        log(f"[baostock] 逐日数据 {len(codes)} 只，{DAILY_START} ~ {end}")
        bad = update_daily_parallel(codes, end, workers=args.workers, log=log)
        log(f"[baostock] 失败 {len(bad)} 只（重跑即补）: {bad[:20]}")
        try:
            log(f"[baostock] 行业快照 {len(fetch_industry())} 只")
        except Exception as e:  # noqa: BLE001
            log(f"[baostock] 行业快照失败: {e}")
    if "market_temp" in args.only:
        from ..data.longbridge_cli import market_temperature

        try:
            mt = market_temperature("2018-01-01", end)
            log(f"[market_temp] {len(mt)} 天，{mt.index.min().date()} ~ {mt.index.max().date()}")
        except Exception as e:  # noqa: BLE001
            log(f"[market_temp] 失败: {e}")
    refresh_panels(log)
    print_status()
    print(f"done {pd.Timestamp.now():%H:%M:%S}", flush=True)


def print_status() -> None:
    from ..data.freshness import freshness
    from ..data.sync import data_version

    print(f"\n数据截止（行情数据版本 {data_version()}）：", flush=True)
    print(freshness().to_string(index=False), flush=True)


def refresh_panels(log=print) -> None:
    """重建 data/panel 派生表（源数据未变化的直接读取，不重建）。"""
    from ..data.baostock_data import is_st_panel
    from ..features.risk import risk_events

    for name, fn in (("is_st", is_st_panel), ("risk_notice", risk_events)):
        try:
            log(f"[panel] {name}: {len(fn())} 行")
        except Exception as e:  # noqa: BLE001 - 缺源数据时只告警
            log(f"[panel] {name} 失败: {e}")
