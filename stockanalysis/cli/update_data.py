"""update-data：抓取 / 补齐 akshare 基本面缓存（只有补充基本面特征和名称/ST 回退需要，日常选股不依赖）。"""
HELP = "更新基本面缓存（akshare）"


def add_arguments(p) -> None:
    p.add_argument("--start-year", type=int, default=2012)


def run(args) -> None:
    from ..data.fundamental import update_all
    from ..data.market import latest_trade_date
    from ..data.sync import data_version

    print(f"行情数据版本 {data_version()}，最新交易日 {latest_trade_date().date()}（行情由 sync 命令管理）")
    update_all(args.start_year)
    print("done")
