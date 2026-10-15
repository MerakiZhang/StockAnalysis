"""backtest：组合回测（默认只到研发区 in_sample_end）。"""
from .common import (add_holdout_arg, add_pool_arg, add_set_arg, add_sync_args, add_tag_arg, preflight,
                     resolve_pools)

HELP = "组合回测（默认只看研发区）"


def add_arguments(p) -> None:
    add_pool_arg(p)
    p.add_argument("--scorer", default=None, help="覆盖打分器（如 momentum / random 冒烟）")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    add_set_arg(p)
    add_holdout_arg(p)
    add_tag_arg(p)
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..backtest.runner import backtest_pool, backtest_window
    from ..config import load_pool
    from ..research.reports import backtest_section, header, write_report

    for name in resolve_pools(args.pool):
        override = {"scorer": {"name": args.scorer}} if args.scorer else None
        pool = load_pool(name, override=override, sets=args.set)
        start, end = backtest_window(pool, args.start, args.end, args.show_holdout)
        res = backtest_pool(pool, start, end)
        write_report(f"backtest_{name}", header(f"回测报告：{name}", f"区间 {start} ~ {end}")
                     + backtest_section(pool, res, start, end), args.tag)
