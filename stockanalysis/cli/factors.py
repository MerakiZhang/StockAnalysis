"""factors：补充基本面特征体检（覆盖率、RankIC、相关性）。"""
from .common import add_holdout_arg, add_pool_arg, add_sync_args, preflight, resolve_pools

HELP = "补充基本面特征体检"


def add_arguments(p) -> None:
    add_pool_arg(p)
    add_holdout_arg(p)
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..research.factor_check import check_pool
    from ..research.reports import header, write_report

    for name in resolve_pools(args.pool):
        write_report(f"factor_report_{name}",
                     header(f"补充基本面特征体检：{name}") + check_pool(name, args.show_holdout))
