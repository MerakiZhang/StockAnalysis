"""quality：烂股剔除效果体检（每条规则的命中只数、命中组与保留组的后续收益差与大跌率）。"""
from .common import add_holdout_arg, add_pool_arg, add_set_arg, add_sync_args, add_tag_arg, preflight, resolve_pools

HELP = "烂股剔除效果体检"


def add_arguments(p) -> None:
    add_pool_arg(p, default="long")
    add_holdout_arg(p)
    add_set_arg(p)
    add_tag_arg(p)
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..research.quality_check import check_pool
    from ..research.reports import header, write_report

    for name in resolve_pools(args.pool):
        write_report(f"quality_report_{name}", header(f"烂股剔除效果体检：{name}")
                     + check_pool(name, args.show_holdout, args.set), tag=args.tag)
