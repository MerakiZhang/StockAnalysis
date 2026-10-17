"""event-study：条件事件研究（条件库逐条逐阈值的后续超额 / 胜率 / 逐年稳定性）+ 连续因子 RankIC + 择时分组。"""
from .common import (add_holdout_arg, add_pool_arg, add_set_arg, add_sync_args, add_tag_arg, preflight,
                     resolve_pools, str_list)

HELP = "条件事件研究（MCP 规则翻译 + qlib 因子）"


def add_arguments(p) -> None:
    add_pool_arg(p, default="long")
    add_holdout_arg(p)
    add_set_arg(p)
    add_tag_arg(p)
    add_sync_args(p)
    p.add_argument("--only", type=str_list, default=None, help="只检验这些条件（逗号分隔），默认全部")


def run(args) -> None:
    preflight(args)
    from ..research.event_study import study
    from ..research.reports import header, write_report

    for name in resolve_pools(args.pool):
        write_report(f"event_study_{name}", header(f"条件事件研究：{name}")
                     + study(name, args.show_holdout, args.set, args.only), tag=args.tag)
