"""rounds：LightGBM 固定树数选择（研发区）。"""
from .common import add_pool_arg, add_sync_args, int_list, preflight, resolve_pools

HELP = "LightGBM 树数选择（研发区）"


def add_arguments(p) -> None:
    add_pool_arg(p, required=True)
    p.add_argument("--rounds", default="25,50,100,200,400")
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..research.reports import header, write_report
    from ..research.rounds import run_rounds

    for name in resolve_pools(args.pool):
        write_report(f"lgbm_rounds_{name}",
                     header(f"LightGBM 树数选择：{name}") + run_rounds(name, sorted(int_list(args.rounds))))
