"""select：生成当期股池（数据最新交易日），写 data/pool/<池>/<日期>.csv 与简报。"""
from .common import add_pool_arg, add_sync_args, preflight, resolve_pools

HELP = "生成当期股池"


def add_arguments(p) -> None:
    add_pool_arg(p)
    p.add_argument("--date", default=None, help="信号日，默认数据最新交易日")
    p.add_argument("--init", action="store_true", help="忽略上期股池，按纯排名重新建仓")
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..data.reference import name_map
    from ..selection.daily import run_selection

    names = name_map()
    for name in resolve_pools(args.pool):
        run_selection(name, args.date, args.init, names)
