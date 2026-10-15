"""train：滚动训练（walk-forward）。中证1000 内存大，多池时建议逐池单独运行。"""
import gc

from .common import add_holdout_arg, add_pool_arg, add_set_arg, add_sync_args, preflight, resolve_pools, str_list

HELP = "滚动训练模型（lgbm / linear）"


def add_arguments(p) -> None:
    add_pool_arg(p)
    p.add_argument("--kinds", default="lgbm", help="lgbm,linear")
    p.add_argument("--first-test", default=None, help="首个测试期起点（默认池 backtest.start）")
    add_set_arg(p)
    add_holdout_arg(p)
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..config import load_pool
    from ..models.training import train_rolling
    from ..research.reports import header, train_section, write_report
    from ..runtime import log_rss

    for name in resolve_pools(args.pool):
        pool = load_pool(name, sets=args.set)
        manifests = train_rolling(pool, tuple(str_list(args.kinds)), args.first_test,
                                  log=lambda s: print(s, flush=True))
        log_rss(f"{name} 训练完成")
        write_report(f"train_{name}", header(f"训练报告：{name}") + train_section(pool, manifests, args.show_holdout))
        del manifests
        gc.collect()  # 特征 handler 数 GB，释放后再进入下一池
