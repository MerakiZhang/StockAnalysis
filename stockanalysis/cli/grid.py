"""grid：换仓规则 × 分数平滑的参数网格（默认只在研发区）。"""
from .common import (add_holdout_arg, add_pool_arg, add_set_arg, add_sync_args, add_tag_arg, int_list, preflight,
                     resolve_pools, str_list)

HELP = "参数敏感性网格"


def add_arguments(p) -> None:
    add_pool_arg(p)
    p.add_argument("--n-drop", default="1")
    p.add_argument("--hold-thresh", default="3,10,20")
    p.add_argument("--score-ema", default="0", help="0 = 不平滑")
    p.add_argument("--freq", default="daily", help="daily,weekly,monthly")
    add_set_arg(p)
    add_holdout_arg(p)
    add_tag_arg(p)
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..research.grid import run_grid
    from ..research.reports import header, write_report

    for name in resolve_pools(args.pool):
        df = run_grid(name, int_list(args.n_drop), int_list(args.hold_thresh), int_list(args.score_ema),
                      str_list(args.freq), args.set, args.show_holdout)
        span = "研发区+保留集" if args.show_holdout else "研发区"
        body = df.to_markdown(index=False, floatfmt=".4f") if not df.empty else "全部失败"
        write_report(f"grid_{name}", header(f"参数网格：{name}（{span}）") + [body, ""], args.tag)
