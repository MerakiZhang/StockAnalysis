"""check：选股与回测同口径检查（逐日回放），有逻辑差异时退出码为 1。"""
from .common import add_pool_arg, add_set_arg, resolve_pools

HELP = "选股与回测同口径检查"


def add_arguments(p) -> None:
    add_pool_arg(p)
    p.add_argument("--start", default="2022-01-04", help="起始信号日（默认在研发区内）")
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--scorer", default=None, help="覆盖打分器（如 momentum）")
    add_set_arg(p)


def run(args) -> int:
    from ..research.consistency import check_pool
    from ..research.reports import header, write_report

    lines, total = header("选股 / 回测同口径检查"), 0
    for name in resolve_pools(args.pool):
        part, n = check_pool(name, args.start, args.days, args.scorer, args.set)
        lines += part
        total += n
    write_report("consistency_check", lines)
    print(f"逻辑差异合计 {total}")
    return 1 if total else 0
