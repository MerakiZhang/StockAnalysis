"""check：防未来检查（数据截止到 T 与延续到之后的模拟，T 日持仓与次日计划须一致），有差异时退出码为 1。"""
from .common import add_pool_arg, add_set_arg, resolve_pools

HELP = "防未来检查（前缀一致性）"


def add_arguments(p) -> None:
    add_pool_arg(p)
    p.add_argument("--start", default="2022-01-04", help="起始信号日（默认在研发区内）")
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--scorer", default=None, help="覆盖打分器（如 momentum）")
    add_set_arg(p)


def run(args) -> int:
    from ..research.consistency import check_pool
    from ..research.reports import header, write_report

    lines, total = header("防未来检查（前缀一致性）"), 0
    for name in resolve_pools(args.pool):
        part, n = check_pool(name, args.start, args.days, args.scorer, args.set)
        lines += part
        total += n
    write_report("consistency_check", lines)
    print(f"逻辑差异合计 {total}")
    return 1 if total else 0
