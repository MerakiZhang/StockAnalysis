"""points-grid：买卖点池参数网格 + 研发区滚动验证（2018~20 选优 → 2021 检验；2018~21 选优 → 2022 检验）。

例：python -X utf8 -m stockanalysis points-grid --pool short \
        --vary "points.stop_loss=0.05,0.08,null" --vary "points.max_hold=5,10,20"
"""
from .common import add_pool_arg, add_set_arg, add_sync_args, add_tag_arg, preflight, resolve_pools

HELP = "买卖点池参数网格 + 滚动验证（只看研发区）"


def add_arguments(p) -> None:
    add_pool_arg(p, default="long")
    p.add_argument("--vary", action="append", default=[], metavar="KEY=V1,V2",
                   help="点路径与候选值（逗号分隔），可重复；方案为各键取值的笛卡尔积")
    p.add_argument("--target", default="ann", choices=("ann", "excess", "calmar"),
                   help="滚动验证选优目标：绝对年化 / 超额年化 / 年化÷最大回撤")
    p.add_argument("--starts", type=int, default=1, help="多起点个数（每隔约 2 个月一个建仓日），合成净值评估")
    add_set_arg(p)
    add_tag_arg(p)
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..research.points_grid import run_grid
    from ..research.reports import header, write_report

    for name in resolve_pools(args.pool):
        write_report(f"points_grid_{name}", header(f"买卖点参数网格：{name}")
                     + run_grid(name, args.vary, args.set, args.target, args.starts), tag=args.tag)
