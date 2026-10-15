"""compare：模型 A/B 对比（RankIC + TopK 未来超额 + 组合回测）。"""
from .common import add_holdout_arg, add_pool_arg, add_sync_args, add_tag_arg, preflight, resolve_pools

HELP = "模型 A/B 对比"


def add_arguments(p) -> None:
    add_pool_arg(p)
    p.add_argument("--variant", action="append", default=None,
                   help="方案，可重复：名称=点路径覆盖;点路径覆盖（默认 lgbm / lgbm_fund / linear）")
    add_holdout_arg(p)
    add_tag_arg(p)
    add_sync_args(p)


def run(args) -> None:
    preflight(args)
    from ..research.compare import DEFAULT_VARIANTS, compare_pool
    from ..research.reports import header, write_report

    for name in resolve_pools(args.pool):
        lines = header(f"模型对比：{name}") + compare_pool(name, args.variant or DEFAULT_VARIANTS, args.show_holdout)
        write_report(f"model_compare_{name}", lines, args.tag)
