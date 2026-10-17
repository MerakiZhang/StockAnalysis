"""track：实盘跟踪报告（data/reports/track_<池>.md），或为改过规则的池开新跟踪段。

    track --pool all                          各池：实盘净值 / 超额、成交、已发布股池核对、与回测同长度区间比较
    track --pool long --new-segment 2026-11-02   当前配置从该信号日起生效（改规则后必须执行，选股才会继续）
"""
HELP = "实盘跟踪报告 / 开新跟踪段"


def add_arguments(p) -> None:
    from .common import add_pool_arg

    add_pool_arg(p)
    p.add_argument("--date", default=None, help="截止信号日，默认数据最新交易日")
    p.add_argument("--new-segment", default=None, metavar="YYYY-MM-DD", help="当前配置从该信号日起生效（追加跟踪段）")
    p.add_argument("--no-backtest", action="store_true", help="不跑回测分布比较（更快）")


def run(args) -> None:
    from ..config import load_pool
    from ..paths import REPORT_DIR
    from ..research.track import track_pool
    from ..selection.segments import new_segment
    from .common import resolve_pools

    for name in resolve_pools(args.pool):
        if args.new_segment:
            segs = new_segment(load_pool(name), args.new_segment)
            print(f"[{name}] 新跟踪段自 {segs[-1]['start']} 起（共 {len(segs)} 段）", flush=True)
            continue
        lines = track_pool(name, args.date, with_backtest=not args.no_backtest)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"track_{name}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines), flush=True)
        print(f"报告: {path}", flush=True)
