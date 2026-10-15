"""sync：qlib 数据与 GitHub release 版本比对 / 更新 / 标记。"""
HELP = "同步 qlib 数据（GitHub release）"


def add_arguments(p) -> None:
    p.add_argument("--check", action="store_true", help="只比对版本，不下载")
    p.add_argument("--adopt", metavar="TAG", help="把现有数据标记为该版本（不下载）")
    p.add_argument("--strict", action="store_true", help="失败时报错退出")


def run(args) -> None:
    from ..config import load_global
    from ..data.sync import DEFAULT_REPO, adopt, sync_if_needed

    if args.adopt:
        print(adopt(args.adopt))
        return
    data_cfg = load_global().get("data") or {}
    print(sync_if_needed(repo=data_cfg.get("github_repo", DEFAULT_REPO), proxy=data_cfg.get("proxy"),
                         strict=args.strict, check_only=args.check))
