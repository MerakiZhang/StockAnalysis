"""命令行公共参数与启动步骤。"""
from __future__ import annotations

import argparse


def str_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def add_pool_arg(p: argparse.ArgumentParser, required: bool = False, default: str | None = "all") -> None:
    from ..config import list_pools

    p.add_argument("--pool", required=required, default=None if required else default,
                   help="池名，逗号分隔；all = 全部 enabled 的池。可选: " + ",".join(list_pools()))


def resolve_pools(text: str) -> list[str]:
    from ..config import list_pools

    if text == "all":
        return list_pools(enabled_only=True)
    names = str_list(text)
    unknown = set(names) - set(list_pools())
    if unknown:
        raise SystemExit(f"未知股池 {sorted(unknown)}，可选: {list_pools()}")
    return names


def add_set_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="点路径覆盖池配置，可重复，如 --set points.stop_loss=0.08 --set points.max_hold=10")


def add_holdout_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--show-holdout", action="store_true",
                   help="延伸到保留集（in_sample_end 之后）。仅用于参数冻结后的最终评估")


def add_tag_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tag", default=None, help="报告文件名后缀（实验用，避免覆盖已有报告）")


def add_sync_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skip-data-check", action="store_true", help="跳过 GitHub 数据版本检查")
    p.add_argument("--strict-data", action="store_true", help="数据同步失败时中止（默认告警后继续）")


def preflight(args: argparse.Namespace):
    """数据版本检查（早于任何 qlib 读取）。按 global.yaml 的 data 段比对 GitHub 最新 release。"""
    from ..config import load_global
    from ..data.sync import DEFAULT_REPO, SyncResult, data_version, sync_if_needed

    data_cfg = load_global().get("data") or {}
    if getattr(args, "skip_data_check", False) or not data_cfg.get("auto_sync", True):
        res = SyncResult("skipped", data_version(), None, "（已跳过检查）")
    else:
        res = sync_if_needed(repo=data_cfg.get("github_repo", DEFAULT_REPO), proxy=data_cfg.get("proxy"),
                             strict=getattr(args, "strict_data", False))
    print(res, flush=True)
    return res
