"""命令行入口：python -X utf8 -m stockanalysis <命令> [参数]

每个命令一个模块（HELP / add_arguments / run），业务逻辑在各功能包内，这里只做参数解析与调度。
新增命令 = 新增模块并加入 COMMANDS。
"""
from __future__ import annotations

import argparse
import importlib

COMMANDS = {  # 命令名 -> 模块名
    "select": "select",
    "train": "train",
    "backtest": "backtest",
    "compare": "compare",
    "grid": "grid",
    "rounds": "rounds",
    "factors": "factors",
    "quality": "quality",
    "check": "check",
    "sync": "sync",
    "update-data": "update_data",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -X utf8 -m stockanalysis",
                                     description="A 股指数成分股 LightGBM 选股")
    sub = parser.add_subparsers(dest="command", required=True, metavar="命令")
    modules = {}
    for cmd, mod_name in COMMANDS.items():
        mod = importlib.import_module(f".{mod_name}", __name__)
        modules[cmd] = mod
        mod.add_arguments(sub.add_parser(cmd, help=mod.HELP, description=mod.__doc__))
    args = parser.parse_args(argv)
    return int(modules[args.command].run(args) or 0)
