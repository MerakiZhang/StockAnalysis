"""运行时工具：进程内存监控（本机提交内存上限约 32G，中证1000 训练类任务需关注）。"""
from __future__ import annotations

import os


def rss_gb() -> float | None:
    """当前进程 RSS（GB）；未安装 psutil 时返回 None。"""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def log_rss(tag: str = "") -> None:
    gb = rss_gb()
    if gb is not None:
        print(f"[mem] {tag} RSS={gb:.2f} GB", flush=True)
