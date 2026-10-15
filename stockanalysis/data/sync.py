"""qlib 数据同步：以 GitHub（chenditc/investment_data）最新 release 为准。

规则：本地版本标记 ≠ GitHub 最新 release → 下载并替换；一致则跳过。
不追求对齐最新交易日（GitHub 更新滞后没关系）。

release 结构（2026-10 核实）：tag = 交易日（如 2026-10-14），assets：
    qlib_bin.tar.gz          数据包（约 570MB，顶层一个目录，内含 calendars/features/instruments）
    qlib_bin.manifest.json   {release_tag, target_trade_date, archive_sha256, ...}

替换流程（原子、可回滚）：
    下载到 <父目录>/.sync_tmp → 校验 sha256 → 解压到 cn_data.new → 校验目录结构与日历 →
    cn_data → cn_data.bak（只留一代）→ cn_data.new → cn_data → 写 .version
任一步失败：告警并继续使用旧数据（strict=True 时抛错）。旧数据在替换完成前不被改动。

必须在 init_qlib() 与 listing_dates() 等缓存之前调用。
任何派生数据不得写入 cn_data 目录（替换时会整体丢弃）。

命令行（见 cli/sync.py）：
    python -X utf8 -m stockanalysis sync --check          只比对版本
    python -X utf8 -m stockanalysis sync                  需要时更新
    python -X utf8 -m stockanalysis sync --adopt 2026-10-14   把现有数据标记为该版本
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_REPO = "chenditc/investment_data"
ARCHIVE_NAME = "qlib_bin.tar.gz"
MANIFEST_NAME = "qlib_bin.manifest.json"
VERSION_FILE = ".version"
REQUIRED_INSTRUMENTS = ("all", "csi300", "csi500", "csi1000")


class SyncError(RuntimeError):
    pass


@dataclass
class SyncResult:
    status: str               # up_to_date / updated / adopted / failed / skipped
    local: str | None
    remote: str | None
    message: str = ""

    def __str__(self) -> str:
        return f"[数据同步] 本地={self.local or '无标记'} GitHub={self.remote or '未知'} -> {self.status} {self.message}".rstrip()


# ------------------------------------------------------------------ 网络
def _opener(proxy: str | None):
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    return urllib.request.build_opener(*handlers)


def _http_get_json(url: str, proxy: str | None = None, timeout: float = 20) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "StockAnalysis-sync"})
    with _opener(proxy).open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dest: Path, proxy: str | None = None, timeout: float = 60,
              retries: int = 5, backoff: float = 5.0) -> None:
    """断点续传下载：dest 已有部分内容时用 HTTP Range 续传；连接中断自动重试。

    国内访问 GitHub 常见连接中途断开（实测约 100KB/s，几十 MB 后断流），不续传几乎无法下完 570MB。
    重试用尽仍不完整则抛 SyncError，已下载部分保留在 dest，下次运行继续。
    """
    total: int | None = None
    for attempt in range(retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have >= total:
            return
        headers = {"User-Agent": "StockAnalysis-sync"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with _opener(proxy).open(req, timeout=timeout) as r:
                resumed = have > 0 and getattr(r, "status", 200) == 206
                if not resumed:
                    have = 0  # 服务器不支持 Range 或首次下载：从头写
                length = int(r.headers.get("Content-Length") or 0)
                total = have + length if length else total
                if attempt or have:
                    print(f"  续传：已有 {have / 1e6:.0f} MB（第 {attempt + 1} 次尝试）", flush=True)
                with open(dest, "ab" if resumed else "wb") as f:
                    done, step = have, -1
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total and done * 10 // total > step:
                            step = done * 10 // total
                            print(f"  下载 {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
            size = dest.stat().st_size
            if total is None or size >= total:
                return
        except (OSError, http.client.HTTPException) as e:
            print(f"  [下载中断] {e}", flush=True)
        if attempt < retries:
            time.sleep(min(60.0, backoff * 2 ** attempt))
    have = dest.stat().st_size if dest.exists() else 0
    raise SyncError(f"下载不完整：{have / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB，已保留，下次运行续传")


def github_latest(repo: str = DEFAULT_REPO, proxy: str | None = None) -> dict:
    """最新 release：{tag, archive_url, manifest}。manifest 拉取失败时为 {}。"""
    rel = _http_get_json(f"https://api.github.com/repos/{repo}/releases/latest", proxy)
    assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    if ARCHIVE_NAME not in assets:
        raise SyncError(f"release {rel.get('tag_name')} 缺少 {ARCHIVE_NAME}，assets={list(assets)}")
    manifest: dict = {}
    if MANIFEST_NAME in assets:
        try:
            manifest = _http_get_json(assets[MANIFEST_NAME], proxy)
        except Exception as e:  # noqa: BLE001 - manifest 只用于校验，缺失不致命
            print(f"[数据同步] manifest 获取失败（跳过 sha256 校验）: {e}")
    return {"tag": rel["tag_name"], "archive_url": assets[ARCHIVE_NAME], "manifest": manifest}


# ------------------------------------------------------------------ 本地
def _data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    from .market import data_dir as default_dir
    return Path(default_dir())


def local_version(data_dir: str | Path | None = None) -> dict | None:
    p = _data_dir(data_dir) / VERSION_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def data_version(data_dir: str | Path | None = None) -> str:
    """报告/简报留痕用：本地数据版本 tag，无标记时返回 'unknown'。"""
    v = local_version(data_dir)
    return v.get("tag", "unknown") if v else "unknown"


def calendar_end(data_dir: str | Path) -> pd.Timestamp | None:
    p = Path(data_dir) / "calendars" / "day.txt"
    if not p.exists():
        return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return pd.Timestamp(lines[-1]) if lines else None


def write_version(data_dir: Path, tag: str, manifest: dict | None = None, how: str = "sync") -> None:
    info = {"tag": tag, "how": how, "synced_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    if manifest:
        info.update({k: manifest.get(k) for k in ("target_trade_date", "archive_sha256") if k in manifest})
    (data_dir / VERSION_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def adopt(tag: str, data_dir: str | Path | None = None) -> SyncResult:
    """把现有数据标记为某版本（不下载）。"""
    d = _data_dir(data_dir)
    if not (d / "calendars").exists():
        raise SyncError(f"{d} 不是 qlib 数据目录")
    write_version(d, tag, how="adopt")
    return SyncResult("adopted", tag, tag, f"日历末日 {calendar_end(d).date()}")


# ------------------------------------------------------------------ 替换
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _extract(archive: Path, dest: Path) -> Path:
    """解压到 dest，返回含 calendars/ 的根目录（兼容包内有无顶层目录）。"""
    dest.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python < 3.11.4 无 filter 参数
            tar.extractall(dest)
    if (dest / "calendars").exists():
        return dest
    subs = [p for p in dest.iterdir() if p.is_dir() and (p / "calendars").exists()]
    if len(subs) != 1:
        raise SyncError(f"数据包结构异常：未找到 calendars/（{[p.name for p in dest.iterdir()]}）")
    return subs[0]


def validate_data_dir(new_dir: Path, old_dir: Path | None) -> None:
    for sub in ("calendars", "features", "instruments"):
        if not (new_dir / sub).is_dir():
            raise SyncError(f"新数据缺少 {sub}/")
    if not any((new_dir / "features").iterdir()):
        raise SyncError("新数据 features/ 为空")
    for inst in REQUIRED_INSTRUMENTS:
        if not (new_dir / "instruments" / f"{inst}.txt").exists():
            raise SyncError(f"新数据缺少成分文件 instruments/{inst}.txt")
    new_end = calendar_end(new_dir)
    old_end = calendar_end(old_dir) if old_dir is not None and old_dir.exists() else None
    if new_end is None:
        raise SyncError("新数据日历为空")
    if old_end is not None and new_end < old_end:
        raise SyncError(f"新数据日历末日 {new_end.date()} 早于本地 {old_end.date()}")


def swap_dirs(target: Path, new_root: Path) -> None:
    """target → target.bak，new_root → target；第二步失败时回滚。"""
    bak = target.with_name(target.name + ".bak")
    if bak.exists():
        shutil.rmtree(bak)
    if target.exists():
        os.replace(target, bak)
    try:
        os.replace(new_root, target)
    except OSError:
        if bak.exists() and not target.exists():
            os.replace(bak, target)
        raise


def _update(d: Path, latest: dict, proxy: str | None) -> None:
    tmp = d.parent / ".sync_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / f"{latest['tag']}_{ARCHIVE_NAME}"
    for f in tmp.iterdir():  # 只保留当前版本的未完成下载，其余清掉
        if f != archive:
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    print(f"[数据同步] 下载 {latest['tag']} ...", flush=True)
    _download(latest["archive_url"], archive, proxy)  # 失败时保留部分文件供续传
    try:  # 下载完整后，无论校验成败都清理临时目录
        want = latest["manifest"].get("archive_sha256")
        if want and _sha256(archive) != want:
            raise SyncError("数据包 sha256 校验失败")
        new_root = _extract(archive, tmp / "extract")
        validate_data_dir(new_root, d)
        staged = d.with_name(d.name + ".new")
        if staged.exists():
            shutil.rmtree(staged)
        os.replace(new_root, staged)
        swap_dirs(d, staged)
        write_version(d, latest["tag"], latest["manifest"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sync_if_needed(data_dir: str | Path | None = None, repo: str = DEFAULT_REPO,
                   proxy: str | None = None, strict: bool = False,
                   check_only: bool = False) -> SyncResult:
    d = _data_dir(data_dir)
    local = local_version(d)
    local_tag = local.get("tag") if local else None
    try:
        latest = github_latest(repo, proxy)
    except Exception as e:  # noqa: BLE001 - 网络问题是常态，不阻断
        if strict:
            raise SyncError(f"无法获取 GitHub 最新版本: {e}") from e
        return SyncResult("failed", local_tag, None, f"无法获取 GitHub 版本，继续使用本地数据（{e}）")
    remote_tag = latest["tag"]
    if local_tag == remote_tag:
        return SyncResult("up_to_date", local_tag, remote_tag)

    # 无标记且本地日历末日 = release 目标交易日：视为同一版本，直接标记，免去全量下载
    target = latest["manifest"].get("target_trade_date")
    if local is None and target and d.exists():
        end = calendar_end(d)
        if end is not None and end == pd.Timestamp(target):
            if check_only:
                return SyncResult("adoptable", None, remote_tag, f"本地日历末日 {end.date()} 与 release 一致，可直接标记")
            write_version(d, remote_tag, latest["manifest"], how="auto_adopt")
            return SyncResult("adopted", remote_tag, remote_tag, f"本地日历末日 {end.date()} 与 release 一致，已标记")

    if check_only:
        return SyncResult("outdated", local_tag, remote_tag, "（--check 模式，未更新）")
    try:
        _update(d, latest, proxy)
    except Exception as e:  # noqa: BLE001
        if strict:
            raise
        return SyncResult("failed", local_tag, remote_tag, f"更新失败，继续使用本地数据（{e}）")
    return SyncResult("updated", remote_tag, remote_tag, f"日历末日 {calendar_end(d).date()}")
