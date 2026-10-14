"""数据同步：版本比对、自动标记、下载失败降级、校验失败不替换、成功替换。全部 mock 网络。"""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from src.data import sync


def make_data_dir(root: Path, cal_end: str, marker: str = "old") -> Path:
    d = root
    (d / "calendars").mkdir(parents=True)
    (d / "calendars" / "day.txt").write_text(f"2026-10-09\n{cal_end}\n", encoding="utf-8")
    (d / "features" / "sh600000").mkdir(parents=True)
    (d / "features" / "sh600000" / "close.day.bin").write_bytes(marker.encode())
    (d / "instruments").mkdir()
    for inst in sync.REQUIRED_INSTRUMENTS:
        (d / "instruments" / f"{inst}.txt").write_text("SH600000\t2005-01-04\t2099-12-31\n", encoding="utf-8")
    return d


def make_archive(tmp: Path, cal_end: str, drop: str | None = None) -> bytes:
    src = make_data_dir(tmp / "build" / "qlib_bin", cal_end, marker="new")
    if drop:
        (src / "instruments" / f"{drop}.txt").unlink()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src, arcname="qlib_bin")
    return buf.getvalue()


def fake_remote(monkeypatch, tag: str, archive: bytes | None, target: str, sha: str | None = None):
    manifest = {"release_tag": tag, "target_trade_date": target,
                "archive_sha256": sha or ("sha256:" + hashlib.sha256(archive or b"").hexdigest())}

    def get_json(url, proxy=None, timeout=20):
        if url.endswith("releases/latest"):
            return {"tag_name": tag, "assets": [
                {"name": sync.ARCHIVE_NAME, "browser_download_url": "http://x/archive"},
                {"name": sync.MANIFEST_NAME, "browser_download_url": "http://x/manifest"}]}
        return manifest

    def download(url, dest, proxy=None, timeout=60):
        if archive is None:
            raise OSError("network down")
        Path(dest).write_bytes(archive)

    monkeypatch.setattr(sync, "_http_get_json", get_json)
    monkeypatch.setattr(sync, "_download", download)


def feature_marker(d: Path) -> str:
    return (d / "features" / "sh600000" / "close.day.bin").read_text()


def test_up_to_date_skips(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")
    sync.write_version(d, "2026-10-13")
    fake_remote(monkeypatch, "2026-10-13", None, "2026-10-13")
    res = sync.sync_if_needed(d)
    assert res.status == "up_to_date"


def test_auto_adopt_when_calendar_matches(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-14")
    fake_remote(monkeypatch, "2026-10-14", None, "2026-10-14")
    assert sync.sync_if_needed(d, check_only=True).status == "adoptable"
    assert sync.local_version(d) is None
    res = sync.sync_if_needed(d)
    assert res.status == "adopted"
    assert sync.local_version(d)["tag"] == "2026-10-14"
    assert feature_marker(d) == "old"


def test_download_failure_keeps_old_data(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")
    sync.write_version(d, "2026-10-13")
    fake_remote(monkeypatch, "2026-10-14", None, "2026-10-14")
    res = sync.sync_if_needed(d)
    assert res.status == "failed"
    assert feature_marker(d) == "old"
    assert sync.local_version(d)["tag"] == "2026-10-13"
    with pytest.raises(OSError):
        sync.sync_if_needed(d, strict=True)


def test_network_failure_non_strict(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")

    def boom(*a, **k):
        raise OSError("timeout")

    monkeypatch.setattr(sync, "_http_get_json", boom)
    assert sync.sync_if_needed(d).status == "failed"
    with pytest.raises(sync.SyncError):
        sync.sync_if_needed(d, strict=True)


def test_validation_failure_does_not_replace(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")
    sync.write_version(d, "2026-10-13")
    archive = make_archive(tmp_path, "2026-10-14", drop="csi1000")
    fake_remote(monkeypatch, "2026-10-14", archive, "2026-10-14")
    res = sync.sync_if_needed(d)
    assert res.status == "failed" and "csi1000" in res.message
    assert feature_marker(d) == "old"
    assert not (tmp_path / ".sync_tmp").exists()


def test_sha_mismatch_does_not_replace(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")
    sync.write_version(d, "2026-10-13")
    archive = make_archive(tmp_path, "2026-10-14")
    fake_remote(monkeypatch, "2026-10-14", archive, "2026-10-14", sha="sha256:bad")
    assert sync.sync_if_needed(d).status == "failed"
    assert feature_marker(d) == "old"


def test_older_calendar_rejected(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")
    sync.write_version(d, "2026-10-13")
    archive = make_archive(tmp_path, "2026-10-10")
    fake_remote(monkeypatch, "2026-10-10", archive, "2026-10-10")
    assert sync.sync_if_needed(d).status == "failed"
    assert feature_marker(d) == "old"


def test_successful_update_swaps_and_keeps_backup(tmp_path, monkeypatch):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")
    sync.write_version(d, "2026-10-13")
    archive = make_archive(tmp_path, "2026-10-14")
    fake_remote(monkeypatch, "2026-10-14", archive, "2026-10-14")
    res = sync.sync_if_needed(d)
    assert res.status == "updated", res.message
    assert feature_marker(d) == "new"
    assert sync.local_version(d)["tag"] == "2026-10-14"
    bak = tmp_path / "cn_data.bak"
    assert feature_marker(bak) == "old"
    assert not (tmp_path / "cn_data.new").exists()


def test_adopt_writes_marker(tmp_path):
    d = make_data_dir(tmp_path / "cn_data", "2026-10-13")
    res = sync.adopt("2026-10-13", d)
    assert res.status == "adopted"
    assert json.loads((d / ".version").read_text(encoding="utf-8"))["how"] == "adopt"
