"""长桥 CLI 封装（本机 `longbridge` 已登录）：市场温度历史（可回测的择时输入）与实盘体检。

    market_temperature(start, end)  A 股市场温度 / 估值 / 情绪（0~100），2018 起逐日；缓存 data/raw/longbridge/market_temp/<yyyy>
    health_check(codes)            入池股的当日资金流向（大 / 中 / 小单净流入）、机构评级、近期新闻与公告标题

CLI 不可用、未登录或网络失败时只告警、返回空结果，不中断选股。
市场温度的计算口径由长桥定义、无法复核，作为择时候选因子须先经研发区检验。
时间戳为 UTC，+8 小时即北京时间交易日。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pandas as pd

from ..paths import LONGBRIDGE_DIR

DEFAULT_EXE = os.path.expanduser(r"~\AppData\Local\Programs\longbridge\longbridge.exe")
TEMP_DIR = LONGBRIDGE_DIR / "market_temp"
TIMEOUT = 60  # 秒


def exe() -> str | None:
    """长桥 CLI 路径：环境变量 LONGBRIDGE_CLI → PATH → 默认安装位置。"""
    for cand in (os.environ.get("LONGBRIDGE_CLI"), shutil.which("longbridge"), DEFAULT_EXE):
        if cand and os.path.exists(cand):
            return cand
    return None


def run(*args: str, timeout: int = TIMEOUT):
    """执行 CLI 并解析 JSON 输出；失败抛 RuntimeError。"""
    path = exe()
    if path is None:
        raise RuntimeError("未找到 longbridge CLI（可设置环境变量 LONGBRIDGE_CLI）")
    proc = subprocess.run([path, *args, "--format", "json"], capture_output=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:300])
    return json.loads(proc.stdout)


def to_lb_symbol(code: str) -> str:
    """SH600519 -> 600519.SH"""
    return f"{code[2:]}.{code[:2]}"


def _to_cn_date(ts: str) -> pd.Timestamp:
    return (pd.Timestamp(ts).tz_convert(None) + pd.Timedelta(hours=8)).normalize()


# ------------------------------------------------------------------ 市场温度
def fetch_market_temperature_year(year: int, use_cache: bool = True) -> pd.DataFrame:
    """单年市场温度（接口单次区间上限 3 年，按年缓存）；当年的缓存每天重抓。"""
    path = TEMP_DIR / f"{year}.parquet"
    complete = path.exists() and pd.Timestamp.fromtimestamp(path.stat().st_mtime) > pd.Timestamp(f"{year + 1}-01-03")
    fresh = path.exists() and pd.Timestamp.now() - pd.Timestamp.fromtimestamp(path.stat().st_mtime) < pd.Timedelta(hours=12)
    if use_cache and (complete or fresh):
        return pd.read_parquet(path)
    rows = run("market-temp", "CN", "--history", "--start", f"{year}-01-01", "--end", f"{year}-12-31")
    df = pd.DataFrame({
        "date": [_to_cn_date(r["time"]) for r in rows],
        "temperature": [float(r["temperature"]) for r in rows],
        "valuation": [float(r["valuation"]) for r in rows],
        "sentiment": [float(r["sentiment"]) for r in rows],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def market_temperature(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """[start, end] 的逐日市场温度，index=date。"""
    years = range(max(pd.Timestamp(start).year, 2018), pd.Timestamp(end).year + 1)
    frames = [fetch_market_temperature_year(y, use_cache) for y in years]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("date").set_index("date").sort_index()
    return df.loc[start:end]


# ------------------------------------------------------------------ 实盘体检
def _capital(code: str) -> dict:
    d = run("capital", to_lb_symbol(code))
    cin, cout = d.get("capital_in") or {}, d.get("capital_out") or {}
    net = {k: float(cin.get(k) or 0) - float(cout.get(k) or 0) for k in ("large", "medium", "small")}
    return {"大单净流入": net["large"], "中单净流入": net["medium"], "小单净流入": net["small"]}


def _rating(code: str) -> dict:
    d = run("institution-rating", to_lb_symbol(code))
    ins = d.get("instratings") or {}
    return {"机构评级": ins.get("recommend") or "", "目标价": ins.get("target") or ""}


def _titles(cmd: str, code: str, n: int, time_key: str) -> list[str]:
    rows = run(cmd, to_lb_symbol(code), "--lang", "zh-CN") or []
    out = []
    for r in rows[:n]:
        day = _to_cn_date(r[time_key]).strftime("%m-%d") if r.get(time_key) else ""
        title = " ".join(str(r.get("title", "")).replace("|", "／").split())   # 竖线 / 换行会破坏 Markdown 表格
        out.append(f"{day} {title}".strip())
    return out


def health_check(codes: list[str], n_news: int = 3, log=print) -> pd.DataFrame:
    """入池股实盘体检（只作人工参考，不参与选股）：资金流向 / 机构评级 / 近期新闻 / 近期公告。"""
    if exe() is None:
        log("[warn] 未找到 longbridge CLI，跳过体检")
        return pd.DataFrame()
    rows = []
    for c in codes:
        row: dict = {"code": c}
        for name, fn in (("资金", _capital), ("评级", _rating)):
            try:
                row.update(fn(c))
            except Exception as e:  # noqa: BLE001 - 体检失败不影响选股
                log(f"[warn] {c} {name}查询失败: {str(e)[:100]}")
        for col, cmd, key in (("近期新闻", "news", "published_at"), ("近期公告", "filing", "publish_at")):
            try:
                row[col] = "；".join(_titles(cmd, c, n_news, key))
            except Exception as e:  # noqa: BLE001
                log(f"[warn] {c} {col}查询失败: {str(e)[:100]}")
        rows.append(row)
    return pd.DataFrame(rows)
