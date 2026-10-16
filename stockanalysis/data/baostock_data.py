"""baostock 数据：逐日 ST / 停牌 / 估值 / 换手率（逐股，2015 起），行业分类（当前快照）。

缓存布局（data/baostock/）：
    daily/<qlib_code>.parquet     date, close_raw, amount, turn, trade_status, pe_ttm, pb_mrq, ps_ttm, pcf_ttm, is_st
    industry_<yyyymmdd>.parquet   qlib_code, industry, classification（证监会分类，只有当前快照）

增量：已有缓存从最后日期的次日续抓；中断后重跑即断点续传。
网络：baostock 原生 send_msg 在服务端断开时 recv 返回空字节会无限空转（表现为"卡住"），
这里替换为带超时、断开即报错的实现，单只股票失败时重新登录重试，仍失败则记入失败清单、继续下一只。
baostock 不含北交所。
"""
from __future__ import annotations

import time
import zlib

import pandas as pd

from ..paths import BAOSTOCK_DIR

DAILY_DIR = BAOSTOCK_DIR / "daily"
DAILY_START = "2015-01-01"   # 回测 2018 起 + 250 日预热 + 两年亏损判定
DAILY_FIELDS = {             # baostock 字段 -> 缓存列名
    "date": "date",
    "close": "close_raw",    # 不复权收盘价（股价 < 2 元判定用）
    "amount": "amount",
    "turn": "turn",          # 换手率（%）
    "tradestatus": "trade_status",  # 1 正常交易 0 停牌
    "peTTM": "pe_ttm",
    "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ttm",
    "isST": "is_st",         # 1 = ST / *ST
}
SOCKET_TIMEOUT = 30          # 秒：单次 recv 无数据即视为断开


def to_bs_code(code: str) -> str:
    """SH600000 -> sh.600000"""
    return f"{code[:2].lower()}.{code[2:]}"


def _patch_socket(timeout: float = SOCKET_TIMEOUT) -> None:
    """替换 baostock.util.socketutil.send_msg（各模块经 `sock.send_msg` 属性调用，替换即生效）。"""
    import baostock.common.contants as cons
    import baostock.common.context as context
    import baostock.util.socketutil as su

    if getattr(su, "_stockanalysis_patched", False):
        return

    def send_msg(msg: str) -> str:
        sock = getattr(context, "default_socket", None)
        if sock is None:
            raise ConnectionError("baostock 未登录")
        sock.settimeout(timeout)
        sock.sendall(bytes(msg + "\n", encoding="utf-8"))
        buf = b""
        while not buf.endswith(b"<![CDATA[]]>\n"):
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("baostock 连接被服务端关闭")
            buf += chunk
        head = bytes.decode(buf[:cons.MESSAGE_HEADER_LENGTH])
        arr = head.split(cons.MESSAGE_SPLIT)
        if arr[1] in cons.COMPRESSED_MESSAGE_TYPE_TUPLE:
            n = int(arr[2])
            body = zlib.decompress(buf[cons.MESSAGE_HEADER_LENGTH:cons.MESSAGE_HEADER_LENGTH + n])
            return head + bytes.decode(body)
        return bytes.decode(buf)

    su.send_msg = send_msg
    su._stockanalysis_patched = True


class Session:
    """baostock 登录会话：`with Session() as s: s.query(...)`，失败自动重新登录重试。"""

    def __init__(self, retries: int = 3, sleep: float = 2.0):
        self.retries, self.sleep = retries, sleep

    def __enter__(self):
        _patch_socket()
        self._login()
        return self

    def __exit__(self, *exc):
        import baostock as bs

        try:
            bs.logout()
        except Exception:  # noqa: BLE001 - 退出失败不影响结果
            pass

    def _login(self) -> None:
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            raise ConnectionError(f"baostock 登录失败: {lg.error_msg}")

    def query(self, fn_name: str, *args, **kwargs) -> pd.DataFrame:
        """调用 baostock.<fn_name>，读完全部分页返回 DataFrame（列为 baostock 原字段名）。"""
        import baostock as bs

        last: Exception | None = None
        for i in range(self.retries):
            try:
                rs = getattr(bs, fn_name)(*args, **kwargs)
                if rs.error_code != "0":
                    raise RuntimeError(f"{fn_name}: {rs.error_code} {rs.error_msg}")
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                return pd.DataFrame(rows, columns=rs.fields)
            except Exception as e:  # noqa: BLE001 - 网络异常类型不统一
                last = e
                time.sleep(self.sleep * (i + 1))
                try:
                    self._login()
                except Exception as le:  # noqa: BLE001
                    last = le
        raise last


# ------------------------------------------------------------------ 逐日
def _clean_daily(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=DAILY_FIELDS)[list(DAILY_FIELDS.values())]
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns.drop("date"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_daily(session: Session, code: str, start: str, end: str) -> pd.DataFrame:
    raw = session.query("query_history_k_data_plus", to_bs_code(code), ",".join(DAILY_FIELDS),
                        start_date=start, end_date=end, frequency="d", adjustflag="3")
    return _clean_daily(raw) if len(raw) else pd.DataFrame(columns=list(DAILY_FIELDS.values()))


def daily_path(code: str):
    return DAILY_DIR / f"{code}.parquet"


def update_daily(codes: list[str], end: str, start: str = DAILY_START, log=print) -> list[str]:
    """逐股补齐 [start, end] 的逐日数据，返回失败代码清单。只处理沪深（baostock 无北交所）。"""
    codes = [c for c in codes if c[:2] in ("SH", "SZ")]
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    failed, t0 = [], time.time()
    with Session() as s:
        for i, code in enumerate(codes, 1):
            path = daily_path(code)
            old = pd.read_parquet(path) if path.exists() else None
            begin = start
            if old is not None and len(old):
                last = pd.Timestamp(old["date"].max())
                if last >= pd.Timestamp(end):
                    continue
                begin = str((last + pd.Timedelta(days=1)).date())
            try:
                new = fetch_daily(s, code, begin, end)
            except Exception as e:  # noqa: BLE001
                failed.append(code)
                log(f"  {code} FAILED: {e}")
                continue
            df = new if old is None else pd.concat([old, new], ignore_index=True)
            df = df.drop_duplicates("date", keep="last").sort_values("date")
            df.to_parquet(path, index=False)
            if i % 200 == 0:
                log(f"  {i}/{len(codes)}  {time.time() - t0:.0f}s  失败 {len(failed)}")
    return failed


def load_daily(field: str, codes: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """逐日字段宽表（index=dates, columns=codes）；无缓存的股票整列为 NaN。"""
    cols = {}
    for c in codes:
        p = daily_path(c)
        if p.exists():
            d = pd.read_parquet(p, columns=["date", field])
            cols[c] = d.set_index("date")[field]
    wide = pd.DataFrame(cols).reindex(index=pd.DatetimeIndex(dates))
    return wide.reindex(columns=list(codes)).astype("float32")


# ------------------------------------------------------------------ 行业
def fetch_industry(use_cache: bool = True) -> pd.DataFrame:
    """当前行业分类快照（证监会分类）。历史行业不可得，用于回测时有轻微前视，研究报告中注明。"""
    today = pd.Timestamp.now().strftime("%Y%m%d")
    path = BAOSTOCK_DIR / f"industry_{today}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    with Session() as s:
        raw = s.query("query_stock_industry")
    df = pd.DataFrame({
        "qlib_code": raw["code"].str.replace(".", "", regex=False).str.upper(),
        "industry": raw["industry"].replace("", pd.NA),
        "classification": raw["industryClassification"],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_industry() -> pd.Series:
    """最新一份行业快照：qlib_code -> 行业。无缓存返回空 Series。"""
    files = sorted(BAOSTOCK_DIR.glob("industry_*.parquet"))
    if not files:
        return pd.Series(dtype=object)
    df = pd.read_parquet(files[-1]).dropna(subset=["industry"])
    return df.set_index("qlib_code")["industry"]
