from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from io import BytesIO
from zoneinfo import ZoneInfo
import zipfile

import pandas as pd
import requests

IST = ZoneInfo("Asia/Kolkata")

ARCHIVE_HOSTS = (
    "https://nsearchives.nseindia.com",
    "https://archives.nseindia.com",
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151 Safari/537.36"
)


def candidate_weekdays(lookback_weekdays: int, include_today_after_hour: int = 19) -> list[date]:
    now = datetime.now(IST)
    cursor = now.date()
    if now.hour < include_today_after_hour:
        cursor -= timedelta(days=1)

    out = []
    while len(out) < lookback_weekdays:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor -= timedelta(days=1)
    return out


def _url(segment: str, d: date, host: str) -> str:
    ds = d.strftime("%Y%m%d")
    return f"{host}/content/{segment}/BhavCopy_NSE_{segment.upper()}_0_0_0_{ds}_F_0000.csv.zip"


def fetch_bhavcopy(segment: str, d: date, timeout: int = 18) -> pd.DataFrame | None:
    """Download one NSE UDiFF final bhavcopy."""
    headers = {
        "User-Agent": UA,
        "Accept": "application/zip,application/octet-stream,*/*",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }

    for host in ARCHIVE_HOSTS:
        try:
            r = requests.get(_url(segment, d, host), headers=headers, timeout=timeout)
            if r.status_code in (403, 404):
                continue
            r.raise_for_status()
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    continue
                with z.open(csvs[0]) as f:
                    df = pd.read_csv(f, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass

    return None


def fetch_recent_sessions(segment: str, sessions: int, candidate_count: int) -> list[tuple[date, pd.DataFrame]]:
    dates = candidate_weekdays(candidate_count)
    found: list[tuple[date, pd.DataFrame]] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_bhavcopy, segment, d): d for d in dates}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is not None and not df.empty:
                found.append((d, df))

    found.sort(key=lambda x: x[0], reverse=True)
    return found[:sessions]


def normalize_cash_frame(d: date, raw: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "TckrSymb": "symbol",
        "FinInstrmTp": "instrument_type",
        "SctySrs": "series",
        "OpnPric": "open",
        "HghPric": "high",
        "LwPric": "low",
        "ClsPric": "close",
        "TtlTradgVol": "volume",
        "TtlTrfVal": "traded_value",
    }
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Cash bhavcopy missing columns: {missing}")

    df = raw[list(needed)].rename(columns=needed).copy()
    df = df[df["instrument_type"].astype(str).str.upper().eq("STK")]
    df = df[df["series"].astype(str).str.upper().eq("EQ")]

    df["date"] = pd.Timestamp(d)
    for c in ["open", "high", "low", "close", "volume", "traded_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["symbol", "close", "high", "low", "volume"])


def normalize_futures_oi(d: date, raw: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "TckrSymb": "symbol",
        "FinInstrmTp": "instrument_type",
        "OpnIntrst": "open_interest",
        "NewBrdLotQty": "lot_size",
        "TtlTradgVol": "contracts_volume",
        "TtlTrfVal": "futures_traded_value",
    }
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"F&O bhavcopy missing columns: {missing}")

    df = raw[list(needed)].rename(columns=needed).copy()
    df = df[df["instrument_type"].astype(str).str.upper().eq("STF")]
    df["date"] = pd.Timestamp(d)

    for c in ["open_interest", "lot_size", "contracts_volume", "futures_traded_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["oi_shares"] = df["open_interest"] * df["lot_size"]
    df["fut_volume_shares"] = df["contracts_volume"] * df["lot_size"]

    return (
        df.groupby(["date", "symbol"], as_index=False)
        .agg(
            oi_shares=("oi_shares", "sum"),
            futures_volume_shares=("fut_volume_shares", "sum"),
            futures_traded_value=("futures_traded_value", "sum"),
        )
    )
