from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_chaikin_indicators


def build_cash_metrics(cash: pd.DataFrame, cmf_window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = add_chaikin_indicators(cash, cmf_window)
    hist = hist.sort_values(["symbol", "date"]).copy()

    g = hist.groupby("symbol", group_keys=False)
    hist["ret_1d_pct"] = g["close"].pct_change() * 100
    hist["ret_3d_pct"] = g["close"].pct_change(3) * 100
    hist["cmf_change_3d"] = g[f"cmf_{cmf_window}"].diff(3)

    hist["avg_volume_20"] = (
        g["volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    )
    hist["avg_traded_value_20"] = (
        g["traded_value"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    )
    hist["volume_ratio"] = hist["volume"] / hist["avg_volume_20"].replace(0, np.nan)

    latest = hist.groupby("symbol", as_index=False).tail(1).copy()
    latest = latest.rename(columns={f"cmf_{cmf_window}": "cmf"})
    return latest, hist


def build_oi_metrics(fo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fo = fo.sort_values(["symbol", "date"]).copy()
    piv = fo.pivot_table(index="symbol", columns="date", values="oi_shares", aggfunc="sum")
    dates = sorted(piv.columns)
    if len(dates) < 4:
        raise ValueError("Need at least 4 successful F&O trading sessions.")

    d0, d1, d2, d3 = dates[-4], dates[-3], dates[-2], dates[-1]
    out = pd.DataFrame(index=piv.index)
    out["oi_t3"] = piv[d0]
    out["oi_t2"] = piv[d1]
    out["oi_t1"] = piv[d2]
    out["oi_latest"] = piv[d3]

    out["oi_3d_pct"] = np.where(
        out["oi_t3"] > 0,
        (out["oi_latest"] / out["oi_t3"] - 1.0) * 100,
        np.nan,
    )
    out["oi_1d_pct"] = np.where(
        out["oi_t1"] > 0,
        (out["oi_latest"] / out["oi_t1"] - 1.0) * 100,
        np.nan,
    )
    out["oi_monotonic_3d"] = (
        (out["oi_t2"] > out["oi_t3"])
        & (out["oi_t1"] > out["oi_t2"])
        & (out["oi_latest"] > out["oi_t1"])
    )
    out["oi_positive_steps"] = (
        (out["oi_t2"] > out["oi_t3"]).astype(int)
        + (out["oi_t1"] > out["oi_t2"]).astype(int)
        + (out["oi_latest"] > out["oi_t1"]).astype(int)
    )

    out["oi_date_t3"] = d0
    out["oi_date_latest"] = d3
    return out.reset_index(), fo


def classify_buildup(price_3d_pct: float, oi_3d_pct: float) -> str:
    if pd.isna(price_3d_pct) or pd.isna(oi_3d_pct):
        return "Unknown"
    if oi_3d_pct >= 0 and price_3d_pct > 0:
        return "Long buildup"
    if oi_3d_pct >= 0 and price_3d_pct < 0:
        return "Short buildup"
    if oi_3d_pct < 0 and price_3d_pct > 0:
        return "Short covering"
    return "Long unwinding"


def score_rows(df: pd.DataFrame) -> pd.Series:
    oi = np.clip(df["oi_3d_pct"].fillna(0), 0, 60) / 60 * 40
    cmf = np.clip((df["cmf"].fillna(-0.25) + 0.05) / 0.35, 0, 1) * 30
    cmf_trend = np.where(df["cmf_change_3d"].fillna(-1) > 0, 10, 0)
    liq = np.clip(df["volume_ratio"].fillna(0), 0, 2) / 2 * 10
    align = np.where(df["buildup"].eq("Long buildup"), 10, 0)
    return np.round(oi + cmf + cmf_trend + liq + align, 1)


def build_scanner(cash: pd.DataFrame, futures_oi: pd.DataFrame, cmf_window: int = 20):
    latest_cash, cash_hist = build_cash_metrics(cash, cmf_window)
    oi_metrics, oi_hist = build_oi_metrics(futures_oi)

    cols = [
        "symbol", "date", "close", "ret_1d_pct", "ret_3d_pct",
        "cmf", "cmf_change_3d", "volume", "volume_ratio",
        "avg_traded_value_20"
    ]
    out = latest_cash[cols].merge(oi_metrics, on="symbol", how="inner")
    out["buildup"] = [
        classify_buildup(p, o) for p, o in zip(out["ret_3d_pct"], out["oi_3d_pct"])
    ]
    out["score"] = score_rows(out)
    return out.sort_values(["score", "oi_3d_pct"], ascending=False), cash_hist, oi_hist
