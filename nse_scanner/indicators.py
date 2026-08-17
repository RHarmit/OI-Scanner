from __future__ import annotations
import numpy as np
import pandas as pd


def add_chaikin_indicators(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add Chaikin Money Flow (CMF), money-flow volume and Accumulation/Distribution Line.
    Input must contain: symbol, date, high, low, close, volume.
    """
    df = frame.sort_values(["symbol", "date"]).copy()

    span = df["high"] - df["low"]
    multiplier = np.where(
        span.abs() > 1e-12,
        ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span,
        0.0,
    )
    df["mf_multiplier"] = multiplier
    df["mf_volume"] = df["mf_multiplier"] * df["volume"]

    g = df.groupby("symbol", group_keys=False)
    mf_sum = g["mf_volume"].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
    vol_sum = g["volume"].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
    df[f"cmf_{window}"] = np.where(vol_sum.abs() > 1e-12, mf_sum / vol_sum, np.nan)

    df["adl"] = g["mf_volume"].cumsum()
    return df
