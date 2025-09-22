# model/features.py
import numpy as np
import pandas as pd


# Core Fib / ATR helpers
def add_fibonacci_levels(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    required = ["High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_fibonacci_levels: missing columns {missing}. Got: {list(df.columns)}")
    
    df = df.copy()
    df["swing_high"] = df["High"].rolling(window=window, min_periods=window//2).max()
    df["swing_low"]  = df["Low"].rolling(window=window, min_periods=window//2).min()

    sh = df["swing_high"]
    sl = df["swing_low"]

    df["Fib_0"]     = sh
    df["Fib_1"]     = sl
    span = (sh - sl)
    df["Fib_0.382"] = sh - span * 0.382
    df["Fib_0.5"]   = sh - span * 0.5
    df["Fib_0.618"] = sh - span * 0.618
    return df.dropna()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return tr.rolling(period).mean()


# Optional indicators
def add_indicators(df: pd.DataFrame, sma_fast=20, sma_slow=50, rsi_period=14) -> pd.DataFrame:
    df = df.copy()
    # SMA
    df[f"SMA_{sma_fast}"] = df["Close"].rolling(sma_fast).mean()
    df[f"SMA_{sma_slow}"] = df["Close"].rolling(sma_slow).mean()
    df["SMA_spread"] = df[f"SMA_{sma_fast}"] - df[f"SMA_{sma_slow}"]
    # RSI (simple rolling mean version)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Fib distances by ATR, add ATR% and short returns + RSI bands."""
    df = df.copy()
    atr = df["ATR"].replace(0, np.nan)
    for lvl in ["0.382", "0.5", "0.618"]:
        df[f"ndist_fib_{lvl}"] = (df[f"Close"] - df[f"Fib_{lvl}"]) / atr
        df[f"ndist_fib_{lvl}"] = df[f"ndist_fib_{lvl}"].clip(-10, 10)  # trim extremes

    df["ATR_pct"] = (df["ATR"] / df["Close"]).clip(0, 0.1)
    df["ret_1"] = df["Close"].pct_change(1).clip(-0.05, 0.05)
    df["ret_3"] = df["Close"].pct_change(3).clip(-0.10, 0.10)

    # Convenience binary bands for RSI, if present
    if "RSI" in df.columns:
        df["RSI_low"] = (df["RSI"] <= 35).astype(int)
        df["RSI_high"] = (df["RSI"] >= 65).astype(int)
    else:
        df["RSI_low"] = np.nan
        df["RSI_high"] = np.nan
    return df

# Master builder with switches for ablation modes
def add_fib_event_context(
    df: pd.DataFrame,
    window: int = 50,
    atr_period: int = 14,
    proximity_k: float = 0.25,
    # switches + params
    use_indicators: bool = True,
    use_normalized: bool = False,
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_period: int = 14,
) -> pd.DataFrame:

    df = add_fibonacci_levels(df, window=window).copy()
    df["ATR"] = _atr(df, period=atr_period)
    df["eps"] = df["ATR"] * proximity_k

    # Robust swing flags
    high_np = np.asarray(df["High"]).reshape(-1)
    low_np  = np.asarray(df["Low"]).reshape(-1)
    sh_np   = np.asarray(df["swing_high"]).reshape(-1)
    sl_np   = np.asarray(df["swing_low"]).reshape(-1)

    eps_eq = np.finfo(float).eps * 10
    df["is_swing_high"] = (np.abs(high_np - sh_np) <= eps_eq).astype(int)
    df["is_swing_low"]  = (np.abs(low_np  - sl_np) <= eps_eq).astype(int)

    # Trend proxy
    ma_len = max(20, window // 2)
    df["MA"] = df["Close"].rolling(ma_len, min_periods=ma_len//2).mean()
    df["trend_up"] = (df["Close"] > df["MA"]).astype(int)

    # Proximity to Fib levels
    for lvl in ["0.382", "0.5", "0.618"]:
        df[f"near_fib_{lvl}"] = (np.abs(df["Close"] - df[f"Fib_{lvl}"]) <= df["eps"]).astype(int)

    # Any near-fib
    df["near_any_fib"] = df[[f"near_fib_{l}" for l in ["0.382", "0.5", "0.618"]]].max(axis=1)

    # Raw distances
    for lvl in ["0.382", "0.5", "0.618"]:
        df[f"dist_fib_{lvl}"] = (df["Close"] - df[f"Fib_{lvl}"])

    # Context label 
    df["is_support_context"] = df["trend_up"]

    # Optional blocks
    if use_indicators:
        df = add_indicators(df, sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period)
    if use_normalized:
        df = add_normalized_features(df)

    return df