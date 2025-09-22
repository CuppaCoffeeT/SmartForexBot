# model/data_loader.py
import pandas as pd
import yfinance as yf

REQUIRED = ["Open", "High", "Low", "Close"]
INTRADAY_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h"}

def _standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    # Map any casing/spacing to names
    norm = {c: str(c).strip().lower().replace(" ", "") for c in df.columns}
    cmap = {}
    for orig, n in norm.items():
        if   n == "open":   cmap[orig] = "Open"
        elif n == "high":   cmap[orig] = "High"
        elif n == "low":    cmap[orig] = "Low"
        elif n == "close":  cmap[orig] = "Close"
        elif n in ("adjclose","adjustedclose"): cmap[orig] = "Adj Close"
        elif n == "volume": cmap[orig] = "Volume"
        else:               cmap[orig] = orig
    df = df.rename(columns=cmap)
    return df

def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]
    # Synthesize missing OHLC from Close 
    for c in ("Open","High","Low"):
        if c not in out.columns and "Close" in out.columns:
            out[c] = out["Close"]
    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
    return out[cols]

def _pick_level_with_ohlc(cols) -> int | None:
    ohlc = {"Open","High","Low","Close","Adj Close","Volume"}
    for i in range(cols.nlevels):
        vals = set(map(str, cols.get_level_values(i)))
        if len(vals & ohlc) >= 3:
            return i
    return None

def _download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        prepost=False,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # If MultiIndex reduce to this symbol 
    if isinstance(df.columns, pd.MultiIndex):
        cols = df.columns
        # is exactly the symbol slice it out
        for lvl in range(cols.nlevels):
            if set(cols.get_level_values(lvl)) == {symbol}:
                try:
                    df = df.xs(symbol, axis=1, level=lvl, drop_level=True)
                    break
                except Exception:
                    pass
        # level containing OHLC names
        if isinstance(df.columns, pd.MultiIndex):
            lvl_ohlc = _pick_level_with_ohlc(df.columns)
            if lvl_ohlc is not None:
                df.columns = df.columns.get_level_values(lvl_ohlc)
            else:
                df.columns = ["_".join([str(x) for x in tup if str(x) != ""])
                              for tup in df.columns.to_list()]

    # Standardize names and keep the usual columns (but don't drop to empty)
    df = _standardize_names(df)
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    if keep:
        df = df[keep].copy()

    # If we still don't have Close, bail
    if "Close" not in df.columns and "Adj Close" not in df.columns:
        return pd.DataFrame()

    # Ensure OHLC exists
    df = _ensure_ohlc(df)

    # index should be datetime for resampling downstream
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df.dropna(subset=["Close"])

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = _ensure_ohlc(df)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:    agg["Volume"] = "sum"
    if "Adj Close" in df.columns: agg["Adj Close"] = "last"
    out = df.resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(subset=["Close"])

def get_data(symbol="EURUSD=X", interval="4h", period="12mo") -> pd.DataFrame:
    """
    Robust download:
      - If '4h' requested, fetch '1h' and resample to 4H.
      - Intraday periods beyond Yahoo limits → 60d.
      - If intraday fails, try daily 12mo then 24mo.
      - If Close-only, synthesize OHLC so the Fib pipeline can run.
    Prints each attempt so you can see what worked.
    """
    requested_interval = interval.lower()
    wants_4h = requested_interval in {"4h","4hr","4hours"}
    dl_interval = "1h" if wants_4h else requested_interval

    def attempt(ivl: str, per: str, post=None, label=""):
        df0 = _download(symbol, ivl, per)
        if df0.empty:
            print(f"[data_loader] ❌ Empty for {symbol} {ivl} {per}")
            return pd.DataFrame()
        df1 = _resample_ohlc(df0, "4H") if post == "4h" else df0
        ok = not df1.empty and ("Close" in df1.columns or "Adj Close" in df1.columns)
        print(f"[data_loader] {'✅' if ok else '❌'} {symbol} {ivl} {per}{(' → '+label) if label else ''} | cols={list(df1.columns)} rows={len(df1)}")
        return df1 if ok else pd.DataFrame()

    # 1) Requested (or 1h→4H)
    df = attempt(dl_interval, period, post=("4h" if wants_4h else None), label=("4H resample" if wants_4h else ""))

    # 2) Intraday cap to 60d
    if df.empty and dl_interval in INTRADAY_INTERVALS and period not in {"60d","30d"}:
        df = attempt(dl_interval, "60d", post=("4h" if wants_4h else None), label=("4H resample" if wants_4h else ""))

    # 3) Daily 12mo
    if df.empty:
        df = attempt("1d", "12mo")

    # 4) Daily 24mo
    if df.empty:
        df = attempt("1d", "24mo")

    if df.empty:
        raise KeyError("Missing Close/OHLC after all fallbacks.")

    return df