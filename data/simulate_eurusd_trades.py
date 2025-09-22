
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.data_loader import get_data
from model.features import add_fib_event_context
from train import create_event_labels

ARTIFACT_PATH = Path("artifacts/EURUSDX_1h.joblib")  
SYMBOL = "EURUSD=X"
INTERVAL = "1h"
# Fetch a slightly longer history to ensure feature windows warm up
DOWNLOAD_PERIOD = "180d"
START_DATE = pd.Timestamp("2025-06-10", tz="UTC")
END_DATE = pd.Timestamp("2025-09-10", tz="UTC")
START_BALANCE = 100.0
POSITION_RISK = 0.10  # 10% of balance per trade


def load_artifact(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    payload = joblib.load(path)
    expected_keys = {"model", "threshold", "features", "meta"}
    missing = expected_keys - payload.keys()
    if missing:
        raise KeyError(f"Artifact missing keys: {missing}")
    return payload


def build_feature_frame(meta: Dict[str, float]) -> pd.DataFrame:
    df_raw = get_data(SYMBOL, interval=INTERVAL, period=DOWNLOAD_PERIOD)
    df_raw = df_raw.sort_index()
    df_feat = add_fib_event_context(
        df_raw,
        window=int(meta.get("window", 40)),
        proximity_k=float(meta.get("proximity_k", 0.35)),
        use_indicators=(meta.get("feature_mode", "fib") in {"fib_ind", "fib_ind_norm"}),
        use_normalized=(meta.get("feature_mode", "fib") == "fib_ind_norm"),
        sma_fast=int(meta.get("sma_fast", 20)),
        sma_slow=int(meta.get("sma_slow", 50)),
        rsi_period=int(meta.get("rsi_period", 14)),
    )
    return df_feat


def attach_event_labels(df_feat: pd.DataFrame, meta: Dict[str, float]) -> pd.DataFrame:
    df_events = create_event_labels(
        df_feat,
        horizon=int(meta.get("horizon", 6)),
        tp=float(meta.get("tp", 0.002)),
        sl=float(meta.get("sl", 0.0015)),
    )
    return df_events


def simulate_trades(events: pd.DataFrame, features: List[str], model, threshold: float) -> pd.DataFrame:
    records = []
    balance = START_BALANCE
    wins = losses = trades = 0
    for ts, row in events.iterrows():
        if ts < START_DATE or ts > END_DATE:
            continue
        feat = row[features]
        if feat.isna().any():
            continue
        proba = float(model.predict_proba(feat.to_frame().T)[0, 1])
        signed_conf = proba - threshold
        context_up = bool(row.get("is_support_context", 1) == 1)
        direction = "BUY" if context_up else "SELL"
        take_trade = signed_conf > 0
        outcome = None
        balance_before = balance
        if take_trade:
            trades += 1
            stake = balance * POSITION_RISK
            success = bool(row.get("signal", 0) == 1)
            if success:
                balance += stake
                wins += 1
                outcome = "win"
            else:
                balance -= stake
                losses += 1
                outcome = "loss"
        records.append(
            {
                "timestamp": ts,
                "prob": proba,
                "threshold": threshold,
                "signed_confidence": signed_conf,
                "direction": direction,
                "take_trade": take_trade,
                "outcome": outcome,
                "balance_before": balance_before,
                "balance_after": balance,
            }
        )
    summary = {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / trades) if trades else None,
        "final_balance": balance,
    }
    return pd.DataFrame(records), summary


def simulate_naive(events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    balance = START_BALANCE
    wins = losses = trades = 0
    recs = []
    for ts, row in events.iterrows():
        if ts < START_DATE or ts > END_DATE:
            continue
        signal = int(row.get("signal", 0))
        stake = balance * POSITION_RISK
        balance_before = balance
        trades += 1
        if signal == 1:
            balance += stake
            wins += 1
            outcome = "win"
        else:
            balance -= stake 
            losses += 1
            outcome = "loss"
        recs.append({
            "timestamp": ts,
            "take_trade": True,
            "outcome": outcome,
            "balance_before": balance_before,
            "balance_after": balance,
        })
    summary = {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / trades) if trades else None,
        "final_balance": balance,
    }
    return pd.DataFrame(recs), summary


def equity_from_trades(trade_log: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.Series:
    if trade_log is None or trade_log.empty:
        return pd.Series(START_BALANCE, index=price_index)
    trade_log = trade_log.set_index("timestamp").sort_index()
    equity = trade_log["balance_after"].reindex(price_index, method="ffill")
    if equity.isna().all():
        equity = pd.Series(START_BALANCE, index=price_index)
    else:
        first_balance = trade_log.iloc[0]["balance_before"]
        equity.iloc[0] = first_balance
        equity = equity.bfill().fillna(START_BALANCE)
    return equity


def plot_comparison(price_series: pd.Series, curves: Dict[str, pd.Series]) -> None:
    base = price_series.iloc[0]
    price_indexed = price_series / base * START_BALANCE
    plt.figure(figsize=(12, 6))
    plt.plot(price_indexed.index, price_indexed.values, label="EURUSD Close (indexed)", color="#1f77b4", linewidth=1.5)
    palette = {
        "naive": "#7f7f7f",
        "fib": "#d62728",
        "fib_ind": "#2ca02c",
        "fib_ind_norm": "#9467bd",
    }
    for name, eq in curves.items():
        eq_idx = eq / eq.iloc[0] * START_BALANCE
        plt.plot(eq_idx.index, eq_idx.values, label=name, color=palette.get(name, None), linewidth=1.6)
    plt.title("EURUSD — Equity vs Price (2025-06-10 to 2025-09-10)")
    plt.xlabel("Date")
    plt.ylabel("Indexed value (start = $100)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def artifact_path_for_mode(mode: str) -> Path:
    clean = SYMBOL.replace("=","")
    p = Path(f"artifacts/{clean}_1h_{mode}.joblib")
    if p.exists():
        return p
    # legacy fallback without mode
    legacy = Path(f"artifacts/{clean}_1h.joblib")
    return legacy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare equity curves across feature modes and a naïve baseline")
    args = parser.parse_args()

    modes = ["fib", "fib_ind", "fib_ind_norm"]
    curves: Dict[str, pd.Series] = {}
    summaries: Dict[str, Dict[str, float]] = {}

    # Build price series once using the fib mode meta (they should share interval/period)
    price_series = None

    for mode in modes:
        art_path = artifact_path_for_mode(mode)
        if not art_path.exists():
            print(f"[skip] Missing artifact for mode={mode}: {art_path}")
            continue
        artifact = load_artifact(art_path)
        model = artifact["model"]
        threshold = float(artifact["threshold"])
        features = artifact["features"]
        meta = artifact["meta"]

        df_feat = build_feature_frame(meta)
        if price_series is None:
            price_series = df_feat.loc[(df_feat.index >= START_DATE) & (df_feat.index <= END_DATE), "Close"].copy()
        df_events = attach_event_labels(df_feat, meta)
        df_events = df_events.loc[(df_events.index >= START_DATE) & (df_events.index <= END_DATE)]
        if df_events.empty:
            print(f"[skip] No events for mode={mode} in window")
            continue

        tlog, summ = simulate_trades(df_events, features, model, threshold)
        eq = equity_from_trades(tlog, price_series.index)
        curves[mode] = eq
        summaries[mode] = summ

    # Naive baseline using fib meta/events (fallback to first available mode)
    baseline_events = None
    for mode in modes:
        if mode in curves:
            # Recompute events for consistency with its meta
            art_path = artifact_path_for_mode(mode)
            art = load_artifact(art_path)
            df_feat = build_feature_frame(art["meta"])  # build using that meta
            events = attach_event_labels(df_feat, art["meta"]) 
            baseline_events = events.loc[(events.index >= START_DATE) & (events.index <= END_DATE)]
            break
    if baseline_events is not None and not baseline_events.empty:
        tlog_naive, summ_naive = simulate_naive(baseline_events)
        curves["naive"] = equity_from_trades(tlog_naive, price_series.index)
        summaries["naive"] = summ_naive

    if price_series is None:
        raise RuntimeError("No price data available for the specified range.")

    # Print summaries
    print("=== Summaries ===")
    for name, summ in summaries.items():
        print(f"{name}: trades={summ['trades']} wins={summ['wins']} losses={summ['losses']} win_rate={summ['win_rate']:.2f} final=${summ['final_balance']:.2f}")

    # Plot
    plot_comparison(price_series, curves)


if __name__ == "__main__":
    main()
