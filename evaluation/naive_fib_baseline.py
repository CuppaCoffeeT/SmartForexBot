
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.data_loader import get_data
from model.features import add_fib_event_context
from train import create_event_labels

DEFAULT_ARTIFACT = Path("artifacts/EURUSDX_1h_fib.joblib")
DEFAULT_SYMBOL = "EURUSD=X"
DEFAULT_INTERVAL = "1h"
DEFAULT_PERIOD = "60d"
START_BALANCE = 100.0
POSITION_RISK = 0.10


def load_meta_from_artifact(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    payload = None
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is required to load artifacts.") from exc
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "meta" not in payload:
        raise ValueError("Artifact missing meta information")
    return payload["meta"]


def build_events(symbol: str, interval: str, period: str, meta: Dict[str, float]) -> pd.DataFrame:
    df_raw = get_data(symbol, interval=interval, period=period)
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
    events = create_event_labels(
        df_feat,
        horizon=int(meta.get("horizon", 6)),
        tp=float(meta.get("tp", 0.002)),
        sl=float(meta.get("sl", 0.0015)),
    )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive Fib-touch baseline evaluation")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Path to model artifact for meta defaults")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL)
    parser.add_argument("--period", type=str, default=DEFAULT_PERIOD, help="Lookback period for evaluation")
    parser.add_argument("--start", type=str, default=None, help="Optional ISO start date for filtering")
    parser.add_argument("--end", type=str, default=None, help="Optional ISO end date for filtering")
    args = parser.parse_args()

    meta = load_meta_from_artifact(args.artifact)
    period = meta.get("period", args.period)

    events = build_events(args.symbol, args.interval, period, meta)
    if args.start:
        start_ts = pd.Timestamp(args.start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        events = events.loc[events.index >= start_ts]
    if args.end:
        end_ts = pd.Timestamp(args.end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        events = events.loc[events.index <= end_ts]

    if events.empty:
        print("No events found for the specified filters.")
        return

    y_true = events["signal"].astype(int)
    y_pred = np.ones_like(y_true)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    auc = 0.5  # deterministic positive predictions 

    print("=== Naive Fib-touch Metrics ===")
    print(f"Total events: {len(events)}")
    print(f"Positive rate (success frequency): {y_true.mean():.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC-AUC (degenerate, defaulting to 0.5): {auc:.3f}")
    print("\nClassification report:\n", report)


if __name__ == "__main__":
    main()
