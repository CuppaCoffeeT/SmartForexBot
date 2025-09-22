
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report, 
    confusion_matrix,
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


def load_artifact(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    payload = joblib.load(path)
    required = {"model", "threshold", "features", "meta"}
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"Artifact missing keys: {missing}")
    return payload


def build_events(symbol: str, interval: str, period: str, meta: Dict[str, float]) -> pd.DataFrame:
    df_raw = get_data(symbol, interval=interval, period=period).sort_index()
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
    events["Close"] = df_feat.loc[events.index, "Close"]
    return events


def score_events(events: pd.DataFrame, features: list[str], model, threshold: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = events.copy()
    missing = [f for f in features if f not in subset.columns]
    if missing:
        raise KeyError(f"Events frame missing features: {missing}")
    mask = subset[features].notna().all(axis=1)
    subset = subset.loc[mask]
    X = subset[features]
    proba = model.predict_proba(X)[:, 1]
    subset["probability"] = proba
    subset["prediction"] = (proba >= threshold).astype(int)
    subset["signed_confidence"] = proba - threshold
    return subset, subset["probability"], subset["prediction"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model metrics on a date window")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Path to model artifact")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL)
    parser.add_argument("--period", type=str, default=DEFAULT_PERIOD)
    parser.add_argument("--start", type=str, default=None, help="Optional ISO start date")
    parser.add_argument("--end", type=str, default=None, help="Optional ISO end date")
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    meta = artifact["meta"]
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
        print("No events found for the specified window.")
        return

    scored, proba, preds = score_events(events, artifact["features"], artifact["model"], float(artifact["threshold"]))
    y_true = scored["signal"].astype(int)

    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, proba)
    except ValueError:
        ap = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, digits=3, zero_division=0)

    print("=== Model Metrics ===")
    print(f"Total events evaluated: {len(scored)}")
    print(f"Positive rate (actual): {y_true.mean():.3f}")
    print(f"Threshold: {artifact['threshold']:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Average Precision: {ap:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("\nClassification report:\n", report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    signed = scored["signed_confidence"]
    if not signed.empty:
        buckets = pd.cut(
            signed,
            bins=[float("-inf"), -0.1, 0, 0.1, 0.2, float("inf")],
            labels=["<-0.10", "-0.10–0", "0–0.10", "0.10–0.20", ">0.20"],
        )
        bucket_summary = pd.concat(
            [
                buckets.value_counts().rename("events"),
                scored.groupby(buckets)["prediction"].mean().rename("predicted_rate"),
                scored.groupby(buckets)["signal"].mean().rename("actual_rate"),
            ],
            axis=1,
        ).fillna(0)
        print("\nSigned-confidence buckets:")
        print(bucket_summary)


if __name__ == "__main__":
    main()
