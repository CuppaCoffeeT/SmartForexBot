
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.data_loader import get_data
from model.features import add_fib_event_context
from train import (
    get_feature_list,
    create_event_labels,
    safe_auc_scorer,
    safe_auc,
    tune_threshold,
    choose_threshold_with_min_trades,
)
from model.model import get_model


PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X",
    "USDCAD=X","USDCHF=X","USDSGD=X",
    "EURJPY=X","EURGBP=X","EURAUD=X",
    "GBPJPY=X","AUDJPY=X","CADJPY=X","CHFJPY=X",
]
FEATURE_MODES = ["fib", "fib_ind", "fib_ind_norm"]


def evaluate_pair_mode(symbol: str, feature_mode: str,
                       interval: str = "1h", period: str = "60d",
                       window: int = 40, proximity_k: float = 0.35,
                       sma_fast: int = 20, sma_slow: int = 50, rsi_period: int = 14,
                       param_grid: Dict = None) -> Dict[str, float]:
    try:
        df0 = get_data(symbol, interval=interval, period=period)
        df1 = add_fib_event_context(
            df0, window=window, proximity_k=proximity_k,
            use_indicators=(feature_mode in {"fib_ind", "fib_ind_norm"}),
            use_normalized=(feature_mode == "fib_ind_norm"),
            sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period,
        )
        dfE = create_event_labels(df1)

        features = get_feature_list(feature_mode, sma_fast=sma_fast, sma_slow=sma_slow)
        X = dfE[features]
        y = dfE["signal"]
        mask = X.notna().all(axis=1)
        X, y = X[mask], y[mask]
        n = len(X)
        if n < 50:
            return {"symbol": symbol, "mode": feature_mode, "events": n, "auc": np.nan, "f1": np.nan,
                    "precision": np.nan, "recall": np.nan, "acc": np.nan, "thr": np.nan, "trades": 0}

        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        classes = np.unique(y_train)
        class_weight = {int(c): float(len(y_train) / (2 * (y_train == c).sum())) for c in classes}

        if param_grid is None:
            param_grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "min_samples_leaf": [5, 10]}

        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            get_model({"class_weight": class_weight, "random_state": 42}),
            param_grid,
            cv=tscv,
            scoring=safe_auc_scorer,
        )
        grid.fit(X_train, y_train)

        proba_train = grid.predict_proba(X_train)[:, 1]
        best_j_thr, best_f1_thr = tune_threshold(y_train, proba_train)
        thr = choose_threshold_with_min_trades(y_train, proba_train, min_trades=max(3, int(0.1 * len(y_train))))

        proba_test = grid.predict_proba(X_test)[:, 1]
        auc = safe_auc(y_test, proba_test)
        y_pred = (proba_test >= thr).astype(int)
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = rep.get("1", {}).get("precision", np.nan)
        recall = rep.get("1", {}).get("recall", np.nan)
        f1 = rep.get("1", {}).get("f1-score", np.nan)
        acc = rep.get("accuracy", np.nan)
        trades = int((proba_test >= thr).sum())

        return {
            "symbol": symbol,
            "mode": feature_mode,
            "events": n,
            "auc": float(auc) if auc is not None else np.nan,
            "precision": float(precision) if precision is not None else np.nan,
            "recall": float(recall) if recall is not None else np.nan,
            "f1": float(f1) if f1 is not None else np.nan,
            "acc": float(acc) if acc is not None else np.nan,
            "thr": float(thr),
            "trades": trades,
        }
    except Exception:
        return {"symbol": symbol, "mode": feature_mode, "events": 0, "auc": np.nan, "f1": np.nan,
                "precision": np.nan, "recall": np.nan, "acc": np.nan, "thr": np.nan, "trades": 0}


def main() -> None:
    rows: List[Dict[str, float]] = []
    for sym in PAIRS:
        for mode in FEATURE_MODES:
            res = evaluate_pair_mode(sym, mode)
            rows.append(res)
            print(f"{sym} {mode}: events={res['events']} auc={res['auc']:.3f} f1={res['f1']:.3f} trades={res['trades']}")

    df = pd.DataFrame(rows)
    out = ROOT / "evaluation" / "mode_compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[write] {out}")

    # Print a compact markdown pivot (AUC by mode)
    try:
        piv = df.pivot_table(index="symbol", columns="mode", values="auc")
        print("\nAUC by mode (markdown):\n")
        print(piv.to_markdown(floatfmt=".3f"))
    except Exception:
        pass


if __name__ == "__main__":
    main()

