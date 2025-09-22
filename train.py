# train.py
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    make_scorer,
)
from pathlib import Path
import joblib
from typing import List, Dict, Any, Optional

#Safe AUC scorer 
def safe_auc_scorer(estimator, X, y):
    """
    GridSearchCV scoring function. Returns 0.5 when AUC is undefined
    (e.g., only one class in y) or when predict_proba is unavailable.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        proba = estimator.predict_proba(X)[:, 1]
    except Exception:
        return 0.5
    try:
        return roc_auc_score(y, proba)
    except Exception:
        return 0.5

# Direct AUC helper for use outside GridSearchCV 
def safe_auc(y_true, y_proba):
    """
    Direct AUC helper used outside GridSearchCV.
    Returns 0.5 when AUC is undefined (single-class y) or on any error.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return roc_auc_score(y_true, y_proba)
    except Exception:
        return 0.5

from model.data_loader import get_data
from model.features import add_fib_event_context
 #Feature set selector for ablation 
def get_feature_list(feature_mode: str, sma_fast: int = 20, sma_slow: int = 50):
    mode = feature_mode.lower()
    if mode == "fib":
        return [
            "ATR",
            "dist_fib_0.382", "dist_fib_0.5", "dist_fib_0.618",
            "is_support_context", "trend_up",
        ]
    elif mode == "fib_ind":
        return [
            "ATR",
            "dist_fib_0.382", "dist_fib_0.5", "dist_fib_0.618",
            "is_support_context", "trend_up",
            f"SMA_{sma_fast}", f"SMA_{sma_slow}", "SMA_spread",
            "RSI",
        ]
    elif mode == "fib_ind_norm":
        return [
            "ret_1", "ret_3",
            "RSI", "RSI_low", "RSI_high",
            f"SMA_{sma_fast}", f"SMA_{sma_slow}", "SMA_spread",
            "ATR", "ATR_pct",
            "ndist_fib_0.382", "ndist_fib_0.5", "ndist_fib_0.618",
            "is_support_context", "trend_up",
        ]
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
from model.model import get_model


def create_event_labels(
    df,
    horizon: int = 6,       
    tp: float = 0.002,      
    sl: float = 0.0015,     
):
    """
    Label only rows where a near-fib event occurs.
    Uptrend (support context): expect bounce up → positive if +tp, negative if -sl.
    Downtrend (resistance context): expect rejection down → positive if -tp, negative if +sl.
    """
    df = df.copy()
    fut_close = df["Close"].shift(-horizon)

    up_ctx = df["is_support_context"] == 1
    dn_ctx = df["is_support_context"] == 0
    hit    = df["near_any_fib"] == 1

    up_tp  = (fut_close >= df["Close"] * (1 + tp))
    up_sl  = (fut_close <= df["Close"] * (1 - sl))

    dn_tp  = (fut_close <= df["Close"] * (1 - tp))
    dn_sl  = (fut_close >= df["Close"] * (1 + sl))

    label = np.full(len(df), np.nan)

    # Positive class when expectation is met
    label[(up_ctx & hit & up_tp)] = 1
    label[(dn_ctx & hit & dn_tp)] = 1

    # Negative class when opposite happens
    label[(up_ctx & hit & up_sl)] = 0
    label[(dn_ctx & hit & dn_sl)] = 0

    df["event"] = ((hit & up_ctx) | (hit & dn_ctx)).astype(int)
    df["signal"] = label
    df = df[df["event"] == 1]
    df = df.dropna(subset=["signal"])
    df["signal"] = df["signal"].astype(int)
    return df


def tune_threshold(y_true, proba):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, thr = roc_curve(y_true, proba)
    j_scores = tpr - fpr
    best_j_thr = thr[j_scores.argmax()]
    prec, rec, thr_pr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_f1_thr = thr_pr[f1s.argmax()]
    return float(best_j_thr), float(best_f1_thr)

def choose_threshold_with_min_trades(y_true, proba, min_trades=5):
    """
    Pick a threshold that (a) favors F1 on TRAIN but (b) guarantees at least `min_trades` predicted positives.
    Falls back to the lowest threshold that satisfies the constraint; if none do, returns the F1-optimal threshold.
    """
    from sklearn.metrics import precision_recall_curve
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    prec, rec, thr = precision_recall_curve(y_true, proba)
    if len(thr) == 0:
        return 0.5
    # F1 for each threshold (skip last point which has no threshold)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    order = np.argsort(-f1s)  
    # Enforce minimum trade count constraint
    for idx in order:
        t = thr[idx]
        trades = int((proba >= t).sum())
        if trades >= min_trades:
            return float(t)
    # If none satisfy 
    trades_per_t = [(t, int((proba >= t).sum())) for t in thr]
    t_best = min(trades_per_t, key=lambda x: (-x[1], x[0]))[0]
    return float(t_best)

def save_model_artifact(model, threshold, feature_names, meta, out_dir="artifacts", name="model"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "threshold": float(threshold),
        "features": list(feature_names),
        "meta": meta,
    }
    fname = out_path / f"{name}.joblib"
    joblib.dump(payload, fname)
    print(f"[save] model artifact -> {fname}")

def train_model(
    symbol="EURUSD=X",
    interval="1h",
    period="60d",
    window=40,
    horizon=6,
    tp=0.002,
    sl=0.0015,
    proximity_k=0.35,
    feature_mode="fib_ind",         
    sma_fast=20, sma_slow=50, rsi_period=14,  
    save_artifact=True,
    artifact_dir="artifacts",
):
   
    df = get_data(symbol, interval, period)

    # Fib event context
    df = add_fib_event_context(
        df,
        window=window,
        proximity_k=proximity_k,
        use_indicators=(feature_mode in {"fib_ind", "fib_ind_norm"}),
        use_normalized=(feature_mode == "fib_ind_norm"),
        sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period,
    )

    # Labels 
    df = create_event_labels(df, horizon=horizon, tp=tp, sl=sl)
    if len(df) < 50:
        print("[events] Not enough Fib events after filtering; "
              "try a larger period, smaller window, or bigger proximity_k.")
        return None, None, None, None

    # Features
    features = get_feature_list(feature_mode, sma_fast=sma_fast, sma_slow=sma_slow)
    X = df[features]
    y = df["signal"]
    
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"[events] total={len(y)}  pos={pos}  neg={neg}  pos%={(pos/len(y))*100:.1f}")

    # Chronological split 
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"[split] train={len(X_train)}, test={len(X_test)}")

    # Baseline
    maj = int(y_train.value_counts().idxmax())
    baseline_acc = (y_test == maj).mean()
    print(f"[baseline] majority acc on test: {baseline_acc:.3f}  (majority_class={maj})")

    #  CV + class weights
    tscv = TimeSeriesSplit(n_splits=3)
    classes = np.unique(y_train)
    class_weight = {int(c): float(len(y_train) / (2 * (y_train == c).sum())) for c in classes}

    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10],
    }
    grid = GridSearchCV(
        get_model({"class_weight": class_weight, "random_state": 42}),
        param_grid,
        cv=tscv,
        scoring=safe_auc_scorer,
    )
    grid.fit(X_train, y_train)
    proba = grid.predict_proba(X_test)[:, 1]
    if hasattr(grid.best_estimator_, "feature_importances_"):
        importances = grid.best_estimator_.feature_importances_
        order = np.argsort(importances)[::-1]
        print("\n[feature importance]")
        for idx in order:
            print(f"  {features[idx]:<18} {importances[idx]:.3f}")

    #  Default 0.5 threshold metrics
    print("Best params:", grid.best_params_)
    print("Test AUC:", roc_auc_score(y_test, proba))
    y_pred_default = (proba >= 0.5).astype(int)
    print("Test accuracy (thr=0.5):", accuracy_score(y_test, y_pred_default))
    print("\nClassification Report (thr=0.5):\n", classification_report(y_test, y_pred_default))

    # Threshold tuning (on train to define an operating point)
    train_proba = grid.predict_proba(X_train)[:, 1]
    best_j_thr, best_f1_thr = tune_threshold(y_train, train_proba)
    chosen_thr = choose_threshold_with_min_trades(y_train, train_proba, min_trades=max(3, int(0.1 * len(y_train))))

    y_pred_adj = (proba >= chosen_thr).astype(int)
    print(f"[thresholds] YoudenJ(train)={best_j_thr:.3f}  F1(train)={best_f1_thr:.3f}  -> using {chosen_thr:.3f}")
    print("[adjusted threshold] acc:", accuracy_score(y_test, y_pred_adj))
    print("[adjusted threshold] report:\n", classification_report(y_test, y_pred_adj))

    # Backtest simulation
    bt_default = backtest(y_test.to_numpy(), proba, threshold=0.5)
    bt_adj = backtest(y_test.to_numpy(), proba, threshold=chosen_thr)

    def _bt_pretty(d: Dict[str, Any]):
        return {k: v for k, v in d.items() if k not in {"returns", "pnl_raw", "expectancy_raw"}}

    print("\n[backtest results]")
    print(" Default thr=0.5 :", _bt_pretty(bt_default))
    print(f" Adjusted thr={chosen_thr:.3f} :", _bt_pretty(bt_adj))

    # Save the operating model + threshold for the web app
    if save_artifact:
        meta = {
            "symbol": symbol,
            "interval": interval,
            "period": period,
            "window": window,
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "proximity_k": proximity_k,
            "best_params": grid.best_params_,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_mode": feature_mode,
            "sma_fast": sma_fast, "sma_slow": sma_slow, "rsi_period": rsi_period,
        }
        # Include feature_mode in artifact name so can have variants 
        model_name = f"{symbol.replace('=','')}_{interval}_{feature_mode}"
        save_model_artifact(grid.best_estimator_, chosen_thr, features, meta, out_dir=artifact_dir, name=model_name)

    return grid, df, X_test, y_test

def backtest(y_true, proba, threshold=0.5, reward=1.0, risk=1.0):

    y_pred = (proba >= threshold).astype(int)
    trades = (y_pred == 1)  
    
    results = []
    for pred, actual in zip(y_pred, y_true):
        if pred == 1:  # take a trade
            if actual == 1:
                results.append(reward)   # win
            else:
                results.append(-risk)    # loss
    
    if not results:
        return {
            "trades": 0,
            "win_rate": None,
            "pnl": 0.0,
            "expectancy": None,
            "returns": [],
            "pnl_raw": 0.0,
            "expectancy_raw": None,
        }

    pnl = float(np.sum(results))
    wins = sum(1 for r in results if r > 0)
    win_rate = wins / len(results)
    expectancy = pnl / len(results)

    return {
        "trades": len(results),
        "win_rate": round(win_rate, 2),
        "pnl": round(pnl, 2),
        "expectancy": round(expectancy, 2),
        "returns": results,
        "pnl_raw": pnl,
        "expectancy_raw": expectancy,
    }


def white_reality_check(returns: List[float], block_size: Optional[int] = None, n_boot: int = 2000,
                        random_state: int = 42) -> Optional[Dict[str, float]]:
    """Simple White's Reality Check using moving-block bootstrap against zero-mean null."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return None

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean / std * np.sqrt(n)) if std > 0 else float("nan")
    t_stat = (np.sqrt(n) * mean / (std + 1e-12)) if std > 0 else float("nan")

    if block_size is None:
        block_size = max(1, min(10, n // 5))
    block_size = max(1, block_size)
    rng = np.random.default_rng(random_state)

    centered = arr - mean
    boot_means = []
    max_start = max(1, n - block_size + 1)
    for _ in range(n_boot):
        sample = []
        while len(sample) < n:
            start = rng.integers(0, max_start)
            block = centered[start:start + block_size]
            sample.extend(block)
        sample = np.asarray(sample[:n])
        boot_means.append(sample.mean())
    boot_means = np.asarray(boot_means)

    pvalue = float((np.sum(boot_means >= mean) + 1) / (len(boot_means) + 1))

    # 95% CI using percentile method around observed mean
    ci_low = mean - np.quantile(boot_means, 0.975)
    ci_high = mean - np.quantile(boot_means, 0.025)

    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "t_stat": t_stat,
        "pvalue": pvalue,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

def walk_forward_evaluate(
    symbol="EURUSD=X",
    interval="1h",
    period="60d",
    window=40,
    horizon=6,
    tp=0.002,
    sl=0.0015,
    proximity_k=0.35,
    n_folds=5,
    param_grid={"n_estimators": [200, 400], "max_depth": [3, 5, 7]},
    min_trades=5,
    fixed_window=False,
    feature_mode="fib_ind",
    sma_fast=20, sma_slow=50, rsi_period=14,
):
    # Prepare dataset once
    df0 = get_data(symbol, interval, period)
    df1 = add_fib_event_context(
        df0, window=window, proximity_k=proximity_k,
        use_indicators=(feature_mode in {"fib_ind", "fib_ind_norm"}),
        use_normalized=(feature_mode == "fib_ind_norm"),
        sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period,
    )
    dfE = create_event_labels(df1, horizon=horizon, tp=tp, sl=sl)

    features = get_feature_list(feature_mode, sma_fast=sma_fast, sma_slow=sma_slow)
    X = dfE[features]
    y = dfE["signal"]
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    n = len(X)
    if n < (n_folds + 1) * 20:
        print(f"[walk-forward] Not enough events ({n}). Reduce n_folds or loosen filters.")
        return

    chunk = n // (n_folds + 1) 
    results = []
    print(f"[walk-forward] total_events={n}  folds={n_folds}  chunk={chunk}  fixed_window={fixed_window}")

    for k in range(n_folds):
        if fixed_window:
            # rolling window of one chunk for train, next chunk for test
            train_start = max(0, chunk * k)
            train_end   = min(chunk * (k + 1), n)
            test_start  = train_end
            test_end    = min(train_end + chunk, n)
        else:
            train_start = 0
            train_end   = min(chunk * (k + 1), n)
            test_start  = train_end
            test_end    = min(chunk * (k + 2), n)

        X_train, y_train = X.iloc[train_start:train_end], y.iloc[train_start:train_end]
        X_test,  y_test  = X.iloc[test_start:test_end],  y.iloc[test_start:test_end]

        if len(X_test) == 0 or len(X_train) < 30:
            continue
        # Skip folds where train or test has only a single class
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[fold {k+1}] skipped (single-class in train/test: train_classes={np.unique(y_train)}, test_classes={np.unique(y_test)})")
            continue

        # class weights
        classes = np.unique(y_train)
        class_weight = {int(c): float(len(y_train) / (2 * (y_train == c).sum())) for c in classes}

        # fit model
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            get_model({"class_weight": class_weight, "random_state": 42}),
            param_grid,
            cv=tscv,
            scoring=safe_auc_scorer,
        )
        grid.fit(X_train, y_train)

        # thresholds based on TRAIN
        train_proba = grid.predict_proba(X_train)[:, 1]
        best_j_thr, best_f1_thr = tune_threshold(y_train, train_proba)
        thr = choose_threshold_with_min_trades(y_train, train_proba, min_trades=max(min_trades, int(0.1 * len(y_train))))

        # evaluate on TEST
        test_proba = grid.predict_proba(X_test)[:, 1]

        # ensure at least a few trades in test; otherwise reuse prior threshold if available
        if (test_proba >= thr).sum() < max(1, int(0.05 * len(X_test))) and results:
            thr = float(results[-1]["thr"])  

        auc = safe_auc(y_test, test_proba)
        y_pred = (test_proba >= thr).astype(int)

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = rep["accuracy"]
        bt = backtest(y_test.to_numpy(), test_proba, threshold=thr)

        results.append({
            "fold": k+1,
            "train": len(X_train),
            "test": len(X_test),
            "auc": round(auc, 3),
            "acc": round(acc, 3),
            "thr": round(thr, 3),
            "trades": bt["trades"],
            "win_rate": bt["win_rate"],
            "expectancy": bt["expectancy"],
            "pnl": bt["pnl"],
            "returns": bt["returns"],
            "pnl_raw": bt["pnl_raw"],
            "expectancy_raw": bt["expectancy_raw"],
        })

        print(f"[fold {k+1}] train={len(X_train)} test={len(X_test)} | AUC={auc:.3f} acc={acc:.3f} thr={thr:.3f} "
              f"| trades={bt['trades']} win%={bt['win_rate']} exp={bt['expectancy']} pnl={bt['pnl']}")

    if results:
        # summary
        avg_auc = np.mean([r["auc"] for r in results])
        avg_acc = np.mean([r["acc"] for r in results])
        exp_vals = [r["expectancy_raw"] for r in results if r["expectancy_raw"] is not None]
        avg_exp = float(np.mean(exp_vals)) if exp_vals else float("nan")
        total_pnl = np.sum([r["pnl_raw"] for r in results])
        total_trades = np.sum([r["trades"] for r in results])

        print("\n[walk-forward summary]")
        print(f" folds={len(results)}  avg_auc={avg_auc:.3f}  avg_acc={avg_acc:.3f}  "
              f"avg_expectancy={avg_exp:.3f}  trades={int(total_trades)}  pnl={total_pnl:.2f}")

        all_returns: List[float] = []
        for r in results:
            if r.get("returns"):
                all_returns.extend(r["returns"])
        stats = white_reality_check(all_returns)
        if stats:
            print("[stat-test] White Reality Check (one-sided, H0: mean return <= 0)")
            print(
                f" trades={stats['n']} mean={stats['mean']:.3f} sharpe={stats['sharpe']:.2f} "
                f"t={stats['t_stat']:.2f} p={stats['pvalue']:.3f} "
                f"ci95=({stats['ci_low']:.3f}, {stats['ci_high']:.3f})"
            )


def _example_single_run():
    """Example single-pair training + walk-forward (not executed by default)."""
    train_model(
        symbol="EURUSD=X",
        interval="1h",
        period="60d",
        window=40,
        horizon=6,
        tp=0.002,
        sl=0.0015,
        proximity_k=0.35,
        feature_mode="fib",
        sma_fast=10, sma_slow=40, rsi_period=10,
    )
    walk_forward_evaluate(
        symbol="EURUSD=X",
        interval="1h",
        period="60d",
        window=40,
        horizon=6,
        tp=0.002,
        sl=0.0015,
        proximity_k=0.35,
        n_folds=3,
        min_trades=5,
        fixed_window=False,
        feature_mode="fib",
        sma_fast=10, sma_slow=40, rsi_period=10,
    )



# scripts/batch_train.py 
from train import train_model

PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X",
    "USDCAD=X","USDCHF=X","USDSGD=X",
    "EURJPY=X","EURGBP=X","EURAUD=X",
    "GBPJPY=X","AUDJPY=X","CADJPY=X","CHFJPY=X",
]

def main():
    interval = "1h"
    period = "60d"
    feature_modes = ["fib", "fib_ind", "fib_ind_norm"]
    for sym in PAIRS:
        for mode in feature_modes:
            print("\n"+"="*60)
            print(f"[batch] training {sym} {interval} {period} mode={mode}")
            try:
                train_model(
                    symbol=sym,
                    interval=interval,
                    period=period,
                    window=40,
                    horizon=6,
                    tp=0.002, sl=0.0015,
                    proximity_k=0.35,
                    feature_mode=mode,
                    sma_fast=10, sma_slow=40, rsi_period=10,
                    save_artifact=True,
                    artifact_dir="artifacts",
                )
            except Exception as e:
                print(f"[batch][{sym} {interval} {mode}] skipped: {e}")

if __name__ == "__main__":
    main()
