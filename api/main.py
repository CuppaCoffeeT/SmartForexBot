from __future__ import annotations
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import numpy as np
import pandas as pd
import joblib

from utils.llm_client import generate_text as llm_generate_text, has_client as llm_has_client

# Re-use your existing modules
from model.data_loader import get_data
from model.features import add_fib_event_context
from train import get_feature_list, safe_auc 

FIB_LEVELS = [0.382, 0.5, 0.618]


# Model Registry / Loader

@dataclass
class ModelArtifact:
    estimator: Any
    threshold: float
    features: List[str]
    meta: Dict[str, Any]

_cached_artifacts: Dict[str, ModelArtifact] = {}


def load_artifact(symbol: str = "EURUSD=X", interval: str = "1h", artifact_dir: str = "artifacts", feature_mode: Optional[str] = None) -> ModelArtifact:
    clean = symbol.replace('=','')
    tried: List[str] = []
    # Prefer mode-specific filename
    if feature_mode:
        key = f"{clean}_{interval}_{feature_mode}"
        tried.append(key)
        if key in _cached_artifacts:
            return _cached_artifacts[key]
        path = os.path.join(artifact_dir, f"{key}.joblib")
        if os.path.exists(path):
            payload = joblib.load(path)
            art = ModelArtifact(
                estimator=payload["model"],
                threshold=float(payload["threshold"]),
                features=list(payload["features"]),
                meta=dict(payload["meta"]),
            )
            _cached_artifacts[key] = art
            return art
    # Legacy fallback
    legacy_key = f"{clean}_{interval}"
    tried.append(legacy_key)
    if legacy_key in _cached_artifacts:
        return _cached_artifacts[legacy_key]
    legacy_path = os.path.join(artifact_dir, f"{legacy_key}.joblib")
    payload = joblib.load(legacy_path)
    art = ModelArtifact(
        estimator=payload["model"],
        threshold=float(payload["threshold"]),
        features=list(payload["features"]),
        meta=dict(payload["meta"]),
    )
    _cached_artifacts[legacy_key] = art
    return art


# Utterance → intent 
INTENTS = {
    "ask_followup_why": [r"why.*(work|setup|trade|signal)", r"how.*work", r"what.*makes.*(setup|signal)"],
    "ask_risk": [r"risk", r"stop", r"position", r"manage.*risk", r"drawdown"],
    "ask_confidence": [r"confidence", r"probability", r"chance", r"sure"],
    "ask_reco": [r"what.*(trade|pair).*now", r"(recommend|reco).*", r"best.*pair", r"top\s*\d+"],
    "ask_pair_status": [r"(eurusd|gbpusd|usdjpy|audusd|usdchf|usdcad|nzdusd|usdsgd|eurjpy|eurgbp|euraud|gbpjpy|audjpy|cadjpy|chfjpy|[A-Z]{3}[\/\s-]?[A-Z]{3})"],
    "ask_explain": [r"why.*(buy|sell|signal|recommend)", r"explain|because|reason|justif"],
    "ask_chart": [r"show.*(chart|candle|visual|plot)"],
    "ask_compare": [r"compare|vs|versus"],
    "ask_topk": [r"top\s*(\d+)\s*(pairs|setups|trades)"],
    "help": [r"help|how.*use|what.*can.*you.*do"],
}

def detect_intent(utterance: str) -> str:
    u = utterance.lower().strip()
    for intent, patterns in INTENTS.items():
        for p in patterns:
            if re.search(p, u):
                return intent
    if extract_symbol(utterance):
        return "ask_pair_status"
    return "ask_reco"


# Currency-only extraction
_CURS = {"EUR","USD","GBP","JPY","AUD","NZD","CAD","CHF","SGD"}

def extract_currency(text: str) -> Optional[str]:
    m = re.search(r"\b(eur|usd|gbp|jpy|aud|nzd|cad|chf|sgd)\b", text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper()

def pairs_with_currency(currency: str, pool: Optional[List[str]] = None) -> List[str]:
    cur = currency.upper()
    pairs = list(pool) if pool is not None else list(DEFAULT_PAIRS)
    want = []
    for p in pairs:
        base = p.replace("=X", "")
        if len(base) != 6:
            continue
        if base[:3] == cur or base[3:] == cur:
            want.append(p)
    return want

def best_for_currency(currency: str, topk: int = 3, feature_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    subset = pairs_with_currency(currency)
    if not subset:
        return []
    return rank_recommendations(pairs=subset, topk=topk, feature_mode=feature_mode)

def compare_pairs(symbol_a: str, symbol_b: str) -> Dict[str, Any]:
    a = get_pair_insight(symbol_a)
    b = get_pair_insight(symbol_b)
    # confidence score
    def score(ins):  
        s = ins["insight"]
        if not s.get("has_event"):
            return 0.0
        return float(abs(s["prob"] - s["threshold"]))
    sa, sb = score(a), score(b)
    better = symbol_a if sa >= sb else symbol_b
    reason = f"{better} has higher confidence margin ({max(sa, sb):.2f})"
    return {"a": a, "b": b, "better": better, "reason": reason, "scores": {"a": sa, "b": sb}}


# Symbol normalization
YF_SUFFIX = "=X"  # e.g., EURUSD -> EURUSD=X
KNOWN = {
    "EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD",
    "USDCAD","USDCHF","USDSGD",
    "EURJPY","EURGBP","EURAUD",
    "GBPJPY","AUDJPY","CADJPY","CHFJPY",
}


def normalize_symbol(token: str) -> Optional[str]:
    t = token.upper().replace("/", "")
    if len(t) == 6 and t.isalpha():
        return t + YF_SUFFIX
    if t.endswith("=X"):
        return t
    return None


def extract_symbol(text: str) -> Optional[str]:
    for s in sorted(KNOWN, key=len, reverse=True):
        if re.search(rf"\b{s}\b", text, re.IGNORECASE):
            return s + YF_SUFFIX
        if re.search(rf"\b{s[:3]}[\/\s-]?{s[3:]}\b", text, re.IGNORECASE):
            return s + YF_SUFFIX
    # Generic XXX/YYY 
    m = re.search(r"\b([A-Za-z]{3})[\/\s-]?([A-Za-z]{3})\b", text)
    if m:
        a, b = m.group(1).upper(), m.group(2).upper()
        if (a in {"EUR","USD","GBP","JPY","AUD","NZD","CAD","CHF","SGD"} and
            b in {"EUR","USD","GBP","JPY","AUD","NZD","CAD","CHF","SGD"}):
            return normalize_symbol(a + b)
    return None


# scoring logic

def prepare_live_frame(symbol: str, interval: str, period: str, window: int, proximity_k: float,
                        feature_mode: str, sma_fast: int, sma_slow: int, rsi_period: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch fresh data and compute features + event labels (without lookahead labeling)."""
    df = get_data(symbol, interval, period)
    df_feat = add_fib_event_context(
        df,
        window=window,
        proximity_k=proximity_k,
        use_indicators=(feature_mode in {"fib_ind", "fib_ind_norm"}),
        use_normalized=(feature_mode == "fib_ind_norm"),
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        rsi_period=rsi_period,
    )
    return df, df_feat


def score_last_event(art: ModelArtifact, df_feat: pd.DataFrame) -> Dict[str, Any]:
    cands = df_feat[(df_feat.get("near_any_fib", 0) == 1) & df_feat[art.features].notna().all(axis=1)]
    if cands.empty:
        return {"has_event": False}
    row = cands.iloc[-1]
    X = row[art.features].to_frame().T
    proba = float(art.estimator.predict_proba(X)[:, 1][0])
    take = int(proba >= art.threshold)
    conf = abs(proba - art.threshold)
    context_up = bool(row.get("is_support_context", 1) == 1)
    direction = "BUY" if (context_up and take) else ("SELL" if (not context_up and take) else "HOLD")

    closest_label = None
    closest_level_value = None
    entry_low = entry_high = None
    eps_val = float(row.get("eps", np.nan))
    nearest = []
    for lvl in FIB_LEVELS:
        dist = row.get(f"dist_fib_{lvl}")
        if dist is None or not np.isfinite(dist):
            continue
        nearest.append((abs(float(dist)), lvl))
    if nearest:
        nearest.sort(key=lambda x: x[0])
        _, closest_label = nearest[0]
        fib_col = f"Fib_{closest_label}"
        if fib_col in row:
            closest_level_value = float(row[fib_col])
            if np.isfinite(eps_val):
                entry_low = closest_level_value - eps_val
                entry_high = closest_level_value + eps_val

    return {
        "has_event": True,
        "prob": round(proba, 3),
        "threshold": round(art.threshold, 3),
        "take": take,
        "direction": direction,
        "context_up": context_up,
        "row_ts": getattr(row, "name", None),
        "fib_dists": {f"dist_fib_{lvl}": row.get(f"dist_fib_{lvl}", np.nan) for lvl in FIB_LEVELS},
        "entry": {
            "fib_label": closest_label,
            "level": closest_level_value,
            "zone_low": entry_low,
            "zone_high": entry_high,
            "half_hint": eps_val if np.isfinite(eps_val) else None,
        },
    }


def _state_description(state: str, direction: Optional[str], fib_label: Optional[float]) -> str:
    label_txt = f"the {fib_label:.3f} retracement" if isinstance(fib_label, (int, float)) else "the retracement"
    if state == "in_zone":
        return f"Price is sitting inside {label_txt} zone."
    if state == "climbing_toward":
        if direction == "SELL":
            return f"Price is climbing into {label_txt} resistance."
        return f"Price is climbing toward {label_txt} from below."
    if state == "dropping_toward":
        if direction == "BUY":
            return f"Price is dropping into {label_txt} support."
        return f"Price is dropping toward {label_txt} from above."
    if state == "leaving_below":
        return f"Price is sliding away below {label_txt} after the test."
    if state == "leaving_above":
        return f"Price is breaking away above {label_txt}."
    if state == "no_event":
        return "No active retracement setup right now."
    if state == "unavailable":
        return "Retracement zone unavailable for this signal."
    return "Price is not near a retracement setup right now."


def _classify_entry_motion(entry: Dict[str, Any], direction: Optional[str], close: Optional[float], prev_close: Optional[float]) -> None:
    if not isinstance(entry, dict):
        return

    entry["close"] = float(close) if close is not None else entry.get("close")
    entry["prev_close"] = float(prev_close) if prev_close is not None else entry.get("prev_close")

    level = entry.get("level")
    zone_low = entry.get("zone_low")
    zone_high = entry.get("zone_high")

    if level is None or zone_low is None or zone_high is None or close is None:
        state = entry.get("state", "unavailable")
        entry["state"] = state
        entry["state_text"] = _state_description(state, direction, entry.get("fib_label"))
        entry.setdefault("proximity", None)
        entry.setdefault("delta", None)
        return

    # Normalise bounds 
    zone_low_f = float(zone_low)
    zone_high_f = float(zone_high)
    if zone_high_f < zone_low_f:
        zone_low_f, zone_high_f = zone_high_f, zone_low_f
    entry["zone_low"] = zone_low_f
    entry["zone_high"] = zone_high_f

    close_f = float(close)
    level_f = float(level)

    delta = close_f - level_f
    entry["delta"] = delta
    entry["abs_delta"] = abs(delta)

    half_width = max(zone_high_f - level_f, level_f - zone_low_f, 0.0)
    proximity = abs(delta) / half_width if half_width > 1e-9 else None
    entry["proximity"] = proximity
    entry["half_width"] = half_width

    prev = prev_close if prev_close is not None else close_f
    recent_move = close_f - float(prev)
    entry["recent_move"] = recent_move

    if zone_low_f <= close_f <= zone_high_f:
        state = "in_zone"
    elif close_f < zone_low_f:
        state = "climbing_toward" if recent_move >= 0 else "leaving_below"
    else:
        state = "dropping_toward" if recent_move <= 0 else "leaving_above"

    entry["state"] = state
    entry["state_text"] = _state_description(state, direction, entry.get("fib_label"))


def fib_levels_from_window(df: pd.DataFrame, window: int) -> Dict[str, float]:

    if len(df) < window:
        window = len(df)
    high = float(df["High"].iloc[-window:].max())
    low = float(df["Low"].iloc[-window:].min())
    return {
        "0.0": low,
        "0.382": low + (high - low) * 0.382,
        "0.5": low + (high - low) * 0.5,
        "0.618": low + (high - low) * 0.618,
        "1.0": high,
    }

# Public service: pair insight
def get_pair_insight(symbol: str, interval: Optional[str] = None, artifacts_dir: str = "artifacts", feature_mode: Optional[str] = None) -> Dict[str, Any]:

    req_interval = (interval or "").lower() or None
    try_interval = req_interval or "1h"
    try:
        art = load_artifact(symbol=symbol, interval=try_interval, artifact_dir=artifacts_dir, feature_mode=feature_mode)
    except Exception:
        if try_interval != "1h":
            art = load_artifact(symbol=symbol, interval="1h", artifact_dir=artifacts_dir, feature_mode=feature_mode)
        else:
            raise
    meta = art.meta
    data_interval = req_interval or meta.get("interval", "1h")
    df, df_feat = prepare_live_frame(
        symbol=symbol,
        interval=data_interval,
        period=meta.get("period", "60d"),
        window=int(meta.get("window", 40)),
        proximity_k=float(meta.get("proximity_k", 0.35)),
        feature_mode=meta.get("feature_mode", feature_mode or "fib"),
        sma_fast=int(meta.get("sma_fast", 20)),
        sma_slow=int(meta.get("sma_slow", 50)),
        rsi_period=int(meta.get("rsi_period", 14)),
    )
    levels = fib_levels_from_window(df, int(meta.get("window", 40)))
    scoring = score_last_event(art, df_feat)
    try:
        close_price = float(df["Close"].iloc[-1])
    except Exception:
        close_price = None
    try:
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
    except Exception:
        prev_close = None
    entry = scoring.get("entry") if isinstance(scoring.get("entry"), dict) else None
    if entry:
        label = entry.get("fib_label")
        level_from_grid = None
        if isinstance(label, (int, float)):
            label = round(float(label), 3)
        if isinstance(label, float):
            level_from_grid = levels.get(f"{label:.3f}")
        if level_from_grid is None and label is not None:
            level_from_grid = levels.get(str(label))
        if level_from_grid is not None and np.isfinite(level_from_grid):
            entry["level"] = float(level_from_grid)
            half_hint = entry.get("half_hint")
            if half_hint is not None and np.isfinite(half_hint) and half_hint > 0:
                half = float(half_hint)
            else:
                lo = entry.get("zone_low")
                hi = entry.get("zone_high")
                half = None
                if lo is not None and hi is not None:
                    try:
                        lo = float(lo)
                        hi = float(hi)
                        half = max(abs(entry["level"] - lo), abs(hi - entry["level"]))
                    except Exception:
                        half = None
            if half is None or not np.isfinite(half) or half <= 0:
                half = 0.0001 * max(entry["level"], 1.0)
            entry["zone_low"] = entry["level"] - half
            entry["zone_high"] = entry["level"] + half
        _classify_entry_motion(entry, scoring.get("direction"), close_price, prev_close)
        if entry.get("state"):
            scoring["entry_state"] = entry["state"]
            scoring["entry_state_text"] = entry.get("state_text")
    return {
        "symbol": symbol,
        "insight": scoring,
        "levels": levels,
        "ohlc": df.tail(300)[["Open", "High", "Low", "Close"]].reset_index(names=["time"]).to_dict(orient="list"),
        "meta": {**meta, "model_interval": try_interval, "data_interval": data_interval},
    }

# Recommendation engine 
DEFAULT_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "USDSGD=X",
    "EURJPY=X", "EURGBP=X", "EURAUD=X",
    "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X",
]


def _is_trending(df_feat: pd.DataFrame) -> bool:
    r = df_feat.tail(50)
    if r.empty: return False
    spread = r.get("SMA_spread")
    if spread is None: return True
    return bool((spread.tail(10).abs().mean()) > (r["ATR"].tail(10).mean() * 0.05))

def rank_recommendations(pairs: List[str] = DEFAULT_PAIRS, topk: int = 3, feature_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    recos = []
    for sym in pairs:
        try:
            info = get_pair_insight(sym, feature_mode=feature_mode)
            ins = info["insight"]
            if not ins.get("has_event"):
                continue
            prob_val = float(ins["prob"])
            thr_val = float(ins["threshold"])
            score = prob_val - thr_val
            bias = 0.0
            for lvl in (0.382, 0.5, 0.618):
                d = ins["fib_dists"].get(f"dist_fib_{lvl}")
                if pd.notna(d): bias += 1.0 / (1e-6 + abs(d))
            recos.append({
                "symbol": sym,
                "direction": ins["direction"],
                "prob": round(prob_val, 3),
                "threshold": round(thr_val, 3),
                "confidence": round(score, 3),
                "bias": round(bias, 3),
            })
        except Exception:
            continue
    recos.sort(key=lambda r: (r["confidence"], r["bias"]), reverse=True)
    return recos[:topk]


# Ollama entry

_LLM_SYSTEM_PROMPT = (
    "You are SmartForexBot, a concise forex assistant focused on Fibonacci retracement trades. "
    "Use the supplied analytics to speak directly to the trader with precise, confident prose."
)
_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.35"))
_LLM_KIND_INSTRUCTIONS = {
    "signal_detail": (
        "Write no more than two tight sentences for a swing trader. Highlight the direction, probability vs "
        "threshold, nearest Fibonacci level, and a quick risk reminder."
    ),
    "signal_simple": (
        "Explain for a beginner in two to three short sentences. Start with the plain recommendation (Buy/Sell/Wait) "
        "and relate it to the Fibonacci checkpoint."
    ),
    "recommendation": (
        "Summarise the trade idea in two sentences, mentioning direction, confidence margin, and Fib interaction."
    ),
    "followup": (
        "Provide two sentences explaining why the setup still makes sense, citing classifier bias and the Fib zone."
    ),
    "risk": (
        "Give one to two sentences outlining the risk plan around the Fib zone, including stop placement guidance."
    ),
    "confidence": (
        "Explain the confidence score in one to two sentences, including what could invalidate the signal."
    ),
}
_LLM_KIND_MAX_TOKENS = {
    "signal_simple": 170,
    "confidence": 180,
    "risk": 160,
}


def _confidence_label(margin: float) -> str:
    if margin >= 0.25:
        return "strong"
    if margin >= 0.15:
        return "moderate"
    if margin >= 0.08:
        return "weak"
    return "very weak"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def _format_entry_zone(entry: Dict[str, Any]) -> Optional[str]:
    lo = _safe_float(entry.get("zone_low"))
    hi = _safe_float(entry.get("zone_high"))
    if lo is not None and hi is not None and hi > lo:
        return f"{lo:.5f} - {hi:.5f}"
    level = _safe_float(entry.get("level"))
    if level is not None:
        return f"{level:.5f}"
    return None


def _compute_nearest_fib_label(info: Dict[str, Any], ins: Dict[str, Any]) -> Optional[str]:
    fibs = ins.get("fib_dists", {}) or {}
    near = []
    for lvl in (0.618, 0.5, 0.382):
        dist_val = _safe_float(fibs.get(f"dist_fib_{lvl}"))
        if dist_val is not None:
            near.append((lvl, abs(dist_val)))
    if near:
        near.sort(key=lambda x: x[1])
        return f"{near[0][0]:.3f}"

    levels = (info or {}).get("levels") or {}
    ohlc = (info or {}).get("ohlc") or {}
    closes = ohlc.get("Close") or []
    try:
        close_val = float(closes[-1])
    except (TypeError, ValueError, IndexError):
        close_val = None
    if close_val is None:
        return None

    cands = []
    for label, lvl_val in levels.items():
        lvl_float = _safe_float(lvl_val)
        if lvl_float is not None:
            cands.append((label, abs(lvl_float - close_val)))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])
    return str(cands[0][0])


def _summarize_for_llm(info: Dict[str, Any]) -> Dict[str, Any]:
    ins = info.get("insight", {}) or {}
    entry = ins.get("entry") or {}
    meta = info.get("meta", {}) or {}
    prob = _safe_float(ins.get("prob"))
    thr = _safe_float(ins.get("threshold"))
    margin = abs(prob - thr) if prob is not None and thr is not None else None
    confidence_tag = _confidence_label(margin) if margin is not None else None
    entry_zone = _format_entry_zone(entry)
    nearest_fib = _compute_nearest_fib_label(info, ins)
    width = None
    zone_low = _safe_float(entry.get("zone_low"))
    zone_high = _safe_float(entry.get("zone_high"))
    if zone_low is not None and zone_high is not None:
        width = zone_high - zone_low

    return {
        "symbol": info.get("symbol", "?"),
        "has_event": bool(ins.get("has_event")),
        "direction": ins.get("direction") or "HOLD",
        "prob": prob,
        "threshold": thr,
        "margin": margin,
        "confidence_label": confidence_tag,
        "context_up": bool(ins.get("context_up")),
        "nearest_fib": nearest_fib,
        "entry_label": entry.get("fib_label"),
        "entry_zone": entry_zone,
        "state_text": ins.get("entry_state_text") or entry.get("state_text"),
        "proximity": _safe_float(entry.get("proximity")),
        "recent_move": _safe_float(entry.get("recent_move")),
        "risk_band": width,
        "half_width": _safe_float(entry.get("half_width")),
        "train_size": meta.get("train_size"),
        "test_size": meta.get("test_size"),
    }


def _build_llm_prompt(kind: str, summary: Dict[str, Any], fallback: str) -> str:
    instructions = _LLM_KIND_INSTRUCTIONS.get(kind, _LLM_KIND_INSTRUCTIONS["signal_detail"])
    lines = [
        f"Pair: {summary['symbol']}",
        f"Direction: {summary['direction']}",
        f"HasActiveEvent: {summary['has_event']}",
        f"TrendContext: {'uptrend pullback' if summary['context_up'] else 'downtrend rally'}",
    ]
    if summary["prob"] is not None and summary["threshold"] is not None:
        lines.append(f"ClassifierProbability: {summary['prob']:.3f}")
        lines.append(f"ClassifierThreshold: {summary['threshold']:.3f}")
    if summary["margin"] is not None:
        lines.append(f"ConfidenceMargin: {summary['margin']:.3f}")
    if summary["confidence_label"]:
        lines.append(f"ConfidenceTag: {summary['confidence_label']}")
    if summary["nearest_fib"]:
        lines.append(f"NearestFibLevel: {summary['nearest_fib']}")
    if summary["entry_label"] is not None:
        lines.append(f"SignalFibLabel: {summary['entry_label']}")
    if summary["entry_zone"]:
        lines.append(f"EntryZone: {summary['entry_zone']}")
    if summary["state_text"]:
        lines.append(f"EntryState: {summary['state_text']}")
    if summary["proximity"] is not None:
        lines.append(f"ZoneProximity: {summary['proximity']:.2f}")
    if summary["recent_move"] is not None:
        lines.append(f"RecentPriceMove: {summary['recent_move']:.5f}")
    if summary["risk_band"] is not None:
        lines.append(f"ZoneWidth: {summary['risk_band']:.5f}")
    if summary["train_size"] is not None and summary["test_size"] is not None:
        lines.append(f"TrainEvents: {summary['train_size']}")
        lines.append(f"ValidationEvents: {summary['test_size']}")

    context_block = "\n".join(f"- {line}" for line in lines)
    reference = repr(fallback.strip()) if fallback else "'No fallback available.'"
    return (
        f"{instructions}\n\n"
        f"Market snapshot:\n{context_block}\n\n"
        "If there is no actionable setup, clearly advise the user to wait. "
        "Respond in natural sentences (no bullet lists or markdown headings).\n"
        f"Reference text for tone only (do not copy): {reference}"
    )


def _llm_explain(kind: str, info: Dict[str, Any], fallback: str) -> str:
    fallback_text = fallback or ""
    if not llm_has_client():
        return fallback_text
    summary = _summarize_for_llm(info)
    prompt = _build_llm_prompt(kind, summary, fallback_text)
    max_tokens = _LLM_KIND_MAX_TOKENS.get(kind, 220)
    response = llm_generate_text(
        prompt,
        system=_LLM_SYSTEM_PROMPT,
        temperature=_LLM_TEMPERATURE,
        max_tokens=max_tokens,
    )
    if response:
        text = response.strip()
        if text:
            return text
    return fallback_text


def _explain_signal_template(info: Dict[str, Any]) -> str:
    sym = info.get("symbol", "?")
    ins = info.get("insight", {})
    if not ins.get("has_event"):
        return f"No fresh Fib event detected for {sym}. Price isn’t near key retracement levels right now."
    dirn = ins["direction"]
    proba, thr = ins["prob"], ins["threshold"]
    ctx = "support (uptrend pullback)" if ins.get("context_up") else "resistance (downtrend rally)"
    fibs = ins.get("fib_dists", {}) or {}
    near = []
    for lvl in (0.618, 0.5, 0.382):
        d = fibs.get(f"dist_fib_{lvl}")
        if d is not None and np.isfinite(d):
            near.append((lvl, abs(float(d))))
    near.sort(key=lambda x: x[1])
    near_txt = f"closest to {near[0][0]:.3f}" if near else "near a retracement"
    margin = abs(proba - thr)
    bias_txt = "Momentum favors continuation" if ins.get("context_up") and dirn == "BUY" else \
               "Momentum favors rejection" if (not ins.get("context_up") and dirn == "SELL") else \
               "Signal is marginal; treat as HOLD risk-managed"
    entry_txt = ""
    entry = ins.get("entry") if isinstance(ins.get("entry"), dict) else None
    if entry and entry.get("level") is not None:
        label = entry.get("fib_label")
        label_txt = f"{label} retracement" if label is not None else "the retracement"
        lvl = float(entry["level"])
        lo = entry.get("zone_low")
        hi = entry.get("zone_high")
        zone = None
        if lo is not None and hi is not None:
            zone = f"{lo:.5f}-{hi:.5f}"
        entry_txt = (
            f" Watch {label_txt} near {lvl:.5f}" +
            (f" (zone {zone})" if zone else "") +
            (" for rejection" if dirn == "SELL" else " for support" if dirn == "BUY" else "") + "."
        )

    return (
        f"{sym} in {ctx}: classifier score {proba:.2f} vs threshold {thr:.2f} (margin {margin:.2f}). "
        f"Price {near_txt}. Action: {dirn}. {bias_txt}.{entry_txt}"
    )


def explain_signal(info: Dict[str, Any]) -> str:
    fallback = _explain_signal_template(info)
    return _llm_explain("signal_detail", info, fallback)


def _explain_recommendation_template(info: Dict[str, Any]) -> str:
    sym = info.get("symbol", "?")
    ins = info.get("insight", {}) or {}
    if not ins.get("has_event"):
        return f"No strong setup for {sym} right now. Price isn’t near key Fib levels."
    dirn = ins.get("direction", "HOLD")
    proba, thr = float(ins.get("prob", 0.0)), float(ins.get("threshold", 0.0))
    margin = abs(proba - thr)
    ctag = _confidence_label(margin)
    ctx = "uptrend pullback (support)" if ins.get("context_up") else "downtrend rally (resistance)"
    fibs = ins.get("fib_dists", {}) or {}
    near = []
    for lvl in (0.618, 0.5, 0.382):
        d = fibs.get(f"dist_fib_{lvl}")
        if d is not None and np.isfinite(d):
            near.append((lvl, abs(float(d))))
    near.sort(key=lambda x: x[1])
    near_txt = f"near {near[0][0]:.3f}" if near else "near a retracement"
    action_txt = (
        "Momentum favors continuation" if ins.get("context_up") and dirn == "BUY" else
        "Momentum favors rejection" if (not ins.get("context_up") and dirn == "SELL") else
        "Signal looks marginal; manage risk"
    )
    entry_txt = ""
    entry = ins.get("entry") if isinstance(ins.get("entry"), dict) else None
    if entry and entry.get("level") is not None:
        label = entry.get("fib_label")
        label_txt = f"{label} Fib" if label is not None else "the Fib"
        lvl = float(entry["level"])
        lo = entry.get("zone_low")
        hi = entry.get("zone_high")
        zone_txt = None
        if lo is not None and hi is not None:
            zone_txt = f"{lo:.5f}-{hi:.5f}"
        focus = "rejection" if dirn == "SELL" else "bounce" if dirn == "BUY" else "test"
        entry_txt = (
            f" Watch {label_txt} near {lvl:.5f}" +
            (f" (entry zone {zone_txt})" if zone_txt else "") +
            f" for a {focus} before committing."
        )

    return (
        f"{sym}: {dirn} setup ({ctag} confidence). Score {proba:.2f} vs {thr:.2f} "
        f"(margin {margin:.2f}), {ctx}, price {near_txt}. {action_txt}.{entry_txt}"
    )


def explain_recommendation(info: Dict[str, Any]) -> str:
    fallback = _explain_recommendation_template(info)
    return _llm_explain("recommendation", info, fallback)


# ---------------------------------
# Beginner-friendly explanations
# ---------------------------------
def _nearest_fib_text(ins: Dict[str, Any]) -> str:
    fibs = ins.get("fib_dists", {}) or {}
    near = []
    for lvl in (0.618, 0.5, 0.382):
        d = fibs.get(f"dist_fib_{lvl}")
        if d is not None and np.isfinite(d):
            near.append((lvl, abs(float(d))))
    near.sort(key=lambda x: x[1])
    return f"{near[0][0]:.3f}" if near else "a common checkpoint"


def _nearest_fib_now(info: Dict[str, Any]) -> Optional[str]:
    """Find the Fib label nearest to the latest close in `info`.
    Falls back to None if data is missing.
    """
    levels = (info or {}).get("levels") or {}
    ohlc = (info or {}).get("ohlc") or {}
    try:
        close = float(ohlc["Close"][-1])
    except Exception:
        return None
    cands = []
    for k in ["0.382", "0.5", "0.618", "0.0", "1.0"]:
        v = levels.get(k)
        if v is None:
            continue
        try:
            cands.append((k, abs(float(v) - close)))
        except Exception:
            pass
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])
    return cands[0][0]


def _explain_signal_simple_template(info: Dict[str, Any]) -> str:
    sym = info.get("symbol", "?")
    ins = info.get("insight", {}) or {}
    if not ins.get("has_event"):
        psym = _pretty_symbol(sym)
        return (
            f"Suggestion: Wait\n"
            f"Confidence: Very Weak\n"
            f"Why: {psym} isn’t near a typical checkpoint; it’s safer to wait."
        )
    dirn = ins.get("direction", "HOLD")
    proba, thr = float(ins.get("prob", 0.0)), float(ins.get("threshold", 0.0))
    margin = abs(proba - thr)
    ctag = _confidence_label(margin)
    ctx_up = bool(ins.get("context_up"))
    fib_txt = _nearest_fib_now(info) or _nearest_fib_text(ins)
    psym = _pretty_symbol(sym)
    trend_txt = "uptrend" if ctx_up else "downtrend"
    zone_txt = "potential pullback" if ctx_up else "potential rejection"
    tail = "Signals support this direction, but manage risk." if dirn != "HOLD" else \
        "But with mixed or weak signals currently, it’s better to wait."

    entry_txt = ""
    entry = ins.get("entry") if isinstance(ins.get("entry"), dict) else None
    if entry and entry.get("level") is not None:
        label = entry.get("fib_label")
        label_txt = f"{label} retracement" if label is not None else "the retracement"
        lvl = float(entry["level"])
        lo = entry.get("zone_low")
        hi = entry.get("zone_high")
        if lo is not None and hi is not None:
            zone_txt_entry = f"{lo:.5f}-{hi:.5f}"
        else:
            zone_txt_entry = f"{lvl:.5f}"
        trigger_txt = "rejection" if dirn == "SELL" else "bounce" if dirn == "BUY" else "reaction"
        entry_txt = f"\nEntry: Watch {label_txt} near {zone_txt_entry} for a {trigger_txt} before entering."

    suggestion = {"BUY": "Buy", "SELL": "Sell", "HOLD": "Wait"}.get(dirn, "Wait")
    return (
        f"Suggestion: {suggestion}\n"
        f"Confidence: {ctag.title()}\n"
        f"Why: {psym} is in an {trend_txt}, {zone_txt} at the {fib_txt} line. {tail}" +
        entry_txt
    )


def explain_signal_simple(info: Dict[str, Any]) -> str:
    fallback = _explain_signal_simple_template(info)
    return _llm_explain("signal_simple", info, fallback)


def explain_recommendation_simple(info: Dict[str, Any]) -> str:
    return explain_signal_simple(info)


def _feature_mode_summary(mode: str) -> str:
    mode = (mode or "").lower()
    return {
        "fib": "pure Fibonacci retracement distance and trend filters",
        "fib_ind": "Fib retracements plus SMA slope and RSI context",
        "fib_ind_norm": "normalized Fib distances, short-term returns, and RSI bands",
    }.get(mode, "the trained feature mix")


def _explain_followup_template(info: Dict[str, Any]) -> str:
    sym = info.get("symbol", "?")
    ins = info.get("insight", {}) or {}
    meta = info.get("meta", {}) or {}
    if not ins.get("has_event"):
        return f"{sym}: The model isn’t tracking an active Fib trigger right now, so there’s nothing deeper to add."
    prob = ins.get("prob")
    thr = ins.get("threshold")
    margin = abs((prob or 0) - (thr or 0)) if prob is not None and thr is not None else None
    ctx = "support bounce" if ins.get("context_up") else "resistance rejection"
    entry = ins.get("entry") or {}
    lvl = entry.get("level")
    label = entry.get("fib_label")
    state_txt = ins.get("entry_state_text") or entry.get("state_text") or ""
    feature_txt = _feature_mode_summary(meta.get("feature_mode"))
    pieces = []
    pieces.append(f"Classifier tags it as a {ctx} because price is on the correct side of its rolling trend filter.")
    if prob is not None and thr is not None:
        margin_txt = f", leaving a {margin:.2f} confidence buffer" if margin is not None else ""
        pieces.append(f"Probability printed {prob:.2f} versus trigger {thr:.2f}{margin_txt}.")
    if lvl is not None:
        lbl_txt = f"the {label} retracement" if label is not None else "that Fib retracement"
        pieces.append(f"Price is leaning on {lbl_txt} near {lvl:.5f}; we expect the level to hold ({state_txt}).")
    pieces.append(f"Under the hood the model blends {feature_txt} to flag statistically biased Fib touches.")
    variants = [f"{sym}: " + " ".join(pieces), f"Here’s why the {sym} setup still makes sense: " + " ".join(pieces)]
    return random.choice(variants)


def explain_followup_reason(info: Dict[str, Any]) -> str:
    fallback = _explain_followup_template(info)
    return _llm_explain("followup", info, fallback)


def _explain_risk_template(info: Dict[str, Any]) -> str:
    sym = info.get("symbol", "?")
    ins = info.get("insight", {}) or {}
    entry = ins.get("entry") or {}
    ctx = "support" if ins.get("context_up") else "resistance"
    zone_low = entry.get("zone_low")
    zone_high = entry.get("zone_high")
    width = (zone_high - zone_low) if zone_high is not None and zone_low is not None else None
    parts = [f"Risk plan for {sym}: treat the Fib band as {ctx}."]
    if width is not None:
        parts.append(f"The zone spans roughly {width:.5f} in price; keep stops just outside that band to let the level breathe.")
    elif entry.get("half_width") is not None:
        parts.append(f"Give the trade at least one band of room (~{entry['half_width']:.5f}) before calling it invalid.")
    else:
        parts.append("Size positions so a single invalidation equals one unit of risk; widen or tighten as volatility changes.")
    parts.append("Scale down size if volatility spikes or if upcoming news could blow through the level.")
    return " ".join(parts)


def explain_risk_guidance(info: Dict[str, Any]) -> str:
    fallback = _explain_risk_template(info)
    return _llm_explain("risk", info, fallback)


def _explain_confidence_template(info: Dict[str, Any]) -> str:
    sym = info.get("symbol", "?")
    ins = info.get("insight", {}) or {}
    meta = info.get("meta", {}) or {}
    if not ins.get("has_event"):
        return f"{sym}: No live confidence score because we don’t have an active Fib trigger right now."
    prob = ins.get("prob")
    thr = ins.get("threshold")
    margin = abs((prob or 0) - (thr or 0)) if prob is not None and thr is not None else None
    entry = ins.get("entry") or {}
    state = entry.get("state_text") or "the current Fib interaction"
    confidence_tag = _confidence_label(margin or 0.0).title() if margin is not None else "Unknown"
    train_size = meta.get("train_size")
    test_size = meta.get("test_size")
    bits = []
    if prob is not None and thr is not None:
        bits.append(f"Score {prob:.2f} vs threshold {thr:.2f} → {confidence_tag.lower()} margin {margin:.2f}.")
    if state:
        bits.append(f"Signal assumes {state} stays in play.")
    if train_size and test_size:
        bits.append(f"Latest training window used {train_size} train events and validated on {test_size} test events.")
    bits.append("Confidence fades quickly if price escapes the zone or macro data shifts, so keep reassessing.")
    return f"{sym} confidence check: " + " ".join(bits)


def explain_confidence_detail(info: Dict[str, Any]) -> str:
    fallback = _explain_confidence_template(info)
    return _llm_explain("confidence", info, fallback)


def get_overview_snapshot(pairs: Optional[List[str]] = None, interval: str = "1h", feature_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return lightweight status for each pair: how price sits relative to the active Fib zone.

    If `feature_mode` is provided, attempts to use artifacts trained with that mode; otherwise falls
    back to legacy artifacts without mode suffix.
    """
    symbols = list(pairs or DEFAULT_PAIRS)
    snapshot: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            info = get_pair_insight(sym, interval=interval, feature_mode=feature_mode)
        except Exception:
            continue
        ins = info.get("insight") or {}
        entry = ins.get("entry") if isinstance(ins.get("entry"), dict) else None
        if not ins.get("has_event"):
            snapshot.append({
                "symbol": sym,
                "state": "no_event",
                "state_text": "No retracement setup right now.",
                "direction": ins.get("direction"),
                "confidence": 0.0,
                "entry": None,
            })
            continue
        prob_val = float(ins.get("prob", 0.0))
        thr_val = float(ins.get("threshold", 0.0))
        margin = prob_val - thr_val
        snapshot.append({
            "symbol": sym,
            "direction": ins.get("direction"),
            "state": (entry or {}).get("state", "unavailable"),
            "state_text": (entry or {}).get("state_text", "Retracement zone unavailable."),
            "proximity": (entry or {}).get("proximity"),
            "confidence": round(margin, 3),
            "entry": entry,
        })
    return snapshot


def _pretty_symbol(sym: str) -> str:
    try:
        clean = str(sym).replace("=X", "")
        if len(clean) == 6:
            return clean[:3] + "/" + clean[3:]
        return clean
    except Exception:
        return str(sym)
