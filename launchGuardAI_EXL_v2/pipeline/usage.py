"""
=============================================================================
API USAGE AND COST TRACKING
=============================================================================

Records every Anthropic API call the app makes, with token counts and cost,
so the user can see what each step spent and what the total is.

WHICH STEPS ACTUALLY COST MONEY
-------------------------------
  column mapping    FREE  - deterministic string matching, no model involved
  clustering        FREE  - k-means runs locally
  persona writing   PAID  - one call per segment
  persona reactions PAID  - one call per segment, per intervention
  A/B simulation    PAID  - one call per simulated shopper

Column mapping and clustering are often assumed to be AI steps. They are not,
and the cost tab says so explicitly rather than leaving a suspicious zero.

PRICES
------
Per million tokens, checked August 2026. Anthropic changes these, so they live
in one dict at the top and can be edited without touching anything else.
Note Sonnet 5 is on introductory pricing until 31 August 2026, after which it
returns to $3.00 / $15.00.
=============================================================================
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# USD per million tokens: (input, output)
PRICING: Dict[str, tuple] = {
    "claude-opus-5":        (5.00, 25.00),
    "claude-sonnet-5":      (2.00, 10.00),   # introductory, rises 1 Sep 2026
    "claude-haiku-4-5":     (1.00,  5.00),
    "claude-fable-5":       (10.00, 50.00),
    "claude-opus-4-8":      (5.00, 25.00),
    "claude-sonnet-4-6":    (3.00, 15.00),
    "claude-sonnet-4-5":    (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
FALLBACK_PRICE = (3.00, 15.00)      # unknown model: assume Sonnet-tier

# cache reads bill at 10% of base input
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25       # 5-minute TTL

MAX_LOG = 60                         # rolling window the UI shows

STEP_LABELS = {
    "personas": "Persona writing",
    "reaction": "Persona reaction",
    "abtest": "Shopper simulation",
}

_LOCK = threading.Lock()
_LOG_PATH: Optional[str] = None


def configure(path: str):
    """Where to persist the log. Called once at app startup."""
    global _LOG_PATH
    _LOG_PATH = path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _price_for(model: str) -> tuple:
    # OpenRouter slugs are vendor-prefixed ("anthropic/claude-sonnet-4.5");
    # the price table is keyed on the bare model name.
    if "/" in model:
        model = model.split("/", 1)[1]
    if model in PRICING:
        return PRICING[model]
    # tolerate dated suffixes like claude-sonnet-5-20260801
    for known, price in PRICING.items():
        if model.startswith(known):
            return price
    return FALLBACK_PRICE


def _load() -> List[dict]:
    if not _LOG_PATH or not os.path.exists(_LOG_PATH):
        return []
    try:
        return json.load(open(_LOG_PATH, encoding="utf-8"))
    except Exception:
        return []


def _save(entries: List[dict]):
    if not _LOG_PATH:
        return
    try:
        with open(_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries[:MAX_LOG], f, indent=2)
    except Exception:
        pass          # cost tracking must never break the actual work


def compute_cost(model: str, inp: int, out: int,
                 cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout = _price_for(model)
    cost = (inp / 1e6) * pin + (out / 1e6) * pout
    cost += (cache_read / 1e6) * pin * CACHE_READ_MULTIPLIER
    cost += (cache_write / 1e6) * pin * CACHE_WRITE_MULTIPLIER
    return round(cost, 6)


def record(step: str, model: str, response: Any, label: str = "") -> dict:
    """
    Log one call. `response` is the raw Anthropic message object.

    Wrapped in a broad try so a change in the SDK's usage shape can never take
    down the pipeline - worst case the entry is skipped.
    """
    try:
        u = getattr(response, "usage", None)
        inp = int(getattr(u, "input_tokens", 0) or 0)
        out = int(getattr(u, "output_tokens", 0) or 0)
        cr = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cw = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    except Exception:
        return {}

    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "step_label": STEP_LABELS.get(step, step),
        "label": label[:90],
        "model": model,
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cr,
        "cache_write_tokens": cw,
        "total_tokens": inp + out + cr + cw,
        "cost_usd": compute_cost(model, inp, out, cr, cw),
    }

    with _LOCK:
        entries = _load()
        entries.insert(0, entry)
        _save(entries)
    return entry


def record_tokens(step: str, model: str, inp: int, out: int,
                  label: str = "") -> dict:
    """
    Log a call from already-extracted token counts.

    The adapter normalises the two SDKs' differing usage shapes, so it passes
    plain integers rather than a provider-specific response object.
    """
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "step_label": STEP_LABELS.get(step, step),
        "label": (label or "")[:90],
        "model": model,
        "input_tokens": int(inp or 0),
        "output_tokens": int(out or 0),
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": int(inp or 0) + int(out or 0),
        "cost_usd": compute_cost(model, int(inp or 0), int(out or 0)),
    }
    with _LOCK:
        entries = _load()
        entries.insert(0, entry)
        _save(entries)
    return entry


def checkpoint() -> str:
    """Marker to measure the cost of one operation. Pair with cost_since()."""
    return datetime.now(timezone.utc).isoformat()


def cost_since(marker: str) -> dict:
    """What was spent after `marker`. Used for the per-run cost badge."""
    with _LOCK:
        entries = [e for e in _load() if e["at"] > marker]
    return {
        "calls": len(entries),
        "input_tokens": sum(e["input_tokens"] for e in entries),
        "output_tokens": sum(e["output_tokens"] for e in entries),
        "total_tokens": sum(e["total_tokens"] for e in entries),
        "cost_usd": round(sum(e["cost_usd"] for e in entries), 6),
    }


def summary() -> dict:
    with _LOCK:
        entries = _load()

    by_step: Dict[str, dict] = {}
    for e in entries:
        s = by_step.setdefault(e["step"], {
            "step": e["step"], "step_label": e.get("step_label", e["step"]),
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0})
        s["calls"] += 1
        s["input_tokens"] += e["input_tokens"]
        s["output_tokens"] += e["output_tokens"]
        s["total_tokens"] += e["total_tokens"]
        s["cost_usd"] = round(s["cost_usd"] + e["cost_usd"], 6)

    return {
        "total_calls": len(entries),
        "total_input_tokens": sum(e["input_tokens"] for e in entries),
        "total_output_tokens": sum(e["output_tokens"] for e in entries),
        "total_tokens": sum(e["total_tokens"] for e in entries),
        "total_cost_usd": round(sum(e["cost_usd"] for e in entries), 6),
        "by_step": sorted(by_step.values(), key=lambda s: -s["cost_usd"]),
        "window": MAX_LOG,
        "free_steps": [
            {"name": "Column mapping",
             "why": "Deterministic string matching, no model call"},
            {"name": "Segmentation",
             "why": "K-means runs locally on your machine"},
            {"name": "Cached results",
             "why": "Re-opening a past run costs nothing"},
        ],
        "pricing_note": "Rates checked August 2026. Sonnet 5 is on "
                        "introductory pricing until 31 August 2026. If you are "
                        "routing through OpenRouter, actual billing includes "
                        "their margin on top of these vendor rates.",
    }


def recent(n: int = MAX_LOG) -> List[dict]:
    with _LOCK:
        return _load()[:n]


def reset():
    with _LOCK:
        _save([])