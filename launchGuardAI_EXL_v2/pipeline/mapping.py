"""
=============================================================================
COLUMN MAPPING
=============================================================================

An uploaded CSV will not use our internal column names. This module guesses
the mapping, scores its confidence, and hands the user something to confirm.

Two passes:

  1. FUZZY   deterministic string matching against known aliases. Free and
             instant. On a well-named file this gets everything.
  2. AI      only for fields the fuzzy pass could not fill. Handles
             abbreviations (rcy, sess_cnt), business jargon and non-English
             headers that string matching will never resolve.

The AI pass is second on purpose. Running it first would add cost and latency
to every upload, including the many where it has nothing to contribute.

Auto-detection is a suggestion, never a decision. The UI shows what was
matched, where each guess came from, and lets the user override every field,
because a silently wrong mapping produces plausible-looking nonsense all the
way through the pipeline.
=============================================================================
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import llm 


# canonical field -> (aliases, required, role, description)
# role: id | cluster | enrich | outcome
FIELDS = {
    "user_id": (
        ["user_id", "userid", "customer_id", "customerid", "id", "cust_id",
         "client_id", "account_id", "customer", "user"],
        True, "id", "Unique customer identifier"),

    # --- clustering features ---
    "monetary_net": (
        ["monetary_net", "monetary", "total_spend", "lifetime_value", "ltv",
         "revenue", "total_revenue", "spend", "total_sales", "sales",
         "lifetime_spend", "monetary_value"],
        False, "cluster", "Total spend per customer"),
    "recency_days": (
        ["recency_days", "recency", "days_since_last_order", "days_since_purchase",
         "last_order_days", "days_since_last_purchase", "last_purchase_days"],
        False, "cluster", "Days since last purchase"),
    "frequency_orders": (
        ["frequency_orders", "frequency", "order_count", "num_orders",
         "total_orders", "orders", "purchase_count", "n_orders"],
        False, "cluster", "Number of orders placed"),
    "avg_item_price": (
        ["avg_item_price", "average_item_price", "mean_item_price",
         "avg_unit_price", "average_price", "item_price"],
        False, "cluster", "Average price per item bought"),
    "total_sessions": (
        ["total_sessions", "sessions", "session_count", "num_sessions",
         "visits", "total_visits", "visit_count"],
        False, "cluster", "Number of browsing sessions"),
    "avg_events_per_session": (
        ["avg_events_per_session", "events_per_session", "pages_per_session",
         "avg_pageviews", "pageviews_per_session", "avg_pages"],
        False, "cluster", "Browsing depth per visit"),
    "avg_session_duration_sec": (
        ["avg_session_duration_sec", "avg_session_duration", "session_duration",
         "time_on_site", "avg_time_on_site", "duration_seconds"],
        False, "cluster", "Average session length in seconds"),
    "category_diversity": (
        ["category_diversity", "categories_shopped", "n_categories",
         "distinct_categories", "category_count", "num_categories"],
        False, "cluster", "Distinct categories purchased from"),

    # --- enrichment ---
    "avg_order_value": (
        ["avg_order_value", "aov", "average_order_value", "mean_order_value"],
        False, "enrich", "Average value per order"),
    "avg_basket_size": (
        ["avg_basket_size", "basket_size", "items_per_order", "avg_items"],
        False, "enrich", "Items per order"),
    "discount_reliance_pct": (
        ["discount_reliance_pct", "discount_reliance", "pct_discounted",
         "discount_rate", "promo_share", "discount_share"],
        False, "enrich", "Share of purchases made on discount"),
    "avg_discount_depth": (
        ["avg_discount_depth", "discount_depth", "avg_markdown", "markdown"],
        False, "enrich", "Typical discount taken"),
    "cart_abandonment_rate": (
        ["cart_abandonment_rate", "abandonment_rate", "cart_abandon_rate",
         "abandon_rate"],
        False, "enrich", "Share of carts left behind"),
    "session_conversion_rate": (
        ["session_conversion_rate", "conversion_rate", "cvr", "conv_rate"],
        False, "enrich", "Share of sessions ending in purchase"),
    "browse_to_cart_rate": (
        ["browse_to_cart_rate", "add_to_cart_rate", "cart_rate"],
        False, "enrich", "Share of browsing that reaches the cart"),
    "return_rate": (
        ["return_rate", "returns_rate", "pct_returned", "return_pct"],
        False, "enrich", "Share of items returned"),
    "tenure_days": (
        ["tenure_days", "tenure", "days_since_signup", "account_age_days",
         "customer_age_days"],
        False, "enrich", "Days since the account was created"),
    "age": (["age", "customer_age", "user_age"], False, "enrich", "Customer age"),
    "gender": (["gender", "sex", "customer_gender"], False, "enrich", "Gender"),
    "country": (["country", "nation", "country_code"], False, "enrich", "Country"),
    "state": (["state", "region", "province"], False, "enrich", "Region"),
    "top_category": (
        ["top_category", "favourite_category", "favorite_category",
         "main_category", "primary_category", "category"],
        False, "enrich", "Most purchased category"),
    "dominant_traffic_source": (
        ["dominant_traffic_source", "traffic_source", "channel", "source",
         "acquisition_channel", "utm_source", "referrer", "marketing_channel"],
        False, "enrich", "Where they usually arrive from"),

    # --- held out, never clustered on ---
    "price_elasticity": (
        ["price_elasticity", "elasticity", "own_price_elasticity"],
        False, "outcome", "Own-price elasticity, used by the pricing engine"),
    "prob_churn_on_price_increase": (
        ["prob_churn_on_price_increase", "churn_prob", "churn_probability",
         "price_churn_risk"],
        False, "outcome", "Chance of leaving after a price rise"),
}

# Human-readable labels for the UI. Kept separate from FIELDS so the internal
# names stay stable — those are what the pipeline and CSVs use.
DISPLAY_NAMES = {
    "user_id": "Customer ID",
    "monetary_net": "Total spend",
    "recency_days": "Days since last purchase",
    "frequency_orders": "Orders placed",
    "avg_item_price": "Average item price",
    "total_sessions": "Browsing sessions",
    "avg_events_per_session": "Pages per session",
    "avg_session_duration_sec": "Session length",
    "category_diversity": "Categories shopped",
    "avg_order_value": "Average order value",
    "avg_basket_size": "Items per order",
    "discount_reliance_pct": "Discount reliance",
    "avg_discount_depth": "Average discount depth",
    "cart_abandonment_rate": "Cart abandonment",
    "session_conversion_rate": "Conversion rate",
    "browse_to_cart_rate": "Browse to cart",
    "return_rate": "Return rate",
    "tenure_days": "Customer tenure",
    "age": "Age",
    "gender": "Gender",
    "country": "Country",
    "state": "Region",
    "top_category": "Top category",
    "dominant_traffic_source": "Main channel",
    "price_elasticity": "Price elasticity",
    "prob_churn_on_price_increase": "Churn risk on price rise",
}


def display_name(field: str) -> str:
    """Label for the UI. Falls back to title-casing the internal name."""
    return DISPLAY_NAMES.get(field, field.replace("_", " ").capitalize())

CLUSTER_FIELDS = [k for k, v in FIELDS.items() if v[2] == "cluster"]
MIN_CLUSTER_FIELDS = 3


def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse separators."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _score(col: str, alias: str) -> float:
    """0-1 similarity between a CSV column and a known alias."""
    c, a = _norm(col), _norm(alias)
    if not c or not a:
        return 0.0
    if c == a:
        return 1.0
    # A prefix only counts when the shorter string is most of the longer one.
    # Without this, alias "market" swallows a column called "Marketing Channel".
    ratio = min(len(c), len(a)) / max(len(c), len(a))
    if (c.startswith(a) or a.startswith(c)) and ratio >= 0.6:
        return 0.85
    if (a in c or c in a) and ratio >= 0.6:
        return 0.72
    # Token match, so "customer_total_spend" still matches "total_spend".
    # Requires EVERY alias token to be present: a partial overlap let
    # "Random Internal Code" match the alias "country_code" on "code" alone.
    ct = set(re.split(r"[^a-z0-9]+", str(col).lower())) - {""}
    at = set(re.split(r"[^a-z0-9]+", alias.lower())) - {""}
    if ct and at and at.issubset(ct):
        return 0.55 + 0.15 * (len(at) / len(ct))
    return 0.0


def detect_mapping(df: pd.DataFrame) -> dict:
    """
    Guess which CSV column feeds which canonical field.

    Greedy best-match, and each CSV column is consumed once so two fields
    cannot both claim the same column.
    """
    cols = list(df.columns)
    numeric = {c for c in cols if pd.api.types.is_numeric_dtype(df[c])}
    taken: set = set()
    mapping: Dict[str, Optional[str]] = {}
    confidence: Dict[str, float] = {}

    # score every (field, column) pair, then assign strongest first
    pairs = []
    for field, (aliases, _req, role, _desc) in FIELDS.items():
        for col in cols:
            best = max(_score(col, a) for a in aliases)
            if best <= 0:
                continue
            # a numeric field matched to a text column is almost always wrong
            if role in ("cluster", "outcome") and col not in numeric:
                best *= 0.35
            pairs.append((best, field, col))

    for score, field, col in sorted(pairs, key=lambda p: -p[0]):
        if field in mapping or col in taken:
            continue
        if score < 0.55:
            continue
        mapping[field] = col
        confidence[field] = round(score, 2)
        taken.add(col)

    for field in FIELDS:
        mapping.setdefault(field, None)
        confidence.setdefault(field, 0.0)

    return {"mapping": mapping, "confidence": confidence}


def validate_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> dict:
    """Can we actually run the pipeline with this mapping?"""
    errors: List[str] = []
    warnings: List[str] = []

    uid = mapping.get("user_id")
    if not uid:
        errors.append("No customer identifier selected. Pick the column that "
                      "uniquely identifies each customer.")
    elif uid in df.columns:
        dupes = int(df[uid].duplicated().sum())
        if dupes:
            errors.append(f"{dupes:,} duplicate values in '{uid}'. This file must "
                          f"have one row per customer.")
        if df[uid].isna().any():
            errors.append(f"'{uid}' has blank values.")

    present = [f for f in CLUSTER_FIELDS if mapping.get(f)]
    if len(present) < MIN_CLUSTER_FIELDS:
        errors.append(
            f"Only {len(present)} behavioural column(s) mapped. At least "
            f"{MIN_CLUSTER_FIELDS} are needed to find meaningful segments.")

    for field in present:
        col = mapping[field]
        if col not in df.columns:
            errors.append(f"Column '{col}' is not in the file.")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            errors.append(f"'{col}' has no usable numbers in it.")
            continue
        null_pct = 100 * s.isna().mean()
        if null_pct > 40:
            warnings.append(f"'{col}' is {null_pct:.0f}% blank — it will carry "
                            f"little weight in segmentation.")
        if s.nunique(dropna=True) <= 1:
            warnings.append(f"'{col}' is the same value for every customer, so "
                            f"it cannot separate anyone. It will be dropped.")

    n = len(df)
    if n < 50:
        errors.append(f"Only {n} rows. Segmentation needs at least 50 customers.")
    elif n < 500:
        warnings.append(f"{n} customers is a small base — segments may be unstable.")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "cluster_fields_mapped": present}


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Rename to canonical names and coerce types. Unmapped columns are dropped."""
    keep = {col: field for field, col in mapping.items() if col and col in df.columns}
    out = df[list(keep.keys())].rename(columns=keep).copy()

    for field in out.columns:
        role = FIELDS[field][2] if field in FIELDS else "enrich"
        if field == "user_id":
            continue
        if role in ("cluster", "outcome") or field in (
                "avg_order_value", "avg_basket_size", "discount_reliance_pct",
                "avg_discount_depth", "cart_abandonment_rate",
                "session_conversion_rate", "browse_to_cart_rate",
                "return_rate", "tenure_days", "age"):
            out[field] = pd.to_numeric(out[field], errors="coerce")

    # drop constant clustering columns - they add dimensions without signal
    for f in [c for c in out.columns if c in CLUSTER_FIELDS]:
        if out[f].nunique(dropna=True) <= 1:
            out = out.drop(columns=[f])

    return out


def field_catalogue() -> List[dict]:
    """Field list for the mapping UI."""
    return [{"field": f, "display": display_name(f), "required": req,
             "role": role, "description": desc, "examples": aliases[:4]}
            for f, (aliases, req, role, desc) in FIELDS.items()]


# =============================================================================
# AI FALLBACK
# =============================================================================

# Column names alone are often ambiguous - "rcy" could be anything. A few
# sample values usually settle it: [12, 340, 891] is plainly a day count.
SAMPLE_VALUES = 4

AI_SYSTEM = """You match messy spreadsheet column headers to a fixed set of \
customer-analytics fields.

You are given the fields still needing a match, and the columns not yet used,
each with a few sample values from the data.

Rules:
1. Match on meaning, not spelling. "rcy", "days_since_last_txn" and \
"letzter_kauf" all mean recency.
2. Sample values are the strongest signal. A column of 0-1 decimals is a rate; \
a column of large integers is a count or a currency amount.
3. Only use column names exactly as given. Never invent one.
4. Each column matches at most one field, and each field at most one column.
5. Leave a field out entirely rather than guessing. A wrong mapping is far \
worse than no mapping, because it produces confident nonsense downstream.
6. Give a confidence between 0 and 1 for each match, and a short reason.
7. Return ONE JSON object, no markdown fences, no preamble."""


def _sample_values(df: pd.DataFrame, col: str, n: int = SAMPLE_VALUES) -> list:
    """A few real values, JSON-safe, so the model can infer the column's meaning."""
    try:
        vals = df[col].dropna().head(n).tolist()
    except Exception:
        return []
    out = []
    for v in vals:
        if hasattr(v, "item"):          # numpy scalar
            v = v.item()
        if isinstance(v, float):
            v = round(v, 3)
        out.append(v if isinstance(v, (int, float, str, bool)) else str(v))
    return out


def _build_ai_prompt(df: pd.DataFrame, fields: List[str],
                     columns: List[str]) -> str:
    want = []
    for f in fields:
        aliases, req, role, desc = FIELDS[f]
        want.append({"field": f, "means": desc,
                     "used_for": "defines the segments" if role == "cluster"
                                 else "describes the segments",
                     "examples_of_names": aliases[:5]})

    have = [{"column": c,
             "dtype": str(df[c].dtype),
             "sample_values": _sample_values(df, c),
             "distinct_values": int(df[c].nunique(dropna=True))}
            for c in columns]

    return (
        "FIELDS STILL NEEDING A COLUMN:\n" + json.dumps(want, indent=2)
        + "\n\nCOLUMNS NOT YET USED:\n" + json.dumps(have, indent=2, default=str)
        + "\n\nReturn JSON shaped exactly like this:\n"
        + json.dumps({"matches": [
            {"field": "<one of the field names above>",
             "column": "<one of the column names above>",
             "confidence": 0.0,
             "reason": "<short - why this column means this field>"}]}, indent=2)
    )


def _extract_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _call_model(system: str, user: str, model: str) -> tuple:
    """Returns (text, None). Usage is recorded inside llm.call."""
    return llm.call(system, user, model=model, max_tokens=2000,
                    step="mapping", label="column matching"), None


def ai_fill_gaps(df: pd.DataFrame, mapping: Dict[str, Optional[str]],
                 model: Optional[str] = None) -> dict:
    """
    Ask a model about the fields the fuzzy pass left empty.

    Returns {"mapping", "confidence", "sources", "ai_notes", "ai_used"}.
    `sources` records whether each field came from "fuzzy" or "ai" so the UI
    can show provenance - the user should know which guesses are deterministic
    and which are a model's opinion.
    """
    model = model or os.getenv("LAUNCHGUARD_MODEL", "claude-sonnet-5")

    mapping = dict(mapping)
    sources = {f: ("fuzzy" if c else None) for f, c in mapping.items()}

    missing = [f for f, c in mapping.items() if not c]
    unused = [c for c in df.columns if c not in mapping.values()]

    if not missing or not unused:
        return {"mapping": mapping, "confidence": {}, "sources": sources,
                "ai_notes": [], "ai_used": False,
                "skip_reason": "nothing left to match"}

    prompt = _build_ai_prompt(df, missing, unused)

    try:
        # usage is now recorded inside llm.call, so no second call here
        text, _ = _call_model(AI_SYSTEM, prompt, model)
        data = _extract_json(text)
    except Exception as e:
        return {"mapping": mapping, "confidence": {}, "sources": sources,
                "ai_notes": [], "ai_used": False,
                "error": f"{type(e).__name__}: {e}"}

    conf, notes = {}, []
    taken = set(c for c in mapping.values() if c)

    for m in data.get("matches", []):
        f, c = m.get("field"), m.get("column")
        # the model is only allowed to pick from what it was shown
        if f not in missing or c not in unused or c in taken:
            continue
        if mapping.get(f):
            continue
        try:
            cf = float(m.get("confidence", 0))
        except (TypeError, ValueError):
            cf = 0.0
        if cf < 0.5:                    # low-confidence guesses stay unmapped
            continue

        mapping[f] = c
        sources[f] = "ai"
        conf[f] = round(cf, 2)
        taken.add(c)
        notes.append({"field": f, "column": c, "confidence": round(cf, 2),
                      "reason": str(m.get("reason", ""))[:160]})

    return {"mapping": mapping, "confidence": conf, "sources": sources,
            "ai_notes": notes, "ai_used": True,
            "fields_offered": len(missing), "fields_matched": len(notes)}


def detect_mapping_smart(df: pd.DataFrame, use_ai: bool = True,
                         model: Optional[str] = None) -> dict:
    """
    Full detection: fuzzy first, then AI on the leftovers.

    Merges both confidence dicts so the UI can show one number per field
    regardless of which pass produced it.
    """
    base = detect_mapping(df)
    result = {
        "mapping": base["mapping"],
        "confidence": base["confidence"],
        "sources": {f: ("fuzzy" if c else None)
                    for f, c in base["mapping"].items()},
        "ai_used": False, "ai_notes": [],
    }

    if not use_ai:
        return result
    import llm
    if not llm.has_key():
        result["skip_reason"] = "no API key configured"
        return result

    ai = ai_fill_gaps(df, base["mapping"], model=model)
    result["mapping"] = ai["mapping"]
    result["sources"] = ai["sources"]
    result["ai_used"] = ai.get("ai_used", False)
    result["ai_notes"] = ai.get("ai_notes", [])
    if ai.get("error"):
        result["ai_error"] = ai["error"]
    result["confidence"] = {**base["confidence"], **ai.get("confidence", {})}
    return result