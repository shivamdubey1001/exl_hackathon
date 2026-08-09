"""
=============================================================================
COLUMN MAPPING
=============================================================================

An uploaded CSV will not use our internal column names. This module guesses
the mapping, scores its confidence, and hands the user something to confirm.

Auto-detection is a suggestion, never a decision. The UI shows what was
matched and lets the user override every field, because a silently wrong
mapping produces plausible-looking nonsense all the way through the pipeline.
=============================================================================
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


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
    return [{"field": f, "required": req, "role": role, "description": desc,
             "examples": aliases[:4]}
            for f, (aliases, req, role, desc) in FIELDS.items()]
