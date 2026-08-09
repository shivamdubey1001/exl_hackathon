"""
=============================================================================
SEGMENTATION
=============================================================================

Clustering and profiling, refactored from the v1 CLI script into functions the
API can call inside a background job.

Two differences from v1:
  - k is chosen by the user (2-6), not by the script
  - features are whatever the uploaded file actually provided, not a fixed list
=============================================================================
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42

# Core behavioural features, in priority order. These define a segment.
# Everything else numeric is enrichment: it describes a segment but should not
# define one. Throwing all numeric columns into k-means adds dimensions without
# adding structure - on our own data that alone dropped silhouette from 0.31 to
# 0.11, because age and tenure are noise relative to spend and browsing.
CORE_FEATURES = [
    "monetary_net",
    "recency_days",
    "avg_item_price",
    "frequency_orders",
    "total_sessions",
    "avg_events_per_session",
    "avg_session_duration_sec",
    "category_diversity",
]

# Used only if the upload is too thin to fill the core set
FALLBACK_FEATURES = [
    "avg_order_value", "avg_basket_size", "return_rate", "tenure_days",
    "session_conversion_rate",
]

MIN_FEATURES = 3

# Columns that must never define a segment. price_elasticity and churn are the
# outcomes the personas are meant to predict - clustering on them would leak
# the answer into the question.
NEVER_CLUSTER = ["price_elasticity", "prob_churn_on_price_increase",
                 "expected_spend_retention", "expected_revenue_after_change",
                 "revenue_delta"]

# Synthetic overlays. Held out by default so segments come from real behaviour
# and any difference in price sensitivity is an emergent finding.
SYNTHETIC = ["discount_reliance_pct", "avg_discount_depth",
             "cart_abandonment_rate", "browse_to_cart_rate"]

# Heavily right-skewed in every retail dataset - log before scaling or a
# handful of big spenders drag every centroid toward themselves.
LOG_TRANSFORM = ["monetary_net", "avg_item_price", "avg_order_value",
                 "avg_session_duration_sec"]

# Session ids often persist across days, producing absurd durations
SESSION_DURATION_CAP = 3600


def prepare_features(df: pd.DataFrame,
                     include_synthetic: bool = False) -> (pd.DataFrame, List[str]):
    """Build the numeric matrix to cluster on, and say which columns it used."""
    work = df.copy()

    def usable(col):
        return (col in work.columns
                and pd.api.types.is_numeric_dtype(work[col])
                and work[col].nunique(dropna=True) > 1)

    # core first, then fallback only if the upload was thin
    candidates = [c for c in CORE_FEATURES if usable(c)]
    if len(candidates) < MIN_FEATURES:
        candidates += [c for c in FALLBACK_FEATURES
                       if usable(c) and c not in candidates]
    if include_synthetic:
        candidates += [c for c in SYNTHETIC if usable(c) and c not in candidates]

    # last resort: an upload using none of our known names
    if len(candidates) < MIN_FEATURES:
        for col in work.columns:
            if col in candidates or col == "user_id" or col in NEVER_CLUSTER:
                continue
            if usable(col):
                candidates.append(col)

    if len(candidates) < 2:
        raise ValueError(
            "Need at least 2 numeric behavioural columns with varying values.")

    for col in candidates:
        if col == "avg_session_duration_sec":
            work[col] = work[col].clip(upper=SESSION_DURATION_CAP)
        if col in LOG_TRANSFORM:
            work[col] = np.log1p(work[col].clip(lower=0))

    feats = work[candidates].copy()
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0)
    return feats, candidates


def evaluate_k(feats: pd.DataFrame, k_range=range(2, 7),
               sample: int = 8000) -> List[dict]:
    """Silhouette and inertia per k, so the UI can guide the choice."""
    X = StandardScaler().fit_transform(feats)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs = X[idx]

    out = []
    for k in k_range:
        if k >= len(Xs):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        kms = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        out.append({"k": k,
                    "silhouette": round(float(silhouette_score(Xs, kms.labels_)), 4),
                    "inertia": round(float(km.inertia_), 1)})
    return out


def run_clustering(df: pd.DataFrame, k: int,
                   include_synthetic: bool = False,
                   progress: Optional[Callable] = None) -> dict:
    """
    Fit k-means and profile the result.

    Returns the labelled frame, per-cluster profiles ready for the LLM, and a
    2D projection for plotting.
    """
    def say(msg, pct):
        if progress:
            progress(msg, pct)

    say("Preparing features", 10)
    feats, used = prepare_features(df, include_synthetic)
    if len(used) < 2:
        raise ValueError("Not enough usable numeric columns to segment on.")

    say("Scaling", 25)
    X = StandardScaler().fit_transform(feats)

    say(f"Finding {k} segments", 45)
    km = KMeans(n_clusters=k, n_init=25, random_state=SEED).fit(X)

    out = df.copy()
    out["cluster_id"] = km.labels_

    say("Scoring segment quality", 65)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(8000, len(X)), replace=False)
    sil = float(silhouette_score(X[idx], km.labels_[idx]))

    say("Profiling segments", 80)
    profiles = profile_clusters(out, used)

    say("Projecting", 92)
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X)
    pidx = rng.choice(len(coords), min(2500, len(coords)), replace=False)
    projection = {
        "points": [{"x": round(float(coords[i, 0]), 3),
                    "y": round(float(coords[i, 1]), 3),
                    "c": int(km.labels_[i])} for i in pidx],
        "explained": [round(float(v), 3) for v in pca.explained_variance_ratio_],
    }

    say("Done", 100)
    return {
        "users": out,
        "meta": {
            "k": int(k),
            "silhouette": round(sil, 4),
            "clustered_on": used,
            "held_out": [c for c in df.columns
                         if c in SYNTHETIC + NEVER_CLUSTER and c not in used],
            "total_users": int(len(out)),
            # a floor of roughly 0.15-0.18 is what random data scores, so this
            # tells the user whether the structure found is real
            "random_baseline": 0.17,
            "clusters": profiles,
        },
        "projection": projection,
    }


def profile_clusters(df: pd.DataFrame, features: List[str]) -> List[dict]:
    """Per-cluster summary. Index vs population is what makes it readable."""
    extra = [c for c in df.columns
             if c not in features + ["cluster_id", "user_id"]
             and pd.api.types.is_numeric_dtype(df[c])]
    numeric = features + extra
    cats = [c for c in ["top_category", "dominant_traffic_source", "gender",
                        "country", "state"] if c in df.columns]

    overall = df[numeric].mean()
    total = len(df)
    has_money = "monetary_net" in df.columns
    total_rev = float(df["monetary_net"].sum()) if has_money else 0.0

    profiles = []
    for cid, grp in df.groupby("cluster_id"):
        p = {"cluster_id": int(cid), "size": int(len(grp)),
             "share_of_customers_pct": round(100 * len(grp) / total, 1)}
        if has_money and total_rev:
            p["share_of_revenue_pct"] = round(
                100 * float(grp["monetary_net"].sum()) / total_rev, 1)

        stats = {}
        for c in numeric:
            mean = float(grp[c].mean())
            base = float(overall[c])
            idx = int(round(100 * mean / base)) if base not in (0.0,) and not np.isnan(base) else None
            stats[c] = {"mean": round(mean, 3), "vs_population_index": idx}
        p["metrics"] = stats

        for c in cats:
            vc = grp[c].value_counts(normalize=True).head(3)
            p[f"top_{c}"] = {str(k): round(float(v), 3) for k, v in vc.items()}

        profiles.append(p)

    return sorted(profiles, key=lambda x: -x["size"])
