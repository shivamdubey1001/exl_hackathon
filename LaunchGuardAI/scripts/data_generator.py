"""
=============================================================================
SYNTHETIC SIGNAL GENERATOR
Customer Persona Project - hackathon build
=============================================================================

WHAT THIS DOES
--------------
TheLook e-commerce data was profiled and found to be missing the signals this
project needs:

  * sale_price == retail_price for all 180,928 order items  -> no discount,
    no price variation, no measurable price sensitivity
  * cart abandonment exists in raw events (249,563 abandoned sessions) but
    those sessions carry no user_id and cannot be attributed to a user, so
    every identified user has conversion rate 1.0 and abandonment 0.0

This script keeps everything real that IS real (users, orders, timings,
categories, prices, session structure) and overlays ONLY the missing signals.

DESIGN PRINCIPLES (these are what stop it being circular)
---------------------------------------------------------
1. Latent traits are CONTINUOUS, never discrete labels. No user is assigned
   to "cluster 2". Segments must EMERGE from the continuous space, otherwise
   clustering just recovers labels we planted and proves nothing.

2. Latent traits are CONDITIONED on each user's real observed behaviour, then
   blended with substantial noise. A user's synthetic price sensitivity is a
   function of their real avg_item_price and spend - not drawn independently.

3. Noise is deliberately heavy (NOISE_WEIGHT below) so trait distributions
   OVERLAP. Perfectly separated clusters are an obvious tell. We target a
   believable silhouette range, not a perfect one.

4. Ground truth is saved separately. Because we know each user's true latent
   traits, we can test whether the pipeline RECOVERS them - which is the
   validation story no team using purely real data can tell.

INPUT
-----
CSV export of lookerpractice-467505.hackathon1.user_feature_table
filtered to has_transaction_data = 1 AND has_session_data = 1

OUTPUT
------
  user_feature_table_enriched.csv  -> real + synthetic features, for clustering
  ground_truth_latent_traits.csv   -> hidden traits, for validation ONLY.
                                      NEVER feed this to the clustering model.
=============================================================================
"""

import argparse
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

SEED = 42

# --- Latent structure -------------------------------------------------------
# Users are drawn from N overlapping modes in a continuous trait space. The
# modes give K-Means something real to find; the wide spread keeps them
# genuinely overlapping so no user carries a clean label.
#
# N_MODES is NOT the number of personas you must end up with. It is latent
# structure only - the clustering step chooses k independently, and if it lands
# on a different k that is a legitimate finding, not an error.
N_MODES = 4

# Standard deviation of each mode. Higher = more overlap = lower silhouette.
# 0.16 lands silhouette around 0.35-0.50, which reads as genuine data.
# Below ~0.10 the clusters separate too cleanly and look planted.
MODE_SPREAD = 0.13

# How much of a trait comes from its latent mode vs the user's real behaviour.
# Keeping real behaviour at ~40% is what stops the synthetic layer being
# disconnected from the actual TheLook data underneath it.
MODE_WEIGHT = 0.65

# Extra independent noise when turning a latent trait into an observable
# behaviour. Stops the mapping being a deterministic giveaway.
BEHAVIOUR_NOISE = 0.18

# Session ids in TheLook persist across days, so duration has absurd outliers
# (Q3 = 115,761s = 32 hours). Cap before it dominates every centroid.
SESSION_DURATION_CAP_SEC = 3600

# Simulated price change used for the elasticity / pricing-friction test
PRICE_CHANGE_PCT = 0.10


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def pct_rank(s: pd.Series) -> pd.Series:
    """Rank-normalise to 0-1. Rank-based so extreme outliers (monetary,
    duration) cannot distort the scale the way min-max would."""
    return s.rank(pct=True, method="average")


def blend_with_noise(signal, mode_draw, rng, mode_weight=MODE_WEIGHT):
    """
    Combine a latent mode draw with a real-behaviour signal, plus noise.

    IMPORTANT: the result is deliberately NOT rank-normalised. Rank-normalising
    forces a uniform distribution, which erases all density structure and leaves
    K-Means nothing to find (an earlier version did this and scored silhouette
    0.15 - i.e. no recoverable segments at all). Real customer populations have
    modes; this keeps them, while the wide mode spread keeps them overlapping.
    """
    noise = rng.normal(0, 0.05, len(signal))
    mixed = mode_weight * mode_draw + (1 - mode_weight) * signal.to_numpy() + noise
    return pd.Series(np.clip(mixed, 0, 1), index=signal.index)


def jitter(x, rng, scale=BEHAVIOUR_NOISE):
    """Additive gaussian noise for trait -> behaviour mapping."""
    return x + rng.normal(0, scale, len(x))


# --------------------------------------------------------------------------
# STEP 1: LATENT TRAITS (the hidden ground truth)
# --------------------------------------------------------------------------

def build_latent_traits(df: pd.DataFrame, rng) -> pd.DataFrame:
    """
    Four continuous latent traits, each conditioned on real observed columns.
    These are the 'true' generating process. Clustering should recover
    structure resembling them - that is the validation test.
    """
    n = len(df)
    lat = pd.DataFrame(index=df.index)
    lat["user_id"] = df["user_id"]

    # ---- latent mode centres in 4D trait space ----------------------------
    # Deliberately not one-per-corner: modes share coordinates on some axes so
    # that no single feature cleanly separates them. A merchant's real customer
    # base does not split neatly along one dimension either.
    #            price_sens  deliberation  exploration  loyalty
    centres = np.array([
        [0.72,       0.30,        0.35,       0.30],   # cheap + quick decisions
        [0.28,       0.72,        0.55,       0.60],   # premium + researches heavily
        [0.55,       0.55,        0.80,       0.35],   # broad browser, middling spend
        [0.35,       0.35,        0.25,       0.80],   # narrow, habitual repeat buyer
    ])
    assert centres.shape[0] == N_MODES

    mode_idx = rng.integers(0, N_MODES, n)
    mode_draw = centres[mode_idx] + rng.normal(0, MODE_SPREAD, (n, N_MODES))
    mode_draw = np.clip(mode_draw, 0, 1)

    # keep the generating mode for validation only - heavy overlap means
    # perfect recovery is impossible by construction, which is the point
    lat["_generating_mode"] = mode_idx

    # ---- PRICE SENSITIVITY -------------------------------------------------
    # Real driver: people buying cheap items and spending little skew price-driven.
    cheap_items = 1 - pct_rank(df["avg_item_price"])
    low_spend = 1 - pct_rank(df["monetary_net"])
    small_basket = 1 - pct_rank(df["avg_basket_size"])
    signal = 0.55 * cheap_items + 0.30 * low_spend + 0.15 * small_basket
    lat["price_sensitivity"] = blend_with_noise(signal, mode_draw[:, 0], rng)

    # ---- DELIBERATION ------------------------------------------------------
    # Real driver: session depth and duration - how much research before buying.
    deep_sessions = pct_rank(df["avg_events_per_session"])
    long_sessions = pct_rank(df["avg_session_duration_sec"].clip(upper=SESSION_DURATION_CAP_SEC))
    many_visits = pct_rank(df["total_sessions"])
    signal = 0.40 * deep_sessions + 0.35 * long_sessions + 0.25 * many_visits
    lat["deliberation"] = blend_with_noise(signal, mode_draw[:, 1], rng)

    # ---- EXPLORATION -------------------------------------------------------
    # Real driver: category diversity - breadth vs specialisation.
    broad = pct_rank(df["category_diversity"])
    varied_sessions = pct_rank(df["total_sessions"])
    signal = 0.70 * broad + 0.30 * varied_sessions
    lat["exploration"] = blend_with_noise(signal, mode_draw[:, 2], rng)

    # ---- LOYALTY -----------------------------------------------------------
    # Real driver: frequency, recency, tenure.
    repeat = pct_rank(df["frequency_orders"])
    fresh = 1 - pct_rank(df["recency_days"])
    tenured = pct_rank(df["tenure_days"])
    signal = 0.45 * repeat + 0.40 * fresh + 0.15 * tenured
    lat["loyalty"] = blend_with_noise(signal, mode_draw[:, 3], rng)

    return lat


# --------------------------------------------------------------------------
# STEP 2: DISCOUNT BEHAVIOUR (replaces the constant-zero columns)
# --------------------------------------------------------------------------

def generate_discount_behaviour(df, lat, rng) -> pd.DataFrame:
    """
    Discount reliance and depth, derived from price_sensitivity plus
    independent noise. Loyal customers get a small dampening - they buy
    regardless of promotion, which is exactly the 'margin given away' segment
    the pitch highlights.
    """
    out = pd.DataFrame(index=df.index)
    ps = lat["price_sensitivity"].to_numpy()
    loy = lat["loyalty"].to_numpy()

    # Share of items bought on discount. Loyalty pulls it down slightly.
    reliance = jitter(0.88 * ps - 0.15 * loy + 0.12, rng)
    out["discount_reliance_pct"] = np.clip(reliance, 0, 1).round(4)

    # How deep a markdown they typically take. Price-sensitive shoppers wait
    # for bigger cuts. Capped at 70% - deeper is implausible retail.
    depth = jitter(0.42 * ps + 0.06, rng, scale=0.10)
    depth = np.clip(depth, 0, 0.70)
    # Users who never take a discount must have zero depth
    depth = np.where(out["discount_reliance_pct"] < 0.02, 0.0, depth)
    out["avg_discount_depth"] = depth.round(4)

    # Deepest markdown ever taken - separates true bargain hunters from
    # people who happened to catch one small sale.
    max_depth = np.clip(jitter(depth + 0.18 * ps, rng, scale=0.08), 0, 0.85)
    out["max_discount_depth"] = np.maximum(max_depth, depth).round(4)

    # Share of spend at full price. The inverse signal - high values here on a
    # user who still receives discounts = margin the merchant threw away.
    out["full_price_spend_share"] = (1 - out["discount_reliance_pct"]).round(4)

    # Absolute discounted spend, grounded in their REAL monetary value
    out["discounted_spend"] = (
        df["monetary_net"].to_numpy() * out["discount_reliance_pct"].to_numpy()
    ).round(2)

    return out


# --------------------------------------------------------------------------
# STEP 3: CART ABANDONMENT (fills the structural attribution gap)
# --------------------------------------------------------------------------

def generate_abandonment_behaviour(df, lat, rng) -> pd.DataFrame:
    """
    Abandonment is real in the raw events (58% of cart sessions) but cannot be
    attributed to identified users. We regenerate it at user level, driven by
    price sensitivity (baulk at checkout price) and deliberation (add to cart
    to think about it). Anchored so the population mean lands near the 58%
    actually observed in the raw event data.
    """
    out = pd.DataFrame(index=df.index)
    ps = lat["price_sensitivity"].to_numpy()
    delib = lat["deliberation"].to_numpy()
    loy = lat["loyalty"].to_numpy()
    n_sessions = df["total_sessions"].to_numpy()

    # Observed population abandonment rate in raw events: 249563/430491 = 0.580
    rate = 0.58 + 0.30 * (ps - 0.5) + 0.20 * (delib - 0.5) - 0.15 * (loy - 0.5)
    rate = np.clip(jitter(rate, rng, scale=0.10), 0.02, 0.95)
    out["cart_abandonment_rate"] = rate.round(4)

    # Cart sessions: most sessions involve a cart, scaled by deliberation
    cart_prob = np.clip(0.55 + 0.30 * delib, 0.3, 0.95)
    cart_sessions = rng.binomial(n_sessions, cart_prob)
    cart_sessions = np.maximum(cart_sessions, 1)
    out["cart_sessions"] = cart_sessions

    abandoned = rng.binomial(cart_sessions, rate)
    out["abandoned_cart_sessions"] = abandoned
    out["converting_sessions"] = cart_sessions - abandoned

    out["session_conversion_rate"] = np.round(
        out["converting_sessions"] / np.maximum(n_sessions, 1), 4
    ).clip(0, 1)

    # Browse -> cart rate: how readily browsing turns into intent
    out["browse_to_cart_rate"] = np.round(
        np.clip(jitter(0.45 + 0.35 * delib - 0.10 * ps, rng, scale=0.10), 0.05, 1.0), 4
    )

    return out


# --------------------------------------------------------------------------
# STEP 4: PRICE-CHANGE RESPONSE (the pricing-friction test)
# --------------------------------------------------------------------------

def generate_price_response(df, lat, rng, price_change=PRICE_CHANGE_PCT) -> pd.DataFrame:
    """
    Simulates a uniform price increase and models heterogeneous response -
    the core claim of the whole project: not all customers react the same way.

    This is HELD-OUT GROUND TRUTH. Do not feed these columns to the clustering
    model. Use them to test whether the personas the pipeline produced actually
    predicted the right response.
    """
    out = pd.DataFrame(index=df.index)
    ps = lat["price_sensitivity"].to_numpy()
    loy = lat["loyalty"].to_numpy()
    delib = lat["deliberation"].to_numpy()

    out["simulated_price_change_pct"] = price_change

    # Probability of churning entirely after the price rise
    churn = 0.55 * ps - 0.30 * loy + 0.10 * delib
    churn = np.clip(jitter(churn + 0.15, rng, scale=0.08), 0.01, 0.95)
    out["prob_churn_on_price_increase"] = churn.round(4)

    # Own-price elasticity: % change in quantity per % change in price.
    # Negative by definition; price-sensitive users are more elastic.
    elasticity = -(0.4 + 2.6 * ps) + 0.6 * loy
    out["price_elasticity"] = jitter(elasticity, rng, scale=0.25).round(4)

    # Expected spend retained after the increase
    qty_change = out["price_elasticity"].to_numpy() * price_change
    retained = np.clip(1 + qty_change, 0, 1.5) * (1 - churn)
    out["expected_spend_retention"] = np.clip(retained, 0, 1.5).round(4)

    out["expected_revenue_after_change"] = (
        df["monetary_net"].to_numpy()
        * out["expected_spend_retention"].to_numpy()
        * (1 + price_change)
    ).round(2)

    out["revenue_delta"] = (
        out["expected_revenue_after_change"].to_numpy() - df["monetary_net"].to_numpy()
    ).round(2)

    return out


# --------------------------------------------------------------------------
# STEP 5: CLEANUP OF REAL COLUMNS
# --------------------------------------------------------------------------

def clean_real_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes to the genuinely-real columns before clustering."""
    out = df.copy()

    # Session ids persist across days in TheLook, producing 32-hour sessions.
    # Cap, and keep a log version - raw duration would dominate every centroid.
    out["avg_session_duration_capped"] = out["avg_session_duration_sec"].clip(
        upper=SESSION_DURATION_CAP_SEC
    )
    out["log_session_duration"] = np.log1p(out["avg_session_duration_capped"]).round(4)

    # Monetary is heavily right-skewed (0 -> 20 -> 59 -> 139 -> 1578)
    out["log_monetary_net"] = np.log1p(out["monetary_net"]).round(4)
    out["log_avg_item_price"] = np.log1p(out["avg_item_price"]).round(4)

    return out


# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------

def report(df, lat, enriched):
    print("\n" + "=" * 70)
    print("GENERATION REPORT")
    print("=" * 70)
    print(f"Users processed: {len(df):,}")

    print("\n--- Latent traits (should each be ~uniform 0-1) ---")
    print(lat[["price_sensitivity", "deliberation", "exploration", "loyalty"]]
          .describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

    print("\n--- Trait correlations (should be LOW - independent axes) ---")
    print(lat[["price_sensitivity", "deliberation", "exploration", "loyalty"]]
          .corr().round(3).to_string())

    print("\n--- Generated signals ---")
    cols = ["discount_reliance_pct", "avg_discount_depth", "cart_abandonment_rate",
            "session_conversion_rate", "prob_churn_on_price_increase", "price_elasticity"]
    print(enriched[cols].describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
          .round(3).to_string())

    print("\n--- Grounding check: synthetic vs REAL behaviour ---")
    print("  (nonzero = synthetic signals are conditioned on real data, not independent)")
    checks = [
        ("discount_reliance_pct", "avg_item_price", "expect NEGATIVE"),
        ("cart_abandonment_rate", "monetary_net", "expect NEGATIVE"),
        ("prob_churn_on_price_increase", "frequency_orders", "expect NEGATIVE"),
    ]
    for syn, real, note in checks:
        r = enriched[syn].corr(enriched[real])
        print(f"  corr({syn}, {real}) = {r:+.3f}   [{note}]")

    ab = enriched["abandoned_cart_sessions"].sum()
    cs = enriched["cart_sessions"].sum()
    print(f"\n  Population abandonment rate: {ab/cs:.3f}  (raw events showed 0.580)")


REAL_FEATURES = ["log_monetary_net", "recency_days", "log_avg_item_price",
                 "total_sessions", "avg_events_per_session", "category_diversity"]
SYNTH_FEATURES = ["discount_reliance_pct", "avg_discount_depth",
                  "cart_abandonment_rate", "browse_to_cart_rate"]


def _sil(enriched, feats, k, n=10000):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = StandardScaler().fit_transform(enriched[feats].fillna(0))
    idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
    X = X[idx]
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    return silhouette_score(X, km.labels_)


def silhouette_check(enriched):
    """
    Decomposes where cluster structure actually comes from.

    This matters more than the headline number. If REAL-only silhouette is near
    zero, the real TheLook features carry no natural segments and every cluster
    you find is being driven by the synthetic overlay - which is circular, and a
    judge can attack it. If REAL-only shows some structure, the synthetic layer
    is sharpening real segments rather than inventing them.

    Reference points for interpretation:
      < 0.15   weak / little separable structure
      0.15-0.35 typical for genuine customer segmentation
      0.35-0.60 strong, still plausible
      > 0.70   suspiciously clean - looks planted, expect to be challenged
    """
    print("\n--- Silhouette decomposition (where does structure come from?) ---")
    print(f"  {'k':<4}{'real only':>12}{'synthetic only':>17}{'combined':>12}")
    for k in range(3, 7):
        r = _sil(enriched, REAL_FEATURES, k)
        s = _sil(enriched, SYNTH_FEATURES, k)
        c = _sil(enriched, REAL_FEATURES + SYNTH_FEATURES, k)
        print(f"  {k:<4}{r:>12.3f}{s:>17.3f}{c:>12.3f}")

    r4 = _sil(enriched, REAL_FEATURES, 4)
    s4 = _sil(enriched, SYNTH_FEATURES, 4)
    print()
    if r4 < 0.12:
        print("  WARNING: real features carry almost no cluster structure.")
        print("  Segments will be driven mainly by the synthetic overlay.")
        print("  State this openly in the pitch rather than letting it be found.")
    if s4 > 0.70:
        print("  WARNING: synthetic features separate too cleanly.")
        print("  Raise MODE_SPREAD to increase overlap before shipping.")
    if 0.12 <= r4 and s4 <= 0.70:
        print("  Structure looks balanced between real and synthetic sources.")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV export of user_feature_table")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-silhouette", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # keep only users with complete data on both sides
    if "has_transaction_data" in df.columns and "has_session_data" in df.columns:
        before = len(df)
        df = df[(df["has_transaction_data"] == 1) & (df["has_session_data"] == 1)]
        print(f"Filtered to complete-data users: {before:,} -> {len(df):,}")

    df = df.reset_index(drop=True)

    required = ["user_id", "monetary_net", "recency_days", "frequency_orders",
                "avg_item_price", "total_sessions", "avg_events_per_session",
                "avg_session_duration_sec", "category_diversity", "avg_basket_size",
                "tenure_days"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # fill any residual nulls in driver columns before ranking
    for c in required[1:]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # ---- build ----
    lat = build_latent_traits(df, rng)
    disc = generate_discount_behaviour(df, lat, rng)
    aband = generate_abandonment_behaviour(df, lat, rng)
    price = generate_price_response(df, lat, rng)
    base = clean_real_columns(df)

    # drop the dead constant columns from the real table if present
    dead = ["discount_reliance_pct", "avg_discount_depth", "max_discount_depth",
            "full_price_spend_share", "cart_abandonment_rate", "cart_sessions",
            "abandoned_cart_sessions", "converting_sessions",
            "session_conversion_rate", "browse_to_cart_rate"]
    base = base.drop(columns=[c for c in dead if c in base.columns])

    enriched = pd.concat([base, disc, aband, price], axis=1)

    # ---- report ----
    report(df, lat, enriched)
    if not args.skip_silhouette:
        silhouette_check(enriched)

    # ---- save ----
    out_main = f"{args.outdir}/user_feature_table_enriched.csv"
    out_truth = f"{args.outdir}/ground_truth_latent_traits.csv"
    enriched.to_csv(out_main, index=False)
    lat.to_csv(out_truth, index=False)

    print("\n" + "=" * 70)
    print(f"WROTE  {out_main}   ({enriched.shape[0]:,} x {enriched.shape[1]})")
    print(f"WROTE  {out_truth}  <- VALIDATION ONLY, never feed to clustering")
    print("=" * 70)


if __name__ == "__main__":
    main()