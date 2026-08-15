"""
=============================================================================
STEP 2 - CLUSTERING AND CLUSTER PROFILING
Customer Persona Project
=============================================================================

Takes user_feature_table_enriched.csv and produces:

  users_clustered.csv    every user with a cluster_id
  cluster_profiles.json  compact per-cluster summary -> input to the LLM
  elbow_silhouette.png   the chart for your slides

WHY DISCOUNT IS HELD OUT BY DEFAULT
-----------------------------------
The discount / abandonment / elasticity columns are SYNTHETIC. If we cluster on
them, we are partly rediscovering structure we planted, and a judge can say so.

Default behaviour: cluster on BEHAVIOURAL features only (spend, browsing,
categories), then measure whether the discovered segments happen to differ on
price sensitivity. If they do, that is a real finding - price sensitivity was
not used to build the segments, it emerged. That is a much stronger claim.

Use --include-synthetic to cluster on everything instead. The script reports
both so you can compare.
=============================================================================
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SEED = 42

# Behavioural features - all REAL, all from TheLook
BEHAVIOURAL = [
    "log_monetary_net",        # spend level (log: raw is heavily right-skewed)
    "recency_days",            # how long since they bought
    "log_avg_item_price",      # what price point they shop at
    "frequency_orders",        # repeat purchasing
    "total_sessions",          # how often they visit
    "avg_events_per_session",  # browsing depth per visit
    "log_session_duration",    # how long they linger (capped, then logged)
    "category_diversity",      # breadth vs specialisation
]

# Synthetic overlay features - held out by default, see docstring
SYNTHETIC = [
    "discount_reliance_pct",
    "avg_discount_depth",
    "cart_abandonment_rate",
    "browse_to_cart_rate",
]

# Never cluster on these - they are the held-out outcome we want to predict
NEVER_CLUSTER = [
    "prob_churn_on_price_increase",
    "price_elasticity",
    "expected_spend_retention",
    "expected_revenue_after_change",
    "revenue_delta",
]


def choose_k(X, k_range, outpath):
    """Elbow (inertia) + silhouette across candidate k. Saves the chart."""
    rng = np.random.default_rng(SEED)
    samp = X[rng.choice(len(X), min(10000, len(X)), replace=False)]

    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        inertias.append(km.inertia_)
        km_s = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(samp)
        sils.append(silhouette_score(samp, km_s.labels_))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(k_range), inertias, "o-")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
    ax[0].set_title("Elbow method")
    ax[1].plot(list(k_range), sils, "o-", color="darkorange")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette")
    ax[1].set_title("Silhouette by k")
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()

    print("\n--- Choosing k ---")
    print(f"  {'k':<5}{'inertia':>14}{'silhouette':>14}")
    for k, i, s in zip(k_range, inertias, sils):
        print(f"  {k:<5}{i:>14,.0f}{s:>14.3f}")
    print(f"  chart -> {outpath}")
    return dict(zip(k_range, sils))


def profile_clusters(df, features, extra_numeric, categoricals):
    """Build the compact per-cluster summary that gets handed to the LLM."""
    profiles = []
    total = len(df)
    overall = df[features + extra_numeric].mean()

    for cid, grp in df.groupby("cluster_id"):
        p = {
            "cluster_id": int(cid),
            "size": int(len(grp)),
            "share_of_customers_pct": round(100 * len(grp) / total, 1),
        }

        # revenue share matters more than headcount share for the pitch
        if "monetary_net" in df.columns:
            p["share_of_revenue_pct"] = round(
                100 * grp["monetary_net"].sum() / df["monetary_net"].sum(), 1)

        stats = {}
        for c in features + extra_numeric:
            if c not in grp.columns:
                continue
            mean = grp[c].mean()
            # index vs population: 100 = average, 150 = 1.5x the average.
            # This is what makes a cluster interpretable to a merchant.
            idx = round(100 * mean / overall[c], 0) if overall[c] not in (0, np.nan) else None
            stats[c] = {"mean": round(float(mean), 3),
                        "vs_population_index": None if idx is None or np.isnan(idx) else int(idx)}
        p["metrics"] = stats

        for c in categoricals:
            if c in grp.columns:
                vc = grp[c].value_counts(normalize=True).head(3)
                p[f"top_{c}"] = {str(k): round(float(v), 3) for k, v in vc.items()}

        profiles.append(p)

    return sorted(profiles, key=lambda x: -x["size"])


def recovery_check(df, truth_path):
    """
    If ground truth latent traits exist, check whether the discovered clusters
    actually separate on them. This is the validation slide: we did not cluster
    on these traits, so differences here mean the pipeline recovered real
    structure rather than inventing it.
    """
    try:
        truth = pd.read_csv(truth_path)
    except Exception:
        print("\n(no ground truth file found - skipping recovery check)")
        return None

    traits = ["price_sensitivity", "deliberation", "exploration", "loyalty"]
    traits = [t for t in traits if t in truth.columns]
    m = df[["user_id", "cluster_id"]].merge(truth, on="user_id", how="inner")

    print("\n--- Recovery check: cluster means on HELD-OUT latent traits ---")
    print("  (these were never fed to the model - spread here = real recovery)")
    tbl = m.groupby("cluster_id")[traits].mean().round(3)
    print(tbl.to_string())

    spread = (tbl.max() - tbl.min()).round(3)
    print("\n  spread across clusters (higher = better separation):")
    for t in traits:
        flag = "  <-- weak" if spread[t] < 0.10 else ""
        print(f"    {t:<20}{spread[t]:.3f}{flag}")
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--truth", default=None, help="ground_truth_latent_traits.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--k", type=int, default=None, help="skip selection, force k")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="cluster on synthetic features too (see docstring)")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} users, {df.shape[1]} columns")

    features = [c for c in BEHAVIOURAL if c in df.columns]
    if args.include_synthetic:
        features += [c for c in SYNTHETIC if c in df.columns]

    missing = [c for c in BEHAVIOURAL if c not in df.columns]
    if missing:
        print(f"  WARNING missing behavioural columns: {missing}")

    leaked = [c for c in NEVER_CLUSTER if c in features]
    if leaked:
        raise SystemExit(f"Outcome columns leaked into features: {leaked}")

    print(f"\nClustering on {len(features)} features:")
    for f in features:
        print(f"  - {f}")
    if not args.include_synthetic:
        print("  (synthetic discount/abandonment HELD OUT - will be tested after)")

    X = df[features].fillna(df[features].median())
    Xs = StandardScaler().fit_transform(X)

    # ---- pick k ----
    if args.k:
        k = args.k
        print(f"\nUsing forced k = {k}")
    else:
        sils = choose_k(Xs, range(2, 9), f"{args.outdir}/elbow_silhouette.png")
        # prefer 3-5 for a demo: fewer is unusable, more is unpitchable
        candidates = {kk: v for kk, v in sils.items() if 3 <= kk <= 5}
        k = max(candidates, key=candidates.get)
        print(f"\nSelected k = {k} (best silhouette in the 3-5 range)")
        print("  note: 3-5 chosen for pitchability. If a k outside that range")
        print("  scores far better, say so on the slide rather than hiding it.")

    # ---- fit ----
    km = KMeans(n_clusters=k, n_init=25, random_state=SEED).fit(Xs)
    df["cluster_id"] = km.labels_
    final_sil = silhouette_score(
        Xs[np.random.default_rng(SEED).choice(len(Xs), min(10000, len(Xs)), replace=False)],
        km.predict(Xs[np.random.default_rng(SEED).choice(len(Xs), min(10000, len(Xs)), replace=False)])
    )
    print(f"\nFinal silhouette at k={k}: {final_sil:.3f}")

    print("\n--- Cluster sizes ---")
    for cid, n in df["cluster_id"].value_counts().sort_index().items():
        print(f"  cluster {cid}: {n:,} users ({100*n/len(df):.1f}%)")

    # ---- did price sensitivity emerge without being clustered on? ----
    if not args.include_synthetic:
        syn_present = [c for c in SYNTHETIC if c in df.columns]
        if syn_present:
            print("\n--- Held-out synthetic signals by cluster ---")
            print("  (spread here = segments differ on price sensitivity even")
            print("   though it was NEVER used to build them)")
            print(df.groupby("cluster_id")[syn_present].mean().round(3).to_string())

    # ---- outcome columns: the thing personas must predict ----
    out_present = [c for c in NEVER_CLUSTER if c in df.columns]
    if out_present:
        print("\n--- Held-out OUTCOMES by cluster (never used anywhere) ---")
        print(df.groupby("cluster_id")[out_present].mean().round(3).to_string())

    # ---- recovery vs ground truth ----
    if args.truth:
        recovery_check(df, args.truth)

    # ---- 2D projection for the slide ----
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(Xs)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(coords), min(6000, len(coords)), replace=False)
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(coords[idx, 0], coords[idx, 1], c=df["cluster_id"].values[idx],
                cmap="tab10", s=5, alpha=0.5)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    plt.title(f"Customer segments (k={k})")
    plt.tight_layout(); plt.savefig(f"{args.outdir}/clusters_pca.png", dpi=140); plt.close()

    # ---- profiles for the LLM ----
    extra = [c for c in ["monetary_net", "avg_item_price", "avg_basket_size",
                         "return_rate", "tenure_days", "age",
                         "discount_reliance_pct", "avg_discount_depth",
                         "cart_abandonment_rate", "session_conversion_rate",
                         "prob_churn_on_price_increase", "price_elasticity"]
             if c in df.columns]
    cats = [c for c in ["top_category", "dominant_traffic_source", "gender", "country"]
            if c in df.columns]

    profiles = profile_clusters(df, features, extra, cats)
    meta = {
        "k": int(k),
        "silhouette": round(float(final_sil), 4),
        "clustered_on": features,
        "held_out": [c for c in SYNTHETIC + NEVER_CLUSTER if c in df.columns
                     and c not in features],
        "total_users": int(len(df)),
        "clusters": profiles,
    }

    with open(f"{args.outdir}/cluster_profiles.json", "w") as f:
        json.dump(meta, f, indent=2)
    df.to_csv(f"{args.outdir}/users_clustered.csv", index=False)

    print("\n" + "=" * 70)
    print(f"WROTE {args.outdir}/users_clustered.csv")
    print(f"WROTE {args.outdir}/cluster_profiles.json   <- feed this to the LLM")
    print(f"WROTE {args.outdir}/elbow_silhouette.png")
    print(f"WROTE {args.outdir}/clusters_pca.png")
    print("=" * 70)


if __name__ == "__main__":
    main()