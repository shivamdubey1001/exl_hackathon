"""
=============================================================================
ASSIGN PERSONAS TO USERS
=============================================================================

Joins personas.json back onto users_clustered.csv so every user row carries
the persona their segment was assigned.

  python assign_personas.py \
      --users out/users_clustered.csv \
      --personas personas/personas.json \
      --out out/users_with_personas.csv

Produces one row per user with persona name, summary, trait scores, and the
qualitative fields flattened for CSV. Also writes a compact
persona_lookup.json for the UI, so the frontend does not have to load the
full user table just to render a persona card.
=============================================================================
"""

import argparse
import json
import os

import pandas as pd


# Fields copied straight onto each user row
SCALAR_FIELDS = [
    "persona_name",
    "one_line_summary",
    "price_sensitivity_score",
    "brand_loyalty_score",
    "deliberation_score",
    "segment_share_percentage",
    "revenue_share_percentage",
    "shopping_style",
    "channel_preference",
    "predicted_price_increase_reaction",
    "merchant_recommendation",
]

# List fields, flattened to a pipe-delimited string so the CSV stays readable.
# Pipe rather than comma so Excel does not split them into columns.
LIST_FIELDS = ["core_buying_triggers", "primary_hesitations", "evidence"]

# Long text kept out of the per-user CSV - it would repeat 46,000 times.
# Lives in persona_lookup.json instead.
EXCLUDE_FROM_CSV = ["system_prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--out", default="users_with_personas.csv")
    ap.add_argument("--lookup", default=None,
                    help="where to write persona_lookup.json "
                         "(default: alongside --out)")
    ap.add_argument("--include-system-prompt", action="store_true",
                    help="also write system_prompt into the CSV (large)")
    args = ap.parse_args()

    users = pd.read_csv(args.users)
    payload = json.load(open(args.personas, encoding="utf-8"))
    personas = payload["personas"]

    if "cluster_id" not in users.columns:
        raise SystemExit("users file has no cluster_id column")

    print(f"Loaded {len(users):,} users and {len(personas)} personas")

    # ---- build one row per persona, then merge ----
    rows = []
    for p in personas:
        row = {"cluster_id": p["cluster_id"]}
        for f in SCALAR_FIELDS:
            if f in p:
                row[f] = p[f]
        for f in LIST_FIELDS:
            if f in p:
                v = p[f]
                row[f] = " | ".join(v) if isinstance(v, list) else v
        if args.include_system_prompt and "system_prompt" in p:
            row["system_prompt"] = p["system_prompt"]
        rows.append(row)

    pmap = pd.DataFrame(rows)

    # ---- check every cluster has a persona before merging ----
    user_clusters = set(users["cluster_id"].dropna().unique())
    persona_clusters = set(pmap["cluster_id"])
    orphans = user_clusters - persona_clusters
    if orphans:
        print(f"  WARNING clusters with no persona: {sorted(orphans)}")
        print("  (those users will get blank persona fields)")
    unused = persona_clusters - user_clusters
    if unused:
        print(f"  WARNING personas matching no users: {sorted(unused)}")

    out = users.merge(pmap, on="cluster_id", how="left")

    if len(out) != len(users):
        raise SystemExit(
            f"Row count changed on merge: {len(users):,} -> {len(out):,}. "
            "personas.json probably has duplicate cluster_ids.")

    # ---- report ----
    print("\n--- Assignment summary ---")
    summary = (out.groupby(["cluster_id", "persona_name"])
                 .agg(users=("user_id", "count"),
                      revenue=("monetary_net", "sum"))
                 .reset_index())
    total_rev = out["monetary_net"].sum() if "monetary_net" in out else None
    for _, r in summary.iterrows():
        rev_txt = ""
        if total_rev:
            rev_txt = (f"  {r['revenue']:>14,.0f} revenue "
                       f"({100*r['revenue']/total_rev:>4.1f}%)")
        print(f"  [{int(r['cluster_id'])}] {r['persona_name']:<28}"
              f"{r['users']:>8,} users ({100*r['users']/len(out):>4.1f}%){rev_txt}")

    missing = out["persona_name"].isna().sum()
    if missing:
        print(f"\n  {missing:,} users have no persona assigned")

    # ---- write ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)

    lookup_path = args.lookup or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "persona_lookup.json")
    lookup = {
        str(p["cluster_id"]): {
            k: v for k, v in p.items() if k != "cluster_id"
        } for p in personas
    }
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)

    print(f"\nWROTE {args.out}  ({len(out):,} rows x {out.shape[1]} cols)")
    print(f"WROTE {lookup_path}  <- full persona objects incl. system_prompt")


if __name__ == "__main__":
    main()