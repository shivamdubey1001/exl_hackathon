"""
=============================================================================
SYNTHETIC A/B TEST ENGINE
=============================================================================

Samples users per persona, asks each one - in character - how they would
respond to two variants, then scales the result to whatever population the
merchant selected.

SAMPLING AND SCALING - why not one call per user
-------------------------------------------------
The obvious design is one API call per simulated customer. It is also wrong,
in two ways:

  Cost. At roughly 1,900 tokens a call, 10,000 users is about $75 and over an
  hour of wall time. One slider drag should not do that.

  Statistics. Users inside a persona are behaviourally similar BY CONSTRUCTION
  - that is what put them in the same cluster. Simulating 500 of them yields
  500 near-identical answers. The spread you would measure is LLM sampling
  noise, not customer variance, so averaging more of it buys precision about
  the model rather than about customers.

So we simulate a fixed sample per persona and project onto the selected
population, weighted by true segment size. That is what every market research
study does, and it is the honest description of what the numbers mean.

The results always report BOTH figures - sample size and population - so the
projection is never hidden behind a large number.

THE A/A TEST
------------
An LLM asked the same question twice gives different answers. Run with
--aa-test and both arms become identical; whatever lift comes back is pure
model noise, and any real result below it is meaningless.
=============================================================================
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import threading
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

import usage

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_MODEL = "claude-sonnet-5"
MAX_WORKERS = 6
REQUESTS_PER_SECOND = 4.0

# --- sample size policy ---------------------------------------------------
# The user gives a TOTAL sample size; it is divided evenly across personas.
# 120 total across 4 personas means 30 each.
#
# 30 per persona is comfortably past the point where extra samples stop
# changing the answer, because within-segment variance is low by construction.
DEFAULT_SAMPLE_SIZE = 120

# No hard ceiling: the live cost estimate is the safeguard, since it is always
# on screen before the run button. A limit that blocks a legitimate large run
# is worse than a number the user can see and reconsider.

# Rough token cost of one shopper call, measured from real runs. Used only for
# the pre-run estimate the UI shows.
EST_INPUT_TOKENS = 1500
EST_OUTPUT_TOKENS = 400


class RateLimiter:
    """Token-bucket rate limit shared across worker threads."""
    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            gap = now - self.last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self.last = time.monotonic()


# =============================================================================
# COST ESTIMATE
# =============================================================================

def split_sample(sample_size: int, n_personas: int, available: int) -> tuple:
    """
    Turn a total sample size into a per-persona count.

    Divided evenly rather than proportionally: a persona holding 58% of the
    base would otherwise swamp the sample and the small high-value segment
    would barely appear - and the small segment is usually the interesting one.
    Aggregates are re-weighted by true segment size afterwards.
    """
    n_personas = max(1, n_personas)
    sample_size = max(n_personas, int(sample_size))      # at least one each
    sample_size = min(sample_size, available)            # cannot exceed the base
    per_persona = max(1, sample_size // n_personas)
    return per_persona, per_persona * n_personas


def estimate_run(n_personas: int, sample_size: int, available: int,
                 model: str = DEFAULT_MODEL) -> dict:
    """
    What a run would cost before committing to it.

    Called on every keystroke in the sample-size box, so it must stay pure
    arithmetic - no API calls, no file reads.
    """
    per_persona, calls = split_sample(sample_size, n_personas, available)

    cost = usage.compute_cost(model, calls * EST_INPUT_TOKENS,
                              calls * EST_OUTPUT_TOKENS)
    seconds = calls / REQUESTS_PER_SECOND

    return {
        "sample_size": calls,
        "per_persona": per_persona,
        "total_calls": calls,
        "estimated_cost_usd": round(cost, 4),
        "estimated_seconds": int(seconds),
        "default_sample_size": DEFAULT_SAMPLE_SIZE,
        "max_sample_size": available,
        # rounding down to an even split can lose a few from the request
        "adjusted": calls != min(int(sample_size), available),
    }


# =============================================================================
# BUILD PER-USER METADATA FROM REAL COLUMNS
# =============================================================================

BASELINE_MAP = {
    "average_order_value_usd": "avg_order_value",
    "lifetime_spend_usd": "monetary_net",
    "avg_item_price_usd": "avg_item_price",
    "orders_placed": "frequency_orders",
    "days_since_last_order": "recency_days",
}

BEHAVIOUR_MAP = {
    "total_sessions": "total_sessions",
    "pages_viewed_per_session": "avg_events_per_session",
    "session_conversion_rate": "session_conversion_rate",
    "cart_abandonment_rate": "cart_abandonment_rate",
    "browse_to_cart_rate": "browse_to_cart_rate",
    "discount_reliance": "discount_reliance_pct",
    "avg_discount_depth": "avg_discount_depth",
    "categories_shopped": "category_diversity",
    "return_rate": "return_rate",
}

DEMO_MAP = {
    "age": "age",
    "gender": "gender",
    "country": "country",
    "top_category": "top_category",
    "acquisition_channel": "dominant_traffic_source",
}

PERSONA_TRAITS = ["price_sensitivity_score", "brand_loyalty_score",
                  "deliberation_score"]


def _num(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


def build_user_metadata(row: pd.Series) -> dict:
    """Everything the model needs about this specific customer."""
    def pull(mapping):
        out = {}
        for label, col in mapping.items():
            if col in row.index:
                v = _num(row[col])
                if v is not None:
                    out[label] = v
        return out

    meta = {
        "persona": row.get("persona_name"),
        "persona_summary": row.get("one_line_summary"),
        "financial_baseline": pull(BASELINE_MAP),
        "measured_behaviour": pull(BEHAVIOUR_MAP),
        "profile": pull(DEMO_MAP),
        "persona_traits": {t: _num(row[t]) for t in PERSONA_TRAITS
                           if t in row.index and not pd.isna(row[t])},
    }

    scr = row.get("session_conversion_rate")
    if scr is not None and not pd.isna(scr):
        meta["financial_baseline"]["baseline_conversion_rate_pct"] = \
            round(float(scr) * 100, 2)

    for f in ["core_buying_triggers", "primary_hesitations"]:
        if f in row.index and isinstance(row[f], str):
            meta[f] = [p.strip() for p in row[f].split("|")]

    return meta


# =============================================================================
# SAMPLING
# =============================================================================

def sample_users(df: pd.DataFrame, per_persona: int, seed: int = 42) -> pd.DataFrame:
    """
    Balanced random sample: `per_persona` users from each persona.

    Balanced rather than proportional on purpose. A persona holding 58% of the
    base would otherwise dominate the sample and the small high-value segment
    would barely appear - and the small segment is usually the interesting one.
    Aggregates are re-weighted by true segment size afterwards.
    """
    if "persona_name" not in df.columns:
        raise ValueError("users file has no persona_name column")

    parts = []
    for _, grp in df.groupby("persona_name"):
        n = min(len(grp), per_persona)
        parts.append(grp.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


# =============================================================================
# PROMPTING
# =============================================================================

SYSTEM_PROMPT = """You are simulating one specific online shopper's response \
to a change on a retail website.

You are given that shopper's REAL measured behaviour: their actual conversion \
rate, order value, browsing depth, and discount reliance. These are facts, not \
estimates.

Your job is to predict how each variant changes their behaviour RELATIVE to \
that measured baseline.

Rules:
1. Anchor on the measured baseline. If their conversion rate is 12%, variant A \
should be near 12% unless the variant itself changes something.
2. Be realistic about magnitude. Most UX changes move conversion by a few \
percent relative, not double it. Reserve large swings for changes that remove \
a real barrier for this specific shopper.
3. Differentiate by person. A shopper with price_sensitivity 0.25 should barely \
react to a discount banner. One at 0.75 should react strongly.
4. It is a valid and common answer that a variant changes nothing for this \
shopper. Do not invent a lift to seem useful.
5. Return ONE JSON object, no markdown fences, no commentary.
6. Inside JSON string values, never use double quotes and avoid apostrophes \
where you can. Refer to variants as variant A and variant B rather than \
quoting their text back."""


RESPONSE_FIELDS = {
    "p_conv_variant_a": "<number 0-1> predicted conversion probability for variant A",
    "p_conv_variant_b": "<number 0-1> predicted conversion probability for variant B",
    "predicted_aov_variant_a": "<number> predicted order value in USD for variant A",
    "predicted_aov_variant_b": "<number> predicted order value in USD for variant B",
    "friction_driver": "<string> one sentence on what drives this shopper's decision",
    "quotable_line": "<string> under 20 words, in this shopper's own voice, about variant B",
}


def build_user_prompt(meta: dict, variant_a: str, variant_b: str) -> str:
    return (
        "SHOPPER PROFILE (all figures are measured, not estimated):\n"
        + json.dumps(meta, indent=2)
        + "\n\nVARIANT A (control):\n  " + variant_a
        + "\n\nVARIANT B (treatment):\n  " + variant_b
        + "\n\nReturn raw JSON in exactly this shape:\n"
        + json.dumps(RESPONSE_FIELDS, indent=2)
    )


# =============================================================================
# LLM
# =============================================================================

def extract_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        raise json.JSONDecodeError("no JSON object found", t, 0)
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Models often echo quoted variant text into a string value, which
        # breaks the JSON. Turn stray double quotes mid-value into apostrophes.
        fixed = re.sub(r'(?<=[\w\s)\-%$.])"(?=[\w\s(\-%$.])', "'", blob)
        return json.loads(fixed)


def evaluate_user(sample: dict, variant_a: str, variant_b: str,
                  model: str, limiter: RateLimiter,
                  system_extra: str = "") -> dict:
    """
    One API call for one simulated shopper.

    On total failure this returns failed=True with NO predictions. It
    deliberately does not fall back to baseline-for-both-variants: that yields
    exactly zero lift, which is indistinguishable from a real null result and
    would quietly poison the aggregate.
    """
    from usage import make_client
    client = make_client()
    ##client = anthropic.Anthropic()

    meta = sample["metadata"]
    system = SYSTEM_PROMPT + (("\n\n" + system_extra) if system_extra else "")

    last_err = None
    for attempt in range(3):
        limiter.wait()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user",
                           "content": build_user_prompt(meta, variant_a, variant_b)}],
            )
            usage.record("abtest", model, resp,
                         label=f"{sample['persona']} - user {sample['user_id']}")
            text = "".join(b.text for b in resp.content if b.type == "text")
            d = extract_json(text)
            return {
                "user_id": sample["user_id"],
                "persona": sample["persona"],
                "cvr_a": float(d["p_conv_variant_a"]),
                "cvr_b": float(d["p_conv_variant_b"]),
                "aov_a": float(d["predicted_aov_variant_a"]),
                "aov_b": float(d["predicted_aov_variant_b"]),
                "reason": d.get("friction_driver", ""),
                "quote": d.get("quotable_line", ""),
                "failed": False,
            }
        except Exception as e:
            last_err = e
            print(f"[ab] user {sample['user_id']} attempt {attempt+1} failed: "
                  f"{type(e).__name__}: {e}")
            time.sleep(2 ** attempt)

    return {
        "user_id": sample["user_id"],
        "persona": sample["persona"],
        "cvr_a": np.nan, "cvr_b": np.nan,
        "aov_a": np.nan, "aov_b": np.nan,
        "reason": f"ERROR: {last_err}", "quote": "",
        "failed": True,
    }


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate(df: pd.DataFrame, persona_sizes: Dict[str, int],
              population: int, per_persona: int) -> dict:
    """
    Roll the sample up to headline figures and project onto the population.

    Sampling is balanced across personas, so a raw mean would misrepresent a
    base where one persona holds most of the customers. Everything is
    re-weighted by true segment size before projection.
    """
    ok = df[~df["failed"]].copy()
    if ok.empty:
        return {"error": "every simulated shopper failed", "failures": len(df)}

    total_customers = sum(persona_sizes.values()) or 1
    weights = {p: persona_sizes.get(p, 0) / total_customers
               for p in ok["persona"].unique()}
    ok["w"] = ok["persona"].map(weights).fillna(0)

    def wmean(col):
        return (float(np.average(ok[col], weights=ok["w"]))
                if ok["w"].sum() else float(ok[col].mean()))

    cvr_a, cvr_b = wmean("cvr_a"), wmean("cvr_b")
    aov_a, aov_b = wmean("aov_a"), wmean("aov_b")
    rpu_a, rpu_b = cvr_a * aov_a, cvr_b * aov_b

    def lift(a, b):
        return ((b - a) / a * 100) if a else 0.0

    ok["user_rpu_a"] = ok["cvr_a"] * ok["aov_a"]
    ok["user_rpu_b"] = ok["cvr_b"] * ok["aov_b"]
    ok["user_lift"] = np.where(
        ok["user_rpu_a"] > 0,
        (ok["user_rpu_b"] - ok["user_rpu_a"]) / ok["user_rpu_a"] * 100, 0.0)

    n = len(ok)
    spread = float(ok["user_lift"].std(ddof=1)) if n > 1 else 0.0
    stderr = spread / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * stderr

    persona_df = ok.groupby("persona").agg(
        n=("user_id", "count"),
        avg_cvr_a=("cvr_a", "mean"), avg_cvr_b=("cvr_b", "mean"),
        avg_aov_a=("aov_a", "mean"), avg_aov_b=("aov_b", "mean"),
    ).reset_index()
    persona_df["cvr_lift_pct"] = ((persona_df["avg_cvr_b"] - persona_df["avg_cvr_a"])
                                  / persona_df["avg_cvr_a"].replace(0, np.nan) * 100).round(2)
    persona_df["aov_lift_pct"] = ((persona_df["avg_aov_b"] - persona_df["avg_aov_a"])
                                  / persona_df["avg_aov_a"].replace(0, np.nan) * 100).round(2)
    persona_df["rpu_lift_pct"] = (
        ((persona_df["avg_cvr_b"] * persona_df["avg_aov_b"])
         - (persona_df["avg_cvr_a"] * persona_df["avg_aov_a"]))
        / (persona_df["avg_cvr_a"] * persona_df["avg_aov_a"]).replace(0, np.nan) * 100
    ).round(2)
    persona_df["segment_size"] = persona_df["persona"].map(persona_sizes)
    # how many of the projected population sit in each segment
    persona_df["projected_customers"] = (
        persona_df["segment_size"] / total_customers * population).round(0).astype(int)

    # --- projection onto the selected population ---
    revenue_a = rpu_a * population
    revenue_b = rpu_b * population

    return {
        "overall_summary": {
            "simulated_users": n,
            "failed_users": int(df["failed"].sum()),
            "per_persona": per_persona,
            "population": population,
            "total_customers_in_dataset": total_customers,
            "control_cvr": round(cvr_a * 100, 2),
            "treatment_cvr": round(cvr_b * 100, 2),
            "cvr_relative_lift_pct": round(lift(cvr_a, cvr_b), 2),
            "control_aov": round(aov_a, 2),
            "treatment_aov": round(aov_b, 2),
            "aov_relative_lift_pct": round(lift(aov_a, aov_b), 2),
            "control_rpu": round(rpu_a, 2),
            "treatment_rpu": round(rpu_b, 2),
            "rpu_relative_lift_pct": round(lift(rpu_a, rpu_b), 2),
            "lift_std_dev_pct": round(spread, 2),
            "lift_std_error_pct": round(stderr, 2),
            "lift_ci95_pct": round(ci95, 2),
            # what the lift is worth across the selected population
            "projected_revenue_control": round(revenue_a, 2),
            "projected_revenue_treatment": round(revenue_b, 2),
            "projected_revenue_delta": round(revenue_b - revenue_a, 2),
        },
        "persona_breakdown": persona_df.to_dict(orient="records"),
        "raw_results": ok.drop(columns=["w"]).to_dict(orient="records"),
    }


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_ab_test(users_csv: str, variant_a: str, variant_b: str,
                sample_size: Optional[int] = None,
                lookup_path: Optional[str] = None,
                model: str = DEFAULT_MODEL, seed: int = 42,
                aa_test: bool = False, in_character: bool = True,
                progress=None) -> dict:
    """
    Single entry point, called directly by the API.

    `sample_size` is the total number of shoppers actually simulated - this is
    the cost. It is split evenly across personas. Results are always projected
    across the whole dataset.
    """
    df = pd.read_csv(users_csv)
    persona_sizes = df["persona_name"].value_counts().to_dict()
    n_personas = df["persona_name"].nunique()

    if sample_size is None:
        sample_size = DEFAULT_SAMPLE_SIZE
    per_persona, _ = split_sample(sample_size, n_personas, len(df))

    pop = len(df)          # always the whole dataset

    lookup = {}
    if lookup_path and os.path.exists(lookup_path):
        lookup = json.load(open(lookup_path, encoding="utf-8"))

    # In an A/A test both arms are identical. Whatever lift comes back is noise.
    b_desc = variant_a if aa_test else variant_b

    sampled = sample_users(df, per_persona, seed)
    samples = []
    for _, row in sampled.iterrows():
        extra = ""
        if in_character and lookup:
            entry = next((v for v in lookup.values()
                          if v.get("persona_name") == row["persona_name"]), None)
            if entry and entry.get("system_prompt"):
                extra = ("The shopper you are simulating is described as "
                         "follows. Let it shape their reaction:\n"
                         + entry["system_prompt"])
        samples.append({
            "user_id": row["user_id"],
            "persona": row["persona_name"],
            "metadata": build_user_metadata(row),
            "system_extra": extra,
        })

    limiter = RateLimiter(REQUESTS_PER_SECOND)
    results = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(evaluate_user, s, variant_a, b_desc, model,
                             limiter, s["system_extra"]) for s in samples]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
            done += 1
            if progress:
                progress(done, len(samples))

    out = aggregate(pd.DataFrame(results), persona_sizes, pop, per_persona)
    out["config"] = {
        "variant_a": variant_a,
        "variant_b": b_desc,
        "aa_test": aa_test,
        "per_persona": per_persona,
        "sample_size": len(samples),
        "population": pop,
        "total_calls": len(samples),
        "model": model,
        "seed": seed,
    }
    return out


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True, help="users_with_personas.csv")
    ap.add_argument("--lookup", default=None, help="persona_lookup.json")
    ap.add_argument("--variant-a", default="Standard product page, no changes")
    ap.add_argument("--variant-b", default="10% discount banner on orders over $100")
    ap.add_argument("--sample-size", type=int, default=None,
                    help=f"total shoppers to simulate, split evenly across "
                         f"personas (default {DEFAULT_SAMPLE_SIZE})")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aa-test", action="store_true")
    ap.add_argument("--no-character", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def prog(done, total):
        print(f"\r  {done}/{total} shoppers simulated", end="", flush=True)

    print("=" * 74)
    print("A/A NOISE TEST" if args.aa_test else "SYNTHETIC A/B TEST")
    print(f"  A: {args.variant_a}")
    print(f"  B: {args.variant_a if args.aa_test else args.variant_b}")
    print("=" * 74)

    res = run_ab_test(
        args.users, args.variant_a, args.variant_b,
        sample_size=args.sample_size, lookup_path=args.lookup,
        model=args.model, seed=args.seed, aa_test=args.aa_test,
        in_character=not args.no_character, progress=prog)
    print()

    if "error" in res:
        print("FAILED:", res["error"])
        return

    s = res["overall_summary"]
    print(f"\n  {s['simulated_users']} shoppers simulated "
          f"({s['per_persona']} per persona, {s['failed_users']} failed)")
    print(f"  projected across {s['population']:,} customers "
          f"of {s['total_customers_in_dataset']:,}\n")
    print(f"  {'':<12}{'control':>12}{'treatment':>12}{'lift':>12}")
    print(f"  {'CVR':<12}{s['control_cvr']:>11.2f}%{s['treatment_cvr']:>11.2f}%"
          f"{s['cvr_relative_lift_pct']:>11.2f}%")
    print(f"  {'AOV':<12}{s['control_aov']:>12.2f}{s['treatment_aov']:>12.2f}"
          f"{s['aov_relative_lift_pct']:>11.2f}%")
    print(f"  {'RPU':<12}{s['control_rpu']:>12.2f}{s['treatment_rpu']:>12.2f}"
          f"{s['rpu_relative_lift_pct']:>11.2f}%")
    print(f"\n  lift 95% CI: +/- {s['lift_ci95_pct']:.2f}pp")
    print(f"  projected revenue delta: {s['projected_revenue_delta']:,.0f}")

    print("\n--- BY PERSONA ---")
    print(f"  {'persona':<28}{'n':>4}{'cvr lift':>11}{'aov lift':>11}{'rpu lift':>11}")
    for p in res["persona_breakdown"]:
        print(f"  {str(p['persona'])[:27]:<28}{p['n']:>4}"
              f"{p['cvr_lift_pct']:>10.2f}%{p['aov_lift_pct']:>10.2f}%"
              f"{p['rpu_lift_pct']:>10.2f}%")

    if args.aa_test:
        print("\n" + "!" * 74)
        print(f"  NOISE FLOOR: {abs(s['rpu_relative_lift_pct']):.2f}% RPU lift on "
              f"IDENTICAL variants.")
        print("  Any A/B result smaller than this is indistinguishable from noise.")
        print("!" * 74)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()