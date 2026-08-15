"""
=============================================================================
AGENT 2 - THE PERSONA AGENT, PLUS RECONCILIATION
=============================================================================

Loads each persona's system_prompt, puts it in front of a proposed merchant
intervention, and gets an in-character reaction. Then checks that reaction
against Agent 1's arithmetic and flags disagreements.

  python persona_agent.py \
      --lookup out/persona_lookup.json \
      --analyst out/sim_coupon15.json \
      --out out/simulation_coupon15.json \
      --model claude-sonnet-5

WHY THE RECONCILIATION MATTERS
------------------------------
Two LLMs agreeing with each other proves nothing. Here the persona produces a
story and the analyst produces a number, and they are generated independently.
If the persona says "I'd buy anyway" while its segment's elasticity implies a
40% volume drop, that contradiction is surfaced rather than smoothed over.

Three checks are run:

  1. DIRECTION   does the persona's verdict match the sign of the volume change?
  2. MAGNITUDE   is the persona's stated purchase likelihood roughly consistent
                 with the elasticity-implied volume change?
  3. RANKING     across all personas, does the order of stated likelihood match
                 the order the analyst predicts? This is the strongest check,
                 because the persona never sees the elasticity figures.

COST
----
ONE API call per persona. Not per user. A merchant simulating 50,000 customers
costs exactly the same as simulating 50, because the persona reaction is
generated once and the analyst engine scales it deterministically.
=============================================================================
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional
import usage

from dotenv import load_dotenv
load_dotenv()
import llm

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    sys.exit("pip install pydantic")


# =============================================================================
# SCHEMA
# =============================================================================

VERDICTS = ["buy_more", "buy_anyway", "buy_less", "hesitate", "walk_away"]


class PersonaReaction(BaseModel):
    gut_reaction: str = Field(
        description="First person, in character, 2-3 sentences. How you feel "
                    "seeing this change. Natural customer language, not marketing speak.")
    verdict: str = Field(
        description=f"Exactly one of: {', '.join(VERDICTS)}")
    purchase_intent_change: float = Field(
        ge=-1.0, le=1.0,
        description="How your buying changes. -1.0 you stop entirely, 0.0 no "
                    "change, +0.5 you buy about 50 percent more, +1.0 you buy "
                    "twice as much. Negative for anything that puts you off, "
                    "positive for anything that tempts you to buy more.")
    main_objection: str = Field(
        description="The single biggest reason you might not go through with it. "
                    "One sentence.")
    what_would_change_my_mind: str = Field(
        description="One concrete thing the retailer could do to keep you. One sentence.")
    quotable_line: str = Field(
        description="One short sentence, under 20 words, that a merchant could "
                    "put on a slide. Must sound like a real customer speaking.")

    @field_validator("verdict", mode="before")
    @classmethod
    def _norm_verdict(cls, v):
        if isinstance(v, str):
            s = v.strip().lower().replace(" ", "_").replace("-", "_")
            if s in VERDICTS:
                return s
            for cand in VERDICTS:          # tolerate near-misses
                if cand in s:
                    return cand
        return v

    @field_validator("purchase_intent_change", mode="before")
    @classmethod
    def _norm_pct(cls, v):
        if isinstance(v, str):
            v = float(v.strip().rstrip("%").replace("+", ""))
        if isinstance(v, (int, float)) and abs(v) > 1.0:
            return float(v) / 100.0      # model wrote -25 instead of -0.25
        return v


# =============================================================================
# PROMPTING
# =============================================================================

def describe_intervention(interv: dict) -> str:
    kind = interv["kind"]
    mag = interv["magnitude"]

    if kind == "price_change":
        base = (f"Prices have gone UP by {abs(mag):.0%}."
                if mag > 0 else f"Prices have come DOWN by {abs(mag):.0%}.")
    elif kind == "coupon":
        base = f"There is a {abs(mag):.0%} off discount code available."
    elif kind == "shipping_change":
        base = ("Shipping is now free." if mag < 0
                else f"A shipping charge of {mag:.2f} has been added at checkout.")
    else:
        base = interv.get("label", "A change has been made.")

    if interv.get("department"):
        base += f" This applies to the {interv['department']} department."
    if interv.get("category"):
        base += f" This applies to {interv['category']}."
    return base


REACTION_INSTRUCTION = """You are roleplaying a real online shopper. Stay \
completely in character.

Rules:
1. React as this specific customer would, not as a marketer or analyst.
2. Do not mention statistics, segments, clusters, or that you are an AI.
3. Be honest. If the change genuinely would not bother you, say so. If it would \
make you leave, say that. Do not soften your reaction to be agreeable.
4. `quotable_line` must sound like something a real person would say out loud.
5. Return ONE JSON object. No markdown fences, no preamble."""


def build_reaction_prompt(interv_text: str, schema_json: str) -> str:
    return (
        "You are shopping at an online retailer you have used before.\n"
        "You have just noticed the following change:\n\n"
        f"  {interv_text}\n\n"
        "How do you react? Answer honestly as yourself.\n\n"
        "Return raw JSON in exactly this shape:\n" + schema_json
    )


# =============================================================================
# LLM
# =============================================================================

def call_anthropic(system: str, user: str, model: str, label: str = "") -> str:
    """Name kept for compatibility; provider is resolved inside llm.call."""
    return llm.call(system, user, model=model, max_tokens=2000,
                    step="reaction", label=label)


def call_openai(system: str, user: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, max_tokens=2000,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def extract_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# =============================================================================
# RECONCILIATION
# =============================================================================

VERDICT_EXPECTED_DIRECTION = {
    "buy_more":   +1,
    "buy_anyway":  0,   # no material change either way
    "buy_less":   -1,
    "hesitate":   -1,
    "walk_away":  -1,
}


def reconcile_one(reaction: PersonaReaction, segment: dict) -> dict:
    """Check one persona's story against its segment's arithmetic."""
    vol = segment["volume_change_pct"]
    dp = segment["effective_price_delta"]
    issues = []

    # ---- check 1: direction ----
    # A discount should raise volume, a price rise should lower it. Compare the
    # persona's stated direction against the analyst's, treating anything
    # within +/-5% as "no material change".
    def sign(x, tol=0.05):
        return 0 if abs(x) < tol else (1 if x > 0 else -1)

    persona_dir = sign(reaction.purchase_intent_change)
    analyst_dir = sign(vol)
    # a stated verdict that flatly contradicts the number is the real problem
    verdict_dir = VERDICT_EXPECTED_DIRECTION.get(reaction.verdict, 0)
    direction_ok = (persona_dir == analyst_dir) or (persona_dir == 0 and analyst_dir == 0)
    if not direction_ok:
        issues.append(
            f"persona intent {reaction.purchase_intent_change:+.0%} points the "
            f"opposite way to predicted volume {vol:+.0%}")
    if verdict_dir != 0 and sign(reaction.purchase_intent_change) != 0 \
            and verdict_dir != persona_dir:
        issues.append(
            f"persona's own verdict '{reaction.verdict}' contradicts its stated "
            f"intent of {reaction.purchase_intent_change:+.0%}")

    # ---- check 2: magnitude ----
    implied_vol = reaction.purchase_intent_change
    gap = abs(implied_vol - vol)
    magnitude_ok = gap <= 0.30
    if not magnitude_ok:
        issues.append(
            f"persona implies {implied_vol:+.0%} volume change, analyst says "
            f"{vol:+.0%} (gap {gap:.0%})")

    # a discount that the persona shrugs at is worth flagging: it means the
    # merchant is paying for nothing in this segment
    if dp < 0 and reaction.verdict == "buy_anyway" and abs(reaction.purchase_intent_change) < 0.05:
        issues.append("MARGIN WARNING: persona would have bought without the "
                      "discount - this segment is pure margin loss")

    return {
        "direction_ok": direction_ok,
        "magnitude_ok": magnitude_ok,
        "implied_volume_change": round(implied_vol, 4),
        "analyst_volume_change": vol,
        "gap": round(gap, 4),
        "issues": issues,
    }


def reconcile_ranking(results: List[dict]) -> dict:
    """
    The strongest check. The personas never saw the elasticity numbers, so if
    they rank the segments in the same order the analyst does, that ordering
    was recovered from character alone.
    """
    usable = [r for r in results if r.get("reaction") and r.get("segment")]
    if len(usable) < 2:
        return {"checked": False, "reason": "need at least 2 personas"}

    by_persona = sorted(usable,
                        key=lambda r: -r["reaction"]["purchase_intent_change"])
    by_analyst = sorted(usable,
                        key=lambda r: -r["segment"]["volume_change_pct"])

    order_p = [r["cluster_id"] for r in by_persona]
    order_a = [r["cluster_id"] for r in by_analyst]

    # count concordant pairs (simple rank agreement)
    n, conc = len(order_p), 0
    total_pairs = n * (n - 1) // 2
    pos_p = {c: i for i, c in enumerate(order_p)}
    pos_a = {c: i for i, c in enumerate(order_a)}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order_p[i], order_p[j]
            if (pos_p[a] < pos_p[b]) == (pos_a[a] < pos_a[b]):
                conc += 1

    return {
        "checked": True,
        "persona_order": order_p,
        "analyst_order": order_a,
        "exact_match": order_p == order_a,
        "concordant_pairs": conc,
        "total_pairs": total_pairs,
        "agreement": round(conc / total_pairs, 3) if total_pairs else None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True, help="persona_lookup.json")
    ap.add_argument("--analyst", required=True,
                    help="analyst_engine output JSON (--out from analyst_engine.py)")
    ap.add_argument("--out", default="simulation_result.json")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--scenario", choices=["blanket", "targeted"], default="blanket")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lookup = json.load(open(args.lookup, encoding="utf-8"))
    analyst = json.load(open(args.analyst, encoding="utf-8"))

    # analyst_engine writes either a plain run or a blanket/targeted comparison
    run = analyst[args.scenario] if args.scenario in analyst else analyst
    interv = run["intervention"]
    segments = {int(s["cluster_id"]): s for s in run["segments"]}

    interv_text = describe_intervention(interv)
    schema_json = json.dumps(
        {f: (info.description or "") for f, info in PersonaReaction.model_fields.items()},
        indent=2)

    print("=" * 74)
    print(f"SIMULATING: {interv['label']}   [{args.scenario}]")
    print(f"  presented to personas as: {interv_text}")
    print("=" * 74)

    results = []
    for cid_str, persona in lookup.items():
        cid = int(cid_str)
        seg = segments.get(cid)
        name = persona.get("persona_name", f"Cluster {cid}")

        print(f"\n--- {name}  (cluster {cid}) ---")

        if seg is None:
            print("  no analyst result for this cluster, skipping")
            continue
        if not seg.get("affected", True):
            print("  not affected by this intervention, skipping")
            results.append({"cluster_id": cid, "persona_name": name,
                            "segment": seg, "reaction": None,
                            "skipped": "not affected"})
            continue

        system = persona["system_prompt"]
        user = build_reaction_prompt(interv_text, schema_json)

        if args.dry_run:
            print("  [dry run] system prompt:")
            print("   ", system[:200].replace("\n", " "), "...")
            continue

        caller = call_anthropic if args.provider == "anthropic" else call_openai
        raw = caller(system + "\n\n" + REACTION_INSTRUCTION, user, args.model)

        try:
            reaction = PersonaReaction(**extract_json(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  FAILED to parse: {e}")
            results.append({"cluster_id": cid, "persona_name": name,
                            "segment": seg, "reaction": None, "error": str(e)})
            continue

        rec = reconcile_one(reaction, seg)

        print(f"  verdict: {reaction.verdict}   "
              f"intent change: {reaction.purchase_intent_change:+.0%}")
        print(f"  \"{reaction.quotable_line}\"")
        print(f"  objection: {reaction.main_objection}")
        print(f"  analyst says volume {seg['volume_change_pct']:+.1%}, "
              f"persona implies {rec['implied_volume_change']:+.1%}")
        if rec["issues"]:
            for i in rec["issues"]:
                print(f"    ! {i}")
        else:
            print("    reconciled: story and numbers agree")

        results.append({
            "cluster_id": cid,
            "persona_name": name,
            "segment": seg,
            "reaction": reaction.model_dump(),
            "reconciliation": rec,
        })

    if args.dry_run:
        print("\n[dry run] no API calls made")
        return

    ranking = reconcile_ranking(results)

    print("\n" + "=" * 74)
    print("RANKING CHECK - personas never saw the elasticity figures")
    if ranking.get("checked"):
        print(f"  persona order (most positive intent first): "
              f"{ranking['persona_order']}")
        print(f"  analyst order:                                    "
              f"{ranking['analyst_order']}")
        print(f"  agreement: {ranking['concordant_pairs']}/{ranking['total_pairs']} "
              f"pairs ({ranking['agreement']:.0%})")
        if ranking["exact_match"]:
            print("  EXACT MATCH - the personas recovered the correct ordering")
    print("=" * 74)

    payload = {
        "intervention": interv,
        "scenario": args.scenario,
        "model": args.model,
        "totals": run["totals"],
        "results": results,
        "ranking_check": ranking,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()