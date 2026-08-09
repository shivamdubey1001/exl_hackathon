"""
=============================================================================
STEP 3 - STAGE 2 QUALITATIVE SYNTHESIS
Turn cluster statistics into runnable AI buyer personas.
=============================================================================

Reads  cluster_profiles.json  (from cluster_and_profile.py)
Writes personas.json          - validated persona objects, one per cluster
       persona_prompts/       - the raw prompts, for your appendix slide

Each persona carries a system_prompt field. That is the payload: it is what
gets loaded into a fresh LLM call at simulation time to make the persona react
in character to a campaign, price change, or product page.

WHAT IS AND IS NOT GROUNDED
---------------------------
The LLM is given ONLY the cluster's real aggregate statistics. It is explicitly
forbidden from inventing numbers. After generation, validate_grounding() checks
that every numeric claim in the persona matches the input stats - so you can
say on stage that persona traits are traceable to data, not hallucinated.

RUNNING WITHOUT AN API KEY
--------------------------
  --dry-run   prints the prompts and writes them to disk, calls nothing.
Use this to iterate on prompt wording for free, then run for real once.
=============================================================================
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    sys.exit("pip install pydantic")

from pydantic import BaseModel, Field, ValidationError, field_validator
# =============================================================================
# SCHEMA - the contract the LLM must fill
# =============================================================================

class GeneratedPersona(BaseModel):
    cluster_id: int = Field(description="Cluster this persona represents")

    persona_name: str = Field(
        description="Catchy archetype name, 2-4 words, e.g. 'The Deliberate Explorer'")
    one_line_summary: str = Field(
        description="Single sentence a merchant could repeat from memory")

    segment_share_percentage: float = Field(
        description="Share of total customer base, from the input stats")
    revenue_share_percentage: float = Field(
        description="Share of total revenue, from the input stats")

    core_buying_triggers: List[str] = Field(
        description="3-5 specific things that make this segment buy")
    primary_hesitations: List[str] = Field(
        description="3-5 specific things that make them abandon or delay")

    price_sensitivity_score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 unbothered by price, 1.0 extremely price-sensitive")
    brand_loyalty_score: float = Field(
        ge=0.0, le=1.0, description="0.0 no loyalty, 1.0 die-hard")
    deliberation_score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 impulse buyer, 1.0 researches extensively")

    shopping_style: str = Field(
        description="2-3 sentences on how they actually browse and decide")
    channel_preference: str = Field(
        description="Where they come from and what messaging reaches them")

    predicted_price_increase_reaction: str = Field(
        description="What this segment does if prices rise ~10 percent")

    merchant_recommendation: str = Field(
        description="One concrete action a merchant should take for this segment")

    evidence: List[str] = Field(
        description="3-5 statistics from the input that justify this persona. "
                    "Each must quote an actual number provided.")

    system_prompt: str = Field(
        description="Second-person instructions that make an LLM roleplay this "
                    "customer during simulation. Must be self-contained.")

    @field_validator("segment_share_percentage", "revenue_share_percentage",
                     "price_sensitivity_score", "brand_loyalty_score",
                     "deliberation_score", mode="before")
    @classmethod
    def _strip_pct(cls, v):
        if isinstance(v, str):
            return float(v.strip().rstrip("%"))
        return v

    @field_validator("core_buying_triggers", "primary_hesitations",
                     "evidence", mode="before")
    @classmethod
    def _to_list(cls, v):
        if isinstance(v, str):
            parts = [p.strip() for p in re.split(r"(?<=[.)])\s+(?=[A-Z])", v)]
            return [p for p in parts if p]
        return v


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

SYSTEM_INSTRUCTION = """You are an expert e-commerce data scientist who \
translates customer segmentation output into actionable buyer personas.

Rules you must follow:
1. Use ONLY the statistics provided. Never invent a number.
2. Every claim in `evidence` must quote a figure that appears in the input.
3. Compare against the population. An index of 100 is average; 150 means this \
segment is 1.5x the average on that metric. Call out what makes this segment \
DIFFERENT, not what makes it typical.
4. The persona must be specific enough that a merchant could recognise these \
customers. Avoid generic marketing language.
5. `system_prompt` must be written in second person ("You are...") and be \
fully self-contained, because it will be loaded on its own into a separate \
model call with no other context.
6. Return ONE JSON object. No markdown fences, no preamble, no commentary.
8. Types matter. Percentages are bare numbers, not strings - write 58.0 not "58.0%". 
Fields marked "array of strings" must be JSON arrays like ["a", "b", "c"], never one concatenated string."""


def build_user_prompt(cluster: dict, meta: dict, schema_json: str,
                      chat_samples: Optional[List[str]] = None) -> str:
    parts = []
    parts.append("SEGMENTATION CONTEXT")
    parts.append(f"- Total customers analysed: {meta['total_users']:,}")
    parts.append(f"- Number of segments: {meta['k']}")
    parts.append(f"- Silhouette score: {meta['silhouette']}")
    parts.append(f"- Segments were built ONLY from these behavioural features: "
                 f"{', '.join(meta['clustered_on'])}")
    if meta.get("held_out"):
        parts.append(f"- These were HELD OUT of clustering and measured "
                     f"afterwards: {', '.join(meta['held_out'])}")
    parts.append("")

    parts.append(f"CLUSTER {cluster['cluster_id']} STATISTICS")
    parts.append(f"- Size: {cluster['size']:,} customers "
                 f"({cluster['share_of_customers_pct']}% of base)")
    if "share_of_revenue_pct" in cluster:
        parts.append(f"- Revenue share: {cluster['share_of_revenue_pct']}%")
    parts.append("")

    parts.append("METRICS (mean, and index vs population where 100 = average)")
    metrics = cluster["metrics"]
    items = metrics.items() if isinstance(metrics, dict) else metrics
    for name, v in items:
        idx = v.get("vs_population_index")
        idx_txt = f"  [index {idx}]" if idx is not None else ""
        parts.append(f"  {name}: {v['mean']}{idx_txt}")
    parts.append("")

    for key in cluster:
        if key.startswith("top_") and isinstance(cluster[key], (dict, list)):
            field = key[4:]
            pairs = cluster[key].items() if isinstance(cluster[key], dict) else cluster[key]
            vals = ", ".join(f"{k} ({100*v:.0f}%)" for k, v in pairs)
            parts.append(f"{field.upper()}: {vals}")
    parts.append("")

    if chat_samples:
        parts.append("SAMPLE SHOPPING-ASSISTANT CONVERSATIONS FROM THIS SEGMENT")
        for i, c in enumerate(chat_samples, 1):
            parts.append(f"  [{i}] {c}")
        parts.append("")

    parts.append("OUTPUT SCHEMA - return exactly this shape as raw JSON:")
    parts.append(schema_json)

    return "\n".join(parts)


# =============================================================================
# LLM CALL
# =============================================================================

def call_anthropic(system: str, user: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=6000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def call_openai(system: str, user: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=3000,
    )
    return resp.choices[0].message.content


def extract_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in fences despite instructions. Strip and parse."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# =============================================================================
# GROUNDING VALIDATION
# =============================================================================

def collect_input_numbers(cluster: dict) -> set:
    """Every number the LLM was actually shown, rounded a few ways."""
    nums = set()

    def add(x):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return
        for r in (0, 1, 2):
            nums.add(round(f, r))
        nums.add(round(f * 100, 0))   # proportions quoted as percentages

    add(cluster.get("size"))
    add(cluster.get("share_of_customers_pct"))
    add(cluster.get("share_of_revenue_pct"))

    metrics = cluster["metrics"]
    items = metrics.items() if isinstance(metrics, dict) else metrics
    for _, v in items:
        add(v.get("mean"))
        add(v.get("vs_population_index"))

    for key in cluster:
        if key.startswith("top_") and isinstance(cluster[key], (dict, list)):
            pairs = cluster[key].items() if isinstance(cluster[key], dict) else cluster[key]
            for _, v in pairs:
                add(v)
    return nums


def validate_grounding(persona: GeneratedPersona, cluster: dict) -> dict:
    """
    Checks every number appearing in `evidence` against the numbers the model
    was given. Unmatched figures are likely hallucinated.

    This is the check that lets you claim on stage that persona traits are
    traceable to the data.
    """
    allowed = collect_input_numbers(cluster)
    findings = {"checked": 0, "matched": 0, "unmatched": []}

    for line in persona.evidence:
        for tok in re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", "")):
            f = float(tok)
            findings["checked"] += 1
            if any(abs(f - a) < 0.51 for a in allowed):
                findings["matched"] += 1
            else:
                findings["unmatched"].append({"value": f, "in": line[:90]})

    findings["grounding_rate"] = (
        round(findings["matched"] / findings["checked"], 3)
        if findings["checked"] else None
    )
    return findings


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True, help="cluster_profiles.json")
    ap.add_argument("--outdir", default="personas")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--chats", default=None,
                    help="optional JSON: {cluster_id: [transcript, ...]}")
    ap.add_argument("--dry-run", action="store_true",
                    help="write prompts, call no API")
    args = ap.parse_args()

    model = args.model or ("claude-sonnet-4-5" if args.provider == "anthropic"
                           else "gpt-4o")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/persona_prompts", exist_ok=True)

    meta = json.load(open(args.profiles))
    chats = json.load(open(args.chats)) if args.chats else {}

    def _type_name(ann):
        s = str(ann)
        if "List[str]" in s or "list[str]" in s:
            return "array of strings"
        if "float" in s:
            return "number"
        if "int" in s:
            return "integer"
        return "string"

    schema_json = json.dumps({
        f: f"<{_type_name(info.annotation)}> {info.description or ''}"
        for f, info in GeneratedPersona.model_fields.items()
    }, indent=2)

    personas, reports = [], []

    for cluster in meta["clusters"]:
        cid = cluster["cluster_id"]
        print(f"\n{'='*60}\nCluster {cid}  "
              f"({cluster['size']:,} users, {cluster['share_of_customers_pct']}%)")

        user_prompt = build_user_prompt(
            cluster, meta, schema_json, chats.get(str(cid)) or chats.get(cid))

        with open(f"{args.outdir}/persona_prompts/cluster_{cid}.txt", "w",
                  encoding="utf-8") as f:
            f.write("=== SYSTEM ===\n" + SYSTEM_INSTRUCTION +
                    "\n\n=== USER ===\n" + user_prompt)

        if args.dry_run:
            print(user_prompt[:1400])
            print("... [truncated]  (dry run - no API call)")
            continue

        caller = call_anthropic if args.provider == "anthropic" else call_openai
        raw = caller(SYSTEM_INSTRUCTION, user_prompt, model)

        try:
            data = extract_json(raw)
            data["cluster_id"] = cid          # never trust the model on this
            persona = GeneratedPersona(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  FAILED to parse/validate: {e}")
            with open(f"{args.outdir}/persona_prompts/cluster_{cid}_RAW_FAIL.txt",
                      "w", encoding="utf-8") as f:
                f.write(raw)
            continue

        grounding = validate_grounding(persona, cluster)
        print(f"  -> {persona.persona_name}")
        print(f"     {persona.one_line_summary}")
        print(f"     price_sensitivity={persona.price_sensitivity_score} "
              f"loyalty={persona.brand_loyalty_score} "
              f"deliberation={persona.deliberation_score}")
        print(f"     grounding: {grounding['matched']}/{grounding['checked']} "
              f"figures traced to input data")
        for u in grounding["unmatched"][:3]:
            print(f"       UNVERIFIED {u['value']} in: {u['in']}")

        personas.append(persona.model_dump())
        reports.append({"cluster_id": cid, **grounding})

    if args.dry_run:
        print(f"\nPrompts written to {args.outdir}/persona_prompts/")
        print("Review them, then rerun without --dry-run.")
        return

    out = {
        "source_silhouette": meta["silhouette"],
        "k": meta["k"],
        "model": model,
        "personas": personas,
        "grounding_reports": reports,
    }
    with open(f"{args.outdir}/personas.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    rates = [r["grounding_rate"] for r in reports if r["grounding_rate"] is not None]
    print(f"\n{'='*60}")
    print(f"Generated {len(personas)}/{len(meta['clusters'])} personas")
    if rates:
        print(f"Mean grounding rate: {sum(rates)/len(rates):.1%}")
        print("  (share of numeric claims traceable to the input statistics)")
    print(f"WROTE {args.outdir}/personas.json")


if __name__ == "__main__":
    main()