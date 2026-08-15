"""
=============================================================================
PERSONA SYNTHESIS AND ASSIGNMENT
=============================================================================

Stage 2, refactored from the v1 CLI script so the API can call it inside a job.

Each cluster's aggregate statistics go to the LLM, which returns a structured
persona. Every numeric claim it makes is then checked back against the numbers
it was given, so "the personas are grounded in the data" is a measurement
rather than a promise.
=============================================================================
"""

import json
import os
import re
from typing import Callable, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
import time
import usage
import llm

DEFAULT_MODEL = os.getenv("LAUNCHGUARD_MODEL", "claude-sonnet-5")


# =============================================================================
# SCHEMA
# =============================================================================

class GeneratedPersona(BaseModel):
    cluster_id: int
    persona_name: str = Field(description="Catchy archetype name, 2-4 words")
    one_line_summary: str = Field(description="One sentence a merchant could repeat from memory")
    segment_share_percentage: float = Field(description="Share of customer base, from the stats")
    revenue_share_percentage: float = Field(description="Share of revenue, from the stats")
    core_buying_triggers: List[str] = Field(description="3-5 specific things that make them buy")
    primary_hesitations: List[str] = Field(description="3-5 specific things that stop them")
    price_sensitivity_score: float = Field(ge=0, le=1, description="0 unbothered by price, 1 extremely sensitive")
    brand_loyalty_score: float = Field(ge=0, le=1, description="0 no loyalty, 1 die-hard")
    deliberation_score: float = Field(ge=0, le=1, description="0 impulse buyer, 1 researches heavily")
    shopping_style: str = Field(description="2-3 sentences on how they browse and decide")
    channel_preference: str = Field(description="Where they arrive from and what reaches them")
    predicted_price_increase_reaction: str = Field(description="What they do if prices rise ~10%")
    merchant_recommendation: str = Field(description="One concrete action for this segment")
    evidence: List[str] = Field(description="3-5 statistics from the input that justify this persona")
    system_prompt: str = Field(description="Second person instructions to roleplay this customer")

    @field_validator("segment_share_percentage", "revenue_share_percentage",
                     "price_sensitivity_score", "brand_loyalty_score",
                     "deliberation_score", mode="before")
    @classmethod
    def _num(cls, v):
        if isinstance(v, str):
            return float(v.strip().rstrip("%").replace("+", ""))
        return v

    @field_validator("core_buying_triggers", "primary_hesitations", "evidence",
                     mode="before")
    @classmethod
    def _list(cls, v):
        if isinstance(v, str):
            parts = [p.strip() for p in re.split(r"(?<=[.)])\s+(?=[A-Z])", v)]
            return [p for p in parts if p]
        return v


SYSTEM_INSTRUCTION = """You are an expert e-commerce data scientist who turns \
customer segmentation output into actionable buyer personas.

Rules:
1. Use ONLY the statistics provided. Never invent a number.
2. Every item in `evidence` must quote a figure that appears in the input.
3. An index of 100 is the population average; 150 means this segment is 1.5x \
the average. Describe what makes this segment DIFFERENT, not what makes it typical.
4. Each persona must be clearly distinguishable from the others in the same \
run. Contrast them explicitly rather than describing each in isolation.
5. `system_prompt` is written in second person and must stand alone, because \
it gets loaded into a separate model call with no other context. Keep it under \
200 words.
6. Return ONE JSON object. No markdown fences, no preamble.
7. Types matter. Percentages are bare numbers (58.0, not "58%"). Array fields \
must be JSON arrays, never one concatenated string."""


def _schema_hint() -> str:
    def tname(ann):
        s = str(ann)
        if "List[str]" in s or "list[str]" in s:
            return "array of strings"
        if "float" in s:
            return "number"
        if "int" in s:
            return "integer"
        return "string"
    return json.dumps(
        {f: f"<{tname(i.annotation)}> {i.description or ''}"
         for f, i in GeneratedPersona.model_fields.items()}, indent=2)


def build_prompt(cluster: dict, meta: dict, siblings: List[dict]) -> str:
    p = ["SEGMENTATION CONTEXT",
         f"- Customers analysed: {meta['total_users']:,}",
         f"- Segments found: {meta['k']}",
         f"- Silhouette score: {meta['silhouette']} "
         f"(random data scores about {meta.get('random_baseline', 0.17)})",
         f"- Segments were built ONLY from: {', '.join(meta['clustered_on'])}"]
    if meta.get("held_out"):
        p.append(f"- Measured afterwards, NOT used to build segments: "
                 f"{', '.join(meta['held_out'])}")
    p.append("")

    p.append(f"THIS SEGMENT — CLUSTER {cluster['cluster_id']}")
    p.append(f"- Size: {cluster['size']:,} customers "
             f"({cluster['share_of_customers_pct']}% of base)")
    if "share_of_revenue_pct" in cluster:
        p.append(f"- Revenue share: {cluster['share_of_revenue_pct']}%")
    p.append("")
    p.append("METRICS (mean, index vs population where 100 = average)")
    for name, v in cluster["metrics"].items():
        idx = v.get("vs_population_index")
        p.append(f"  {name}: {v['mean']}" + (f"  [index {idx}]" if idx is not None else ""))
    p.append("")

    for key in cluster:
        if key.startswith("top_") and isinstance(cluster[key], dict):
            vals = ", ".join(f"{k} ({100*v:.0f}%)" for k, v in cluster[key].items())
            p.append(f"{key[4:].upper()}: {vals}")
    p.append("")

    # sibling summaries force contrast and reduce the mode collapse that makes
    # every persona sound like the same agreeable shopper
    others = [c for c in siblings if c["cluster_id"] != cluster["cluster_id"]]
    if others:
        p.append("THE OTHER SEGMENTS IN THIS RUN — make yours clearly distinct:")
        for o in others:
            bits = []
            for k in ["monetary_net", "recency_days", "total_sessions",
                      "avg_events_per_session", "avg_item_price"]:
                if k in o["metrics"]:
                    bits.append(f"{k} index {o['metrics'][k].get('vs_population_index')}")
            p.append(f"  Cluster {o['cluster_id']} "
                     f"({o['share_of_customers_pct']}% of base): {', '.join(bits)}")
        p.append("")

    p.append("Return raw JSON in exactly this shape:")
    p.append(_schema_hint())
    return "\n".join(p)


# =============================================================================
# LLM
# =============================================================================

def call_llm(system: str, user: str, model: str, label: str = "") -> str:
    """Retries the whole operation, not just the network call."""
    import time
    last = None
    for attempt in range(3):
        try:
            return llm.call(system, user, model=model, max_tokens=8000,
                            step="personas", label=label)
        except Exception as e:
            last = e
            print(f"[personas] call attempt {attempt+1} failed: "
                  f"{type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    raise last


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
# GROUNDING
# =============================================================================

def _allowed_numbers(cluster: dict) -> set:
    nums = set()

    def add(x):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return
        for r in (0, 1, 2):
            nums.add(round(f, r))
        nums.add(round(f * 100, 0))

    add(cluster.get("size"))
    add(cluster.get("share_of_customers_pct"))
    add(cluster.get("share_of_revenue_pct"))
    for _, v in cluster["metrics"].items():
        add(v.get("mean"))
        add(v.get("vs_population_index"))
    for key in cluster:
        if key.startswith("top_") and isinstance(cluster[key], dict):
            for _, v in cluster[key].items():
                add(v)
    return nums


def check_grounding(persona: GeneratedPersona, cluster: dict) -> dict:
    allowed = _allowed_numbers(cluster)
    checked = matched = 0
    unmatched = []
    for line in persona.evidence:
        for tok in re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", "")):
            f = float(tok)
            checked += 1
            if any(abs(f - a) < 0.51 for a in allowed):
                matched += 1
            else:
                unmatched.append({"value": f, "in": line[:90]})
    return {"checked": checked, "matched": matched, "unmatched": unmatched,
            "rate": round(matched / checked, 3) if checked else None}


# =============================================================================
# PUBLIC
# =============================================================================

def generate_personas(meta: dict, model: str = DEFAULT_MODEL,
                      progress: Optional[Callable] = None) -> dict:
    """
    One persona per cluster, up to 3 attempts each.

    The retry has to wrap parsing and validation, not just the API call. A
    truncated response raises JSONDecodeError long after the request itself
    succeeded, and the earlier version treated that as permanent — which is how
    a k=4 run quietly produced 2 personas.
    """
    clusters = meta["clusters"]
    personas, reports, failures = [], [], []

    for i, cluster in enumerate(clusters):
        cid = cluster["cluster_id"]
        if progress:
            progress(f"Writing persona {i+1} of {len(clusters)}",
                     int(100 * i / max(len(clusters), 1)))

        prompt = build_prompt(cluster, meta, clusters)
        persona, last_err = None, None

        for attempt in range(3):
            raw = ""
            try:
                raw = raw = call_llm(SYSTEM_INSTRUCTION, prompt, model,label=f"Cluster {cid}")
                data = extract_json(raw)
                data["cluster_id"] = cid          # never trust the model on this
                persona = GeneratedPersona(**data)
                break
            except Exception as e:
                last_err = e
                print(f"[personas] cluster {cid} attempt {attempt + 1}/3 failed: "
                      f"{type(e).__name__}: {e}")
                if raw:
                    print(f"[personas]   response was {len(raw)} chars, "
                          f"ends: ...{raw[-120:]!r}")
                time.sleep(1.5 * (attempt + 1))

        if persona is None:
            failures.append({"cluster_id": cid,
                             "error": f"{type(last_err).__name__}: {last_err}"})
            continue

        personas.append(persona.model_dump())
        reports.append({"cluster_id": cid, **check_grounding(persona, cluster)})

    if progress:
        progress("Personas ready", 100)

    if failures:
        print(f"[personas] {len(failures)} of {len(clusters)} clusters produced "
              f"no persona: {[f['cluster_id'] for f in failures]}")

    rates = [r["rate"] for r in reports if r["rate"] is not None]
    return {
        "personas": personas,
        "grounding_reports": reports,
        "mean_grounding_rate": round(sum(rates) / len(rates), 3) if rates else None,
        "failures": failures,
        "requested_k": meta.get("k"),
        "personas_generated": len(personas),
        "model": model,
        "k": meta["k"],
        "silhouette": meta["silhouette"],
    }

SCALAR_FIELDS = ["persona_name", "one_line_summary", "price_sensitivity_score",
                 "brand_loyalty_score", "deliberation_score",
                 "segment_share_percentage", "revenue_share_percentage",
                 "shopping_style", "channel_preference",
                 "predicted_price_increase_reaction", "merchant_recommendation"]
LIST_FIELDS = ["core_buying_triggers", "primary_hesitations", "evidence"]


def assign_to_users(users: pd.DataFrame, personas: List[dict]) -> pd.DataFrame:
    """
    Attach each persona to every customer in its cluster.

    system_prompt is deliberately excluded - it is a paragraph, and repeating it
    across tens of thousands of rows bloats the file for no benefit. It lives in
    the lookup instead.
    """
    rows = []
    for p in personas:
        row = {"cluster_id": p["cluster_id"]}
        for f in SCALAR_FIELDS:
            if f in p:
                row[f] = p[f]
        for f in LIST_FIELDS:
            v = p.get(f)
            row[f] = " | ".join(v) if isinstance(v, list) else v
        rows.append(row)

    pmap = pd.DataFrame(rows)
    before = len(users)
    out = users.merge(pmap, on="cluster_id", how="left")
    if len(out) != before:
        raise ValueError(
            f"Row count changed on merge ({before:,} -> {len(out):,}). "
            "personas probably contain a duplicate cluster_id.")
    return out


def build_lookup(personas: List[dict]) -> dict:
    """Full persona objects keyed by cluster id, including system_prompt."""
    return {str(p["cluster_id"]): {k: v for k, v in p.items() if k != "cluster_id"}
            for p in personas}
