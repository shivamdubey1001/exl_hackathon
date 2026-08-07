import os
import json
import concurrent.futures
import numpy as np
import pandas as pd
import time

import anthropic

# ==========================================
# CONFIGURATION
# ==========================================
# Path to the CSV export (this replaces the old BigQuery SQL_QUERY pull —
# the CSV has the same columns the query produced: user_id, checkout_persona,
# persona_metadata_json).
CSV_FILE_PATH = r"c:\Users\Shiva\OneDrive\Desktop\hackathon\ABtestingcodechandan\user_personas.csv"

# ============================================
# CLAUDE (ANTHROPIC) API CONFIGURATION
# ============================================
# Paste your Anthropic API key below, OR set it as an environment variable
# named ANTHROPIC_API_KEY before running this script (recommended).
os.environ["ANTHROPIC_API_KEY"] = ""  # <-- INSERT YOUR CLAUDE API KEY HERE

CLAUDE_MODEL = "claude-opus-5"

# Initialize Claude client (reads ANTHROPIC_API_KEY from the environment automatically)
claude_client = anthropic.Anthropic()


# ==========================================
# 1. LOAD PERSONA SAMPLES FROM CSV
# ==========================================
def fetch_persona_samples(samples_per_persona=5):
    """Loads user persona samples from the local CSV export and draws a
    balanced random sample per persona (replaces the old BigQuery ROW_NUMBER
    sampling)."""
    print(f"Loading persona samples from {CSV_FILE_PATH} ...")
    df = pd.read_csv(CSV_FILE_PATH)

    sampled_df = (
        df.groupby("checkout_persona", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), samples_per_persona)))
    )

    samples = []
    for _, row in sampled_df.iterrows():
        samples.append({
            "user_id": row["user_id"],
            "persona": row["checkout_persona"],
            "metadata": json.loads(row["persona_metadata_json"])
        })
    print(f"Loaded {len(samples)} total sampled profiles across personas.\n")
    return samples


# ==========================================
# 2. CLAUDE EVALUATOR ENGINE
# ==========================================
SYSTEM_PROMPT = """You are an advanced behavioral decision simulator for e-commerce UX evaluation.
Analyze the user persona metadata against Variant A (Control) and Variant B (Treatment)."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "p_conv_variant_a": {"type": "number", "description": "Predicted conversion probability for Variant A, between 0.0 and 1.0"},
        "p_conv_variant_b": {"type": "number", "description": "Predicted conversion probability for Variant B, between 0.0 and 1.0"},
        "predicted_aov_variant_a": {"type": "number", "description": "Predicted average order value for Variant A"},
        "predicted_aov_variant_b": {"type": "number", "description": "Predicted average order value for Variant B"},
        "friction_driver": {"type": "string", "description": "1-sentence summary of decision logic"}
    },
    "required": ["p_conv_variant_a", "p_conv_variant_b", "predicted_aov_variant_a", "predicted_aov_variant_b", "friction_driver"],
    "additionalProperties": False
}

def evaluate_user(sample, variant_a_desc, variant_b_desc):
    """Evaluates micro-conversions for a single user using Claude."""
    payload = {
        "user_profile": sample["metadata"],
        "variant_a_control": variant_a_desc,
        "variant_b_treatment": variant_b_desc
    }
    time.sleep(1.5)

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}
            },
            messages=[{"role": "user", "content": json.dumps(payload)}]
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return {
            "user_id": sample["user_id"],
            "persona": sample["persona"],
            "cvr_a": float(data["p_conv_variant_a"]),
            "cvr_b": float(data["p_conv_variant_b"]),
            "aov_a": float(data["predicted_aov_variant_a"]),
            "aov_b": float(data["predicted_aov_variant_b"]),
            "reason": data["friction_driver"]
        }

    except Exception as e:
        print(f"⚠️ Claude API Error for user {sample['user_id']}: {e}")  # Print the real error!
        base_aov = sample["metadata"]["financial_baseline"]["average_order_value_usd"]
        base_cvr = sample["metadata"]["financial_baseline"]["baseline_conversion_rate_pct"] / 100.0
        return {
            "user_id": sample["user_id"],
            "persona": sample["persona"],
            "cvr_a": base_cvr,
            "cvr_b": base_cvr,
            "aov_a": base_aov,
            "aov_b": base_aov,
            "reason": f"Execution Error: {str(e)}"
        }

def run_synthetic_ab_test(samples, variant_a_desc, variant_b_desc, max_workers=1):
    """Runs parallel evaluations and calculates aggregate business impact."""
    print(f"Running Claude Monte Carlo simulation across N={len(samples)} profiles...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_user, sample, variant_a_desc, variant_b_desc)
            for sample in samples
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    df = pd.DataFrame(results)

    # Calculate Overall Aggregates
    cvr_a = df["cvr_a"].mean()
    cvr_b = df["cvr_b"].mean()
    cvr_lift = ((cvr_b - cvr_a) / cvr_a) * 100 if cvr_a > 0 else 0

    aov_a = df["aov_a"].mean()
    aov_b = df["aov_b"].mean()
    aov_lift = ((aov_b - aov_a) / aov_a) * 100 if aov_a > 0 else 0

    # Revenue Per User (RPU = CVR * AOV)
    rpu_a = cvr_a * aov_a
    rpu_b = cvr_b * aov_b
    rpu_lift = ((rpu_b - rpu_a) / rpu_a) * 100 if rpu_a > 0 else 0

    # Persona Summary
    persona_df = df.groupby("persona").agg(
        avg_cvr_a=("cvr_a", "mean"),
        avg_cvr_b=("cvr_b", "mean"),
        avg_aov_a=("aov_a", "mean"),
        avg_aov_b=("aov_b", "mean")
    ).reset_index()

    persona_df["cvr_lift_pct"] = ((persona_df["avg_cvr_b"] - persona_df["avg_cvr_a"]) / persona_df["avg_cvr_a"]) * 100
    persona_df["aov_lift_pct"] = ((persona_df["avg_aov_b"] - persona_df["avg_aov_a"]) / persona_df["avg_aov_a"]) * 100

    return {
        "overall_summary": {
            "total_simulated_users": len(df),
            "control_cvr": f"{cvr_a * 100:.2f}%",
            "treatment_cvr": f"{cvr_b * 100:.2f}%",
            "cvr_relative_lift": f"{cvr_lift:+.2f}%",
            "control_aov": f"${aov_a:.2f}",
            "treatment_aov": f"${aov_b:.2f}",
            "aov_relative_lift": f"{aov_lift:+.2f}%",
            "control_rpu": f"${rpu_a:.2f}",
            "treatment_rpu": f"${rpu_b:.2f}",
            "rpu_relative_lift": f"{rpu_lift:+.2f}%"
        },
        "persona_breakdown": persona_df,
        "raw_results": df
    }

# ==========================================
# 3. RUN SIMULATION
# ==========================================
if __name__ == "__main__":
    # Define Variant A and Variant B UX Changes
    variant_a = "Standard Product description page - BAU"
    variant_b = "We implemented 10% discount banner on the PDP which contains message 'Get 10% discount on purchase of more then 100 dollar'"

    # 1. Load 5 users per persona (50 total runs) from the CSV
    samples = fetch_persona_samples(samples_per_persona=1)

    # 2. Execute pipeline
    output = run_synthetic_ab_test(samples, variant_a, variant_b, max_workers=3)

    # 3. Display Results
    print("==========================================")
    print("      SYNTHETIC A/B TEST RESULTS          ")
    print("==========================================")
    print(json.dumps(output["overall_summary"], indent=2))
    print("\n--- PERSONA BREAKDOWN ---")
    print(output["persona_breakdown"].to_string(index=False))
