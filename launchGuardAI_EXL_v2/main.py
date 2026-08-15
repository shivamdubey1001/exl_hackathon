"""
=============================================================================
LAUNCHGUARD AI v2 — BACKEND
=============================================================================

  LaunchGuardAI/
    main.py
    pipeline/
      mapping.py            column detection
      segment.py            clustering
      personas.py           persona synthesis + assignment
      analyst_engine.py     copied from v1
      persona_agent.py      copied from v1
      synthetic_ab_test.py  copied from v1
    datasets/               created at runtime, one folder per upload
    samples/                pre-loaded CSVs for the demo
    static/index.html
    .env

  python -m uvicorn main:app --reload --port 8000

WHAT CHANGED FROM v1
--------------------
v1 loaded ONE dataset at startup. v2 is multi-dataset: everything is scoped by
dataset_id, including the simulation cache. Clustering and persona generation
run as background jobs with progress, because they take 30-90 seconds and a
blocking request would time out.

DEMO SAFETY
-----------
Live upload in front of judges is a risk: one malformed CSV and you are
debugging on stage. Drop 2-3 known-good files into samples/ and demo from
those. Keep upload available for anyone who asks to try their own.
=============================================================================
"""

import glob
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field



from dotenv import load_dotenv
load_dotenv()

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "pipeline"))

import mapping as MAP          # noqa: E402
import segment as SEG          # noqa: E402
import personas as PER         # noqa: E402
import usage as USAGE          # noqa: E402
import llm as LLM              # noqa: E402


DATASETS = os.path.join(BASE, "datasets")
SAMPLES = os.path.join(BASE, "samples")
os.makedirs(DATASETS, exist_ok=True)
os.makedirs(SAMPLES, exist_ok=True)
USAGE.configure(os.path.join(BASE, "usage_log.json"))

MAX_ROWS = 200_000
MAX_K = 6
MIN_K = 2

app = FastAPI(title="LaunchGuard AI v2", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# =============================================================================
# DATASET STORAGE
# =============================================================================

def json_safe(df: pd.DataFrame) -> list:
    """Rows as plain dicts with NaN/inf replaced by None.

    JSON has no NaN literal, so pandas nulls must become None before they
    reach the response serialiser.
    """
    import numpy as np
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.astype(object).where(pd.notna(clean), None).to_dict(orient="records")


def ds_dir(did: str) -> str:
    if not did or "/" in did or "\\" in did or ".." in did:
        raise HTTPException(400, "bad dataset id")
    return os.path.join(DATASETS, did)


def ds_path(did: str, name: str) -> str:
    return os.path.join(ds_dir(did), name)


def read_meta(did: str) -> dict:
    p = ds_path(did, "meta.json")
    if not os.path.exists(p):
        raise HTTPException(404, f"dataset '{did}' not found")
    return json.load(open(p, encoding="utf-8"))


def write_meta(did: str, meta: dict):
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(ds_path(did, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def patch_meta(did: str, **kw) -> dict:
    m = read_meta(did)
    m.update(kw)
    write_meta(did, m)
    return m


# =============================================================================
# JOBS
# =============================================================================

JOBS: Dict[str, dict] = {}
JOB_LOCK = threading.Lock()


def new_job(dataset_id: str, kind: str) -> str:
    jid = uuid.uuid4().hex[:12]
    with JOB_LOCK:
        JOBS[jid] = {"job_id": jid, "dataset_id": dataset_id, "kind": kind,
                     "status": "queued", "stage": "waiting", "percent": 0,
                     "error": None, "started_at": time.time()}
    return jid


def job_progress(jid: str, stage: str, percent: int):
    with JOB_LOCK:
        if jid in JOBS:
            JOBS[jid].update({"status": "running", "stage": stage,
                              "percent": max(0, min(100, int(percent)))})


def job_done(jid: str, **extra):
    with JOB_LOCK:
        if jid in JOBS:
            JOBS[jid].update({"status": "done", "percent": 100,
                              "stage": "complete",
                              "elapsed": round(time.time() - JOBS[jid]["started_at"], 1),
                              **extra})


def job_failed(jid: str, err: str):
    with JOB_LOCK:
        if jid in JOBS:
            JOBS[jid].update({"status": "failed", "error": err, "stage": "failed"})


# =============================================================================
# MODELS
# =============================================================================

class MappingBody(BaseModel):
    mapping: Dict[str, Optional[str]]


class ClusterBody(BaseModel):
    k: int = Field(ge=MIN_K, le=MAX_K, description=f"number of segments, {MIN_K}-{MAX_K}")
    include_synthetic: bool = False
    features: Optional[List[str]] = Field(
        default=None,
        description="columns to cluster on; omit to use the automatic choice")
    generate_personas: bool = True


class InterventionBody(BaseModel):
    kind: str
    magnitude: float
    label: Optional[str] = None
    department: Optional[str] = None
    targeted_only: bool = False
    force_live: bool = False
    area: Optional[str] = None


class ABBody(BaseModel):
    variant_a: str
    variant_b: str
    sample_size: Optional[int] = Field(
        default=None,
        description="total shoppers to simulate, split evenly across personas")
    aa_test: bool = False
    in_character: bool = True
    seed: int = 42
    area: Optional[str] = None
    force_live: bool = False


# =============================================================================
# UPLOAD AND MAPPING
# =============================================================================

@app.get("/api/health")
def health():
    prov = LLM.status()
    return {"status": "ok",
            "datasets": len(os.listdir(DATASETS)) if os.path.isdir(DATASETS) else 0,
            "samples": len(glob.glob(os.path.join(SAMPLES, "*.csv"))),
            "api_key_present": prov["has_key"],
            "provider": prov["provider"],
            "model": prov["model"],
            "max_k": MAX_K}


@app.get("/api/fields")
def fields():
    """Field catalogue for the mapping dropdowns."""
    return {"fields": MAP.field_catalogue(),
            "cluster_fields": MAP.CLUSTER_FIELDS,
            "min_cluster_fields": MAP.MIN_CLUSTER_FIELDS}


@app.get("/api/samples")
def samples():
    out = []
    for p in sorted(glob.glob(os.path.join(SAMPLES, "*.csv"))):
        try:
            n = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
        except Exception:
            n = None
        out.append({"filename": os.path.basename(p),
                    "name": os.path.basename(p).replace(".csv", "").replace("_", " ").title(),
                    "rows": n})
    return {"samples": out}


def _ingest(df: pd.DataFrame, name: str) -> dict:
    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"{len(df):,} rows exceeds the {MAX_ROWS:,} limit")
    if df.empty:
        raise HTTPException(400, "That file has no rows in it")

    did = uuid.uuid4().hex[:12]
    os.makedirs(ds_dir(did), exist_ok=True)
    os.makedirs(ds_path(did, "cache"), exist_ok=True)
    df.to_csv(ds_path(did, "raw.csv"), index=False)

    # Fuzzy first, AI only on what it could not match. On a well-named file
    # the AI pass is skipped entirely and costs nothing.
    detected = MAP.detect_mapping_smart(df, use_ai=True)
    meta = {
        "dataset_id": did, "name": name, "rows": int(len(df)),
        "columns": list(df.columns),
        "status": "awaiting_mapping",
        "mapping": detected["mapping"], "confidence": detected["confidence"],
        "sources": detected.get("sources", {}),
        "ai_used": detected.get("ai_used", False),
        "ai_notes": detected.get("ai_notes", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_meta(did, meta)

    preview = json_safe(df.head(5))
    return {**meta, "preview": preview,
            "validation": MAP.validate_mapping(df, detected["mapping"])}


@app.post("/api/datasets/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".tsv")):
        raise HTTPException(400, "Upload a .csv or .tsv file")
    try:
        sep = "\t" if file.filename.lower().endswith(".tsv") else ","
        df = pd.read_csv(file.file, sep=sep)
    except Exception as e:
        raise HTTPException(400, f"Could not read that file: {e}")
    return _ingest(df, file.filename)


@app.post("/api/datasets/from-sample/{filename}")
def from_sample(filename: str):
    p = os.path.join(SAMPLES, os.path.basename(filename))
    if not os.path.exists(p):
        raise HTTPException(404, "sample not found")
    return _ingest(pd.read_csv(p), os.path.basename(filename))


@app.get("/api/datasets")
def list_datasets():
    out = []
    for did in os.listdir(DATASETS):
        try:
            m = read_meta(did)
            out.append({k: m.get(k) for k in
                        ["dataset_id", "name", "rows", "status", "k",
                         "silhouette", "created_at", "n_personas"]})
        except Exception:
            continue
    return {"datasets": sorted(out, key=lambda d: d.get("created_at") or "", reverse=True)}


@app.get("/api/datasets/{did}")
def get_dataset(did: str):
    m = read_meta(did)
    raw = pd.read_csv(ds_path(did, "raw.csv"), nrows=5)
    m["preview"] = json_safe(raw)
    return m


@app.delete("/api/datasets/{did}")
def delete_dataset(did: str):
    shutil.rmtree(ds_dir(did), ignore_errors=True)
    return {"deleted": did}


@app.post("/api/datasets/{did}/mapping")
def set_mapping(did: str, body: MappingBody):
    """Confirm or override the detected mapping. Validation runs before saving."""
    read_meta(did)
    df = pd.read_csv(ds_path(did, "raw.csv"))
    v = MAP.validate_mapping(df, body.mapping)
    if v["ok"]:
        patch_meta(did, mapping=body.mapping, status="ready_to_cluster")
    return {"validation": v, "mapping": body.mapping}

@app.post("/api/datasets/{did}/redetect")
def redetect(did: str, use_ai: bool = True):
    """Re-run column detection. Useful when the first pass had no API key."""
    m = read_meta(did)
    df = pd.read_csv(ds_path(did, "raw.csv"))
    d = MAP.detect_mapping_smart(df, use_ai=use_ai)
    m.update({"mapping": d["mapping"], "confidence": d["confidence"],
              "sources": d.get("sources", {}), "ai_used": d.get("ai_used", False),
              "ai_notes": d.get("ai_notes", [])})
    write_meta(did, m)
    return {"mapping": d["mapping"], "confidence": d["confidence"],
            "sources": d.get("sources", {}), "ai_used": d.get("ai_used", False),
            "ai_notes": d.get("ai_notes", []),
            "validation": MAP.validate_mapping(df, d["mapping"])}

@app.get("/api/datasets/{did}/suggest-k")
def suggest_k(did: str, features: Optional[str] = None):
    """
    Silhouette per k, plus the columns available to cluster on.

    `features` is a comma-separated override. The UI re-calls this whenever the
    user changes the selection, so the cost of adding a column is visible
    rather than theoretical.
    """
    m = read_meta(did)
    if not m.get("mapping"):
        raise HTTPException(400, "confirm the column mapping first")
    df = MAP.apply_mapping(pd.read_csv(ds_path(did, "raw.csv")), m["mapping"])

    chosen = [c.strip() for c in features.split(",") if c.strip()] if features else None

    try:
        feats, used = SEG.prepare_features(df, features=chosen)
    except ValueError as e:
        raise HTTPException(400, str(e))

    scores = SEG.evaluate_k(feats, range(MIN_K, MAX_K + 1))
    best = max(scores, key=lambda s: s["silhouette"]) if scores else None

    return {
        "scores": scores,
        "features_used": used,
        "available_features": SEG.available_features(df),
        "recommended_k": best["k"] if best else 3,
        "random_baseline": 0.17,
        "note": "A silhouette near 0.17 is what random data scores. "
                "Higher means the segments are real.",
    }


# =============================================================================
# THE PIPELINE JOB
# =============================================================================

def _run_pipeline(did: str, jid: str, k: int, include_synthetic: bool,
                  do_personas: bool, features: Optional[List[str]] = None):
    try:
        meta = read_meta(did)
        job_progress(jid, "Reading data", 3)
        raw = pd.read_csv(ds_path(did, "raw.csv"))
        df = MAP.apply_mapping(raw, meta["mapping"])

        # ---- clustering ----
        def cl_prog(stage, pct):
            job_progress(jid, stage, 5 + int(pct * 0.35))       # 5-40%

        res = SEG.run_clustering(df, k=k, include_synthetic=include_synthetic,
                                 features=features, progress=cl_prog)
        users = res["users"]
        cmeta = res["meta"]

        users.to_csv(ds_path(did, "users_clustered.csv"), index=False)
        json.dump(cmeta, open(ds_path(did, "cluster_profiles.json"), "w",
                              encoding="utf-8"), indent=2, default=str)
        json.dump(res["projection"], open(ds_path(did, "projection.json"), "w",
                                          encoding="utf-8"), default=str)
        patch_meta(did, status="clustered", k=int(k),
                   silhouette=cmeta["silhouette"],
                   clustered_on=cmeta["clustered_on"])

        if not do_personas:
            job_done(jid, k=k, silhouette=cmeta["silhouette"])
            return

        # ---- personas ----
        def p_prog(stage, pct):
            job_progress(jid, stage, 42 + int(pct * 0.5))       # 42-92%

        marker = USAGE.checkpoint()
        out = PER.generate_personas(cmeta, progress=p_prog)
        if not out["personas"]:
            raise RuntimeError(
                "No personas were generated. " +
                (out["failures"][0]["error"] if out["failures"] else "unknown error"))

        json.dump(out, open(ds_path(did, "personas.json"), "w", encoding="utf-8"),
                  indent=2, default=str)
        json.dump(PER.build_lookup(out["personas"]),
                  open(ds_path(did, "persona_lookup.json"), "w", encoding="utf-8"),
                  indent=2, default=str)

        job_progress(jid, "Attaching personas to customers", 95)
        merged = PER.assign_to_users(users, out["personas"])
        merged.to_csv(ds_path(did, "users_with_personas.csv"), index=False)

        patch_meta(did, status="ready", n_personas=len(out["personas"]),
                   mean_grounding_rate=out["mean_grounding_rate"],
                   persona_cost=USAGE.cost_since(marker)["cost_usd"])
        job_done(jid, k=k, silhouette=cmeta["silhouette"],
                 n_personas=len(out["personas"]),
                 grounding=out["mean_grounding_rate"])

    except Exception as e:
        traceback.print_exc()
        job_failed(jid, f"{type(e).__name__}: {e}")
        try:
            patch_meta(did, status="failed", last_error=str(e))
        except Exception:
            pass


@app.post("/api/datasets/{did}/run")
def run_pipeline(did: str, body: ClusterBody, background: BackgroundTasks):
    """Cluster, then write personas. Returns a job id to poll."""
    m = read_meta(did)
    if not m.get("mapping"):
        raise HTTPException(400, "confirm the column mapping first")
    if body.generate_personas and not LLM.has_key():
        raise HTTPException(400, "No API key found. Add ANTHROPIC_API_KEY or "
                                 "OPENROUTER_API_KEY to .env, or run with "
                                 "generate_personas = false.")

    jid = new_job(did, "pipeline")
    patch_meta(did, status="processing")
    background.add_task(_run_pipeline, did, jid, body.k,
                        body.include_synthetic, body.generate_personas,
                        body.features)
    return {"job_id": jid, "dataset_id": did}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    with JOB_LOCK:
        j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


# =============================================================================
# RESULTS
# =============================================================================

@app.get("/api/datasets/{did}/personas")
def dataset_personas(did: str):
    m = read_meta(did)
    p = ds_path(did, "personas.json")
    if not os.path.exists(p):
        raise HTTPException(409, f"dataset is '{m.get('status')}' — no personas yet")

    payload = json.load(open(p, encoding="utf-8"))
    users_p = ds_path(did, "users_with_personas.csv")
    cards = []

    if os.path.exists(users_p):
        df = pd.read_csv(users_p)
        total = len(df)
        has_money = "monetary_net" in df.columns
        total_rev = float(df["monetary_net"].sum()) if has_money else 0.0
        for persona in payload["personas"]:
            grp = df[df["cluster_id"] == persona["cluster_id"]]
            card = {k: v for k, v in persona.items() if k != "system_prompt"}
            card["n_customers"] = int(len(grp))
            card["pct_of_customers"] = round(100 * len(grp) / total, 1) if total else 0
            if has_money and total_rev:
                card["revenue"] = round(float(grp["monetary_net"].sum()), 2)
                card["pct_of_revenue"] = round(
                    100 * float(grp["monetary_net"].sum()) / total_rev, 1)
            cards.append(card)
    else:
        cards = [{k: v for k, v in p_.items() if k != "system_prompt"}
                 for p_ in payload["personas"]]

    cards.sort(key=lambda c: -c.get("pct_of_revenue", c.get("pct_of_customers", 0)))
    return {"personas": cards,
            "meta": {"k": payload.get("k"), "silhouette": payload.get("silhouette"),
                     "total_users": m.get("rows"),
                     "clustered_on": m.get("clustered_on", []),
                     "mean_grounding_rate": payload.get("mean_grounding_rate"),
                     "random_baseline": 0.17},
            "grounding_reports": payload.get("grounding_reports", [])}


@app.get("/api/datasets/{did}/projection")
def projection(did: str):
    p = ds_path(did, "projection.json")
    if not os.path.exists(p):
        raise HTTPException(404, "not clustered yet")
    return json.load(open(p, encoding="utf-8"))


@app.get("/api/datasets/{did}/download")
def download_info(did: str):
    """Which artefacts exist, so the UI can offer them."""
    files = {}
    for name in ["users_with_personas.csv", "users_clustered.csv",
                 "personas.json", "cluster_profiles.json", "persona_lookup.json"]:
        p = ds_path(did, name)
        if os.path.exists(p):
            files[name] = {"bytes": os.path.getsize(p),
                           "url": f"/api/datasets/{did}/file/{name}"}
    return {"files": files}


@app.get("/api/datasets/{did}/file/{name}")
def get_file(did: str, name: str):
    from fastapi.responses import FileResponse
    safe = os.path.basename(name)
    p = ds_path(did, safe)
    if not os.path.exists(p):
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=f"{did}_{safe}")


# =============================================================================
# SIMULATION — scoped per dataset
# =============================================================================

def _require_ready(did: str) -> dict:
    m = read_meta(did)
    if not os.path.exists(ds_path(did, "users_with_personas.csv")):
        raise HTTPException(409, f"dataset is '{m.get('status')}' — "
                                 f"run segmentation and personas first")
    return m


def _cache_key(did: str, kind: str, payload: dict) -> str:
    import hashlib
    blob = json.dumps({"d": did, "k": kind, **payload}, sort_keys=True, default=str)
    return f"{kind}_{hashlib.sha1(blob.encode()).hexdigest()[:16]}"


def _cache_get(did: str, key: str):
    p = ds_path(did, os.path.join("cache", key + ".json"))
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_put(did: str, key: str, val: dict):
    os.makedirs(ds_path(did, "cache"), exist_ok=True)
    with open(ds_path(did, os.path.join("cache", key + ".json")),
              "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, default=str)

def _record_run(did: str, key: str, kind: str, area: str, label: str, headline: str):
    """Append to a per-dataset run index so the UI can list past results."""
    p = ds_path(did, "runs.json")
    runs = []
    if os.path.exists(p):
        try:
            runs = json.load(open(p, encoding="utf-8"))
        except Exception:
            runs = []
    runs = [r for r in runs if r.get("key") != key]      # dedupe reruns
    runs.insert(0, {"key": key, "kind": kind, "area": area, "label": label,
                    "headline": headline,
                    "at": datetime.now(timezone.utc).isoformat()})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(runs[:60], f, indent=2)                 # keep the last 60

@app.get("/api/datasets/{did}/runs")
def list_runs(did: str, area: Optional[str] = None):
    p = ds_path(did, "runs.json")
    if not os.path.exists(p):
        return {"runs": []}
    runs = json.load(open(p, encoding="utf-8"))
    if area:
        # entries written before areas existed have no area field; show them
        # rather than silently hiding history the user knows they created
        runs = [r for r in runs if r.get("area") in (area, None)]
    return {"runs": runs}


@app.get("/api/datasets/{did}/runs/{key}")
def get_run(did: str, key: str):
    hit = _cache_get(did, os.path.basename(key))
    if not hit:
        raise HTTPException(404, "that run is no longer cached")
    hit["_cached"] = True
    return hit

@app.post("/api/datasets/{did}/simulate/intervention")
def simulate_intervention(did: str, body: InterventionBody):
    _require_ready(did)
    key = _cache_key(did, "intervention", body.model_dump(exclude={"force_live"}))
    if not body.force_live:
        hit = _cache_get(did, key)
        if hit:
            hit["_cached"] = True
            hit["cache_key"] = key 
            return hit
    
    marker = USAGE.checkpoint() 
    from analyst_engine import AnalystEngine, Intervention

    users_csv = ds_path(did, "users_with_personas.csv")
    df = pd.read_csv(users_csv, nrows=1)
    if "price_elasticity" not in df.columns:
        raise HTTPException(
            400, "This dataset has no price_elasticity column, so pricing cannot "
                 "be costed. Use the A/B test instead, or map an elasticity column.")

    interv = Intervention(kind=body.kind,
                          label=body.label or f"{body.kind} {body.magnitude}",
                          magnitude=body.magnitude, department=body.department)
    analyst = AnalystEngine(users_csv).compare_blanket_vs_targeted(interv)
    scenario = "targeted" if body.targeted_only else "blanket"
    run = analyst[scenario]

    reactions, ranking = [], {"checked": False}
    try:
        import persona_agent as pa
        lookup = json.load(open(ds_path(did, "persona_lookup.json"), encoding="utf-8"))
        segs = {int(s["cluster_id"]): s for s in run["segments"]}
        text = pa.describe_intervention(run["intervention"])
        schema = json.dumps({f: (i.description or "")
                             for f, i in pa.PersonaReaction.model_fields.items()}, indent=2)
        for cid_str, persona in lookup.items():
            seg = segs.get(int(cid_str))
            if not seg or not seg.get("affected", True):
                continue
            raw = pa.call_anthropic(persona["system_prompt"] + "\n\n" + pa.REACTION_INSTRUCTION,
                                    pa.build_reaction_prompt(text, schema),
                                    os.getenv("LAUNCHGUARD_MODEL", "claude-sonnet-5"))
            r = pa.PersonaReaction(**pa.extract_json(raw))
            reactions.append({"cluster_id": int(cid_str),
                              "persona_name": persona.get("persona_name"),
                              "segment": seg, "reaction": r.model_dump(),
                              "reconciliation": pa.reconcile_one(r, seg)})
        ranking = pa.reconcile_ranking(reactions)
    except Exception as e:
        ranking = {"checked": False, "reason": f"persona agent unavailable: {e}"}

    result = {"intervention": run["intervention"], "scenario": scenario,
              "totals": run["totals"], "segments": run["segments"],
              "comparison": {"recommended_clusters": analyst["recommended_clusters"],
                             "savings": analyst["savings"],
                             "blanket_totals": analyst["blanket"]["totals"],
                             "targeted_totals": analyst["targeted"]["totals"]},
              "reactions": reactions, "ranking_check": ranking, "_cached": False}
    result["usage"] = USAGE.cost_since(marker)
    result["cache_key"] = key
    _cache_put(did, key, result)
    _cache_put(did, key, result)
    _record_run(did, key, "intervention", body.area or "pricing",
                run["intervention"]["label"],
                f"net {round(run['totals']['net_profit_impact']):,}")
    return result


@app.post("/api/datasets/{did}/simulate/abtest")
def simulate_abtest(did: str, body: ABBody):
    _require_ready(did)
    key = _cache_key(did, "abtest", body.model_dump(exclude={"force_live"}))
    if not body.force_live:
        hit = _cache_get(did, key)
        if hit:
            hit["_cached"] = True
            hit["cache_key"] = key 
            return hit
    marker = USAGE.checkpoint()
    from synthetic_ab_test import run_ab_test

    nf_existing = _cache_get(did, "noise_floor")
    nf_value = (nf_existing or {}).get("rpu_relative_lift_pct") if nf_existing else None

    result = run_ab_test(ds_path(did, "users_with_personas.csv"),
                         body.variant_a, body.variant_b,
                         sample_size=body.sample_size,
                         lookup_path=ds_path(did, "persona_lookup.json"),
                         model=os.getenv("LAUNCHGUARD_MODEL", "claude-sonnet-5"),
                         seed=body.seed, aa_test=body.aa_test,
                         in_character=body.in_character,
                         noise_floor=None if body.aa_test else nf_value)

    nf = _cache_get(did, "noise_floor")
    if body.aa_test:
        _cache_put(did, "noise_floor", result.get("overall_summary", {}))
    elif nf_existing:
        result["noise_floor"] = nf_existing
        result["above_noise_floor"] = (
            abs(result.get("overall_summary", {}).get("rpu_relative_lift_pct", 0))
            > abs(nf_existing.get("rpu_relative_lift_pct", 0)))

    result["_cached"] = False
    result["usage"] = USAGE.cost_since(marker)
    result["cache_key"] = key
    _cache_put(did, key, result)
    lift = result.get("overall_summary", {}).get("rpu_relative_lift_pct", 0)
    _record_run(did, key, "aatest" if body.aa_test else "abtest",
                body.area or "pdp",
                "A/A noise check" if body.aa_test else body.variant_b[:60],
                f"{lift:+.1f}% RPU")
    return result

@app.get("/api/datasets/{did}/ab-estimate")
def ab_estimate(did: str, sample_size: int = 120):
    """Cost and the actual per-persona allocation, before committing."""
    _require_ready(did)
    from synthetic_ab_test import estimate_run, allocate_sample

    df = pd.read_csv(ds_path(did, "users_with_personas.csv"),
                     usecols=["persona_name"])
    sizes = df["persona_name"].value_counts().to_dict()
    total = len(df)

    # the real proportional allocation, not an even-split approximation
    alloc = allocate_sample(sample_size, sizes, dict(sizes))
    actual = sum(alloc.values())

    est = estimate_run(len(sizes), actual, total,
                       os.getenv("LAUNCHGUARD_MODEL", "claude-sonnet-5"))
    est.update({
        "total_calls": actual,
        "sample_size": actual,
        "personas": len(sizes),
        "total_customers": total,
        "allocation": [
            {"persona": p, "n": n,
             "segment_share_pct": round(100 * sizes[p] / total, 1),
             "sample_share_pct": round(100 * n / actual, 1)}
            for p, n in sorted(alloc.items(), key=lambda x: -x[1])
        ],
    })
    return est

@app.get("/api/datasets/{did}/noise-floor")
def get_noise_floor(did: str):
    nf = _cache_get(did, "noise_floor")
    if not nf:
        raise HTTPException(404, "no A/A test has been run for this dataset")
    return nf


@app.delete("/api/datasets/{did}/cache")
def clear_cache(did: str):
    d = ds_path(did, "cache")
    n = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json"):
                os.remove(os.path.join(d, f))
                n += 1
    return {"cleared": n}


@app.get("/api/usage")
def get_usage(limit: int = 60):
    """Rolling window of API calls, plus totals by step."""
    return {"summary": USAGE.summary(), "calls": USAGE.recent(limit)}


@app.delete("/api/usage")
def clear_usage():
    USAGE.reset()
    return {"cleared": True}


@app.get("/api/areas")
def areas():
    return {"areas": [
        {"id": "pricing", "name": "Pricing", "engine": "intervention",
         "blurb": "Test a price rise or cut before committing margin.",
         "presets": [
             {"label": "Raise prices 5%", "kind": "price_change", "magnitude": 0.05},
             {"label": "Raise prices 10%", "kind": "price_change", "magnitude": 0.10}]},
        {"id": "promotions", "name": "Promotions", "engine": "intervention",
         "blurb": "See who needs the discount and who would buy anyway.",
         "presets": [
             {"label": "10% off site-wide", "kind": "coupon", "magnitude": 0.10},
             {"label": "15% off site-wide", "kind": "coupon", "magnitude": 0.15}]},
        {"id": "shipping", "name": "Shipping", "engine": "intervention",
         "blurb": "Model adding or removing a delivery charge.",
         "presets": [
             {"label": "Free shipping", "kind": "shipping_change", "magnitude": -5.0},
             {"label": "Add $3 shipping", "kind": "shipping_change", "magnitude": 3.0}]},
        {"id": "pdp", "name": "Product Page", "engine": "abtest",
         "blurb": "Test banners, copy and layout as an A/B test.",
         "presets": [{"label": "Discount banner",
                      "variant_a": "Standard product detail page, no promotional messaging.",
                      "variant_b": "Product page with a banner reading 'Get 10% off orders over $100'."}]},
        {"id": "crm", "name": "Email & CRM", "engine": "abtest",
         "blurb": "Compare two campaign messages across your segments.",
         "presets": [{"label": "Urgency vs value",
                      "variant_a": "Email: 'New arrivals now in stock.'",
                      "variant_b": "Email: 'Last chance — 24 hours left on your favourites.'"}]},
    ]}

@app.get("/api/datasets/{did}/runs/{key}/raw.csv")
def run_raw_csv(did: str, key: str):
    """
    Per-shopper detail from a cached run, as CSV.

    The inline table caps at a few hundred rows for browser performance;
    anyone who wants all of it probably wants it in a spreadsheet anyway.
    """
    from fastapi.responses import StreamingResponse
    import io

    hit = _cache_get(did, os.path.basename(key))
    if not hit or not hit.get("raw_results"):
        raise HTTPException(404, "no per-shopper detail for that run")

    df = pd.DataFrame(hit["raw_results"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="shoppers_{key}.csv"'})


STATIC = os.path.join(BASE, "static")
if os.path.isdir(STATIC):
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
