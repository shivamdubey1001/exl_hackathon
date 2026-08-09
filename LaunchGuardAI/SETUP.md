# LaunchGuard AI v2 — setup

## Folder layout

```
LaunchGuardAI/
  main.py
  requirements.txt
  .env                       ANTHROPIC_API_KEY=sk-ant-...
  pipeline/
    mapping.py               (new)
    segment.py               (new)
    personas.py              (new)
    analyst_engine.py        copy from v1
    persona_agent.py         copy from v1
    synthetic_ab_test.py     copy from v1
  static/
    index.html               (frontend)
  samples/
    *.csv                    pre-loaded demo files
  datasets/                  created at runtime
```

## Install and run

```
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Check http://127.0.0.1:8000/api/health first.

## Seed the samples folder

Copy your v1 artefacts in so there is always something known-good to demo:

```
copy ..\LaunchGuardAI\artifacts\user_feature_table_enriched.csv samples\thelook_80k.csv
```

Also make a small one — a few thousand rows loads and clusters in seconds,
which matters when a judge is watching.

## The flow

1. `POST /api/datasets/upload` or `/from-sample/{filename}` → dataset_id +
   detected column mapping + 5-row preview
2. `POST /api/datasets/{id}/mapping` → confirm or override, returns validation
3. `GET  /api/datasets/{id}/suggest-k` → silhouette per k, so the user picks
   with evidence
4. `POST /api/datasets/{id}/run` with `{"k": 3}` → job_id
5. `GET  /api/jobs/{job_id}` → poll until status is done
6. `GET  /api/datasets/{id}/personas` → the review screen
7. `POST /api/datasets/{id}/simulate/...` → same as v1, scoped to the dataset

Step 4 also writes users_with_personas.csv, so the "apply personas to every
user" action happens automatically at the end of the job rather than needing a
separate button.

## Before demoing

- Run one dataset all the way through, then run an A/A test on it so a noise
  floor exists.
- Run each intervention preset once so results come from cache.
- Demo from `samples/`, not a live upload. Keep upload for anyone who asks.
