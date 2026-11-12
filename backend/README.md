# Backend

FastAPI + PostgreSQL in front of the trained tri-modal model.

## Run it

Open the folder in VS Code and run **Run GenoHybrid demo** (F5): it starts
PostgreSQL, the API on port 8000 and the frontend on 5173. By hand, from the repo root:

```bash
brew services start postgresql@16          # once per login
.venv/bin/python -m uvicorn backend.main:app --port 8000
npm run dev                                # second terminal
```

`.env.local` holds `VITE_USE_MOCK=false`, so the UI talks to this API. API docs:
<http://localhost:8000/docs>. Tests: `.venv/bin/python -m pytest tests -q`.

First-time setup on a new machine:

```bash
brew install postgresql@16 && brew services start postgresql@16
/opt/homebrew/opt/postgresql@16/bin/createdb genohybrid
python3 -m venv .venv.nosync && ln -sfn .venv.nosync .venv
.venv/bin/pip install -r backend/requirements.txt
```

Tables are created by Alembic on startup (`backend/migrations`). Set
`DATABASE_URL` to use another database.

## Endpoints

| Method | Path | Does |
| --- | --- | --- |
| `POST` | `/api/predict` | scores one variant, stores it (422 with a reason if it can't be scored) |
| `GET` | `/api/predictions?limit=` | prediction log, newest first |
| `GET` | `/api/predictions/{id}/report` | PDF report |
| `POST` | `/api/batch` | VCF / VCF.GZ / CSV / TSV upload (max 5,000 variants), scored in the background |
| `GET` | `/api/batch` | jobs with live progress; skipped variants are explained in `error` |
| `GET` | `/api/metrics` | measured metrics from `ml/artifacts/metrics.json` |
| `GET` | `/api/health` | database and model status |

## Where the numbers come from

- **Model:** `sequence_cnn_best.pt`, `biomarker_mlp_best.pt` and `gcn_v2_best.pt` from
  github.com/jayatigupta05/VEP (commit 38f68f5), fused 0.40 / 0.30 / 0.30 as in
  `predict.py`. `ml/genohybrid.py` matches that script's recorded predictions on 30
  demo variants (same class every time, probabilities within 0.015).
- **Inputs per variant:** 201 bp of GRCh38 sequence (UCSC API, Ensembl fallback),
  phyloP and phastCons (UCSC bigWig tracks read over HTTP), gene and consequence
  (from the VCF's CSQ/ANN field, otherwise Ensembl VEP). Lookups are cached in
  `ml/artifacts/annotation_cache.sqlite`; a new variant needs internet and can take
  up to ~40 s when Ensembl is slow, after which it is instant. ClinVar review,
  origin and somatic fields are unknown for a new variant and take the training
  medians, as in `predict.py`.
- **Confidence:** temperature-scaled (T = 0.286). The raw fused probabilities were
  badly under-confident (ECE 0.334); after scaling ECE is 0.016, measured on folds
  the temperature was not fitted to. `python -m ml.compute_metrics` refits it.
- **Uncertainty:** std-dev of the predicted class over 30 MC-Dropout passes
  (seeded, so a variant always reports the same value).
- **Attributions:** exact Shapley values over five inputs (phyloP, phastCons,
  consequence, sequence context, gene-graph links) on the log-odds of the
  predicted class. Absent means training median / unknown consequence / no
  flanking sequence / no graph edges.
- **Metrics:** out-of-fold predictions from the repo's 3-fold GroupKFold-by-gene
  validation (100,000 variants). The random, rare/novel and held-out-VUS splits in
  the original plan have not been evaluated, so the Performance page shows only the
  unseen-genes split.

Research prototype: not a clinical diagnostic.
