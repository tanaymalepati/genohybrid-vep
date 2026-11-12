# GenoHybrid-VEP

Uncertainty-aware variant effect prediction. Three-way classification
(Pathogenic / Benign / VUS) with calibrated confidence.

| I'm here to… | Read |
| --- | --- |
| Understand the whole project | **[PROJECT-CONTEXT.md](PROJECT-CONTEXT.md)** — start here, humans and AI alike |
| Present the demo | [DEMO.md](DEMO.md) |
| Build the backend | [backend/README.md](backend/README.md) |
| Build the ML pipeline | [ml/README.md](ml/README.md) |
| Work on the frontend | the rest of this file |

The repo holds all three parts: the React frontend (`src/`), the FastAPI
backend (`backend/`) and the trained model with its artifacts (`ml/`).

---

## Frontend

React + TypeScript (Vite) clinician-facing interface.

**Presenting this?** See [DEMO.md](DEMO.md) — open the folder in VS Code and the
server starts on its own.

## Running it

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production bundle
```

Runs against a mock inference engine by default, so it works with no backend
and no internet (Poppins is self-hosted in `public/fonts`). Point it at the real
service with a `.env.local`:

```
VITE_USE_MOCK=false
VITE_API_BASE=/api
```

`vite.config.ts` proxies `/api` → `http://localhost:8000`.

## Pages

| Route | What it does |
| --- | --- |
| `/` | Single-variant scoring: three-way class probabilities, calibrated confidence, MC-dropout σ, SHAP attributions |
| `/batch` | VCF/CSV upload, job queue with live progress |
| `/performance` | Measured metrics on the unseen-genes split, confusion matrix, reliability diagram |
| `/history` | Persisted prediction log with class filters; click a row for detail |

The three-way output and the uncertainty readout carry the project's
differentiator — VUS is a first-class class with a calibrated confidence, not a
discarded label — so they're the most prominent elements on the prediction
screen.

## The mock

`src/api/mock.ts` is a stand-in for the inference engine. It is deterministic:
the same variant always returns the same result.

Well-known variants have curated feature values, and SHAP contributions are
derived from those values rather than randomised, so the explanations are
internally consistent — a CADD of 32 and absence from gnomAD push toward
pathogenic; a popmax allele frequency of 0.24 pushes hard toward benign. The
CFTR and MSH2 entries are deliberately conflicting so they land in VUS.
Anything not in the curated set falls back to a hash of the coordinates.

**The numbers in the mock are illustrative, not measured results.** Replace them
with real evaluation output before any of this goes in a report.

## Backend contract

Wire types are in [`src/api/types.ts`](src/api/types.ts) — mirror them with
Pydantic models on the FastAPI side.

| Method | Path | Body → Response |
| --- | --- | --- |
| `POST` | `/api/predict` | `VariantInput` → `Prediction` |
| `GET` | `/api/predictions?limit=` | → `Prediction[]` |
| `GET` | `/api/metrics` | → `ModelMetrics[]` (one per split) |
| `GET` | `/api/batch` | → `BatchJob[]` |
| `POST` | `/api/batch` | multipart `file` → `BatchJob` |

Fields that carry modelling meaning:

- `probabilities` — post-temperature-scaling softmax over
  `Pathogenic | Benign | VUS`. Must sum to 1.
- `confidence` — the calibrated probability of `predicted_class`.
- `uncertainty` — predictive std-dev across the MC-dropout forward passes
  (`mc_passes`, 30–100 per the roadmap), *not* 1 − confidence.
- `shap[].value` — signed log-odds contribution toward `predicted_class`, so a
  positive value always means "supports the call". `shap[].raw` is the feature
  value as scored and is displayed next to the bar. Sort order doesn't matter;
  the UI sorts by magnitude and shows the top 10.

## Not built yet

- Auth. Every route is currently open.
- The random, rare/novel and held-out-VUS evaluation splits, and the baseline
  comparison table. See `PROJECT-CONTEXT.md` §7.

Charts are hand-rolled SVG (`ShapPlot`, `CalibrationCurve`) to keep the bundle
small — swap in Recharts or plotly if the visualization module grows.
