# GenoHybrid-VEP — project context

**Read this first.** It's the handoff document for anyone joining the project,
human or AI. It explains what the system is, what is already built, what is
still missing, and the contracts you must not break.

If you are an AI assistant: this file is your context. Read it fully before
changing anything, then read the file it points you at for your specific area.

---

## 0. If you just received this as a zip

Unzip it and open the folder in VS Code. The frontend installs its dependencies
and starts on its own (say yes to the trust and automatic-tasks prompts the
first time); it then runs at <http://localhost:5173> against a built-in mock, so
you can see the whole app working before writing a line of code. To run it
against the real model, follow `backend/README.md`.

`node_modules` and Python virtual environments are not in the zip. Nothing else
is missing.

Then go to §2 to find your part, and the README in your folder for the details.

---

## 1. What this project is

GenoHybrid-VEP predicts whether a human genetic variant is **Pathogenic**,
**Benign**, or a **Variant of Uncertain Significance (VUS)**, and reports a
calibrated confidence with every prediction.

It is a final-year B.E. project. It is not a clinical device.

### The one thing that makes it different

Every comparable tool in the literature either forces a binary
pathogenic/benign decision and treats VUS as noise to discard, or needs
compute (multi-billion-parameter language models, AlphaFold-scale
infrastructure) that a university lab does not have.

GenoHybrid-VEP is deliberately positioned as **three-way, uncertainty-aware,
and resource-efficient**:

- **VUS is a real output class**, not a discarded label.
- **Confidence is calibrated** (temperature scaling), so "90% confident" means
  right about 90% of the time — measured by Expected Calibration Error.
- **Uncertainty is separately quantified** via Monte Carlo Dropout variance and
  reported alongside every prediction, so a weakly-supported call is visibly
  weak rather than dressed up as a firm answer.
- **It runs on ordinary hardware**, because the protein language model is used
  frozen for embeddings rather than fine-tuned end to end.

Every design decision below serves one of those four points. If a change breaks
one of them, it is the wrong change no matter how much it improves accuracy.

---

## 2. Who is building what

| Area | Owner | Status |
| --- | --- | --- |
| Frontend (React + TypeScript) | Tanay | **Built.** Also runs standalone against a mock. |
| Backend (FastAPI + PostgreSQL) | Team | **Built.** All endpoints, batch jobs, PDF reports. |
| ML (features, models, calibration) | Jayati | **Built.** Trained tri-modal model wrapped in `ml/genohybrid.py`. |

The three parts are decoupled by two contracts. Keep them and the parts can be
built in any order, by anyone, without coordination:

- **Contract A — HTTP/JSON**, between frontend and backend.
  Defined in `src/api/types.ts`, mirrored in `backend/models.py`.
- **Contract B — Python**, between backend and ML.
  Defined in `ml/predictor.py` as the `Predictor` protocol.

**These two files are the load-bearing parts of the repo.** Change them only by
agreement, and change both sides together.

---

## 3. Repository layout

```
genohybrid-vep/
├── PROJECT-CONTEXT.md     ← you are here
├── DEMO.md                presentation script
├── README.md              frontend setup + API contract
│
├── src/                   FRONTEND (built, working)
│   ├── api/
│   │   ├── types.ts       ★ CONTRACT A — the wire format
│   │   ├── client.ts      fetch layer; mock/real switch
│   │   └── mock.ts        fake inference engine for running without the backend
│   ├── components/        PredictionCard, ShapPlot, CalibrationCurve, ConfusionMatrix
│   ├── pages/             PredictPage, BatchPage, PerformancePage, HistoryPage
│   ├── App.tsx            layout + nav
│   └── styles.css         all styling; light theme, Poppins, CSS variables
│
├── backend/               BACKEND (built)
│   ├── models.py          ★ Pydantic mirror of CONTRACT A
│   ├── main.py            FastAPI app and all endpoints
│   ├── db.py, migrations/ PostgreSQL tables (Alembic)
│   ├── parsing.py         VCF / CSV upload parsing
│   ├── report.py          PDF report
│   └── README.md          how to run it, where every number comes from
│
├── ml/                    ML (built)
│   ├── predictor.py       ★ CONTRACT B
│   ├── genohybrid.py      the tri-modal model behind the contract
│   ├── annotation.py      sequence, conservation and consequence lookups
│   ├── compute_metrics.py metrics and temperature fitting
│   └── artifacts/         checkpoints, features, validation output
│
├── samples/               small ClinVar VCF files for the demo
└── tests/                 backend tests
```

---

## 4. Contract A — frontend ↔ backend

Source of truth: `src/api/types.ts` (TypeScript) and `backend/models.py`
(Pydantic). They must stay identical.

| Method | Path | Request → Response |
| --- | --- | --- |
| `POST` | `/api/predict` | `VariantInput` → `Prediction` |
| `GET` | `/api/predictions?limit=` | → `Prediction[]` |
| `GET` | `/api/metrics` | → `ModelMetrics[]` (one per eval split) |
| `GET` | `/api/batch` | → `BatchJob[]` |
| `POST` | `/api/batch` | multipart `file` → `BatchJob` |

### Fields that carry real meaning

Getting these wrong silently breaks the project's whole argument:

- **`probabilities`** — softmax over `Pathogenic | Benign | VUS`, *after*
  temperature scaling. Must sum to 1.
- **`confidence`** — the calibrated probability of `predicted_class`. It is just
  `max(probabilities)`. It is **not** a measure of stability.
- **`uncertainty`** — standard deviation across the MC-Dropout forward passes.
  It is **not** `1 - confidence`. This distinction is the point of the project:
  a prediction can be 60% confident and stable, or 60% confident and wildly
  unstable, and the second is far less trustworthy even though both read 60%.
- **`shap[].value`** — signed log-odds contribution *toward `predicted_class`*.
  Positive always means "supports the call", whatever the call is. The UI
  colours by this convention; get the sign wrong and a benign prediction renders
  as if it were pathogenic.
- **`shap[].raw`** — the feature's actual value (CADD 32.0, gnomAD AF 0.243).
  Displayed next to each bar, so it must be the real scored value.

### Switching the frontend to the real backend

One environment variable. Create `.env.local`:

```
VITE_USE_MOCK=false
```

Vite proxies `/api` to `http://localhost:8000` (see `vite.config.ts`). Nothing
else changes, and `src/api/mock.ts` can then be deleted.

---

## 5. Contract B — backend ↔ ML

Source of truth: `ml/predictor.py`.

The ML side delivers **one class** with **one method**:

```python
class Predictor(Protocol):
    def predict(self, variants: list[VariantInput]) -> list[PredictionResult]:
        ...
```

Batched, because scoring one variant at a time wastes the GPU. The backend
handles HTTP, validation, database, and job queueing; it never imports torch or
xgboost directly. The ML side never imports FastAPI.

This means the ML developer can work entirely in notebooks and only has to wrap
the result in `Predictor` at the end.

---

## 6. Original plan (kept for reference)

This was the plan at the start. What was actually built is in §7; the parts of
this plan that were not done are listed there.

**Data sources.** ClinVar (labels), dbNSFP (bundles SIFT, PolyPhen-2, CADD,
REVEL — download once, don't fetch each separately), gnomAD (population
frequency), UCSC phyloP/phastCons (conservation).

**Features.** Tabular legacy scores + conservation + population frequency, plus
frozen ESM-2 embeddings for the sequence window around the variant. *Cache the
embeddings.* Recomputing them per epoch is the single biggest waste of compute
in the whole pipeline.

**Models.** Baselines first — Logistic Regression, Random Forest, XGBoost on
tabular features. Then a CNN + attention branch over the sequence encoding.
Then fuse both into dense layers with an MC-Dropout output head.

Do the baselines properly and report them. The literature survey found no
unified benchmark of these methods, so the comparison table is a genuine
contribution and it is also the cheapest part to build.

**Uncertainty.** 30–100 stochastic forward passes at inference. Mean → the
prediction, variance → the confidence score. Then temperature scaling on a
validation split to make the probabilities calibrated.

**Evaluation.** Three splits, all three reported:

1. **Random** — comparable to published numbers.
2. **Rare/novel** — variants unseen and sparsely represented. Scores will be
   noticeably worse. *Report it anyway.* Benchmarking studies show most tools
   overfit to well-studied variants and hide it behind a random split; showing
   the drop is a deliberate methodological argument, not a weakness.
3. **Held-out VUS** — variants labelled VUS at training time, checked against
   later ClinVar reclassifications.

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, and **ECE**.
Report PR-AUC alongside ROC-AUC — with this much class imbalance ROC-AUC alone
flatters the model.

---

## 7. Current state, honestly

**Working end to end:** frontend → FastAPI → PostgreSQL → the trained tri-modal
model (sequence CNN + biomarker MLP + GCN) from github.com/jayatigupta05/VEP.
Single variants, batch uploads, the prediction log, PDF reports and the
Performance page all use real model output. See `backend/README.md` for exactly
where every number comes from.

**Measured** (3-fold GroupKFold by gene, 100,000 ClinVar variants): accuracy
0.879, macro F1 0.787, MCC 0.771, ROC-AUC 0.936, PR-AUC 0.846, ECE 0.016 after
temperature scaling.

**Not done:** the random, rare/novel and held-out-VUS evaluation splits; the
baseline comparison table (hidden when the real backend is on, since its numbers
were placeholders); ESM-2 embeddings and dbNSFP/gnomAD features from the original
plan (the trained model uses sequence, conservation and consequence instead);
authentication.

The mock (`VITE_USE_MOCK=true`) still works and still shows invented numbers. Do
not use it for anything you report.

## 8. Conventions

- TypeScript is strict; `npm run build` typechecks and must pass.
- No UI or chart libraries on the frontend — charts are hand-rolled SVG. Don't
  add a dependency without a reason that survives "could this be 40 lines of
  SVG?"
- Styling is CSS variables in `src/styles.css`. No inline colours; use
  `var(--path)`, `var(--benign)`, `var(--vus)` so the three classes stay
  consistent everywhere.
- Python: type hints throughout, Pydantic for anything crossing the HTTP
  boundary.
- Fixed random seeds everywhere (numpy, torch, sklearn) — reproducibility is an
  explicit project requirement.

---

## 9. Prompt to give your AI assistant

Drop this repo into your tool of choice and start with:

> Read `PROJECT-CONTEXT.md` in full, then read the README in the folder for my
> area (`backend/` or `ml/`). I'm building the [backend / ML] part. Do not change
> `src/api/types.ts` or `ml/predictor.py` without telling me — they're the
> contracts the other developers are coding against. Confirm you understand the
> difference between `confidence` and `uncertainty` in this project before you
> write anything.

That last sentence is worth keeping. It is the detail AI assistants most often
get wrong here, and getting it wrong quietly destroys the project's main claim.
