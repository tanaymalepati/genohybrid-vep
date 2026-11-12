# ML

Read `../PROJECT-CONTEXT.md` first if you haven't.

**Status:** built. `genohybrid.py` wraps the trained tri-modal model (sequence
CNN, biomarker MLP, GCN) behind the `Predictor` contract below, and
`compute_metrics.py` writes `artifacts/metrics.json`. `../backend/README.md`
says where every number comes from. The rest of this file is the original brief
and is kept for the contract details and the parts of the plan not done yet.

## Your deliverable

One class with one method. That's the entire interface:

```python
class Predictor(Protocol):
    def predict(self, variants: list[VariantInput]) -> list[PredictionResult]:
        ...
```

It's defined in `predictor.py`, along with `StubPredictor`, a working fake so
the rest of the system runs while you build the real thing. **Keep the stub
working** — it's what lets the app be demoed before your model is done.

When your model is ready, point `load_predictor()` at it. Nothing else in the
codebase changes.

You can work entirely in notebooks and only wrap the result in `Predictor` at
the very end. Nothing here requires you to touch FastAPI or React.

## What `PredictionResult` needs from you

- **`probabilities`** — over all three classes, summing to 1, **after
  temperature scaling**. Raw softmax outputs are overconfident, and ECE is one
  of the reported metrics.
- **`uncertainty`** — std-dev of the predicted-class probability across your MC
  Dropout passes. **Not `1 - confidence`.** These are different quantities and
  the project's whole argument rests on the difference: a prediction can be
  confident and unstable, and the second is far less trustworthy even though
  both read the same confidence.
- **`shap`** — signed log-odds contributions **toward the predicted class**, so
  positive always means "supports the call". If you compute SHAP toward the
  pathogenic class, flip the sign when the call is benign. `raw` is the
  feature's actual value (CADD 32.0, gnomAD AF 0.243) and is displayed next to
  each bar.

`PredictionResult.validate()` checks the easy mistakes. Call it in your tests.

## Build order

**1. Data pipeline.** ClinVar for labels, dbNSFP for legacy scores (it bundles
SIFT, PolyPhen-2, CADD and REVEL in one download — don't fetch them separately),
gnomAD for frequency, UCSC for phyloP/phastCons. Three-way label extraction:
Pathogenic / Benign / **VUS held out separately**, not discarded.

**2. Features.** Tabular scores + conservation + frequency, plus frozen ESM-2
embeddings of the sequence window around the variant. **Cache the embeddings to
disk.** Recomputing them every epoch is the single biggest waste of compute in
the pipeline, and the frozen-embedding choice is what keeps this project
runnable on university hardware.

**3. Baselines first.** Logistic Regression, Random Forest, XGBoost on tabular
features alone. Do these properly and report them — the literature survey found
no unified benchmark of these methods, so the comparison table is a real
contribution and it's the cheapest part to build. It also tells you early
whether the deep branch is earning its complexity.

**4. Fusion model.** CNN + attention over the sequence encoding, concatenated
with the tabular branch, into dense layers with an MC-Dropout output head.

**5. Uncertainty + calibration.** 30–100 stochastic forward passes at
inference: mean → prediction, std-dev → uncertainty. Then fit temperature
scaling on a validation split.

**6. Evaluation.** Three splits, all three reported:

- **random** — comparable to published numbers
- **rare_novel** — unseen, sparsely-represented variants
- **held_out_vus** — VUS at training time, checked against later ClinVar
  reclassifications

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, ECE. Report
PR-AUC alongside ROC-AUC — with this class imbalance, ROC-AUC alone flatters the
model.

**The rare/novel split will score worse than the random split. Report it
anyway.** Benchmarking studies show most published tools overfit to well-studied
variants and hide it behind a random split. Showing the drop is a deliberate
methodological argument in your favour, not a weakness to bury.

## Handing metrics to the app

The Performance page reads `GET /api/metrics`, which currently returns
placeholders. Write your evaluation output somewhere the backend can read — a
JSON artifact or a results table — and tell your backend teammate where. The
shape it needs is `ModelMetrics` in `backend/models.py`.

## Practical notes

- Fixed seeds everywhere (numpy, torch, sklearn). Reproducibility is an explicit
  project requirement.
- Use Optuna rather than GridSearchCV for the hyperparameter sweep — it's more
  sample-efficient and it strengthens the methodology section.
- MLflow for experiment tracking is a cheap win that maps directly onto the
  reproducibility requirement.
- `predict()` is batched deliberately. Batch scoring sends thousands of variants
  at once; per-variant GPU calls will be the bottleneck.
