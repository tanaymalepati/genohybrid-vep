# Demo script

## To start

Open the folder in VS Code. The dev server starts by itself and the browser
opens at <http://localhost:5173>. If VS Code asks whether to allow automatic
tasks, click **Allow**.

The very first time you open the folder, VS Code asks two one-time questions —
**"Do you trust the authors of the files in this folder?"** (Yes) and
**"Allow automatic tasks in this folder?"** (Allow). Answer them once now, not
in front of your teacher; after that the server starts on every open.

If it doesn't start on its own: **Terminal → Run Task… → Start demo**, or press
`F5`. Nothing needs to be installed first and it works with no internet.

Last-resort backup, straight from the terminal:

```bash
npm run dev
```

## What to click, in order

Every number below is real output from the trained model.

**1. Predict: pathogenic.** MYBPC3 is preloaded. Click **Run prediction**.
→ Pathogenic, 99.6% confidence, ±0.7% uncertainty. The top attributions are the
splice consequence and high phyloP conservation. **PDF report** downloads it.

**2. Predict: benign.** Click **ASL**, then **Run prediction**.
→ Benign, 99.8%. A synonymous change: the consequence now pushes toward benign.

**3. Predict: VUS, the differentiator.** Click **MACF1**, then **Run prediction**.
→ VUS, 94.1%, with the highest uncertainty of the three (±9.4%). The model
reports VUS as a real answer instead of forcing a binary call.

**4. Batch.** Upload `samples/clinvar_mixed_8.vcf`. Eight real ClinVar variants
score in a few seconds with a live progress bar, then appear in **Log**.

**5. Performance.** Unseen genes, 3-fold: accuracy 0.879, macro F1 0.787,
ECE 0.016. Every test gene was absent from training. The confusion matrix shows
the honest weakness: Pathogenic is the hardest class (precision 0.45).

**6. Log.** Every prediction is stored in PostgreSQL with confidence and
uncertainty. Filter to VUS to show the curation queue.

A brand-new variant needs internet for its first lookup (up to ~40 s); the
examples and sample files above are cached and work offline.

## If asked

**"Is this hitting a real model?"** Yes. With `VITE_USE_MOCK=false` in
`.env.local` every number comes from the trained tri-modal model through the
FastAPI backend. The mock is only a fallback for running the frontend alone.

**"Why is VUS a class instead of a threshold?"** A threshold on a binary
probability tells you the model is unsure. A trained third class plus MC-dropout
variance tells you *how* unsure and *why* — which is what the σ readout and the
SHAP panel are showing.

**"What is σ?"** The standard deviation across 30 Monte Carlo dropout forward
passes. It's separate from confidence: a prediction can be 60% confident and
stable, or 60% confident and wildly unstable. Both read 60%, but the second is
far less trustworthy — and only σ tells you which one you're looking at.
