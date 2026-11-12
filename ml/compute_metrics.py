"""Fits temperature scaling and writes artifacts/calibration.json and metrics.json
from the out-of-fold predictions of the 3-fold GroupKFold validation.
Run: python -m ml.compute_metrics

Each fold's ECE uses a temperature fitted on the other two folds, so the reported
calibration is measured on data the temperature never saw."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ART = Path(__file__).resolve().parent / "artifacts"
df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(ART / "validation/fast_fold_*_predictions.csv"))))
y = df["true_label"].to_numpy()
raw = df[["prob_benign", "prob_pathogenic", "prob_vus"]].to_numpy()
folds = df["fold"].to_numpy()


def scale(p, t):
    q = np.exp(np.log(np.clip(p, 1e-9, 1)) / t)
    return q / q.sum(1, keepdims=True)


def fit(p, labels):
    nll = lambda t: -np.log(scale(p, t)[np.arange(len(p)), labels]).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 5), method="bounded").x)


def top_label_ece(p, labels):
    c, ok = p.max(1), p.argmax(1) == labels
    total = 0.0
    for i in range(10):
        m = (c > i / 10) & (c <= (i + 1) / 10)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - c[m].mean())
    return float(total)


P = np.zeros_like(raw)
for f in np.unique(folds):
    P[folds == f] = scale(raw[folds == f], fit(raw[folds != f], y[folds != f]))
temperature = fit(raw, y)
(ART / "calibration.json").write_text(json.dumps({"temperature": temperature}))
print("temperature", round(temperature, 3), "ECE before", round(top_label_ece(raw, y), 4))
pred = P.argmax(1)
conf, correct = P.max(1), pred == y
bins, ece = [], 0.0
for i in range(10):
    lo, hi = i / 10, (i + 1) / 10
    m = (conf > lo) & (conf <= hi)
    if m.sum():
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    bins.append({"bin_lower": lo, "bin_upper": hi,
                 "mean_confidence": float(conf[m].mean()) if m.sum() else (lo + hi) / 2,
                 "observed_accuracy": float(correct[m].mean()) if m.sum() else 0.0,
                 "count": int(m.sum())})
order = [1, 0, 2]  # Pathogenic, Benign, VUS: the order the frontend displays
out = [{
    "model_version": "genohybrid-trimodal-38f68f5",
    "evaluated_at": "2026-10-11T00:00:00Z",
    "split": "gene_holdout",
    "accuracy": accuracy_score(y, pred),
    "precision": precision_score(y, pred, average="macro"),
    "recall": recall_score(y, pred, average="macro"),
    "f1": f1_score(y, pred, average="macro"),
    "roc_auc": roc_auc_score(y, P, multi_class="ovr", average="macro"),
    "pr_auc": float(np.mean([average_precision_score(y == c, P[:, c]) for c in range(3)])),
    "mcc": matthews_corrcoef(y, pred),
    "ece": float(ece),
    "confusion_matrix": confusion_matrix(y, pred, labels=order).tolist(),
    "classes": ["Pathogenic", "Benign", "VUS"],
    "calibration": bins,
}]
(ART / "metrics.json").write_text(json.dumps(out, indent=2))
print({k: round(v, 4) for k, v in out[0].items() if isinstance(v, float)}, len(y))
