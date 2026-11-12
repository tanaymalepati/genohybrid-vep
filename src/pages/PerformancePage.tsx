import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { MOCK_BASELINES } from '../api/mock';
import type { ModelMetrics } from '../api/types';
import CalibrationCurve from '../components/CalibrationCurve';
import ConfusionMatrix from '../components/ConfusionMatrix';

const SPLIT_LABEL: Record<ModelMetrics['split'], string> = {
  random: 'Random split',
  rare_novel: 'Rare / novel variants',
  held_out_vus: 'Held-out VUS',
  gene_holdout: 'Unseen genes (3-fold)',
};

const SPLIT_NOTE: Record<ModelMetrics['split'], string> = {
  random: 'Standard stratified split — directly comparable to numbers published by other tools.',
  rare_novel:
    'Variants unseen in training and sparsely represented in ClinVar. Reported deliberately: benchmarking studies show most tools overfit to well-studied variants, and a random split hides that.',
  held_out_vus:
    'Variants labelled VUS at training time, scored against their later ClinVar reclassifications.',
  gene_holdout:
    '3-fold GroupKFold by gene on 100,000 ClinVar variants: every test gene was unseen in training. Calibration is measured on folds the temperature was not fitted to.',
};

export default function PerformancePage() {
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [split, setSplit] = useState<ModelMetrics['split'] | null>(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => setMetrics([]));
  }, []);

  // Only offer splits that have actually been evaluated.
  const m = metrics.find((x) => x.split === split) ?? metrics[0];

  return (
    <>
      <h1>Model performance</h1>
      <p className="sub">
        Measured evaluation results, with calibration reported alongside
        discrimination.
      </p>

      <div className="segmented">
        {metrics.map(({ split: s }) => (
          <button key={s} type="button" aria-pressed={s === m?.split} onClick={() => setSplit(s)}>
            {SPLIT_LABEL[s]}
          </button>
        ))}
      </div>

      {!m ? (
        <div className="panel">
          <p className="empty">Loading metrics…</p>
        </div>
      ) : (
        <div className="stack">
          <p className="note">{SPLIT_NOTE[m.split]}</p>

          <div className="panel">
            <h2>Metrics</h2>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
                gap: '22px 16px',
              }}
            >
              {(
                [
                  ['Accuracy', m.accuracy],
                  ['Precision', m.precision],
                  ['Recall', m.recall],
                  ['F1', m.f1],
                  ['ROC-AUC', m.roc_auc],
                  ['PR-AUC', m.pr_auc],
                  ['MCC', m.mcc],
                  ['ECE', m.ece],
                ] as const
              ).map(([label, value]) => (
                <div key={label}>
                  <div className="readout">{value.toFixed(3)}</div>
                  <div className="readout-label">{label}</div>
                </div>
              ))}
            </div>
            <p className="caption">
              Evaluated {new Date(m.evaluated_at).toLocaleDateString()} on{' '}
              <span className="mono">{m.model_version}</span>. Lower is better for ECE;
              higher for everything else.
            </p>
          </div>

          <div className="grid cols-2">
            <div className="panel">
              <h2>Confusion matrix</h2>
              <ConfusionMatrix matrix={m.confusion_matrix} classes={m.classes} />
            </div>
            <div className="panel">
              <h2>Calibration</h2>
              <CalibrationCurve bins={m.calibration} ece={m.ece} />
            </div>
          </div>

          {api.usingMock && (
          <div className="panel panel-flush">
            <div className="panel-head">
              <h2>Baseline comparison</h2>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">ROC-AUC</th>
                  <th className="num">PR-AUC</th>
                  <th className="num">F1</th>
                  <th className="num">MCC</th>
                  <th className="num">ECE</th>
                </tr>
              </thead>
              <tbody>
                {MOCK_BASELINES.map((b, i) => (
                  <tr key={b.model} className={i === MOCK_BASELINES.length - 1 ? 'highlight' : ''}>
                    <td>{b.model}</td>
                    <td className="num mono">{b.roc_auc.toFixed(3)}</td>
                    <td className="num mono">{b.pr_auc.toFixed(3)}</td>
                    <td className="num mono">{b.f1.toFixed(3)}</td>
                    <td className="num mono">{b.mcc.toFixed(3)}</td>
                    <td className="num mono">{b.ece === null ? '—' : b.ece.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="caption" style={{ padding: '0 24px 18px' }}>
              CADD and REVEL emit uncalibrated scores, so ECE is undefined for them.
            </p>
          </div>
          )}
        </div>
      )}
    </>
  );
}
