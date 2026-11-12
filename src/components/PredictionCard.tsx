import { api } from '../api/client';
import type { Prediction, VariantClass } from '../api/types';
import ShapPlot from './ShapPlot';

const CLASS_VAR: Record<VariantClass, string> = {
  Pathogenic: 'var(--path)',
  Benign: 'var(--benign)',
  VUS: 'var(--vus)',
};

export function variantLabel(p: Prediction) {
  const v = p.variant;
  return `${v.gene ? `${v.gene} · ` : ''}chr${v.chrom}:${v.pos} ${v.ref}>${v.alt}`;
}

export default function PredictionCard({ prediction }: { prediction: Prediction }) {
  const p = prediction;
  const order: VariantClass[] = ['Pathogenic', 'VUS', 'Benign'];

  return (
    <div>
      <div className="panel">
        <div className="spread">
          <div>
            <div className="mono" style={{ fontSize: 15, fontWeight: 500 }}>
              {variantLabel(p)}
            </div>
            <div className="muted" style={{ fontSize: 12 }}>
              {p.variant.assembly} · {p.model_version} · {p.mc_passes} MC-dropout passes
            </div>
          </div>
          <div className="row" style={{ gap: 10 }}>
            {!api.usingMock && (
              <a className="muted" style={{ fontSize: 12 }} href={api.reportUrl(p.id)}>
                PDF report
              </a>
            )}
            <span className={`tag ${p.predicted_class}`}>{p.predicted_class}</span>
          </div>
        </div>

        <div className="divider" />

        <div className="grid cols-2">
          <div>
            <div className="readout">{(p.confidence * 100).toFixed(1)}%</div>
            <div className="readout-label">Calibrated confidence</div>
          </div>
          <div>
            <div className="readout">±{(p.uncertainty * 100).toFixed(1)}%</div>
            <div className="readout-label">Predictive uncertainty (σ)</div>
          </div>
        </div>

        <div className="divider" />

        <h2>Class probabilities</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 11 }}>
          {order.map((cls) => (
            <div key={cls}>
              <div className="spread" style={{ fontSize: 12.5, marginBottom: 4 }}>
                <span>{cls}</span>
                <span className="mono">{(p.probabilities[cls] * 100).toFixed(1)}%</span>
              </div>
              <div className="bar">
                <span
                  style={{
                    width: `${p.probabilities[cls] * 100}%`,
                    background: CLASS_VAR[cls],
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <h2>Feature attribution (SHAP)</h2>
        <ShapPlot contributions={p.shap} predictedClass={p.predicted_class} />
      </div>
    </div>
  );
}
