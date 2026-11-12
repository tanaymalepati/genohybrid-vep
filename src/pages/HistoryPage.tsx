import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { Prediction, VariantClass } from '../api/types';
import PredictionCard, { variantLabel } from '../components/PredictionCard';

type Filter = 'all' | VariantClass;

const FILTERS: Filter[] = ['all', 'Pathogenic', 'Benign', 'VUS'];

export default function HistoryPage() {
  const [rows, setRows] = useState<Prediction[]>([]);
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [filter, setFilter] = useState<Filter>('all');

  useEffect(() => {
    api.history().then(setRows).catch(() => setRows([]));
  }, []);

  const visible = rows.filter(
    (r) => filter === 'all' || r.predicted_class === filter,
  );

  return (
    <>
      <h1>Prediction log</h1>
      <p className="sub">
        Every scored variant is persisted with its confidence and uncertainty —
        the audit trail behind performance monitoring.
      </p>

      <div className="segmented">
        {FILTERS.map((f) => (
          <button key={f} type="button" aria-pressed={f === filter} onClick={() => setFilter(f)}>
            {f === 'all' ? 'All' : f}
          </button>
        ))}
      </div>

      <div className="panel panel-flush">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Call</th>
              <th className="num">Confidence</th>
              <th className="num">σ</th>
              <th>Scored</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr key={r.id} className="clickable" onClick={() => setSelected(r)}>
                <td className="mono">{variantLabel(r)}</td>
                <td>
                  <span className={`tag ${r.predicted_class}`}>{r.predicted_class}</span>
                </td>
                <td className="num mono">{(r.confidence * 100).toFixed(1)}%</td>
                <td className="num mono">{r.uncertainty.toFixed(3)}</td>
                <td className="muted">{new Date(r.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {visible.length === 0 && <p className="empty">No predictions match this filter.</p>}
      </div>

      {selected && (
        <div style={{ marginTop: 24 }}>
          <div className="spread" style={{ marginBottom: 12 }}>
            <h3 style={{ margin: 0 }}>{variantLabel(selected)}</h3>
            <button type="button" className="ghost small" onClick={() => setSelected(null)}>
              Close
            </button>
          </div>
          <PredictionCard prediction={selected} />
        </div>
      )}

      {!selected && visible.length > 0 && (
        <p className="caption">Select a row to see its full attribution breakdown.</p>
      )}
    </>
  );
}
