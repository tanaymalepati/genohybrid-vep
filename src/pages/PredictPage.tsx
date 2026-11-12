import { useState } from 'react';
import { api } from '../api/client';
import type { Prediction, VariantInput } from '../api/types';
import PredictionCard from '../components/PredictionCard';

/** Real ClinVar variants, one per predicted class from the trained model. */
const EXAMPLES: Array<{ variant: VariantInput; note: string }> = [
  {
    variant: { chrom: '11', pos: 47347855, ref: 'A', alt: 'T', assembly: 'GRCh38', gene: 'MYBPC3' },
    note: 'Splice variant, ClinVar pathogenic',
  },
  {
    variant: { chrom: '7', pos: 66089689, ref: 'G', alt: 'C', assembly: 'GRCh38', gene: 'ASL' },
    note: 'Synonymous variant, ClinVar benign',
  },
  {
    variant: { chrom: '1', pos: 39432550, ref: 'G', alt: 'A', assembly: 'GRCh38', gene: 'MACF1' },
    note: 'Missense variant the model is unsure about',
  },
];

export default function PredictPage() {
  const [form, setForm] = useState<VariantInput>(EXAMPLES[0].variant);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof VariantInput>(key: K, value: VariantInput[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predict(form));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }

  const valid =
    form.chrom && form.pos > 0 && /^[ACGT]+$/i.test(form.ref) && /^[ACGT]+$/i.test(form.alt);

  return (
    <>
      <h1>Variant prediction</h1>
      <p className="sub">
        Three-way classification with a calibrated confidence score. Variants of
        uncertain significance are a reported class, not a discarded label.
      </p>

      <div className="split">
        <form className="panel" onSubmit={submit}>
          <h2>Variant</h2>

          <div className="grid cols-2" style={{ gap: '0 14px' }}>
            <div className="field">
              <label htmlFor="chrom">Chromosome</label>
              <select id="chrom" value={form.chrom} onChange={(e) => set('chrom', e.target.value)}>
                {[...Array.from({ length: 22 }, (_, i) => String(i + 1)), 'X', 'Y', 'MT'].map(
                  (c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ),
                )}
              </select>
            </div>
            <div className="field">
              <label htmlFor="assembly">Assembly</label>
              <select
                id="assembly"
                value={form.assembly}
                onChange={(e) => set('assembly', e.target.value as VariantInput['assembly'])}
              >
                <option>GRCh38</option>
                <option>GRCh37</option>
              </select>
            </div>
          </div>

          <div className="field">
            <label htmlFor="pos">Position</label>
            <input
              id="pos"
              className="mono"
              type="number"
              min={1}
              value={form.pos}
              onChange={(e) => set('pos', Number(e.target.value))}
            />
          </div>

          <div className="grid cols-2" style={{ gap: '0 14px' }}>
            <div className="field">
              <label htmlFor="ref">Reference allele</label>
              <input
                id="ref"
                className="mono"
                value={form.ref}
                onChange={(e) => set('ref', e.target.value.toUpperCase())}
              />
            </div>
            <div className="field">
              <label htmlFor="alt">Alternate allele</label>
              <input
                id="alt"
                className="mono"
                value={form.alt}
                onChange={(e) => set('alt', e.target.value.toUpperCase())}
              />
            </div>
          </div>

          <div className="field">
            <label htmlFor="gene">Gene symbol (optional)</label>
            <input
              id="gene"
              value={form.gene ?? ''}
              onChange={(e) => set('gene', e.target.value)}
              placeholder="BRCA1"
            />
          </div>

          <button type="submit" disabled={!valid || loading} style={{ width: '100%' }}>
            {loading ? 'Scoring…' : 'Run prediction'}
          </button>

          {error && (
            <p className="note error" style={{ marginTop: 14 }}>
              {error}
            </p>
          )}

          <div className="divider" />

          <label>Load an example</label>
          <div className="row">
            {EXAMPLES.map((ex) => (
              <button
                key={`${ex.variant.chrom}:${ex.variant.pos}`}
                type="button"
                className="ghost small"
                title={ex.note}
                onClick={() => {
                  setForm(ex.variant);
                  setResult(null);
                }}
              >
                {ex.variant.gene}
              </button>
            ))}
          </div>
          <p className="caption">
            {EXAMPLES.map((e) => `${e.variant.gene} — ${e.note.toLowerCase()}`).join(' · ')}
          </p>
        </form>

        {result ? (
          <PredictionCard prediction={result} />
        ) : (
          <div className="panel">
            <h2>Result</h2>
            <p className="empty" style={{ padding: '60px 20px' }}>
              Enter a variant and run a prediction to see class probabilities,
              the uncertainty band, and the feature attributions behind the call.
            </p>
          </div>
        )}
      </div>
    </>
  );
}
