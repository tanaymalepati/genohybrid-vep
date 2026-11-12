import type { ShapContribution, VariantClass } from '../api/types';

const CLASS_VAR: Record<VariantClass, string> = {
  Pathogenic: 'var(--path)',
  Benign: 'var(--benign)',
  VUS: 'var(--vus)',
};

/**
 * Horizontal signed-contribution plot. Bars right of centre support the
 * predicted class and take its colour; bars left of centre argue against it
 * and are drawn neutral, so a benign call never renders in the pathogenic
 * colour.
 */
export default function ShapPlot({
  contributions,
  predictedClass,
  top = 10,
}: {
  contributions: ShapContribution[];
  predictedClass: VariantClass;
  top?: number;
}) {
  const items = contributions.slice(0, top);
  const max = Math.max(...items.map((c) => Math.abs(c.value)), 0.001);
  const supportColor = CLASS_VAR[predictedClass];

  return (
    <div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map((c) => {
          const pct = (Math.abs(c.value) / max) * 50;
          const positive = c.value >= 0;
          return (
            <div key={c.feature} className="row" style={{ gap: 14, flexWrap: 'nowrap' }}>
              <div
                className="mono"
                style={{
                  width: 148,
                  flex: '0 0 148px',
                  textAlign: 'right',
                  color: 'var(--ink-2)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                title={`${c.feature} (${c.group.replace('_', ' ')})`}
              >
                {c.feature}
              </div>
              <div
                className="mono"
                style={{
                  width: 56,
                  flex: '0 0 56px',
                  textAlign: 'right',
                  color: 'var(--ink-3)',
                  fontVariantNumeric: 'tabular-nums',
                }}
                title="Feature value as scored"
              >
                {c.raw === null || c.raw === undefined ? '—' : c.raw}
              </div>
              <div style={{ flex: 1, position: 'relative', height: 15, minWidth: 0 }}>
                <div
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: -1,
                    bottom: -1,
                    width: 1,
                    background: 'var(--rule-strong)',
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    top: 3,
                    height: 9,
                    borderRadius: 2,
                    background: positive ? supportColor : 'var(--rule-strong)',
                    width: `${pct}%`,
                    left: positive ? '50%' : `${50 - pct}%`,
                  }}
                />
              </div>
              <div
                className="mono"
                style={{
                  width: 54,
                  flex: '0 0 54px',
                  textAlign: 'right',
                  color: 'var(--ink-3)',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {c.value > 0 ? '+' : ''}
                {c.value.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
      <p className="caption">
        Columns: feature, its value for this variant, and its signed log-odds
        contribution. Coloured bars support the{' '}
        <strong style={{ color: supportColor }}>{predictedClass}</strong> call; grey
        bars argue against it. Top {items.length} features by magnitude.
      </p>
    </div>
  );
}
