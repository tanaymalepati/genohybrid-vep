import type { CalibrationBin } from '../api/types';

/** Reliability diagram: mean predicted confidence vs. observed accuracy. */
export default function CalibrationCurve({
  bins,
  ece,
}: {
  bins: CalibrationBin[];
  ece: number;
}) {
  const W = 340;
  const H = 250;
  const padL = 38;
  const padB = 32;
  const x = (v: number) => padL + v * (W - padL - 12);
  const y = (v: number) => H - padB - v * (H - padB - 14);

  const path = bins
    .map((b, i) => `${i === 0 ? 'M' : 'L'}${x(b.mean_confidence)},${y(b.observed_accuracy)}`)
    .join(' ');

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Calibration curve">
        {[0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={t}
            x1={padL}
            y1={y(t)}
            x2={W - 12}
            y2={y(t)}
            stroke="var(--rule)"
            strokeWidth={1}
          />
        ))}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(1)}
          y2={y(1)}
          stroke="var(--rule-strong)"
          strokeDasharray="4 4"
        />
        <line x1={padL} y1={y(0)} x2={W - 12} y2={y(0)} stroke="var(--rule-strong)" />
        <line x1={padL} y1={y(0)} x2={padL} y2={12} stroke="var(--rule-strong)" />
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth={1.8} />
        {bins.map((b) => (
          <circle
            key={b.bin_lower}
            cx={x(b.mean_confidence)}
            cy={y(b.observed_accuracy)}
            r={2.8}
            fill="var(--accent)"
          >
            <title>
              {`confidence ${b.mean_confidence.toFixed(2)} · accuracy ${b.observed_accuracy.toFixed(2)} · n=${b.count}`}
            </title>
          </circle>
        ))}
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <text x={x(t)} y={H - 14} fill="var(--ink-3)" fontSize="10" textAnchor="middle">
              {t.toFixed(1)}
            </text>
            <text x={padL - 7} y={y(t) + 3} fill="var(--ink-3)" fontSize="10" textAnchor="end">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text
          x={(W + padL) / 2}
          y={H - 1}
          fill="var(--ink-3)"
          fontSize="10.5"
          textAnchor="middle"
        >
          Mean predicted confidence
        </text>
      </svg>
      <p className="caption">
        Observed accuracy per confidence bin. Dashed line is perfect calibration;
        ECE after temperature scaling is <strong className="mono">{ece.toFixed(3)}</strong>.
      </p>
    </div>
  );
}
