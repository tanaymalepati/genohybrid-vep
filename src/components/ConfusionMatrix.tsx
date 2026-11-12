import type { VariantClass } from '../api/types';

export default function ConfusionMatrix({
  matrix,
  classes,
}: {
  matrix: number[][];
  classes: VariantClass[];
}) {
  const max = Math.max(...matrix.flat(), 1);

  return (
    <div>
      <table>
        <thead>
          <tr>
            <th />
            {classes.map((c) => (
              <th key={c} className="num">
                pred {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={classes[i]}>
              <th scope="row" style={{ background: 'var(--sunken)', fontWeight: 500 }}>
                true {classes[i]}
              </th>
              {row.map((n, j) => {
                const intensity = n / max;
                return (
                  <td
                    key={classes[j]}
                    className="num mono"
                    style={{
                      // Capped so the darkest cell still keeps text contrast.
                      background: `color-mix(in srgb, var(--accent) ${intensity * 30}%, var(--surface))`,
                      color: 'var(--ink)',
                      fontWeight: i === j ? 600 : 400,
                    }}
                  >
                    {n.toLocaleString()}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">Rows are true labels; the diagonal is correct calls.</p>
    </div>
  );
}
