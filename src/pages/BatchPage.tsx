import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { BatchJob } from '../api/types';

const STATUS_COLOR: Record<BatchJob['status'], string> = {
  queued: 'var(--ink-3)',
  running: 'var(--accent)',
  complete: 'var(--benign)',
  failed: 'var(--path)',
};

export default function BatchPage() {
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.batchJobs().then(setJobs).catch(() => setError('Could not load batch jobs'));
  }, []);

  // Keep the progress bars moving while any job is still working.
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running');
  useEffect(() => {
    if (!active || api.usingMock) return;
    const timer = setInterval(() => {
      api.batchJobs().then(setJobs).catch(() => undefined);
    }, 1500);
    return () => clearInterval(timer);
  }, [active]);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const job = await api.submitBatch(file);
      setJobs((j) => [job, ...j]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  return (
    <>
      <h1>Batch scoring</h1>
      <p className="sub">
        Upload a VCF or CSV of variants. Each job is scored by the inference
        worker and written to the predictions table.
      </p>

      <div className="panel">
        <h2>New job</h2>
        <div className="field" style={{ maxWidth: 520, marginBottom: 0 }}>
          <label htmlFor="file">
            VCF (.vcf, .vcf.gz) or CSV with chrom, pos, ref, alt columns
          </label>
          <input
            id="file"
            ref={fileRef}
            type="file"
            accept=".vcf,.gz,.csv,.tsv"
            onChange={onFile}
            disabled={uploading}
          />
        </div>
        {uploading && <p className="caption">Uploading…</p>}
        {error && (
          <p className="note error" style={{ marginTop: 14 }}>
            {error}
          </p>
        )}
      </div>

      <div className="panel panel-flush">
        <div className="panel-head">
          <h2>Jobs</h2>
        </div>
        {jobs.length === 0 ? (
          <p className="empty">No batch jobs submitted yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Job</th>
                <th>File</th>
                <th>Status</th>
                <th style={{ width: 170 }}>Progress</th>
                <th className="num">Variants</th>
                <th>Submitted</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => {
                const pct = j.total ? (j.completed / j.total) * 100 : 0;
                return (
                  <tr key={j.job_id}>
                    <td className="mono">{j.job_id}</td>
                    <td className="mono">{j.filename}</td>
                    <td style={{ color: STATUS_COLOR[j.status], fontWeight: 500 }} title={j.error ?? undefined}>
                      {j.status}
                      {j.error && <div className="caption" style={{ fontWeight: 400 }}>{j.error}</div>}
                    </td>
                    <td>
                      <div className="bar">
                        <span style={{ width: `${pct}%`, background: STATUS_COLOR[j.status] }} />
                      </div>
                    </td>
                    <td className="num mono">
                      {j.completed.toLocaleString()} / {j.total.toLocaleString()}
                    </td>
                    <td className="muted">{new Date(j.submitted_at).toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
