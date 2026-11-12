import type { BatchJob, ModelMetrics, Prediction, VariantInput } from './types';
import { MOCK_METRICS, mockBatchJobs, mockHistory, mockPredict } from './mock';

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';
const BASE = import.meta.env.VITE_API_BASE ?? '/api';

/** FastAPI puts the reason in `detail`; show that, not the raw JSON. */
async function errorText(res: Response): Promise<string> {
  const text = await res.text().catch(() => res.statusText);
  try {
    const { detail } = JSON.parse(text);
    if (typeof detail === 'string') return detail;
  } catch {
    // not JSON
  }
  return `${res.status} ${text}`;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    throw new Error(await errorText(res));
  }
  return res.json() as Promise<T>;
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const api = {
  usingMock: USE_MOCK,

  reportUrl: (id: string) => `${BASE}/predictions/${id}/report`,

  async predict(variant: VariantInput): Promise<Prediction> {
    if (USE_MOCK) {
      await delay(600);
      return mockPredict(variant);
    }
    return req<Prediction>('/predict', {
      method: 'POST',
      body: JSON.stringify(variant),
    });
  },

  async history(limit = 50): Promise<Prediction[]> {
    if (USE_MOCK) {
      await delay(300);
      return mockHistory();
    }
    return req<Prediction[]>(`/predictions?limit=${limit}`);
  },

  async metrics(): Promise<ModelMetrics[]> {
    if (USE_MOCK) {
      await delay(300);
      return MOCK_METRICS;
    }
    return req<ModelMetrics[]>('/metrics');
  },

  async batchJobs(): Promise<BatchJob[]> {
    if (USE_MOCK) {
      await delay(300);
      return mockBatchJobs();
    }
    return req<BatchJob[]>('/batch');
  },

  async submitBatch(file: File): Promise<BatchJob> {
    if (USE_MOCK) {
      await delay(800);
      return {
        job_id: `job_${Math.random().toString(16).slice(2, 6)}`,
        filename: file.name,
        status: 'queued',
        total: 0,
        completed: 0,
        submitted_at: new Date().toISOString(),
      };
    }
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/batch`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(await errorText(res));
    return res.json() as Promise<BatchJob>;
  },
};
