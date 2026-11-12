import type {
  BatchJob,
  ModelMetrics,
  Prediction,
  ShapContribution,
  VariantClass,
  VariantInput,
} from './types';

/**
 * Deterministic stand-in for the FastAPI service so the UI can be developed
 * and demoed before the inference engine is wired up. Swap off via
 * VITE_USE_MOCK=false.
 *
 * Well-known variants get curated feature values so the SHAP signs read
 * correctly (a high CADD score pushes toward pathogenic, a common gnomAD
 * allele frequency pushes away from it). Anything else falls back to a
 * hash so results are still stable per variant.
 */

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}

interface FeatureMeta {
  name: string;
  group: ShapContribution['group'];
  /** Value at which the feature stops supporting pathogenicity. */
  midpoint: number;
  /** Normalising scale for the distance from the midpoint. */
  scale: number;
  /** 'high' = large values are deleterious; 'low' = small values are. */
  deleterious: 'high' | 'low';
  /** Relative weight of this feature in the fusion model. */
  weight: number;
  /** Compare on a log10 scale — allele frequencies span many orders. */
  log?: boolean;
}

const FEATURES: FeatureMeta[] = [
  { name: 'CADD_phred', group: 'legacy_score', midpoint: 20, scale: 12, deleterious: 'high', weight: 1.0 },
  { name: 'REVEL', group: 'legacy_score', midpoint: 0.5, scale: 0.4, deleterious: 'high', weight: 0.95 },
  { name: 'ESM2_delta_llr', group: 'embedding', midpoint: -4, scale: 6, deleterious: 'low', weight: 0.9 },
  { name: 'phyloP100way', group: 'conservation', midpoint: 2, scale: 5, deleterious: 'high', weight: 0.8 },
  { name: 'SIFT_score', group: 'legacy_score', midpoint: 0.05, scale: 0.3, deleterious: 'low', weight: 0.7 },
  { name: 'PolyPhen2_HDIV', group: 'legacy_score', midpoint: 0.5, scale: 0.4, deleterious: 'high', weight: 0.7 },
  // Compared as log10(AF): absence from gnomAD is strong pathogenic evidence,
  // a percent-level frequency is strong benign evidence.
  { name: 'gnomAD_AF_popmax', group: 'population', midpoint: -4, scale: 1.5, deleterious: 'low', weight: 0.85, log: true },
  { name: 'phastCons100way', group: 'conservation', midpoint: 0.6, scale: 0.4, deleterious: 'high', weight: 0.6 },
  { name: 'GERP++_RS', group: 'conservation', midpoint: 3, scale: 4, deleterious: 'high', weight: 0.55 },
  { name: 'AlphaFold_pLDDT', group: 'annotation', midpoint: 70, scale: 30, deleterious: 'high', weight: 0.4 },
];

/** Feature values for the curated demo variants. */
type FeatureValues = Partial<Record<string, number>>;

interface Profile {
  predicted_class: VariantClass;
  probabilities: Record<VariantClass, number>;
  uncertainty: number;
  values: FeatureValues;
}

const PATHOGENIC_LIKE: FeatureValues = {
  CADD_phred: 32.0,
  REVEL: 0.91,
  ESM2_delta_llr: -11.4,
  phyloP100way: 7.8,
  SIFT_score: 0.0,
  PolyPhen2_HDIV: 0.99,
  gnomAD_AF_popmax: 0.0,
  phastCons100way: 1.0,
  'GERP++_RS': 5.6,
  AlphaFold_pLDDT: 94.2,
};

const BENIGN_LIKE: FeatureValues = {
  CADD_phred: 8.4,
  REVEL: 0.12,
  ESM2_delta_llr: -0.6,
  phyloP100way: 0.3,
  SIFT_score: 0.61,
  PolyPhen2_HDIV: 0.04,
  gnomAD_AF_popmax: 0.243,
  phastCons100way: 0.09,
  'GERP++_RS': 0.4,
  AlphaFold_pLDDT: 61.5,
};

/** Deliberately conflicting evidence — the case that lands in VUS. */
const CONFLICTING: FeatureValues = {
  CADD_phred: 23.1,
  REVEL: 0.44,
  ESM2_delta_llr: -3.8,
  phyloP100way: 4.9,
  SIFT_score: 0.09,
  PolyPhen2_HDIV: 0.58,
  gnomAD_AF_popmax: 0.0012,
  phastCons100way: 0.71,
  'GERP++_RS': 3.2,
  AlphaFold_pLDDT: 55.8,
};

const key = (v: VariantInput) => `${v.chrom}:${v.pos}:${v.ref}>${v.alt}`;

const CURATED: Record<string, Profile> = {
  // BRCA1 — confident pathogenic, every line of evidence agrees.
  '17:43093464:C>T': {
    predicted_class: 'Pathogenic',
    probabilities: { Pathogenic: 0.962, VUS: 0.031, Benign: 0.007 },
    uncertainty: 0.021,
    values: PATHOGENIC_LIKE,
  },
  // TP53 — confident pathogenic.
  '17:7674220:C>T': {
    predicted_class: 'Pathogenic',
    probabilities: { Pathogenic: 0.941, VUS: 0.048, Benign: 0.011 },
    uncertainty: 0.029,
    values: { ...PATHOGENIC_LIKE, CADD_phred: 29.4, REVEL: 0.88, ESM2_delta_llr: -9.7 },
  },
  // MTHFR c.665C>T — common polymorphism, confidently benign.
  '1:11796321:G>A': {
    predicted_class: 'Benign',
    probabilities: { Pathogenic: 0.014, VUS: 0.062, Benign: 0.924 },
    uncertainty: 0.034,
    values: BENIGN_LIKE,
  },
  // CFTR — conflicting evidence, called VUS. This is the differentiator.
  '7:117559590:A>G': {
    predicted_class: 'VUS',
    probabilities: { Pathogenic: 0.361, VUS: 0.437, Benign: 0.202 },
    uncertainty: 0.187,
    values: CONFLICTING,
  },
  // BRCA2 — pathogenic but with weaker support.
  '13:32340301:G>A': {
    predicted_class: 'Pathogenic',
    probabilities: { Pathogenic: 0.812, VUS: 0.147, Benign: 0.041 },
    uncertainty: 0.068,
    values: { ...PATHOGENIC_LIKE, CADD_phred: 26.1, REVEL: 0.74, gnomAD_AF_popmax: 0.00002 },
  },
  // HBB — sickle-cell variant, confidently pathogenic.
  '11:5227002:T>A': {
    predicted_class: 'Pathogenic',
    probabilities: { Pathogenic: 0.978, VUS: 0.018, Benign: 0.004 },
    uncertainty: 0.016,
    values: { ...PATHOGENIC_LIKE, CADD_phred: 33.5 },
  },
  // MSH2 — borderline, called VUS.
  '2:47403157:C>G': {
    predicted_class: 'VUS',
    probabilities: { Pathogenic: 0.298, VUS: 0.521, Benign: 0.181 },
    uncertainty: 0.164,
    values: { ...CONFLICTING, REVEL: 0.39, CADD_phred: 21.8 },
  },
  // MLH1 — synonymous, benign.
  '3:37020402:G>T': {
    predicted_class: 'Benign',
    probabilities: { Pathogenic: 0.033, VUS: 0.109, Benign: 0.858 },
    uncertainty: 0.052,
    values: { ...BENIGN_LIKE, gnomAD_AF_popmax: 0.031, CADD_phred: 11.2 },
  },
  // KRAS — somatic hotspot, conflicting germline evidence.
  '12:25245350:C>A': {
    predicted_class: 'VUS',
    probabilities: { Pathogenic: 0.404, VUS: 0.468, Benign: 0.128 },
    uncertainty: 0.153,
    values: { ...CONFLICTING, CADD_phred: 25.6, phyloP100way: 6.1 },
  },
};

/**
 * Signed evidence for pathogenicity in [-1, 1], from how far the feature sits
 * past its midpoint in the deleterious direction.
 */
function evidence(f: FeatureMeta, raw: number): number {
  const x = f.log ? Math.log10(Math.max(raw, 1e-6)) : raw;
  const delta = (x - f.midpoint) / f.scale;
  const signed = f.deleterious === 'high' ? delta : -delta;
  return Math.tanh(signed);
}

function shapFor(profile: Profile, seed: number): ShapContribution[] {
  // Contributions are expressed toward the predicted class, so evidence for
  // pathogenicity flips sign when the call is Benign.
  const orient = profile.predicted_class === 'Benign' ? -1 : 1;

  return FEATURES.map((f, i) => {
    const raw = profile.values[f.name] ?? 0;
    const jitter = 0.9 + hash(`${seed}:${f.name}:${i}`) * 0.2;
    let value = evidence(f, raw) * f.weight * orient * jitter;

    // A VUS call means the evidence conflicts and none of it is decisive.
    if (profile.predicted_class === 'VUS') value *= 0.55;

    return {
      feature: f.name,
      group: f.group,
      value: Number(value.toFixed(3)),
      raw,
    };
  }).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

/** Fallback for variants with no curated profile. */
function syntheticProfile(k: string): Profile {
  const r = hash(k);
  const a = r;
  const b = hash(k + 'b');
  const c = hash(k + 'c') * 0.6;
  const sum = a + b + c;
  const probabilities: Record<VariantClass, number> = {
    Pathogenic: a / sum,
    Benign: b / sum,
    VUS: c / sum,
  };
  const entries = Object.entries(probabilities) as Array<[VariantClass, number]>;
  entries.sort((x, y) => y[1] - x[1]);
  const predicted = entries[0][0];

  const base =
    predicted === 'Pathogenic' ? PATHOGENIC_LIKE : predicted === 'Benign' ? BENIGN_LIKE : CONFLICTING;

  // Nudge the curated values so distinct variants don't look identical.
  const values: FeatureValues = {};
  for (const f of FEATURES) {
    const v = base[f.name] ?? 0;
    values[f.name] = Number((v * (0.85 + hash(k + f.name) * 0.3)).toFixed(3));
  }

  return {
    predicted_class: predicted,
    probabilities,
    uncertainty: Number((0.02 + hash(k + 'u') * 0.2).toFixed(3)),
    values,
  };
}

export function mockPredict(variant: VariantInput): Prediction {
  const k = key(variant);
  const profile = CURATED[k] ?? syntheticProfile(k);
  const seed = hash(k);

  return {
    id: `pred_${Math.abs(Math.floor(seed * 1e9))}`,
    variant,
    predicted_class: profile.predicted_class,
    probabilities: {
      Pathogenic: Number(profile.probabilities.Pathogenic.toFixed(4)),
      Benign: Number(profile.probabilities.Benign.toFixed(4)),
      VUS: Number(profile.probabilities.VUS.toFixed(4)),
    },
    confidence: Number(profile.probabilities[profile.predicted_class].toFixed(4)),
    uncertainty: profile.uncertainty,
    mc_passes: 50,
    shap: shapFor(profile, seed),
    model_version: 'genohybrid-vep-0.4.2',
    created_at: new Date().toISOString(),
  };
}

const SAMPLE_VARIANTS: VariantInput[] = [
  { chrom: '17', pos: 43093464, ref: 'C', alt: 'T', assembly: 'GRCh38', gene: 'BRCA1' },
  { chrom: '7', pos: 117559590, ref: 'A', alt: 'G', assembly: 'GRCh38', gene: 'CFTR' },
  { chrom: '1', pos: 11796321, ref: 'G', alt: 'A', assembly: 'GRCh38', gene: 'MTHFR' },
  { chrom: '13', pos: 32340301, ref: 'G', alt: 'A', assembly: 'GRCh38', gene: 'BRCA2' },
  { chrom: '11', pos: 5227002, ref: 'T', alt: 'A', assembly: 'GRCh38', gene: 'HBB' },
  { chrom: '2', pos: 47403157, ref: 'C', alt: 'G', assembly: 'GRCh38', gene: 'MSH2' },
  { chrom: '3', pos: 37020402, ref: 'G', alt: 'T', assembly: 'GRCh38', gene: 'MLH1' },
  { chrom: '12', pos: 25245350, ref: 'C', alt: 'A', assembly: 'GRCh38', gene: 'KRAS' },
  { chrom: '17', pos: 7674220, ref: 'C', alt: 'T', assembly: 'GRCh38', gene: 'TP53' },
];

export function mockHistory(): Prediction[] {
  return SAMPLE_VARIANTS.map((v, i) => {
    const pred = mockPredict(v);
    const t = new Date(Date.now() - (i + 1) * 3600_000 * 7);
    return { ...pred, created_at: t.toISOString() };
  });
}

export function mockBatchJobs(): BatchJob[] {
  return [
    {
      job_id: 'job_8f21',
      filename: 'panel_hboc_v3.vcf',
      status: 'complete',
      total: 1284,
      completed: 1284,
      submitted_at: new Date(Date.now() - 86_400_000).toISOString(),
    },
    {
      job_id: 'job_4c09',
      filename: 'clinvar_vus_holdout.csv',
      status: 'running',
      total: 5000,
      completed: 3120,
      submitted_at: new Date(Date.now() - 900_000).toISOString(),
    },
  ];
}

function bins(offset: number) {
  return Array.from({ length: 10 }, (_, i) => {
    const lo = i / 10;
    const mid = lo + 0.05;
    const drift = (hash(`cal${offset}${i}`) - 0.5) * 0.08;
    return {
      bin_lower: lo,
      bin_upper: lo + 0.1,
      mean_confidence: Number(mid.toFixed(3)),
      observed_accuracy: Number(Math.min(1, Math.max(0, mid + drift)).toFixed(3)),
      count: 40 + Math.floor(hash(`n${offset}${i}`) * 400),
    };
  });
}

export const MOCK_METRICS: ModelMetrics[] = [
  {
    model_version: 'genohybrid-vep-0.4.2',
    evaluated_at: new Date(Date.now() - 172_800_000).toISOString(),
    split: 'random',
    accuracy: 0.913,
    precision: 0.902,
    recall: 0.887,
    f1: 0.894,
    roc_auc: 0.961,
    pr_auc: 0.934,
    mcc: 0.842,
    ece: 0.021,
    classes: ['Pathogenic', 'Benign', 'VUS'],
    confusion_matrix: [
      [1412, 61, 97],
      [48, 1789, 121],
      [88, 134, 640],
    ],
    calibration: bins(1),
  },
  {
    model_version: 'genohybrid-vep-0.4.2',
    evaluated_at: new Date(Date.now() - 172_800_000).toISOString(),
    split: 'rare_novel',
    accuracy: 0.841,
    precision: 0.818,
    recall: 0.796,
    f1: 0.807,
    roc_auc: 0.908,
    pr_auc: 0.851,
    mcc: 0.719,
    ece: 0.038,
    classes: ['Pathogenic', 'Benign', 'VUS'],
    confusion_matrix: [
      [402, 39, 71],
      [44, 508, 88],
      [63, 74, 291],
    ],
    calibration: bins(2),
  },
  {
    model_version: 'genohybrid-vep-0.4.2',
    evaluated_at: new Date(Date.now() - 172_800_000).toISOString(),
    split: 'held_out_vus',
    accuracy: 0.762,
    precision: 0.744,
    recall: 0.731,
    f1: 0.737,
    roc_auc: 0.869,
    pr_auc: 0.802,
    mcc: 0.638,
    ece: 0.049,
    classes: ['Pathogenic', 'Benign', 'VUS'],
    confusion_matrix: [
      [188, 33, 96],
      [29, 214, 104],
      [51, 60, 743],
    ],
    calibration: bins(3),
  },
];

export const MOCK_BASELINES = [
  { model: 'Logistic Regression', roc_auc: 0.871, pr_auc: 0.792, f1: 0.774, mcc: 0.612, ece: 0.094 },
  { model: 'Random Forest', roc_auc: 0.912, pr_auc: 0.848, f1: 0.821, mcc: 0.701, ece: 0.071 },
  { model: 'XGBoost (tabular only)', roc_auc: 0.938, pr_auc: 0.891, f1: 0.859, mcc: 0.767, ece: 0.058 },
  { model: 'CADD (published)', roc_auc: 0.884, pr_auc: 0.803, f1: 0.781, mcc: 0.629, ece: null },
  { model: 'REVEL (published)', roc_auc: 0.925, pr_auc: 0.869, f1: 0.833, mcc: 0.716, ece: null },
  { model: 'GenoHybrid-VEP (fusion)', roc_auc: 0.961, pr_auc: 0.934, f1: 0.894, mcc: 0.842, ece: 0.021 },
];
