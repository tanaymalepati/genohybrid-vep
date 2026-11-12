// Wire types — these mirror the FastAPI response models.

export type VariantClass = 'Pathogenic' | 'Benign' | 'VUS';

/** Single-variant request. Either a coordinate tuple or an HGVS string. */
export interface VariantInput {
  chrom: string;
  pos: number;
  ref: string;
  alt: string;
  assembly: 'GRCh37' | 'GRCh38';
  gene?: string;
  hgvs?: string;
}

/** Per-feature SHAP attribution for one prediction. */
export interface ShapContribution {
  feature: string;
  /** Signed contribution toward the predicted class, in log-odds. */
  value: number;
  /** Raw feature value as scored (e.g. CADD 24.1, REVEL 0.83). */
  raw?: number | string | null;
  group: 'conservation' | 'legacy_score' | 'population' | 'embedding' | 'annotation';
}

/**
 * Confidence is calibrated (temperature scaling); uncertainty is the
 * predictive std-dev over MC Dropout forward passes.
 */
export interface Prediction {
  id: string;
  variant: VariantInput;
  predicted_class: VariantClass;
  probabilities: Record<VariantClass, number>;
  confidence: number;
  uncertainty: number;
  mc_passes: number;
  shap: ShapContribution[];
  model_version: string;
  created_at: string;
}

export interface BatchJob {
  job_id: string;
  filename: string;
  status: 'queued' | 'running' | 'complete' | 'failed';
  total: number;
  completed: number;
  submitted_at: string;
  error?: string | null;
}

export interface CalibrationBin {
  bin_lower: number;
  bin_upper: number;
  mean_confidence: number;
  observed_accuracy: number;
  count: number;
}

export interface ModelMetrics {
  model_version: string;
  evaluated_at: string;
  split: 'random' | 'rare_novel' | 'held_out_vus' | 'gene_holdout';
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  mcc: number;
  ece: number;
  /** Row = true class, col = predicted class, ordered as `classes`. */
  confusion_matrix: number[][];
  classes: VariantClass[];
  calibration: CalibrationBin[];
}
