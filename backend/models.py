"""
CONTRACT A (Python side) — Pydantic mirror of `src/api/types.ts`.

These models are what the frontend deserialises. If you change a field name or
type here, change it in `src/api/types.ts` in the same commit, or the UI breaks
silently.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

VariantClass = Literal["Pathogenic", "Benign", "VUS"]
Assembly = Literal["GRCh37", "GRCh38"]
FeatureGroup = Literal[
    "conservation", "legacy_score", "population", "embedding", "annotation"
]


class VariantInput(BaseModel):
    chrom: str
    pos: int = Field(gt=0)
    ref: str
    alt: str
    assembly: Assembly = "GRCh38"
    gene: str | None = None
    hgvs: str | None = None

    @field_validator("ref", "alt")
    @classmethod
    def _bases_only(cls, v: str) -> str:
        v = v.upper()
        if not v or set(v) - set("ACGT"):
            raise ValueError("alleles must be non-empty and contain only A, C, G, T")
        return v


class ShapContribution(BaseModel):
    feature: str
    #: Signed log-odds contribution toward the predicted class; positive always
    #: means "supports the call".
    value: float
    group: FeatureGroup
    #: The feature value as scored, displayed beside the bar in the UI.
    raw: float | str | None = None


class Prediction(BaseModel):
    id: str
    variant: VariantInput
    predicted_class: VariantClass
    #: Calibrated probabilities; must sum to 1 and cover all three classes.
    probabilities: dict[VariantClass, float]
    #: max(probabilities) — how strongly one class is favoured.
    confidence: float
    #: MC-Dropout predictive std-dev. NOT 1 - confidence. See PROJECT-CONTEXT.md.
    uncertainty: float
    mc_passes: int
    shap: list[ShapContribution] = Field(default_factory=list)
    model_version: str
    created_at: datetime


class BatchJob(BaseModel):
    job_id: str
    filename: str
    status: Literal["queued", "running", "complete", "failed"]
    total: int
    completed: int
    submitted_at: datetime
    error: str | None = None


class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    mean_confidence: float
    observed_accuracy: float
    count: int


class ModelMetrics(BaseModel):
    model_version: str
    evaluated_at: datetime
    #: Report all three. The rare/novel drop is a deliberate part of the
    #: project's argument, not something to hide.
    split: Literal["random", "rare_novel", "held_out_vus", "gene_holdout"]
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    mcc: float
    #: Expected Calibration Error — lower is better, unlike every metric above.
    ece: float
    #: Row = true class, column = predicted class, ordered as `classes`.
    confusion_matrix: list[list[int]]
    classes: list[VariantClass]
    calibration: list[CalibrationBin]

    # `model_version` would otherwise collide with Pydantic's protected
    # `model_` namespace.
    model_config = {"protected_namespaces": ()}
