"""
CONTRACT B — the boundary between the ML pipeline and the backend.

The backend calls `Predictor.predict()` and nothing else. It never imports
torch, xgboost, or transformers; you never import FastAPI. That separation is
what lets the two halves be built independently.

To plug the real model in:

    1. Implement `Predictor` (any class with a matching `predict` method works
       — this is a structural Protocol, you don't need to subclass it).
    2. Point `load_predictor()` at your implementation.
    3. Change nothing else. The backend picks it up automatically.

Keep `StubPredictor` working. It's what lets the backend and frontend be
developed and demoed while the real model is still training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

VariantClass = Literal["Pathogenic", "Benign", "VUS"]

FeatureGroup = Literal[
    "conservation", "legacy_score", "population", "embedding", "annotation"
]


@dataclass
class VariantInput:
    """A single variant to score."""

    chrom: str
    pos: int
    ref: str
    alt: str
    assembly: Literal["GRCh37", "GRCh38"] = "GRCh38"
    gene: str | None = None
    hgvs: str | None = None
    #: Raw VEP/SnpEff consequence terms if the input file carried them. None means
    #: the predictor looks them up; "" means the file had no annotation.
    consequence: str | None = None


class VariantRejected(ValueError):
    """A single variant cannot be scored; the message says why."""


class BatchRejected(VariantRejected):
    """Some variants in a batch cannot be scored. Maps input index to reason."""

    def __init__(self, reasons: dict[int, str]) -> None:
        self.reasons = reasons
        super().__init__(next(iter(reasons.values())))


@dataclass
class ShapContribution:
    """One feature's contribution to one prediction."""

    feature: str
    #: Signed log-odds contribution TOWARD the predicted class. Positive always
    #: means "supports the call", whichever class was called. Get this sign
    #: wrong and the UI renders benign predictions in the pathogenic colour.
    value: float
    group: FeatureGroup
    #: The feature's actual value as scored (CADD 32.0, gnomAD AF 0.243). Shown
    #: next to the bar in the UI, so it must be the real value, not normalised.
    raw: float | str | None = None


@dataclass
class PredictionResult:
    """What the model returns for one variant."""

    predicted_class: VariantClass

    #: Calibrated (post-temperature-scaling) probabilities. Must sum to 1.0 and
    #: must contain all three classes.
    probabilities: dict[VariantClass, float]

    #: Standard deviation of the predicted-class probability across the MC
    #: Dropout forward passes.
    #:
    #: This is NOT 1 - confidence. Confidence says how strongly the model
    #: favours one class; uncertainty says how much that answer moves when you
    #: resample the dropout mask. A prediction can be confident and unstable.
    #: Reporting these separately is the core claim of the project.
    uncertainty: float

    #: Number of stochastic forward passes used (the plan says 30-100).
    mc_passes: int

    shap: list[ShapContribution] = field(default_factory=list)

    model_version: str = "genohybrid-vep-dev"

    #: Gene the variant was scored under, when the predictor resolved one.
    gene: str | None = None

    #: Each branch's own class probabilities, for comparison with the fused call.
    arm_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)

    def validate(self) -> None:
        """Cheap sanity checks. Call this in your tests."""
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"probabilities sum to {total}, expected 1.0")
        missing = {"Pathogenic", "Benign", "VUS"} - set(self.probabilities)
        if missing:
            raise ValueError(f"missing probability for {missing}")
        if self.predicted_class != max(self.probabilities, key=self.probabilities.get):
            raise ValueError("predicted_class is not the argmax of probabilities")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(f"uncertainty {self.uncertainty} outside [0, 1]")


@runtime_checkable
class Predictor(Protocol):
    """The only thing the backend needs from the ML side."""

    def predict(self, variants: list[VariantInput]) -> list[PredictionResult]:
        """
        Score a batch of variants.

        Batched on purpose: ESM-2 embedding lookup and GPU inference are far
        cheaper per variant in batches, and the batch-scoring endpoint sends
        thousands at a time.

        Must return one result per input, in the same order.
        """
        ...


# --------------------------------------------------------------------------
# Stub implementation — replace, but keep it working.
# --------------------------------------------------------------------------


class StubPredictor:
    """
    Deterministic placeholder so the backend runs before the model exists.

    Hashes the coordinates into a plausible-looking result. Every number it
    produces is fake.
    """

    def __init__(self, model_version: str = "stub-0.0.0") -> None:
        self.model_version = model_version

    def predict(self, variants: list[VariantInput]) -> list[PredictionResult]:
        return [self._one(v) for v in variants]

    def _one(self, v: VariantInput) -> PredictionResult:
        import hashlib

        key = f"{v.chrom}:{v.pos}:{v.ref}>{v.alt}"
        digest = hashlib.sha256(key.encode()).digest()
        a, b, c = (digest[i] + 1 for i in range(3))
        total = a + b + c

        probabilities: dict[VariantClass, float] = {
            "Pathogenic": a / total,
            "Benign": b / total,
            "VUS": c / total,
        }
        predicted = max(probabilities, key=probabilities.get)
        uncertainty = round(0.02 + (digest[3] / 255) * 0.2, 3)

        result = PredictionResult(
            predicted_class=predicted,
            probabilities={k: round(p, 4) for k, p in probabilities.items()},
            uncertainty=uncertainty,
            mc_passes=50,
            shap=[
                ShapContribution(
                    feature=name,
                    value=round((digest[4 + i] / 255 - 0.5) * 1.6, 3),
                    group=group,
                    raw=round(digest[8 + i] / 10, 2),
                )
                for i, (name, group) in enumerate(
                    [
                        ("CADD_phred", "legacy_score"),
                        ("REVEL", "legacy_score"),
                        ("ESM2_delta_llr", "embedding"),
                        ("phyloP100way", "conservation"),
                        ("gnomAD_AF_popmax", "population"),
                    ]
                )
            ],
            model_version=self.model_version,
        )
        result.validate()
        return result


def load_predictor() -> Predictor:
    """
    The tri-modal model. Set GENOHYBRID_PREDICTOR=stub for the fake one, which
    has to be asked for so invented numbers can never pass as real results.
    """
    import os

    if os.environ.get("GENOHYBRID_PREDICTOR") == "stub":
        return StubPredictor()

    from .genohybrid import GenoHybridPredictor

    return GenoHybridPredictor()
