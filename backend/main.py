"""
GenoHybrid-VEP backend: FastAPI over PostgreSQL and the tri-modal model.

    uvicorn backend.main:app --port 8000     (from the repo root)
"""
from __future__ import annotations

import json
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Response, UploadFile
from sqlalchemy import select, update
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.annotation import AnnotationUnavailable  # noqa: E402
from ml.predictor import BatchRejected, PredictionResult, VariantRejected, load_predictor  # noqa: E402
from ml.predictor import VariantInput as MLVariantInput  # noqa: E402

from .db import BatchJobRow, PredictionRow, SessionLocal, get_session, migrate  # noqa: E402
from .models import BatchJob, ModelMetrics, Prediction, ShapContribution, VariantInput  # noqa: E402
from .parsing import UploadError, parse_upload  # noqa: E402
from .report import build_report  # noqa: E402

METRICS_FILE = ROOT / "ml" / "artifacts" / "metrics.json"
CHUNK = 10
predictor = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor
    migrate()
    # Background jobs live in this process, so any still marked active were cut off.
    with SessionLocal() as s:
        s.execute(
            update(BatchJobRow).where(BatchJobRow.status.in_(("queued", "running")))
            .values(status="failed", error="The server restarted before this job finished.")
        )
        s.commit()
    predictor = load_predictor()
    yield


app = FastAPI(title="GenoHybrid-VEP", version="1.0.0", lifespan=lifespan)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_row(variant: MLVariantInput, r: PredictionResult, job_id: str | None) -> PredictionRow:
    return PredictionRow(
        id=f"pred_{uuid.uuid4().hex[:12]}",
        chrom=variant.chrom.removeprefix("chr"), pos=variant.pos,
        ref=variant.ref.upper(), alt=variant.alt.upper(), assembly=variant.assembly,
        gene=variant.gene or r.gene, hgvs=variant.hgvs,
        predicted_class=r.predicted_class, probabilities=r.probabilities,
        confidence=max(r.probabilities.values()), uncertainty=r.uncertainty, mc_passes=r.mc_passes,
        shap=[{"feature": c.feature, "value": c.value, "group": c.group, "raw": c.raw} for c in r.shap],
        arm_probabilities=r.arm_probabilities,
        model_version=r.model_version, created_at=_now(), job_id=job_id,
    )


def _to_prediction(row: PredictionRow) -> Prediction:
    return Prediction(
        id=row.id,
        variant=VariantInput(chrom=row.chrom, pos=row.pos, ref=row.ref, alt=row.alt,
                             assembly=row.assembly, gene=row.gene, hgvs=row.hgvs),
        predicted_class=row.predicted_class, probabilities=row.probabilities,
        confidence=row.confidence, uncertainty=row.uncertainty, mc_passes=row.mc_passes,
        shap=[ShapContribution(**c) for c in row.shap],
        model_version=row.model_version, created_at=row.created_at,
    )


def _to_job(row: BatchJobRow) -> BatchJob:
    return BatchJob(job_id=row.job_id, filename=row.filename, status=row.status, total=row.total,
                    completed=row.completed, submitted_at=row.submitted_at, error=row.error)


@app.get("/api/health")
def health(session: Session = Depends(get_session)) -> dict[str, str]:
    session.execute(select(1))
    return {"status": "ok", "database": "ok", "model": predictor.model_version}


@app.post("/api/predict", response_model=Prediction)
def predict(variant: VariantInput, session: Session = Depends(get_session)) -> Prediction:
    ml_variant = MLVariantInput(chrom=variant.chrom, pos=variant.pos, ref=variant.ref, alt=variant.alt,
                                assembly=variant.assembly, gene=variant.gene or None, hgvs=variant.hgvs)
    try:
        result = predictor.predict([ml_variant])[0]
    except VariantRejected as e:
        raise HTTPException(422, str(e)) from e
    except AnnotationUnavailable as e:
        raise HTTPException(503, f"Genome reference lookup failed: {e}. Check the internet connection.") from e
    row = _to_row(ml_variant, result, None)
    session.add(row)
    session.commit()
    return _to_prediction(row)


@app.get("/api/predictions", response_model=list[Prediction])
def list_predictions(limit: int = 50, session: Session = Depends(get_session)) -> list[Prediction]:
    limit = max(1, min(limit, 1000))
    rows = session.scalars(select(PredictionRow).order_by(PredictionRow.created_at.desc()).limit(limit))
    return [_to_prediction(r) for r in rows]


@app.get("/api/predictions/{prediction_id}/report")
def prediction_report(prediction_id: str, session: Session = Depends(get_session)) -> Response:
    row = session.get(PredictionRow, prediction_id)
    if row is None:
        raise HTTPException(404, "Prediction not found")
    return Response(
        build_report(_to_prediction(row)), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{prediction_id}.pdf"'},
    )


def _run_batch(job_id: str, variants: list[MLVariantInput], malformed: int) -> None:
    skipped = malformed
    reasons: list[str] = []
    done = 0
    with SessionLocal() as s:
        job = s.get(BatchJobRow, job_id)
        job.status = "running"
        s.commit()
        try:
            for start in range(0, len(variants), CHUNK):
                chunk = variants[start:start + CHUNK]
                try:
                    results = predictor.predict(chunk)
                    scored = chunk
                except BatchRejected as e:
                    skipped += len(e.reasons)
                    reasons.extend(e.reasons.values())
                    scored = [v for i, v in enumerate(chunk) if i not in e.reasons]
                    results = predictor.predict(scored)
                s.add_all(_to_row(v, r, job_id) for v, r in zip(scored, results))
                done += len(chunk)
                job.completed = done
                s.commit()
            job.status = "complete"
        except Exception as e:  # the job row is how the user finds out, so record anything
            s.rollback()
            job.status, job.error = "failed", f"Stopped after {done} variants: {e}"
            s.commit()
            return
        if skipped:
            example = f" Example: {reasons[0]}." if reasons else ""
            job.error = f"{skipped} skipped.{example}"
        s.commit()


@app.get("/api/batch", response_model=list[BatchJob])
def list_batch_jobs(session: Session = Depends(get_session)) -> list[BatchJob]:
    rows = session.scalars(select(BatchJobRow).order_by(BatchJobRow.submitted_at.desc()).limit(100))
    return [_to_job(r) for r in rows]


@app.post("/api/batch", response_model=BatchJob)
async def submit_batch(tasks: BackgroundTasks, file: UploadFile = File(...),
                       session: Session = Depends(get_session)) -> BatchJob:
    filename = file.filename or "upload"
    try:
        variants, malformed = parse_upload(filename, await file.read())
    except UploadError as e:
        raise HTTPException(400, str(e)) from e
    row = BatchJobRow(job_id=f"job_{uuid.uuid4().hex[:8]}", filename=filename, status="queued",
                      total=len(variants), completed=0, submitted_at=_now())
    session.add(row)
    session.commit()
    tasks.add_task(_run_batch, row.job_id, variants, malformed)
    return _to_job(row)


@app.get("/api/metrics", response_model=list[ModelMetrics])
def metrics() -> list[ModelMetrics]:
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else []
