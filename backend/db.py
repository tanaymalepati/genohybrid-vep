"""PostgreSQL tables and session handling."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg:///genohybrid")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class BatchJobRow(Base):
    __tablename__ = "batch_jobs"
    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    total: Mapped[int] = mapped_column(Integer, default=0)
    completed: Mapped[int] = mapped_column(Integer, default=0)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class PredictionRow(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    chrom: Mapped[str] = mapped_column(String(8))
    pos: Mapped[int] = mapped_column(Integer)
    ref: Mapped[str] = mapped_column(Text)
    alt: Mapped[str] = mapped_column(Text)
    assembly: Mapped[str] = mapped_column(String(8))
    gene: Mapped[str | None] = mapped_column(Text, nullable=True)
    hgvs: Mapped[str | None] = mapped_column(Text, nullable=True)
    predicted_class: Mapped[str] = mapped_column(String(16))
    probabilities: Mapped[dict] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    uncertainty: Mapped[float] = mapped_column(Float)
    mc_passes: Mapped[int] = mapped_column(Integer)
    shap: Mapped[list] = mapped_column(JSONB)
    arm_probabilities: Mapped[dict] = mapped_column(JSONB, default=dict)
    model_version: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    job_id: Mapped[str | None] = mapped_column(ForeignKey("batch_jobs.job_id"), nullable=True)

    __table_args__ = (Index("ix_predictions_created_at", "created_at"),)


def migrate() -> None:
    here = Path(__file__).resolve().parent
    cfg = Config()
    cfg.set_main_option("script_location", str(here / "migrations"))
    command.upgrade(cfg, "head")


def get_session():
    with SessionLocal() as session:
        yield session
