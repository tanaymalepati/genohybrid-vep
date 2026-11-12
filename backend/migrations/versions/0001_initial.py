"""predictions and batch_jobs"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "batch_jobs",
        sa.Column("job_id", sa.String(32), primary_key=True),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("total", sa.Integer, nullable=False),
        sa.Column("completed", sa.Integer, nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error", sa.Text),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("chrom", sa.String(8), nullable=False),
        sa.Column("pos", sa.Integer, nullable=False),
        sa.Column("ref", sa.Text, nullable=False),
        sa.Column("alt", sa.Text, nullable=False),
        sa.Column("assembly", sa.String(8), nullable=False),
        sa.Column("gene", sa.Text),
        sa.Column("hgvs", sa.Text),
        sa.Column("predicted_class", sa.String(16), nullable=False),
        sa.Column("probabilities", JSONB, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("uncertainty", sa.Float, nullable=False),
        sa.Column("mc_passes", sa.Integer, nullable=False),
        sa.Column("shap", JSONB, nullable=False),
        sa.Column("arm_probabilities", JSONB, nullable=False),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("job_id", sa.String(32), sa.ForeignKey("batch_jobs.job_id")),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("batch_jobs")
