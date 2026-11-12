"""PDF report for one prediction."""
from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import Prediction


def _table(rows: list[list[str]]) -> Table:
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.lightgrey),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
    ]))
    return table


def build_report(p: Prediction) -> bytes:
    styles = getSampleStyleSheet()
    v = p.variant
    story = [
        Paragraph("GenoHybrid-VEP variant report", styles["Title"]),
        Paragraph(
            f"chr{v.chrom}:{v.pos} {v.ref}&gt;{v.alt} ({v.assembly})"
            + (f", gene {v.gene}" if v.gene else ""), styles["Heading3"],
        ),
        Spacer(1, 8),
        Paragraph(f"Predicted class: <b>{p.predicted_class}</b>", styles["Normal"]),
        Paragraph(f"Confidence: {p.confidence:.1%}", styles["Normal"]),
        Paragraph(f"Uncertainty (MC Dropout std-dev, {p.mc_passes} passes): {p.uncertainty:.1%}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Class probabilities", styles["Heading4"]),
        _table([["Class", "Probability"]] + [[c, f"{x:.1%}"] for c, x in p.probabilities.items()]),
        Spacer(1, 12),
        Paragraph("Feature attributions (log-odds toward the predicted class)", styles["Heading4"]),
        _table([["Feature", "Value", "Contribution"]]
               + [[s.feature, "-" if s.raw is None else str(s.raw), f"{s.value:+.3f}"] for s in p.shap]),
        Spacer(1, 16),
        Paragraph(
            f"Model {p.model_version}. Generated from prediction {p.id}, scored "
            f"{p.created_at:%Y-%m-%d %H:%M} UTC. Research prototype: not a clinical "
            "diagnostic and not to be used as the basis for medical decisions.",
            styles["Italic"],
        ),
    ]
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title="GenoHybrid-VEP report").build(story)
    return buffer.getvalue()
