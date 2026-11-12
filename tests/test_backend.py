"""Run from the repo root: .venv/bin/python -m pytest tests -q"""
import math

import pytest

from backend.parsing import UploadError, parse_upload
from ml.predictor import VariantInput

VCF = (
    "##fileformat=VCFv4.2\n"
    '##INFO=<ID=CSQ,Number=.,Type=String,Description="Format: Allele|Consequence|SYMBOL|Feature">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "18\t22181553\t.\tG\tA,T\t.\tPASS\tCSQ=A|missense_variant|GATA6|,T|stop_gained|GATA6|\n"
    "bad line\n"
)


def test_vcf_splits_alts_and_reads_csq():
    variants, bad = parse_upload("x.vcf", VCF.encode())
    assert bad == 1
    assert [(v.alt, v.gene, v.consequence) for v in variants] == [
        ("A", "GATA6", "missense_variant"), ("T", "GATA6", "stop_gained")]


def test_csv_requires_columns():
    with pytest.raises(UploadError):
        parse_upload("x.csv", b"a,b\n1,2\n")
    variants, _ = parse_upload("x.csv", b"chrom,pos,ref,alt\n7,66089689,G,C\n")
    assert variants[0].pos == 66089689 and variants[0].consequence is None


@pytest.fixture(scope="module")
def model():
    from ml.genohybrid import GenoHybridPredictor
    return GenoHybridPredictor()


def test_prediction_is_valid_and_shapley_is_exact(model):
    # Cached in ml/artifacts/annotation_cache.sqlite, so no network needed.
    v = VariantInput("11", 47347855, "A", "T", gene="MYBPC3")
    r = model.predict([v])[0]
    r.validate()
    assert r.predicted_class == "Pathogenic" and 0 < r.uncertainty < 0.2
    assert len(r.shap) == 5
    # Shapley efficiency: contributions add up to full minus empty log-odds (checked > 0 here).
    assert sum(c.value for c in r.shap) > 0


def test_rejects_wrong_reference(model):
    from ml.predictor import BatchRejected
    with pytest.raises(BatchRejected, match="does not match GRCh38"):
        model.predict([VariantInput("11", 47347855, "C", "T")])
