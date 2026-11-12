"""Turns an uploaded VCF or CSV into variants to score."""
from __future__ import annotations

import csv
import gzip
import io
import re

from ml.predictor import VariantInput

MAX_VARIANTS = 5000
MAX_BYTES = 50 * 1024 * 1024


class UploadError(ValueError):
    pass


def _read(raw: bytes) -> str:
    if raw[:2] == b"\x1f\x8b":
        try:
            with gzip.open(io.BytesIO(raw)) as f:
                raw = f.read(MAX_BYTES + 1)
        except OSError as e:
            raise UploadError(f"could not decompress the file: {e}") from e
        if len(raw) > MAX_BYTES:
            raise UploadError("the decompressed file is larger than 50 MB")
    return raw.decode("utf-8", errors="replace")


def _annotation(info: str, alt: str, csq_fields: list[str] | None) -> tuple[str | None, str | None]:
    """(gene, raw consequence) from VEP CSQ or SnpEff ANN; (None, None) if neither is present."""
    fields = dict(p.split("=", 1) for p in info.split(";") if "=" in p)
    gene, terms = "", set()
    if csq_fields and "CSQ" in fields:
        for entry in fields["CSQ"].split(","):
            row = dict(zip(csq_fields, entry.split("|")))
            if row.get("Allele") and row["Allele"] != alt:
                continue
            gene = gene or row.get("SYMBOL") or row.get("Gene") or ""
            terms.update(t for t in [row.get("Consequence")] if t)
    if not terms and "ANN" in fields:
        for entry in fields["ANN"].split(","):
            values = entry.split("|")
            if len(values) >= 4 and (not values[0] or values[0] == alt):
                gene = gene or values[3]
                terms.update(t for t in [values[1]] if t)
    if not terms:
        return None, None
    return gene or None, "&".join(sorted(terms))


def _parse_vcf(text: str) -> tuple[list[VariantInput], int]:
    variants, bad, csq_fields = [], 0, None
    for line in text.splitlines():
        if line.startswith("##INFO=<ID=CSQ"):
            m = re.search(r"Format:\s*([^\">]+)", line)
            csq_fields = [f.strip() for f in m.group(1).split("|")] if m else None
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5 or not cols[1].isdigit():
            bad += 1
            continue
        info = cols[7] if len(cols) > 7 else ""
        for alt in cols[4].split(","):
            gene, cons = _annotation(info, alt, csq_fields)
            variants.append(VariantInput(cols[0], int(cols[1]), cols[3], alt, gene=gene, consequence=cons))
    return variants, bad


def _parse_table(text: str, delimiter: str) -> tuple[list[VariantInput], int]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    columns = {(c or "").strip().lower() for c in reader.fieldnames or []}
    if not {"chrom", "pos", "ref", "alt"} <= columns:
        raise UploadError("the file needs chrom, pos, ref and alt columns")
    variants, bad = [], 0
    for row in reader:
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items() if k}
        if not row["pos"].isdigit() or not row["chrom"]:
            bad += 1
            continue
        variants.append(VariantInput(row["chrom"], int(row["pos"]), row["ref"], row["alt"], gene=row.get("gene") or None))
    return variants, bad


def parse_upload(filename: str, raw: bytes) -> tuple[list[VariantInput], int]:
    """Returns the variants and the number of malformed lines that were skipped."""
    if len(raw) > MAX_BYTES:
        raise UploadError("the file is larger than 50 MB")
    text = _read(raw)
    name = filename.lower().removesuffix(".gz")
    if name.endswith(".csv"):
        variants, bad = _parse_table(text, ",")
    elif name.endswith((".tsv", ".txt")):
        variants, bad = _parse_table(text, "\t")
    elif name.endswith(".vcf"):
        variants, bad = _parse_vcf(text)
    else:
        raise UploadError("upload a .vcf, .vcf.gz, .csv or .tsv file")
    if not variants:
        raise UploadError("no variants were found in the file")
    if len(variants) > MAX_VARIANTS:
        raise UploadError(f"the file has {len(variants):,} variants; the limit is {MAX_VARIANTS:,}")
    return variants, bad
