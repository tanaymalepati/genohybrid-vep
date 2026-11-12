"""Reference lookups for scoring a variant, with a local cache.

Sequence comes from the UCSC API (Ensembl as fallback), conservation from UCSC's
public bigWig tracks read over HTTP, gene and consequence from Ensembl VEP.
Answers never change for a fixed assembly, so each is fetched once.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyBigWig

UCSC_SEQ = "https://api.genome.ucsc.edu/getData/sequence"
ENSEMBL = "https://rest.ensembl.org"
TRACKS = {
    "phylop": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
    "phastcons": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
}
CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


class AnnotationUnavailable(RuntimeError):
    """A reference service could not be reached; the variant itself may be fine."""


def normalise_chrom(chrom: str) -> str:
    c = str(chrom).strip()
    c = c[3:] if c.lower().startswith("chr") else c
    c = c.upper()
    return "MT" if c == "M" else c


def _ucsc_name(chrom: str) -> str:
    return "chrM" if chrom == "MT" else f"chr{chrom}"


class Annotator:
    def __init__(self, cache_path: Path) -> None:
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS position (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vep (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._db_lock = threading.Lock()
        self._local = threading.local()
        # Kept alive so each worker reuses its open bigWig connections.
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._http = httpx.Client(timeout=12, headers={"Accept": "application/json"})

    def _get(self, table: str, key: str):
        with self._db_lock:
            row = self._db.execute(f"SELECT value FROM {table} WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, table: str, key: str, value) -> None:
        with self._db_lock:
            self._db.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?,?)", (key, json.dumps(value))
            )
            self._db.commit()

    def _request(self, method: str, url: str, attempts: int = 3, **kw) -> httpx.Response | None:
        """Retries transient failures. Returns None for a definite 4xx rejection."""
        for attempt in range(attempts):
            try:
                r = self._http.request(method, url, **kw)
            except httpx.HTTPError:
                r = None
            if r is not None:
                if r.status_code == 200:
                    return r
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return None
            if attempt < attempts - 1:
                time.sleep(1.0 * 2**attempt)
        raise AnnotationUnavailable(f"{url} is not responding")

    def _sequence(self, chrom: str, pos: int) -> str | None:
        start = pos - 101  # 0-based start of the 201 bp window centred on pos
        if start < 0:
            return None
        try:
            r = self._request(
                "GET", UCSC_SEQ, attempts=1,
                params={"genome": "hg38", "chrom": _ucsc_name(chrom), "start": start, "end": start + 201},
            )
            return r.json().get("dna", "").upper() if r else None
        except AnnotationUnavailable:
            r = self._request(
                "GET", f"{ENSEMBL}/sequence/region/human/{chrom}:{start + 1}..{start + 201}:1",
                params={"coord_system_version": "GRCh38"},
            )
            return r.json().get("seq", "").upper() if r else None

    def _track(self, name: str, chrom: str, pos: int) -> float | None:
        for attempt in range(3):
            handles = getattr(self._local, "handles", None)
            if handles is None:
                handles = self._local.handles = {}
            try:
                if name not in handles:
                    handles[name] = pyBigWig.open(TRACKS[name])
                value = handles[name].values(_ucsc_name(chrom), pos - 1, pos)[0]
                return None if value is None or math.isnan(value) else float(value)
            except (RuntimeError, OSError, TypeError):
                handles.pop(name, None)
                time.sleep(1.5 * 2**attempt)
        raise AnnotationUnavailable(f"{name} track is not responding")

    def position(self, chrom: str, pos: int) -> dict | None:
        """Returns seq201/phylop/phastcons, or None if the position is off the chromosome."""
        key = f"{chrom}:{pos}"
        cached = self._get("position", key)
        if cached is not None:
            return cached
        seq = self._sequence(chrom, pos)
        if not seq or len(seq) != 201:
            return None
        value = {
            "seq201": seq,
            "phylop": self._track("phylop", chrom, pos),
            "phastcons": self._track("phastcons", chrom, pos),
        }
        self._put("position", key, value)
        return value

    def positions(self, keys: list[tuple[str, int]]) -> dict[tuple[str, int], dict | None]:
        unique = list(dict.fromkeys(keys))
        return dict(zip(unique, self._pool.map(lambda k: self.position(*k), unique)))

    def vep(self, items: list[tuple[str, int, str, str]]) -> dict[tuple, dict]:
        """(chrom, pos, ref, alt) -> {gene, raw_consequence} from Ensembl VEP, batched."""
        out, missing = {}, []
        for item in dict.fromkeys(items):
            cached = self._get("vep", "%s:%s:%s>%s" % item)
            if cached is not None:
                out[item] = cached
            else:
                missing.append(item)
        for i in range(0, len(missing), 200):
            part = missing[i:i + 200]
            r = self._request(
                "POST", f"{ENSEMBL}/vep/human/region", timeout=60,
                json={"variants": ["%s %s . %s %s . . ." % v for v in part]},
            )
            found = {}
            for rec in r.json() if r else []:
                allele = rec.get("allele_string", "/").split("/")
                found[(str(rec.get("seq_region_name")), int(rec.get("start", 0)), allele[0], allele[-1])] = rec
            for item in part:
                rec = found.get(item)
                value = _summarise_vep(rec, item[3]) if rec else {"gene": "", "raw_consequence": ""}
                self._put("vep", "%s:%s:%s>%s" % item, value)
                out[item] = value
        return out


def _summarise_vep(rec: dict, alt: str) -> dict:
    tx = [t for t in rec.get("transcript_consequences", []) if t.get("variant_allele") in (None, alt)]
    terms = sorted({c for t in tx for c in t.get("consequence_terms", [])})
    named = [t for t in tx if t.get("gene_symbol")]
    coding = [t for t in named if t.get("biotype") == "protein_coding"]
    gene = (coding or named or [{}])[0].get("gene_symbol", "")
    return {"gene": gene, "raw_consequence": "&".join(terms) or rec.get("most_severe_consequence", "")}
