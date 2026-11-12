"""The tri-modal GenoHybrid-VEP model behind the `Predictor` contract.

Model definitions, preprocessing and graph construction follow predict.py from
github.com/jayatigupta05/VEP (commit 38f68f5), so results match that pipeline.

- probabilities: 0.40 biomarker + 0.30 graph + 0.30 sequence, dropout off, then
  temperature scaled (fitted by compute_metrics.py). Scaling never changes the call.
- uncertainty: std-dev of the predicted class over MC-Dropout passes.
- shap: exact Shapley values over five input groups, on the log-odds of the
  predicted class, where a group is "absent" when replaced by its training
  median / unknown category / no flanking sequence / no graph neighbours.
"""
from __future__ import annotations

import itertools
import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

from .annotation import CHROMS, Annotator, normalise_chrom
from .predictor import BatchRejected, PredictionResult, ShapContribution, VariantInput

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_VERSION = "genohybrid-trimodal-38f68f5"
SEED = 42
MC_PASSES = 30
WINDOW = 101
W_BIO, W_GRAPH, W_SEQ = 0.40, 0.30, 0.30
CLASSES = ["Benign", "Pathogenic", "VUS"]
BIO_NUMERIC = ["phylop", "phastcons", "Tab_Review_Score", "Tab_Origin_Numeric", "Tab_Somatic_Impact"]
GRAPH_NUMERIC = ["phylop", "phastcons", "Tab_Origin_Numeric", "Tab_Somatic_Impact"]
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
K_SAME, K_CONS, K_PROX, MAX_BP = 4, 3, 2, 5000


class SequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv1d(4, 32, 7, padding=3), nn.BatchNorm1d(32)
        self.conv2, self.bn2 = nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64)
        self.conv3, self.bn3 = nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64)
        self.pool, self.dropout = nn.MaxPool1d(2), nn.Dropout(0.30)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(self.dropout(torch.cat([x.mean(2), x.max(2).values], 1)))


class SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SequenceEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.30), nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


class BiomarkerEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 32), nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class BiomarkerModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = BiomarkerEncoder(dim)
        self.classifier = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.30), nn.Linear(32, 3))

    def forward(self, x):
        return self.classifier(self.encoder(x))


class GraphEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1, self.norm1 = GCNConv(dim, 32), nn.BatchNorm1d(32)
        self.conv2, self.norm2 = GCNConv(32, 32), nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.30)

    def forward(self, x, edges):
        x = self.dropout(F.relu(self.norm1(self.conv1(x, edges))))
        return F.relu(self.norm2(self.conv2(x, edges)))


class GraphModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = GraphEncoder(dim)
        self.classifier = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.30), nn.Linear(32, 3))

    def forward(self, x, edges):
        return self.classifier(self.encoder(x, edges))


def consequence_category(raw: str) -> str:
    t = str(raw).lower()
    if "missense_variant" in t:
        return "missense"
    if "stop_gained" in t or "stop_lost" in t or "start_lost" in t:
        return "nonsense_or_stop"
    if "synonymous_variant" in t:
        return "synonymous"
    if "splice" in t:
        return "splice"
    if "intron_variant" in t:
        return "intron"
    if "5_prime_utr_variant" in t or "3_prime_utr_variant" in t:
        return "utr"
    if "upstream_gene_variant" in t or "downstream_gene_variant" in t:
        return "near_gene"
    return "unknown" if not t.strip() else "other"


def train_split(df: pd.DataFrame) -> pd.DataFrame:
    """The development training rows: the same two gene-grouped splits and seed."""
    df = df.copy()
    df["group_id"] = df["GeneSymbol"].astype("string")
    missing = df["group_id"].isna() | (df["group_id"].fillna("").str.len() == 0)
    df.loc[missing, "group_id"] = (
        "NO_GENE_" + df.loc[missing, "Chromosome"].astype(str) + "_"
        + (df.loc[missing, "Start"] // 1_000_000).astype(str)
    )
    outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    idx, _ = next(outer.split(df, y=df["Label"], groups=df["group_id"]))
    trainval = df.iloc[idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1111, random_state=SEED)
    idx, _ = next(inner.split(trainval, y=trainval["Label"], groups=trainval["group_id"]))
    return trainval.iloc[idx].copy().reset_index(drop=True)


class Preprocessor:
    def __init__(self, train: pd.DataFrame, numeric: list[str]):
        self.numeric = numeric
        self.medians = {
            f: (0.0 if pd.isna(train[f].median()) else float(train[f].median())) for f in numeric
        }
        self.categories = sorted(train["consequence_category"].fillna("unknown").astype(str).unique())
        self.scaler = StandardScaler().fit(self._numeric(train).astype(np.float32))

    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.numeric].copy()
        for f in ("phylop", "phastcons"):
            out[f"{f}_missing"] = df[f].isna().astype(np.float32)
        for f in self.numeric:
            out[f] = out[f].astype(float).fillna(self.medians[f])
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.reset_index(drop=True)
        scaled = self.scaler.transform(self._numeric(df).astype(np.float32)).astype(np.float32)
        index = {c: i for i, c in enumerate(self.categories)}
        onehot = np.zeros((len(df), len(self.categories)), dtype=np.float32)
        for i, c in enumerate(df["consequence_category"].fillna("unknown").astype(str)):
            if c in index:
                onehot[i, index[c]] = 1.0
        return np.concatenate([scaled, onehot], axis=1)

    @property
    def dim(self) -> int:
        return len(self.numeric) + 2 + len(self.categories)


def encode_sequence(seq201: str, ref: str, alt: str, flanks: bool = True) -> np.ndarray:
    half = WINDOW // 2
    x = np.zeros((4, WINDOW), dtype=np.float32)
    if flanks:
        for i, base in enumerate(seq201[100 - half:100 + half + 1]):
            if base in BASE:
                x[BASE[base], i] = 1.0
    x[:, half] = 0.0
    x[BASE[ref], half] = 1.0
    x[BASE[alt], half] = -1.0
    return x


def build_edges(df: pd.DataFrame, medians: dict) -> torch.Tensor:
    """Same-gene edges: same consequence, similar conservation, nearby position."""
    t = df.reset_index(drop=True)
    gene = t["GeneSymbol"].fillna("").astype(str).str.strip()
    cons = t["consequence_category"].fillna("unknown").astype(str)
    phylop = t["phylop"].astype(float).fillna(medians["phylop"])
    phast = t["phastcons"].astype(float).fillna(medians["phastcons"])
    start = t["Start"].astype(int)
    valid = (gene != "") & (gene != "-") & (gene.str.lower() != "nan")
    edges: set[tuple[int, int]] = set()

    def link(i, j):
        if int(i) != int(j):
            edges.add((int(i), int(j)))
            edges.add((int(j), int(i)))

    def nearest(idx, pos, reach, keep, limit=None):
        for i in range(len(idx)):
            near = []
            for off in range(1, reach + 1):
                for j in (i - off, i + off):
                    if 0 <= j < len(idx):
                        d = abs(int(pos[i]) - int(pos[j]))
                        if limit is None or d <= limit:
                            near.append((d, idx[j]))
            near.sort(key=lambda x: x[0])
            for _, n in near[:keep]:
                link(idx[i], n)

    for _, members in gene[valid].groupby(gene[valid], sort=False):
        rows = members.index.to_numpy()
        if len(rows) < 2:
            continue
        for _, sub in cons.loc[rows].groupby(cons.loc[rows], sort=False):
            if len(sub) >= 2:
                ordered = start.loc[sub.index].sort_values()
                nearest(ordered.index.to_numpy(), ordered.to_numpy(), K_SAME, K_SAME)
        c = np.column_stack([phylop.loc[rows], phast.loc[rows]]).astype(np.float32)
        s = c.std(axis=0, keepdims=True)
        s[s < 1e-6] = 1.0
        c = (c - c.mean(axis=0, keepdims=True)) / s
        k = min(K_CONS + 1, len(rows))
        if k > 1:
            _, nn_idx = NearestNeighbors(n_neighbors=k).fit(c).kneighbors(c)
            for i, row in enumerate(nn_idx):
                for j in [j for j in row if j != i][:K_CONS]:
                    link(rows[i], rows[j])
        ordered = start.loc[rows].sort_values()
        nearest(ordered.index.to_numpy(), ordered.to_numpy(), K_PROX + 2, K_PROX, MAX_BP)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.asarray(list(edges), dtype=np.int64).T, dtype=torch.long)


def _load(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))


class GenoHybridPredictor:
    model_version = MODEL_VERSION

    def __init__(self, artifacts: Path = ARTIFACTS) -> None:
        full = pd.read_parquet(artifacts / "genohybrid_features.parquet")
        self.train = train_split(full)
        self.bio_pre = Preprocessor(self.train, BIO_NUMERIC)
        self.graph_pre = Preprocessor(self.train, GRAPH_NUMERIC)
        ck = artifacts / "checkpoints"
        seq_state = _load(ck / "sequence_cnn_best.pt")
        for suffix in ("weight", "bias"):
            if f"encoder.embedding.{suffix}" in seq_state:
                seq_state[f"encoder.fc.0.{suffix}"] = seq_state.pop(f"encoder.embedding.{suffix}")
        self.seq_model = SequenceModel()
        self.seq_model.load_state_dict(seq_state)
        self.bio_model = BiomarkerModel(self.bio_pre.dim)
        self.bio_model.load_state_dict(_load(ck / "biomarker_mlp_best.pt"))
        self.graph_model = GraphModel(self.graph_pre.dim)
        self.graph_model.load_state_dict(_load(ck / "gcn_v2_best.pt"))
        self.models = (self.seq_model, self.bio_model, self.graph_model)
        self.temperature = json.loads((artifacts / "calibration.json").read_text())["temperature"]
        self.annotator = Annotator(artifacts / "annotation_cache.sqlite")
        self._lock = threading.Lock()

    def _calibrate(self, p: np.ndarray) -> np.ndarray:
        q = np.exp(np.log(np.clip(p, 1e-9, 1)) / self.temperature)
        return q / q.sum(-1, keepdims=True)

    def _dropout(self, on: bool) -> None:
        for model in self.models:
            model.eval()
            if on:
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.train()

    def _annotate(self, variants: list[VariantInput]) -> pd.DataFrame:
        rejected: dict[int, str] = {}
        norm = []
        for i, v in enumerate(variants):
            chrom, ref, alt = normalise_chrom(v.chrom), str(v.ref).upper(), str(v.alt).upper()
            if v.assembly != "GRCh38":
                rejected[i] = "this model only supports GRCh38 coordinates"
            elif chrom not in CHROMS:
                rejected[i] = f"unknown chromosome {v.chrom}"
            elif len(ref) != 1 or len(alt) != 1 or ref not in BASE or alt not in BASE:
                rejected[i] = "only single-nucleotide variants (A, C, G, T) can be scored"
            elif ref == alt:
                rejected[i] = "reference and alternate alleles are the same"
            norm.append((chrom, int(v.pos), ref, alt))
        ok = [i for i in range(len(variants)) if i not in rejected]
        positions = self.annotator.positions([norm[i][:2] for i in ok])
        for i in ok:
            info = positions[norm[i][:2]]
            if info is None:
                rejected[i] = f"position {norm[i][1]} is outside chromosome {norm[i][0]}"
            elif info["seq201"][100] != norm[i][2]:
                rejected[i] = (
                    f"reference allele {norm[i][2]} does not match GRCh38, "
                    f"which has {info['seq201'][100]} at this position"
                )
        if rejected:
            raise BatchRejected(rejected)
        need_vep = [norm[i] for i in ok if variants[i].consequence is None]
        vep = self.annotator.vep(need_vep) if need_vep else {}
        rows = []
        for i in ok:
            chrom, pos, ref, alt = norm[i]
            info, v = positions[(chrom, pos)], variants[i]
            raw = v.consequence if v.consequence is not None else vep[norm[i]]["raw_consequence"]
            gene = v.gene or ("" if v.consequence is not None else vep[norm[i]]["gene"])
            rows.append({
                "Chromosome": chrom, "Start": pos,
                "ReferenceAlleleVCF": ref, "AlternateAlleleVCF": alt,
                "GeneSymbol": gene, "molecular_consequence_raw": raw,
                "consequence_category": consequence_category(raw), "seq201": info["seq201"],
                "phylop": np.nan if info["phylop"] is None else info["phylop"],
                "phastcons": np.nan if info["phastcons"] is None else info["phastcons"],
                # ClinVar metadata is unknown for a new variant: training medians apply.
                "Tab_Review_Score": np.nan, "Tab_Origin_Numeric": np.nan, "Tab_Somatic_Impact": np.nan,
            })
        return pd.DataFrame(rows)

    def _mask(self, df: pd.DataFrame, pre: Preprocessor, phylop, phast, cons) -> pd.DataFrame:
        out = df.copy()
        if not phylop:
            out["phylop"] = pre.medians["phylop"]
        if not phast:
            out["phastcons"] = pre.medians["phastcons"]
        if not cons:
            out["consequence_category"] = "unknown"
        return out

    @torch.no_grad()
    def predict(self, variants: list[VariantInput]) -> list[PredictionResult]:
        if not variants:
            return []
        with self._lock:
            q = self._annotate(variants)
            return self._score(q)

    def _score(self, q: pd.DataFrame) -> list[PredictionResult]:
        n = len(q)
        self._dropout(False)
        refs, alts = q["ReferenceAlleleVCF"], q["AlternateAlleleVCF"]
        seq_x = {
            f: torch.tensor(np.stack([
                encode_sequence(s, r, a, f) for s, r, a in zip(q["seq201"], refs, alts)
            ])) for f in (True, False)
        }
        genes = {g for g in q["GeneSymbol"].astype(str).str.strip() if g not in ("", "-")}
        context = self.train[self.train["GeneSymbol"].fillna("").astype(str).isin(genes)]
        combined = pd.concat([context, q], ignore_index=True, sort=False)
        is_q = np.zeros(len(combined), dtype=bool)
        is_q[len(context):] = True
        edges = build_edges(combined, self.graph_pre.medians)
        q_ids = torch.tensor(np.flatnonzero(is_q))
        no_q_edges = (
            edges[:, ~(torch.isin(edges[0], q_ids) | torch.isin(edges[1], q_ids))]
            if edges.numel() else edges
        )
        base_x = self.graph_pre.transform(combined)

        def graph_input(p, h, c):
            x = base_x.copy()
            x[is_q] = self.graph_pre.transform(self._mask(q, self.graph_pre, p, h, c))
            return torch.tensor(x)

        def graph_probs(x, e):
            return torch.softmax(self.graph_model(x, e), 1)[torch.from_numpy(is_q)].numpy()

        full_bio = torch.tensor(self.bio_pre.transform(q))
        full_graph = graph_input(True, True, True)

        # Coalition tables for the Shapley values: bio has 3 inputs, graph 4, sequence 1.
        seq_tab = {f: torch.softmax(self.seq_model(seq_x[f]), 1).numpy() for f in (True, False)}
        bio_tab, graph_tab = {}, {}
        for p, h, c in itertools.product((True, False), repeat=3):
            x = torch.tensor(self.bio_pre.transform(self._mask(q, self.bio_pre, p, h, c)))
            bio_tab[(p, h, c)] = torch.softmax(self.bio_model(x), 1).numpy()
            gx = graph_input(p, h, c)
            for e, ee in ((True, edges), (False, no_q_edges)):
                graph_tab[(p, h, c, e)] = graph_probs(gx, ee)

        def fused(s):  # s = (phylop, phastcons, consequence, sequence, neighbours)
            return self._calibrate(
                W_BIO * bio_tab[s[:3]] + W_GRAPH * graph_tab[(*s[:3], s[4])] + W_SEQ * seq_tab[s[3]]
            )

        final = fused((True, True, True, True, True))
        predicted = final.argmax(1)

        # MC Dropout: repeatable, so the same variant always reports the same sigma.
        torch.manual_seed(SEED)
        self._dropout(True)
        passes = [
            self._calibrate(
                W_BIO * torch.softmax(self.bio_model(full_bio), 1).numpy()
                + W_GRAPH * graph_probs(full_graph, edges)
                + W_SEQ * torch.softmax(self.seq_model(seq_x[True]), 1).numpy()
            )
            for _ in range(MC_PASSES)
        ]
        self._dropout(False)
        spread = np.stack(passes).std(axis=0)

        # Exact Shapley values on the log-odds of the predicted class.
        members = 5
        weight = [
            math.factorial(k) * math.factorial(members - k - 1) / math.factorial(members)
            for k in range(members)
        ]
        coalitions = list(itertools.product((True, False), repeat=members))
        tables = {s: fused(s) for s in coalitions}
        rows = np.arange(n)

        def logit(s):
            p = np.clip(tables[s][rows, predicted], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        logits = {s: logit(s) for s in coalitions}
        phi = np.zeros((n, members))
        for s in coalitions:
            for f in range(members):
                if s[f]:
                    off = list(s)
                    off[f] = False
                    phi[:, f] += weight[sum(off)] * (logits[s] - logits[tuple(off)])

        neighbours = np.zeros(n, dtype=int)
        if edges.numel():
            start = len(context)
            ids, counts = torch.unique(edges[0], return_counts=True)
            for i, c in zip(ids.tolist(), counts.tolist()):
                if i >= start:
                    neighbours[i - start] = c

        results = []
        for i, row in enumerate(q.itertuples(index=False)):
            label = CLASSES[int(predicted[i])]
            shap = [
                ShapContribution("phyloP100way", _r(phi[i, 0]), "conservation", _raw(row.phylop)),
                ShapContribution("phastCons100way", _r(phi[i, 1]), "conservation", _raw(row.phastcons)),
                ShapContribution("consequence", _r(phi[i, 2]), "annotation", row.consequence_category),
                ShapContribution("sequence_context", _r(phi[i, 3]), "embedding", f"{WINDOW} bp"),
                ShapContribution("gene_graph_links", _r(phi[i, 4]), "annotation", int(neighbours[i])),
            ]
            shap.sort(key=lambda c: -abs(c.value))
            probs = {c: float(final[i, j]) for j, c in enumerate(CLASSES)}
            total = sum(probs.values())
            result = PredictionResult(
                predicted_class=label,
                probabilities={c: round(p / total, 6) for c, p in probs.items()},
                uncertainty=round(float(spread[i, int(predicted[i])]), 6),
                mc_passes=MC_PASSES,
                shap=shap,
                model_version=MODEL_VERSION,
                gene=row.GeneSymbol or None,
                arm_probabilities={
                    "biomarker": _arm(bio_tab[(True, True, True)][i]),
                    "graph": _arm(graph_tab[(True, True, True, True)][i]),
                    "sequence": _arm(seq_tab[True][i]),
                },
            )
            result.validate()
            results.append(result)
        return results


def _r(x: float) -> float:
    return round(float(x), 4)


def _raw(x: float):
    return None if pd.isna(x) else round(float(x), 3)


def _arm(p) -> dict[str, float]:
    return {c: float(p[j]) for j, c in enumerate(CLASSES)}
