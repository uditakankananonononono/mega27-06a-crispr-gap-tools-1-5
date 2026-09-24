"""Loaders for the public datasets used across gaps 1-5.

Sources (see scripts/fetch_data.sh):
- Doench et al. 2016 FC+RES (5,310 30-mer guides, Azimuth mirror of the
  published dataset): score_drug_gene_rank in [0,1] is the measured efficacy.
- Doench et al. 2014/2016 V1 (2,144 mouse guides): 34-mer extended context,
  trimmed to the Azimuth 30-mer convention ([4bp][20 guide][NGG][3bp]);
  Percent Rank in [0,1] is the measured efficacy.
- crisprSQL (Stortz & Minary 2021, NAR gkaa885) 100720 release: 25,632
  experimentally measured guide/off-target pairs with cleavage frequencies,
  epigenetic tracks (CTCF, DNase, RRBS, H3K4me3, DRIP), thermodynamic
  energies, cell line and study provenance.

Alignment note: crisprSQL sequence lengths vary by study (20-25 nt). All rows
are left-aligned to the 5' end of the protospacer, verified empirically: for
the Finkelstein library (25-mers) o[:20] has the lowest mean mismatch count
vs the guide (10.2 vs 11.2-15.7 for shifted frames). We use the first 20 nt
as the protospacer for every pair; guide 23-mers carry the canonical NGG PAM
in positions 21-23 (100% end in .GG).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

EPIGEN_COLS = ("epigen_ctcf", "epigen_dnase", "epigen_rrbs", "epigen_h3k4me3", "epigen_drip")
ENERGY_COLS = ("energy_1", "energy_2", "energy_3", "energy_4", "energy_5")


@dataclass
class EfficacyDataset:
    sequences: list          # 30-mer guide contexts
    scores: np.ndarray       # float32 measured efficacy in [0, 1]
    source: str


@dataclass
class OfftargetDataset:
    guides: list             # 20-mer protospacers
    offtargets: list         # 20-mer protospacers (left-aligned, see module docstring)
    labels: np.ndarray       # int8: 1 if cleavage_freq > 0
    cleavage_freq: np.ndarray  # float32 measured cleavage frequency
    studies: list
    cell_lines: list
    pam: list                # guide 3-nt PAM where reported (else "")
    off_pam: list            # off-target 3-nt PAM where reported (else "")
    epigen: dict = field(default_factory=dict)   # col -> float32 array
    energies: dict = field(default_factory=dict)  # col -> float32 array
    source: str = "crisprsql_100720"


def load_doench_fcres() -> EfficacyDataset:
    df = pd.read_csv(os.path.join(DATA_DIR, "FC_plus_RES_withPredictions.csv"), index_col=0)
    return EfficacyDataset(
        sequences=df["30mer"].tolist(),
        scores=df["score_drug_gene_rank"].to_numpy(dtype=np.float32),
        source="doench2016_fcres",
    )


def load_doench_v1() -> EfficacyDataset:
    df = pd.read_csv(os.path.join(DATA_DIR, "V1_suppl_data.txt"), sep="\t")
    ext = df["Extended Spacer(NNNN[20nt]NGGNNNNNNN)"]
    assert (ext.str.len() == 34).all(), "expected 34-mer extended spacers"
    return EfficacyDataset(
        sequences=[s[:30] for s in ext],  # Azimuth 30-mer convention
        scores=df["Percent Rank"].to_numpy(dtype=np.float32),
        source="doench2016_v1",
    )


def load_crisprsql() -> OfftargetDataset:
    df = pd.read_csv(os.path.join(DATA_DIR, "crisprsql", "100720.csv"))
    assert (df.grna_target_sequence.str.len() >= 20).all()
    assert (df.target_sequence.str.len() >= 20).all()
    pam = [g[20:23] if len(g) >= 23 else "" for g in df.grna_target_sequence]
    off_pam = [o[20:23] if len(o) >= 23 else "" for o in df.target_sequence]
    return OfftargetDataset(
        guides=[g[:20] for g in df.grna_target_sequence],
        offtargets=[o[:20] for o in df.target_sequence],
        labels=(df.cleavage_freq > 0).to_numpy(dtype=np.int8),
        cleavage_freq=df.cleavage_freq.to_numpy(dtype=np.float32),
        studies=df.study_name.tolist(),
        cell_lines=df.cell_line.tolist(),
        pam=pam,
        off_pam=off_pam,
        epigen={c: df[c].to_numpy(dtype=np.float32) for c in EPIGEN_COLS},
        energies={c: df[c].to_numpy(dtype=np.float32) for c in ENERGY_COLS},
    )


_INT2BASE = {1: "A", 2: "C", 3: "G", 4: "T", 0: "N", 5: "N"}


def load_deephf(variant: str = "wt", max_n: int | None = None, seed: int = 0) -> EfficacyDataset:
    """DeepHF screen data (Wang et al. 2019, Nat Commun): ~55.6k guides with
    measured indel frequencies for WT-SpCas9 ('wt'), eSpCas9 ('esp') or
    SpCas9-HF1 ('hf') in U2OS cells. A genuinely divergent assay from the
    Doench family - the hard case for cross-dataset generalization (gap 1).
    """
    import pickle
    path = os.path.join(DATA_DIR, "raw", "public_data_crisprCas9", "data", "deepHF",
                        f"{variant}_seq_data_array.pkl")
    with open(path, "rb") as fh:
        seq_int, _biofeat, indel = pickle.load(fh)
    seqs = ["".join(_INT2BASE[b] for b in row) for row in seq_int]
    scores = np.clip(np.asarray(indel, dtype=np.float32), 0.0, 1.0)
    if max_n is not None and len(seqs) > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(seqs), max_n, replace=False))
        seqs = [seqs[i] for i in idx]
        scores = scores[idx]
    return EfficacyDataset(sequences=seqs, scores=scores, source=f"deephf_{variant}")
