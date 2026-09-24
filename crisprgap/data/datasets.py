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


# DeepHF pkl encoding, verified against the bundled biofeature GC column
# (corr -0.966): array is (n, 22); position 0 is a fixed sentinel (value 1);
# positions 1..21 are the 21-mer with 2=C, 3=G, 4=T, 5=A. An earlier
# mapping read 5 as N, zeroing ~25%% of positions - that was a bug.
_INT2BASE = {2: "C", 3: "G", 4: "T", 5: "A"}


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
    arr = np.asarray(seq_int)
    assert (arr[:, 0] == 1).all(), "DeepHF sentinel column changed - re-verify encoding"
    seqs = ["".join(_INT2BASE[b] for b in row[1:]) for row in arr]
    scores = np.clip(np.asarray(indel, dtype=np.float32), 0.0, 1.0)
    if max_n is not None and len(seqs) > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(seqs), max_n, replace=False))
        seqs = [seqs[i] for i in idx]
        scores = scores[idx]
    return EfficacyDataset(sequences=seqs, scores=scores, source=f"deephf_{variant}")


def load_deepspcas9(sheet: str = "HT_Cas9_Train", max_n: int | None = None,
                    seed: int = 0) -> "EfficacyDataset":
    """DeepSpCas9 high-throughput dataset (Kim et al. 2019, Sci Adv 5:eaax9249,
    Table S1; SRA SRP150719): 12,832 guide-contexts (30-mer: 4+20+3+3) with
    background-subtracted indel frequencies in HEK293T cells. A third assay
    family for cross-dataset generalization (gap 1).
    """
    import pandas as pd
    df = pd.read_excel(os.path.join(DATA_DIR, "aax9249_Table_S1.xlsx"), sheet_name=sheet)
    seq_col = "Target context sequence (4+20+3+3)"
    y_col = "Background subtracted indel (%)"
    df = df.dropna(subset=[seq_col, y_col])
    seqs = [s.upper().replace("U", "T") for s in df[seq_col]]
    scores = np.clip(df[y_col].to_numpy(dtype=np.float32) / 100.0, 0.0, 1.0)
    if max_n is not None and len(seqs) > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(seqs), max_n, replace=False))
        seqs = [seqs[i] for i in idx]
        scores = scores[idx]
    return EfficacyDataset(sequences=seqs, scores=scores, source=f"deepspcas9_{sheet.lower()}")


def load_crispron(sheet: str = "SpCas9_eff_Day 2", max_n: int | None = None,
                  seed: int = 0) -> "EfficacyDataset":
    """CRISPRon TRAP12K dataset (Xiang et al. 2021, Nat Commun 12:3238,
    Suppl. Data 4 / PMC8163799): 11,488 gRNAs with total indel efficiency
    in HEK293T (TRAP-seq surrogate assay, Day 2). A fourth assay family
    for cross-dataset generalization (gap 1). Sequences are 20-mer guides
    (no flanking context in the published table).
    """
    import pandas as pd
    df = pd.read_excel(os.path.join(DATA_DIR, "41467_2021_23576_MOESM4_ESM.xlsx"),
                       sheet_name=sheet)
    df = df.dropna(subset=["gRNA", "total_indel_eff"])
    seqs = [s.upper().replace("U", "T") for s in df["gRNA"]]
    scores = np.clip(df["total_indel_eff"].to_numpy(dtype=np.float32) / 100.0, 0.0, 1.0)
    if max_n is not None and len(seqs) > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(seqs), max_n, replace=False))
        seqs = [seqs[i] for i in idx]
        scores = scores[idx]
    return EfficacyDataset(sequences=seqs, scores=scores,
                           source="crispron_trap12k_" + sheet.lower().replace(" ", "_"))


PEGRNA_NUMERIC_COLS = (
    "Correction_Length", "Correction_Deletion", "Correction_Insertion", "Correction_Replacement",
    "RToverhangmatches", "RToverhanglength", "RTlength", "PBSlength",
    "RTmt", "RToverhangmt", "PBSmt", "protospacermt", "extensionmt",
    "deepeditposition", "Editing_Position",
)

PEGRNA_INTERVAL_COLS = (
    "protospacerlocation_only_initial", "PBSlocation",
    "RT_initial_location", "RT_mutated_location",
)


@dataclass
class PegrnaDataset:
    sequences: list          # 99-mer wide initial target context
    mutated: list            # 99-mer wide mutated target
    numerics: np.ndarray     # (n, len(PEGRNA_NUMERIC_COLS) + 3) float32: raw numerics + Correction_Type one-hot
    targets: dict            # {"HEK": (n,) float32, "K562": (n,) float32} edited fractions
    grp_ids: list            # pegRNA group ids for grouped splits
    numeric_col_names: list = field(default_factory=list)
    source: str = "pridict2_23k_v1"


def load_pridict() -> PegrnaDataset:
    """PRIDICT2 processed pegRNA library (Koeppel et al., bioRxiv 2023.10.09.561414;
    underlying screens from Mathis et al. 2023, Nat Biotechnol): 22,956 pegRNAs,
    99-mer wide target contexts, engineered features, and measured editing
    efficiencies in HEK293T and K562 cells. Basis of the gap-3 transfer tool.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "pridict", "data_23k_v1.csv"))
    ctype = pd.get_dummies(df["Correction_Type"]).astype(np.float32)
    num = df[list(PEGRNA_NUMERIC_COLS)].astype(np.float32)
    import ast
    interval_feats, interval_names = [], []
    for c in PEGRNA_INTERVAL_COLS:
        parsed = df[c].map(ast.literal_eval)
        interval_feats.append(np.array([p[0] for p in parsed], dtype=np.float32))
        interval_feats.append(np.array([p[1] for p in parsed], dtype=np.float32))
        interval_names += [f"{c}_start", f"{c}_end"]
    numerics = np.concatenate([num.to_numpy()] + [np.stack(interval_feats, 1).astype(np.float32)] + [ctype.to_numpy()], axis=1)
    names = list(PEGRNA_NUMERIC_COLS) + interval_names + [f"ctype_{c}" for c in ctype.columns]
    assert not np.isnan(numerics).any(), "unexpected NaN in pegRNA numerics"
    return PegrnaDataset(
        sequences=df["wide_initial_target"].tolist(),
        mutated=df["wide_mutated_target"].tolist(),
        numerics=numerics,
        targets={"HEK": df["HEKaverageedited_clamped"].to_numpy(dtype=np.float32),
                 "K562": df["K562averageedited_clamped"].to_numpy(dtype=np.float32)},
        grp_ids=df["grp_id"].tolist(),
        numeric_col_names=names,
    )
