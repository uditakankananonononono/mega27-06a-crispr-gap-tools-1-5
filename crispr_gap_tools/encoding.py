"""Sequence encodings for guide-RNA models. Real, shared by every gap tool."""
from __future__ import annotations
import numpy as np

BASES = ("A", "C", "G", "T")
_BASE2IDX = {b: i for i, b in enumerate(BASES)}
_COMP = str.maketrans("ACGTU", "TGCAA")


def revcom(seq: str) -> str:
    """Reverse complement; U is treated as T's RNA equivalent."""
    return seq.upper().translate(_COMP)[::-1]


def normalize(seq: str) -> str:
    """Uppercase DNA; RNA input converted (U->T)."""
    return seq.upper().replace("U", "T")


def gc_content(seq: str) -> float:
    s = normalize(seq)
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def one_hot(seq: str) -> np.ndarray:
    """(L, 4) one-hot in A, C, G, T order. Unknown bases -> all-zero row."""
    s = normalize(seq)
    out = np.zeros((len(s), 4), dtype=np.float32)
    for i, b in enumerate(s):
        j = _BASE2IDX.get(b)
        if j is not None:
            out[i, j] = 1.0
    return out


def dinuc_one_hot(seq: str) -> np.ndarray:
    """(L-1, 16) dinucleotide one-hot."""
    s = normalize(seq)
    out = np.zeros((max(len(s) - 1, 0), 16), dtype=np.float32)
    for i in range(len(s) - 1):
        a, b = _BASE2IDX.get(s[i]), _BASE2IDX.get(s[i + 1])
        if a is not None and b is not None:
            out[i, a * 4 + b] = 1.0
    return out


def pair_mismatch_mask(wt: str, off: str) -> np.ndarray:
    """Boolean mask of mismatched positions between guide and off-target (same length)."""
    w, o = normalize(wt), normalize(off)
    if len(w) != len(o):
        raise ValueError(f"length mismatch: {len(w)} vs {len(o)}")
    return np.array([a != b for a, b in zip(w, o)])


def encode_pair(wt: str, off: str) -> np.ndarray:
    """(L, 8) channel stack of one-hot(wt) and one-hot(off) for off-target CNNs."""
    w, o = normalize(wt), normalize(off)
    if len(w) != len(o):
        raise ValueError(f"length mismatch: {len(w)} vs {len(o)}")
    return np.concatenate([one_hot(w), one_hot(o)], axis=1)
