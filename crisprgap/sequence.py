"""Sequence encoding utilities for CRISPR guide/target modeling."""
from __future__ import annotations

import numpy as np

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(COMPLEMENT)[::-1]


def one_hot(seq: str, length: int | None = None) -> np.ndarray:
    """One-hot encode a DNA string into (4, L) float32. Ambiguous bases -> all-zeros row."""
    seq = seq.upper()
    L = length or len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    for i, b in enumerate(seq[:L]):
        j = BASE_TO_IDX.get(b)
        if j is not None:
            arr[j, i] = 1.0
    return arr


def mismatch_mask(guide: str, offtarget: str) -> np.ndarray:
    """Binary (L,) mask of mismatch positions between equal-length guide and off-target."""
    assert len(guide) == len(offtarget), "guide and off-target must be equal length"
    return np.array(
        [1.0 if g != o else 0.0 for g, o in zip(guide.upper(), offtarget.upper())],
        dtype=np.float32,
    )


def gc_content(seq: str) -> float:
    s = seq.upper()
    acgt = sum(s.count(b) for b in "ACGT")
    return 0.0 if acgt == 0 else (s.count("G") + s.count("C")) / acgt


def dinucleotide_features(seq: str) -> np.ndarray:
    """16-dim dinucleotide composition feature vector."""
    s = seq.upper()
    feats = np.zeros(16, dtype=np.float32)
    counts = 0
    for i in range(len(s) - 1):
        a, b = BASE_TO_IDX.get(s[i]), BASE_TO_IDX.get(s[i + 1])
        if a is not None and b is not None:
            feats[a * 4 + b] += 1.0
            counts += 1
    if counts:
        feats /= counts
    return feats
