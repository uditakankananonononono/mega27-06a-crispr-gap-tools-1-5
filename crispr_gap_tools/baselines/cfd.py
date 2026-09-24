"""Cutting Frequency Determination (CFD) score - Doench et al. 2016, Nat Biotechnol.

Python 3 port of the reference calculator published with CRISPOR
(CFD_Scoring/cfd-score-calculator.py) using the vendored mismatch/PAM matrices
fitted in the original paper. Used as a benchmark baseline for off-target tools.
"""
from __future__ import annotations
import pickle
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"
_cache: tuple[dict, dict] | None = None


def _mm_pam_scores() -> tuple[dict, dict]:
    global _cache
    if _cache is None:
        with open(_DATA / "mismatch_score.pkl", "rb") as fh:
            mm = pickle.load(fh)
        with open(_DATA / "pam_scores.pkl", "rb") as fh:
            pam = pickle.load(fh)
        _cache = (mm, pam)
    return _cache


def revcom(s: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}
    return "".join(comp[b] for b in s[::-1])


def cfd_score(wt_23mer: str, off_23mer: str) -> float:
    """CFD specificity score in [0, 1]; 1.0 = perfect match with canonical PAM.

    wt_23mer / off_23mer: 23 nt sequences (20 nt guide + 3 nt PAM), DNA alphabet.
    """
    mm_scores, pam_scores = _mm_pam_scores()
    wt, off = wt_23mer.upper(), off_23mer.upper()
    if len(wt) != 23 or len(off) != 23:
        raise ValueError("CFD expects 23-mers (20 nt guide + NGG PAM)")
    sg, pam = off[:-3].replace("T", "U"), off[-2:]
    wt_u = wt.replace("T", "U")
    score = 1.0
    for i, sl in enumerate(sg):
        if wt_u[i] != sl:
            score *= mm_scores["r" + wt_u[i] + ":d" + revcom(sl) + "," + str(i + 1)]
    return score * pam_scores[pam]
