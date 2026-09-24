"""Cutting Frequency Determination (CFD) score - Doench et al. 2016, Nat Biotechnol.

Python 3 port of the reference calculator published with CRISPOR
(CFD_Scoring/cfd-score-calculator.py) using the vendored mismatch/PAM matrices
fitted in the original paper. Benchmark baseline for the gap-2/4/5 tools
alongside the Hsu 2013 MIT score in baselines.py.
"""
from __future__ import annotations
import pickle
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "matrices"
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
    """CFD specificity score in [0, 1]; 1.0 = perfect match with canonical PAM."""
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


_ACGT = set("ACGT")


def cfd_applicable(wt_23mer: str, off_23mer: str) -> bool:
    """CFD is defined for substitution-only 23-mers over ACGT (guide+PAM)."""
    w, o = wt_23mer.upper(), off_23mer.upper()
    return (len(w) == 23 and len(o) == 23
            and set(w) <= _ACGT and set(o) <= _ACGT)
