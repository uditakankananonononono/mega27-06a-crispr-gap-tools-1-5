"""Published off-target scoring baselines, reimplemented from the source papers.

- Hsu 2013 (MIT) score: Hsu et al., Nat Biotechnol 2013, doi:10.1038/nbt.2647.
  Per-position mismatch penalty weights are taken from the paper's supplement.
"""
from __future__ import annotations

import numpy as np

# Hsu et al. 2013 per-position mismatch weights (5'->3' across the 20-nt protospacer)
HSU_WEIGHTS = np.array([
    0.0, 0.0, 0.014, 0.0, 0.0, 0.395, 0.317, 0.0, 0.389, 0.079,
    0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804, 0.685, 0.583,
])


def mit_score(guide: str, offtarget: str) -> float:
    """Hsu 2013 MIT specificity score for one guide/off-target pair (0..1)."""
    assert len(guide) == len(offtarget) == 20
    mm = [i for i, (a, b) in enumerate(zip(guide.upper(), offtarget.upper())) if a != b]
    if not mm:
        return 1.0
    term1 = float(np.prod(1.0 - HSU_WEIGHTS[mm]))
    mean_pairwise = np.mean([(b - a) for i, a in enumerate(mm) for b in mm[i + 1:]]) if len(mm) > 1 else 0.0
    term2 = 1.0 / ((19.0 - mean_pairwise) / 19.0 * 4.0 + 1.0)
    term3 = 1.0 / (len(mm) ** 2)
    return term1 * term2 * term3
