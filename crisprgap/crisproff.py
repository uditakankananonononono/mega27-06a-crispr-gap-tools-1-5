"""CRISPRoff binding-energy scorer (Alkan et al. 2018, Nat Commun), vendored
from RTH-tools/crisproff (GPLv3, see vendor/CRISPROFF_LICENSE). Published-model
baseline for gap 2. Configuration: positional weights + DNA opening with
positional weights + PAM correction; guide self-folding term omitted (needs
the RNAfold binary; the fold term is guide-constant within a guide-grouped
evaluation and cancels in ranking)."""
import os

from crisprgap.vendor import crisproff_pipeline as cp

cp.read_energy_parameters(os.path.join(os.path.dirname(__file__), "vendor", "energy_dics.pkl"))


def crisproff_score(guide23: str, off23: str) -> float:
    """Higher = stronger predicted binding. Inputs are 23-mers (protospacer+PAM)."""
    assert len(guide23) == 23 and len(off23) == 23
    return cp.get_eng(guide23.upper(), off23.upper(), cp.calcRNADNAenergy,
                      pos_weight=True, pam_corr=True,
                      dna_opening=True, dna_pos_wgh=True)
