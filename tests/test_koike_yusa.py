import json

import numpy as np

from crisprgap.data.datasets import load_koike_yusa_miseq


def test_loader_shape_and_counts():
    ds = load_koike_yusa_miseq()
    assert len(ds.offtargets) == 190
    assert len(set(ds.guides)) == 1
    assert ds.guides[0] == "GAAGAGAGCATCATGGGCCA"
    n_ngg = sum(p.upper().endswith("GG") for p in ds.off_pam)
    n_nag = sum(p.upper().endswith("AG") and not p.upper().endswith("GG")
                for p in ds.off_pam)
    assert (n_ngg, n_nag) == (95, 95)


def test_mismatch_reconstruction():
    ds = load_koike_yusa_miseq()
    mm = [sum(a != b for a, b in zip(g, o))
          for g, o in zip(ds.guides, ds.offtargets)]
    assert min(mm) == 2 and max(mm) == 5


def test_results_file_values():
    d = json.load(open("results/koike_yusa_nag.json"))
    assert d["n_sites"] == 190
    assert d["cfd"]["n_scored"] == 190
    # NGG 3-mismatch sites cleave strongly, NAG sites do not
    assert d["gradient"]["ngg"]["3"]["mean_cleavage"] > 5.0
    assert d["gradient"]["nag"]["3"]["mean_cleavage"] < 0.1
    # subclone replicate structure agrees with transient transfection
    assert d["subclone_corr"] > 0.9
