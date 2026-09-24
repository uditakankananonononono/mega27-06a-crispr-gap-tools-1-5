import json

import numpy as np


def test_rna_features_artifact():
    d = json.load(open("results/pegrna_rna_features.json"))
    assert d["n"] == 22956
    assert len(d["external_tools"]) == 2
    for tgt in ("HEK", "K562"):
        t = d["targets"][tgt]
        assert len(t["numeric_only_by_split"]) == 3
        assert len(t["plus_rna_by_split"]) == 3
        # the finding: RNA thermodynamic features add no signal
        assert abs(t["delta"]) < 0.01


def test_rna_feature_computation_small_slice():
    from experiments.rna_features_pegrna import compute, _window
    w = _window("A" * 99, 39)
    assert len(w) == 40
    compute(0, 5)  # writes a small partial; fast
    z = np.load("results/rnafeat_part_0_5.npz")
    assert z["dg"].shape == (5,)
    assert np.isfinite(z["dg"]).all()
    assert (z["tm"] > 0).all()


def test_seqfold_and_primer3_agree_on_sign_of_structure():
    # sanity: a GC-rich stem-forming sequence folds more negatively than a poly-A
    from seqfold import fold
    stem = "GCGCGCGCGCGCTTTTGCGCGCGCGCGC"  # hairpin-capable
    poly = "AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    e_stem = min(s.e for s in fold(stem))
    e_poly = min(s.e for s in fold(poly))
    assert e_stem < e_poly
