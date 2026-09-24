import os
import numpy as np

from crisprgap.data.datasets import (load_doench_fcres, load_doench_v1,
                                     load_crisprsql, DATA_DIR)

pytestmark = []  # data files are committed fixtures for CI hermeticity? see below


def _have(name):
    return os.path.exists(os.path.join(DATA_DIR, name))


import pytest


@pytest.mark.skipif(not _have("FC_plus_RES_withPredictions.csv"), reason="data not fetched")
def test_doench_fcres_loads():
    ds = load_doench_fcres()
    assert len(ds.sequences) > 5000
    assert all(len(s) == 30 for s in ds.sequences)
    assert ds.scores.min() >= 0 and ds.scores.max() <= 1


@pytest.mark.skipif(not _have("V1_suppl_data.txt"), reason="data not fetched")
def test_doench_v1_loads():
    ds = load_doench_v1()
    assert len(ds.sequences) > 2000
    assert all(len(s) == 30 for s in ds.sequences)


@pytest.mark.skipif(not _have("crisprsql/100720.csv"), reason="data not fetched")
def test_crisprsql_loads():
    ds = load_crisprsql()
    assert len(ds.guides) > 20000
    assert all(len(g) == 20 and len(o) == 20 for g, o in zip(ds.guides, ds.offtargets))
    assert ds.labels.sum() > 1000  # plenty of true cleaved off-targets
    assert len(set(ds.studies)) >= 10  # aggregated from many studies


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA_DIR, "raw/public_data_crisprCas9/data/deepHF/wt_seq_data_array.pkl")),
                    reason="deepHF data not fetched")
def test_deephf_loads():
    from crisprgap.data.datasets import load_deephf
    ds = load_deephf("wt", max_n=500)
    assert len(ds.sequences) == 500
    assert all(len(s) == 21 for s in ds.sequences)
    assert all(set(s) <= set("ACGT") for s in ds.sequences)  # encoding fix: no Ns
    assert ds.scores.min() >= 0 and ds.scores.max() <= 1


@pytest.mark.skipif(not _have("raw/crisprscan/crisprscan_MOESM633.xlsx"), reason="data not fetched")
def test_crisprscan_loads():
    from crisprgap.data.datasets import load_crisprscan
    ds = load_crisprscan()
    assert len(ds.sequences) == 492
    assert all(len(s) == 20 and set(s) <= set("ACGT") for s in ds.sequences)
    assert ds.scores.min() >= 0 and ds.scores.max() <= 1


@pytest.mark.skipif(not _have("deepcrispr/hct116.repisgt"), reason="data not fetched")
def test_deepcrispr_loads():
    from crisprgap.data.datasets import load_deepcrispr
    for cl, n in (("hct116", 4239), ("hek293t", 4566), ("hela", 8101), ("hl60", 2076)):
        ds = load_deepcrispr(cl)
        assert len(ds.sequences) == n
        assert all(len(s) == 23 and set(s) <= set("ACGT") for s in ds.sequences)
        assert ds.scores.min() >= 0 and ds.scores.max() <= 1
