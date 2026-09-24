import json
import subprocess
import sys

import numpy as np


def run_cli(*args):
    r = subprocess.run([sys.executable, "-m", "crisprgap", *args],
                       capture_output=True, text=True)
    return r


def test_version():
    r = run_cli("--version")
    assert r.returncode == 0
    assert "crisprgap" in json.loads(r.stdout)


def test_cfd_valid():
    r = run_cli("cfd", "GAGTCCGAGCAGAAGAAGAAGGG", "GAGTCCGAGCAGAAGAAAAAGGG")
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["applicable"] is True
    assert 0.0 < out["score"] <= 1.0


def test_cfd_identical_scores_one():
    r = run_cli("cfd", "GAGTCCGAGCAGAAGAAGAAGGG", "GAGTCCGAGCAGAAGAAGAAGGG")
    assert json.loads(r.stdout)["score"] == 1.0


def test_cfd_indel_rejected():
    r = run_cli("cfd", "GAGTCCGAGCAGAAGAAGAAGGG", "GAGTCCGAGCAGAAGAAGAA-GG")
    assert r.returncode == 2
    assert json.loads(r.stdout)["applicable"] is False


def test_cfd_non_acgt_rejected():
    r = run_cli("cfd", "GAGTCCGAGCAGAAGAAGAANGG", "GAGTCCGAGCAGAAGAAGAAGGG")
    assert r.returncode == 2


def test_mit_valid():
    r = run_cli("mit", "GAGTCCGAGCAGAAGAAGAA", "GAGTCCGAGCAGAAGAAAAA")
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["n_mismatches"] == 1
    assert 0.0 <= out["score"] <= 1.0


def test_mit_length_mismatch_rejected():
    r = run_cli("mit", "GAGTCCGAGCAGAAGAAGAA", "GAGTCCGAGCAGAAGAA")
    assert r.returncode == 2


def test_calibrate_csv(tmp_path):
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < scores).astype(int)
    csv = tmp_path / "scores.csv"
    csv.write_text("score,label\n" + "\n".join(f"{s},{l}" for s, l in zip(scores, labels)))
    r = run_cli("calibrate", str(csv))
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["n"] == 200
    assert 0.0 <= out["raw"]["ece"] <= 1.0
    assert 0.0 <= out["raw"]["brier"] <= 1.0


def test_calibrate_platt_fit(tmp_path):
    rng = np.random.default_rng(1)
    scores = rng.random(200)
    labels = (rng.random(200) < scores).astype(int)
    csv = tmp_path / "scores.csv"
    csv.write_text("score,label\n" + "\n".join(f"{s},{l}" for s, l in zip(scores, labels)))
    r = run_cli("calibrate", str(csv), "--fit", "platt")
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert "platt" in out and "a" in out["platt"] and "b" in out["platt"]


def test_calibrate_small_sample_warns(tmp_path):
    csv = tmp_path / "tiny.csv"
    csv.write_text("score,label\n0.1,0\n0.9,1\n0.2,0\n")
    r = run_cli("calibrate", str(csv))
    assert "warning" in json.loads(r.stdout)


def test_seqstats():
    r = run_cli("seqstats", "GGGGCCCC")
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["gc_content"] == 1.0
    assert out["length"] == 8
    assert abs(sum(out["dinucleotide_frequencies"].values()) - 1.0) < 1e-5


def test_no_command_prints_help():
    r = run_cli()
    assert r.returncode == 2
