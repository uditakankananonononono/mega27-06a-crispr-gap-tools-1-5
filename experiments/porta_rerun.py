"""Regenerate PORTA + discovery results after the DeepHF encoding fix.

Pipeline (matches paper Sec. 12): ridge source model on Doench FC+RES ->
predict DeepHF screens -> per-guide rank residual -> portability model
(ridge on sequence features + 11 DeepHF biofeatures, WT screen only) ->
zero-shot residual prediction and corrected transfer on HF1 / eSp.
Writes portability_cross_screen.json, portability_correction.json,
portability_correction_esp.json, inversion_signature_3way.json.
"""
import json

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

from crisprgap.crossdata import featurize
from crisprgap.data.datasets import load_doench_fcres, load_deephf

import pickle
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_screen(variant):
    path = os.path.join(DATA, "raw/public_data_crisprCas9/data/deepHF",
                        f"{variant}_seq_data_array.pkl")
    seq_int, biofeat, indel = pickle.load(open(path, "rb"))
    arr = np.asarray(seq_int)
    assert (arr[:, 0] == 1).all()
    m = {2: "C", 3: "G", 4: "T", 5: "A"}
    seqs = ["".join(m[b] for b in row[1:]) for row in arr]
    return seqs, np.asarray(biofeat, dtype=np.float32), np.clip(
        np.asarray(indel, dtype=np.float32), 0, 1)


def main():
    fc = load_doench_fcres()
    Xs = featurize(fc.sequences)
    src = Ridge(alpha=1.0).fit(Xs, fc.scores)

    screens = {}
    for v in ("wt", "esp", "hf"):
        seqs, bf, y = load_screen(v)
        pred = src.predict(featurize(seqs))
        resid = rankdata(pred) - rankdata(y)
        screens[v] = dict(seqs=seqs, bf=bf, y=y, pred=pred, resid=resid)
        print(v, len(seqs), flush=True)

    # 3-way sequence-matched residual concordance (discovery)
    idx = {}
    for v, s in screens.items():
        idx[v] = {q: i for i, q in enumerate(s["seqs"])}
    common = set(idx["wt"]) & set(idx["esp"]) & set(idx["hf"])
    common = sorted(common)
    def take(v, key):
        return np.array([screens[v][key][idx[v][q]] for q in common])
    inv = {"n_common": len(common),
           "resid_esp_vs_wt": float(spearmanr(take("esp", "resid"), take("wt", "resid")).statistic),
           "resid_esp_vs_hf": float(spearmanr(take("esp", "resid"), take("hf", "resid")).statistic),
           "resid_hf_vs_wt": float(spearmanr(take("hf", "resid"), take("wt", "resid")).statistic),
           "score_wt_vs_hf": float(spearmanr(take("wt", "y"), take("hf", "y")).statistic)}
    json.dump(inv, open("results/inversion_signature_3way.json", "w"), indent=2)
    print("inv", inv, flush=True)

    # portability model on WT only: sequence features + biofeatures -> residual
    w = screens["wt"]
    Fw = np.concatenate([featurize(w["seqs"]), w["bf"]], axis=1)
    pm = Ridge(alpha=1.0).fit(Fw, w["resid"])

    out = {}
    for v, tag in (("hf", "hf"), ("esp", "esp")):
        s = screens[v]
        F = np.concatenate([featurize(s["seqs"]), s["bf"]], axis=1)
        pred_resid = pm.predict(F)
        cs = float(spearmanr(pred_resid, s["resid"]).statistic)
        base = float(spearmanr(s["pred"], s["y"]).statistic)
        corrected = rankdata(s["pred"]) - pred_resid
        corr_sp = float(spearmanr(corrected, s["y"]).statistic)
        out[tag] = dict(cross_screen=cs, base=base, corrected=corr_sp, n=len(s["seqs"]))
        print(tag, out[tag], flush=True)

    # unseen guides: HF1 sequences absent from the WT screen
    h, wseq = screens["hf"], set(screens["wt"]["seqs"])
    mask = np.array([q not in wseq for q in h["seqs"]])
    Fh = np.concatenate([featurize(h["seqs"]), h["bf"]], axis=1)
    pr = pm.predict(Fh)
    unseen = dict(cross_screen=float(spearmanr(pr[mask], h["resid"][mask]).statistic),
                  base=float(spearmanr(h["pred"][mask], h["y"][mask]).statistic),
                  corrected=float(spearmanr((rankdata(h["pred"]) - pr)[mask], h["y"][mask]).statistic),
                  n=int(mask.sum()))
    print("unseen", unseen, flush=True)

    json.dump({"cross_screen_spearman_all_hf": out["hf"]["cross_screen"],
               "n_hf": out["hf"]["n"],
               "cross_screen_spearman_unseen_guides": unseen["cross_screen"],
               "n_unseen": unseen["n"]},
              open("results/portability_cross_screen.json", "w"), indent=2)
    json.dump({"hf_base_transfer": out["hf"]["base"],
               "hf_residual_corrected": out["hf"]["corrected"],
               "unseen_guides": {"base": unseen["base"],
                                 "corrected": unseen["corrected"],
                                 "n": unseen["n"]}},
              open("results/portability_correction.json", "w"), indent=2)
    json.dump({"esp_base_transfer": out["esp"]["base"],
               "esp_residual_corrected": out["esp"]["corrected"],
               "n": out["esp"]["n"]},
              open("results/portability_correction_esp.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
