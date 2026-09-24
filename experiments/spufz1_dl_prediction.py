"""Deep-learning-guided SpuFz1 engineering (Nat Commun 2026,
10.1038/s41467-026-74624-6, MOESM5): 7,448 single-point mutants scored by 5
independently trained models (300k epochs each). NOTE: the 'mean' column equals
the mean of pred_1..5 - this sheet contains model predictions only, no measured
activity. The honest analysis is ensemble consistency and landscape shape.
Gap-1: how consistent are 5 independently trained activity models across a
7,448-mutant landscape, and how sharp is the predicted fitness peak?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/spufz1_dl/moesm5.xlsx', read_only=True)
rows = list(wb['prediction result'].iter_rows(values_only=True))
recs = []
for r in rows[2:]:
    if r[0] is None:
        continue
    try:
        preds = [float(r[k]) for k in range(2, 7)]
    except (TypeError, ValueError):
        continue
    recs.append((str(r[0]), preds))
n = len(recs)
assert n > 7000, n
means = np.array([np.mean(p) for _, p in recs])
stds = np.array([np.std(p) for _, p in recs])
# pairwise model agreement
P = np.array([p for _, p in recs])  # n x 5
pair_rho = []
for a in range(5):
    for b in range(a + 1, 5):
        pair_rho.append(float(spearmanr(P[:, a], P[:, b]).statistic))
order = np.argsort(means)[::-1]
top10 = [(recs[i][0], float(means[i]), float(stds[i])) for i in order[:10]]
out = {
    'n_mutants': n,
    'ensemble_pairwise_spearman_min': float(min(pair_rho)),
    'ensemble_pairwise_spearman_median': float(np.median(pair_rho)),
    'predicted_mean_median': float(np.median(means)),
    'frac_predicted_above_0': float(np.mean(means > 0)),
    'top10_candidates_(mutation,mean,std)': top10,
    'top10_mean_std': float(np.mean([s for _, _, s in top10])),
    'overall_mean_std': float(np.mean(stds)),
    'note': "sheet contains model predictions only; 'mean' column equals mean of the 5 model outputs (verified), no measured activity in this table",
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/spufz1_dl_prediction.json', 'w'), indent=1)
