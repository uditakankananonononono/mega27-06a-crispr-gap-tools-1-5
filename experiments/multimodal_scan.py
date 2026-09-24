"""Multimodal variant scanning (Nat Biotechnol 2024,
10.1038/s41587-024-02439-1, MOESM4/6): same sgRNA library screened with
ABE8e and BE3.9Max; cross-editor concordance of per-sgRNA LFC."""
import json
import openpyxl
import numpy as np
from scipy.stats import spearmanr


def load(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        try:
            out[str(r[0])] = float(r[3])
        except (TypeError, ValueError, IndexError):
            pass
    return out

out = {}
for tag, path, s1, s2 in (
    ('activation', 'data/raw/multimodal_scan/moesm4.xlsx',
     'Activation_fitness_ABE8e', 'Activation_fitness_BE3.9Max'),
    ('gefitinib', 'data/raw/multimodal_scan/moesm6.xlsx',
     'Gefitinib_fitness_ABE8e', 'Gefitinib_fitness_BE3.9Max'),
):
    a, b = load(path, s1), load(path, s2)
    common = sorted(set(a) & set(b))
    x = np.array([a[k] for k in common]); y = np.array([b[k] for k in common])
    rho = spearmanr(x, y)
    out[tag] = dict(n_common=len(common), spearman=round(float(rho.statistic), 3),
                    pvalue=float(rho.pvalue))
json.dump(out, open('results/multimodal_scan.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
