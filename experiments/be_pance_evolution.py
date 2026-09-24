"""Directed-evolution mutational landscape of a base editor (Nat Biotechnol 2025,
10.1038/s41587-025-02937-w, MOESM3 Supp Table 1): PANCE round-10 triplicates
(10A/10B/10C) vs T0 input library; 20 amino acids x 167 residue positions,
position-normalized percentages (each position sums to ~100).
Gap-1/3: which mutant substitutions are genuinely enriched by selection
(round10/T0 fold-change), and how reproducible is the landscape across
replicate selections? The WT residue dominates every position (~97.7%), so
max-AA profiles are uninformative - the signal is in the mutant fraction."""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/be_pance/moesm3.xlsx', read_only=True)
rows = list(wb['Supp_table_1_PANCE_norm_R10-n=3'].iter_rows(values_only=True))
AAS = list('ACDEFGHIKLMNPQRSTVWY')
blocks = {}
i = 0
while i < len(rows):
    c0 = str(rows[i][0]).strip() if rows[i][0] is not None else ''
    if c0 in ('10A', '10B', '10C', 'T0'):
        mat = {}
        j = i + 2
        while j < len(rows) and rows[j][0] is not None and str(rows[j][0]).strip() in AAS:
            aa = str(rows[j][0]).strip()
            mat[aa] = [float(x) if x is not None else 0.0 for x in rows[j][1:168]]
            j += 1
        blocks[c0] = mat
        i = j
    else:
        i += 1
assert set(blocks) == {'10A', '10B', '10C', 'T0'}
NP = 167

# WT residue = dominant AA in T0 at each position
wt = [max(AAS, key=lambda aa: blocks['T0'][aa][p]) for p in range(NP)]
wt_frac_t0 = np.mean([blocks['T0'][wt[p]][p] for p in range(NP)])

def mutant_vector(round_name):
    """log2 fold-change over T0 for every non-WT substitution, flattened"""
    v = []
    for p in range(NP):
        for aa in AAS:
            if aa == wt[p]:
                continue
            v.append(np.log2((blocks[round_name][aa][p] + 0.001) / (blocks['T0'][aa][p] + 0.001)))
    return np.array(v)

vecs = {r: mutant_vector(r) for r in ('10A', '10B', '10C')}
rhos = {}
for a, b in (('10A', '10B'), ('10A', '10C'), ('10B', '10C')):
    rhos[f'{a}_vs_{b}'] = float(spearmanr(vecs[a], vecs[b]).statistic)

# mean fold-change across replicates per substitution
subs = []
for p in range(NP):
    for aa in AAS:
        if aa == wt[p]:
            continue
        fc = np.mean([(blocks[r][aa][p] + 0.001) / (blocks['T0'][aa][p] + 0.001) for r in ('10A', '10B', '10C')])
        subs.append((float(fc), f'{wt[p]}{p + 1}{aa}'))
subs.sort(reverse=True)
n_enriched_3x = sum(1 for fc, _ in subs if fc > 3)
# top substitution's round-10 mean percentage
top_fc, top_name = subs[0]
p0 = int(''.join(ch for ch in top_name if ch.isdigit())) - 1
aa0 = top_name[-1]
top_pct = float(np.mean([blocks[r][aa0][p0] for r in ('10A', '10B', '10C')]))
out = {
    'n_positions': NP,
    'wt_residue_mean_pct_in_T0': float(wt_frac_t0),
    'replicate_mutant_landscape_spearman': rhos,
    'n_substitutions_enriched_gt3x': n_enriched_3x,
    'top_enriched_substitutions_(fc,name)': subs[:8],
    'top_substitution_round10_mean_pct': top_pct,
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/be_pance_evolution.json', 'w'), indent=1)
