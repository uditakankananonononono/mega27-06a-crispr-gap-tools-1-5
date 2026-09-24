"""Promoter + gRNA scaffold parts list (Nat Biotechnol 2025,
10.1038/s41587-025-02896-2, MOESM3): Table_S1 promoters x 4 cell types +
synHEK3, Table_S9 promoters x 5 loci, Table_S10 scaffold variants x 5 loci.
Gap-1/3: how well do promoter/scaffold editing scores transfer across cell
types and genomic loci (cross-context rank transfer)?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/parts_list/moesm3.xlsx', read_only=True)

def sheet_rows(name):
    rows = list(wb[name].iter_rows(values_only=True))
    hdr = [str(c) if c is not None else '' for c in rows[0]]
    out = []
    for r in rows[1:]:
        if r[0] is None:
            continue
        out.append(r)
    return hdr, out

# Table_S1: promoter scores across 4 cell types (cols 3-6)
hdr, s1 = sheet_rows('Table_S1')
ct_scores = {}
for r in s1:
    try:
        ct_scores[str(r[0])] = [float(r[3]), float(r[4]), float(r[5]), float(r[6])]
    except (TypeError, ValueError):
        continue
cts = ['iPSC', 'mESC', 'K562', 'HEK293T']
ct_rho = {}
for a in range(4):
    for b in range(a + 1, 4):
        x = [v[a] for v in ct_scores.values()]
        y = [v[b] for v in ct_scores.values()]
        ct_rho[f'{cts[a]}_vs_{cts[b]}'] = float(spearmanr(x, y).statistic)

# Table_S9: promoters x 5 loci
hdr9, s9 = sheet_rows('Table_S9')
locus_scores = {}
for r in s9:
    try:
        locus_scores[str(r[0])] = [float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])]
    except (TypeError, ValueError):
        continue
loci = ['synHEK3', 'HBB', 'EMX1', 'FANCF', 'CLYBL']
loc_rho = {}
for a in range(5):
    for b in range(a + 1, 5):
        x = [v[a] for v in locus_scores.values()]
        y = [v[b] for v in locus_scores.values()]
        loc_rho[f'{loci[a]}_vs_{loci[b]}'] = float(spearmanr(x, y).statistic)

# Table_S10: scaffold variants x 5 loci
hdr10, s10 = sheet_rows('Table_S10')
scaf_scores = {}
for r in s10:
    try:
        scaf_scores[str(r[0])] = [float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7])]
    except (TypeError, ValueError):
        continue
scaf_rho = {}
for a in range(5):
    for b in range(a + 1, 5):
        x = [v[a] for v in scaf_scores.values()]
        y = [v[b] for v in scaf_scores.values()]
        scaf_rho[f'{loci[a]}_vs_{loci[b]}'] = float(spearmanr(x, y).statistic)

out = {
    'n_promoters_celltype_panel': len(ct_scores),
    'n_promoters_locus_panel': len(locus_scores),
    'n_scaffold_variants_locus_panel': len(scaf_scores),
    'promoter_cross_celltype_spearman': ct_rho,
    'promoter_cross_locus_spearman': loc_rho,
    'scaffold_cross_locus_spearman': scaf_rho,
    'promoter_cross_celltype_min': float(min(ct_rho.values())),
    'promoter_cross_locus_min': float(min(loc_rho.values())),
    'scaffold_cross_locus_min': float(min(scaf_rho.values())),
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/parts_list_transfer.json', 'w'), indent=1)
