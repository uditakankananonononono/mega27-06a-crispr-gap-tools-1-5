"""High-throughput histone H3 lysine editing by prime editing (Nat Genet 2026,
10.1038/s41588-026-02675-y, MOESM4-6): per-pegRNA editing efficiency, indels and
PAM-only edits for six lysine edits (K23R, K14R, K18R, K14K, K18K, R23K).
Gap-3: how large is the within-site spread of pegRNA efficiency, and does higher
on-target editing trade off against purity (indels)?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

SHEETS = [
    ('moesm4.xlsx', 'Fig.1i', 'K23R'),
    ('moesm5.xlsx', '2c', 'K14R'),
    ('moesm5.xlsx', '2d', 'K18R'),
    ('moesm5.xlsx', '2k', 'K14K'),
    ('moesm5.xlsx', '2l', 'K18K'),
    ('moesm6.xlsx', '3c', 'R23K'),
]
sites = {}
for fn, sheet, label in SHEETS:
    wb = openpyxl.load_workbook(f'data/raw/h3_lysine_pe/{fn}', read_only=True)
    ws = wb[sheet]
    recs = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0 or row[0] is None:
            continue
        try:
            edit, indel, pamonly = float(row[1]), float(row[2]), float(row[3])
        except (TypeError, ValueError):
            continue
        recs.append({'pegrna': str(row[0]), 'edit': edit, 'indel': indel, 'pam_only': pamonly})
    sites[label] = recs

per_site = {}
all_edit, all_indel = [], []
for label, recs in sites.items():
    edits = np.array([r['edit'] for r in recs])
    indels = np.array([r['indel'] for r in recs])
    all_edit.extend(edits); all_indel.extend(indels)
    per_site[label] = {
        'n_pegrnas': len(recs),
        'edit_median': float(np.median(edits)),
        'edit_max': float(edits.max()),
        'edit_min': float(edits.min()),
        'spread_max_over_min': float(edits.max() / max(edits.min(), 1e-9)),
        'indel_median': float(np.median(indels)),
    }
rho = float(spearmanr(all_edit, all_indel).statistic)
best_worst = max(s['edit_median'] for s in per_site.values()) / min(s['edit_median'] for s in per_site.values())
out = {
    'n_sites': len(per_site),
    'n_pegrnas_total': sum(s['n_pegrnas'] for s in per_site.values()),
    'per_site': per_site,
    'edit_vs_indel_spearman_all_pegrnas': rho,
    'best_vs_worst_site_median_ratio': float(best_worst),
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/h3_lysine_pe.json', 'w'), indent=1)
