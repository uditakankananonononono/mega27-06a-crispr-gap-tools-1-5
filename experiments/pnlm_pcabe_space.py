"""Protein-nucleic-acid language-model designed precise compact ABE
(Nat Commun 2025, 10.1038/s41467-025-65311-z, MOESM6): ClinVar pathogenic
G>A variants classed as targetable / precise (no bystander) / bystander /
pathogenic-bystander, per editor (ABE8e, ABE9, PNLM-pcABE) x PAM (NGG, NG).
Gap-2: precision vs targetable-space tradeoff across editors."""
import json
import openpyxl

wb = openpyxl.load_workbook('data/raw/pnlm_pcabe/moesm6.xlsx', read_only=True)
counts = {}
for ws in wb.worksheets:
    n = 0
    for r in ws.iter_rows(values_only=True):
        if r[0] is not None and str(r[0]).strip().isdigit():
            n += 1
    counts[ws.title] = n
print(json.dumps(counts, indent=1))
res = {}
for pam in ('NGG', 'NG'):
    for ed in ('ABE8e', 'ABE9', 'PNLM-pcABE'):
        t = counts.get(f'{pam}-{ed}-Targetable')
        p = counts.get(f'{pam}-{ed}-Precise')
        b = counts.get(f'{pam}-{ed}-ByStander')
        path = counts.get(f'{pam}-{ed}-Pathogenic')
        if p is None:
            continue
        entry = {'precise': p, 'pathogenic_bystander': path}
        if t:
            entry['targetable'] = t
            entry['precision_fraction'] = round(p / t, 4)
        if b is not None:
            entry['bystander'] = b
        if path is not None and p:
            entry['pathogenic_bystander_per_precise'] = round(path / p, 4)
        res[f'{pam}-{ed}'] = entry
expansion = round(res['NG-ABE8e']['precise'] / res['NGG-ABE8e']['precise'], 3)
out = {'sheet_counts': counts, 'per_editor_pam': res,
       'abe8e_precise_ng_vs_ngg_expansion': expansion,
       'note': 'Only ABE8e sheets include Targetable/Bystander denominators; ABE9 and PNLM-pcABE contribute Precise + pathogenic-bystander counts.'}
print(json.dumps(res, indent=1))
json.dump(out, open('results/pnlm_pcabe_space.json', 'w'), indent=1)
