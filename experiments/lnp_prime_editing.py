"""In-vivo LNP prime editing (Nat Nanotechnol 2026, 10.1038/s41565-026-02200-6,
MOESM5): Pcsk9 +1 TTAC installation by LNP-delivered PE vs PE-AAV, dose
response, tissue/cell-type confinement, guide-motif and editor variants, and
14-site off-target panel.
Gap-2/3: does potency gain cost precision, and does delivery confine editing?"""
import json
import numpy as np
import openpyxl

wb = openpyxl.load_workbook('data/raw/lnp_prime_editing/moesm5.xlsx', read_only=True)

def groups(sheet, header_row=0, first_col=1, width=3):
    """Parse sheets laid out as groups of `width` triplicate columns."""
    rows = list(wb[sheet].iter_rows(values_only=True))
    hdr = rows[header_row]
    names, cur = [], None
    for c in range(first_col, len(hdr)):
        v = hdr[c]
        if v is not None and str(v).strip():
            cur = str(v).replace('\xa0', ' ').strip()
        names.append(cur)
    data = {}
    for r in rows[header_row + 1:]:
        label = r[0]
        if label is None or not str(label).strip():
            continue
        label = str(label).replace('\xa0', ' ').strip()
        vals = {}
        for i, c in enumerate(range(first_col, len(r))):
            try:
                v = float(r[c])
            except (TypeError, ValueError):
                continue
            vals.setdefault(names[i], []).append(v)
        data[label] = {k: list(map(float, v)) for k, v in vals.items()}
    return data

def mean(xs):
    return float(np.mean(xs))

# --- in vivo dose response, Figure 4b (groups: editing, Indels) ---
dose = groups('Figure 4b')
dose_out = {}
for d, g in dose.items():
    if d.lower() == 'untreated':
        continue
    ed = mean(g['Pcsk9 +1 TTAC insertion'])
    ind = mean(g['Indels'])
    dose_out[d] = {'editing': ed, 'indels': ind, 'precision': ed / ind}

# --- tissue confinement, Figure 4a ---
tissue = groups('Figure 4a')
treated_tissue = {t: mean(g['s3 PE-LNP']) for t, g in tissue.items()}
liver = treated_tissue['Liver']
others = {t: v for t, v in treated_tissue.items() if t != 'Liver'}
max_other = max(others.values())

# --- cell types, Extended Data 6 ---
cells = groups('Extended Data 6')
cell_treated = {t: mean(g['s3 PE-LNP']) for t, g in cells.items()}
nonhep_max = max(v for k, v in cell_treated.items() if k != 'Bulk liver')

# --- AAV vs LNP biodistribution, Figure 6f ---
biod = groups('Figure 6f')
aav_heart = mean(biod['Heart']['PE-AAV'])
lnp_heart = mean(biod['Heart']['PE-LNP'])
aav_liver = mean(biod['Liver']['PE-AAV'])
lnp_liver = mean(biod['Liver']['PE-LNP'])

# --- off-target panel, Figure 6g ---
ot = groups('Figure 6g')
ot_rows = {}
max_delta = {'PE-AAV': 0.0, 'PE-LNP': 0.0}
for site, g in ot.items():
    un = mean(g['Untreated'])
    rec = {'untreated': un}
    for arm in ('PE-AAV', 'PE-LNP'):
        m = mean(g[arm])
        rec[arm] = m
        max_delta[arm] = max(max_delta[arm], abs(m - un))
    ot_rows[site] = rec

# --- guide 3' motifs, Figure 5c; editors, Figure 5b (in vitro, dose rows) ---
motifs = groups('Figure 5c')
motif_top = {m: mean(v) for m, v in motifs['100'].items()}
eds = groups('Figure 5b')
editor_top = {e: mean(v) for e, v in eds['100'].items()}

# --- editor variants incl. La fusion, Extended Data 9 ---
ed9 = groups('Extended Data 9')
ed9_out = {e: {'editing': mean(g['Pcsk9 +1 TTAC insertion']), 'indels': mean(g['Indels'])}
           for e, g in ed9.items() if e.lower() != 'untreated'}

out = {
    'dose_response': dose_out,
    'peak_precision_dose': max(dose_out, key=lambda d: dose_out[d]['precision']),
    'editing_gain_2_to_4_mgkg_pct': 100 * (dose_out['4 mg/kg']['editing'] / dose_out['2 mg/kg']['editing'] - 1),
    'indel_gain_2_to_4_mgkg_pct': 100 * (dose_out['4 mg/kg']['indels'] / dose_out['2 mg/kg']['indels'] - 1),
    'tissue_confinement': {'liver': liver, 'max_other_tissue': max_other,
                           'fold_liver_over_max_other': liver / max_other},
    'celltype_confinement': {'bulk_liver': cell_treated['Bulk liver'],
                             'max_nonhepatocyte': nonhep_max},
    'aav_vs_lnp': {'aav_liver': aav_liver, 'lnp_liver': lnp_liver,
                   'aav_heart': aav_heart, 'lnp_heart': lnp_heart,
                   'heart_fold_aav_over_lnp': aav_heart / max(lnp_heart, 1e-9)},
    'offtarget_sites': ot_rows,
    'offtarget_max_abs_delta': max_delta,
    'motif_topdose_editing': motif_top,
    'editor_topdose_editing': editor_top,
    'editor_variants_ed9': ed9_out,
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/lnp_prime_editing.json', 'w'), indent=1)
