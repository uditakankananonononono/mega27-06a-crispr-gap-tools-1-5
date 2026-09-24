"""Extru-seq (Kim lab 2022, Genome Biology 10.1186/s13059-022-02842-4)
MOESM5 targeted amplicon validation of Extru-seq candidate off-targets.
7 targets (human PCSK9/Albumin/HBB/VEGFA/FANCF, mouse PCSK9/Albumin),
19-nt spacers -> CFD (20-mer positional matrix) not applicable; we test
mismatch-count decay, validation rate, and seed vs PAM-distal tolerance."""
import json
import openpyxl
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

wb = openpyxl.load_workbook('data/raw/extruseq/moesm5.xlsx', read_only=True)
rows = [list(r) for r in wb.active.iter_rows(values_only=True)]

sections, cur = [], None
for r in rows:
    c1 = r[1]
    if isinstance(c1, str) and c1.strip() and (r[2] is None or 'Position' in str(r[2])):
        cur = c1.strip(); continue
    pos = r[2]
    if cur and isinstance(pos, str) and pos.startswith('chr'):
        sections.append(dict(target_section=cur, pos=pos, seq=str(r[3]).strip(),
                             pam=str(r[4]).strip(), n_mm=int(r[5]),
                             indel=float(r[6]), ctrl=float(r[7]) if r[7] not in (None, '') else 0.0,
                             valid=str(r[8]).strip() == 'Yes'))

sites = sections
n_val = sum(s['valid'] for s in sites)
by_mm = {}
for s in sites:
    by_mm.setdefault(s['n_mm'], []).append(s)

mm_curve = {str(k): dict(n=len(v), val_rate=round(sum(x['valid'] for x in v)/len(v), 3),
                         median_indel=float(np.median([x['indel'] for x in v])))
            for k, v in sorted(by_mm.items())}

rho, p_rho = spearmanr([s['n_mm'] for s in sites], [s['indel'] for s in sites])

# seed vs PAM-distal: 19-mer, PAM-proximal = 3' end; seed = last 6 nt (pos 14-19, 1-based)
seed_indel, distal_indel = [], []
for s in sites:
    mm_pos = [i for i, ch in enumerate(s['seq']) if ch.islower()]
    if not mm_pos:
        continue
    if all(i >= 13 for i in mm_pos):
        seed_indel.append(s['indel'])
    elif all(i < 13 for i in mm_pos):
        distal_indel.append(s['indel'])
u, p_seed = mannwhitneyu(seed_indel, distal_indel) if seed_indel and distal_indel else (None, None)

# mouse sections have mmseq2-style? just count species
human = [s for s in sites if s['target_section'].startswith('Human')]
mouse = [s for s in sites if s['target_section'].startswith('Mouse')]

out = dict(
    source='Extru-seq Kim lab 2022 Genome Biology 10.1186/s13059-022-02842-4 MOESM5',
    n_sites=len(sites), n_targets=len({s['target_section'] for s in sites}),
    n_validated=n_val, val_rate=round(n_val/len(sites), 3),
    n_human=len(human), n_mouse=len(mouse),
    mismatch_curve=mm_curve,
    spearman_mm_indel=dict(rho=round(float(rho), 3), p=float(p_rho)),
    seed_vs_distal=dict(n_seed=len(seed_indel), n_distal=len(distal_indel),
                        median_seed=float(np.median(seed_indel)),
                        median_distal=float(np.median(distal_indel)),
                        mwu_p=float(p_seed) if p_seed is not None else None),
    spacer_note='19-nt spacers; CFD 20-mer matrix not applicable',
)
json.dump(out, open('results/extruseq_validation.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
