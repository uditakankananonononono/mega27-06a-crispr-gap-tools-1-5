"""PEAC-seq (Nat Commun 2022, 10.1038/s41467-022-35086-8, MOESM4-8): prime-
editor-based off-target detection on the classic GUIDE-seq panel (VEGFA
TS1/2/3, EMX1, RNF2). Cross-method question: do PE-based detections replicate
the Cas9-nuclease GUIDE-seq sets for the same guides (crisprSQL Tsai study)?
Site-guide assignment by exact 20-mer match against crisprSQL Tsai guides."""
import json
import csv
import numpy as np
import openpyxl
from collections import defaultdict
from scipy.stats import mannwhitneyu

COMP = str.maketrans('ACGT', 'TGCA')
def rc(s):
    return s.translate(COMP)[::-1]

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

# crisprSQL Tsai: guide20 -> off-target 20mer set
guides = defaultdict(set)
with open('data/crisprsql/100720.csv') as f:
    for row in csv.DictReader(f):
        if row['study_name'] == 'Tsai':
            g = row['grna_target_sequence'].strip().upper()[:20]
            t = row['target_sequence'].strip().upper()[:20]
            guides[g].add(t)
guide_list = sorted(guides, key=lambda g: -len(guides[g]))[:12]

files = {'VEGFA_TS1': 'moesm4.xlsx', 'VEGFA_TS2': 'moesm5.xlsx', 'VEGFA_TS3': 'moesm6.xlsx',
         'EMX1': 'moesm7.xlsx', 'RNF2': 'moesm8.xlsx'}
out = {}
for target, fn in files.items():
    wb = openpyxl.load_workbook(f'data/raw/peacseq/{fn}', read_only=True)
    ws = wb.active
    sites = []
    for r in ws.iter_rows(min_row=4, values_only=True):
        seq = r[8]
        if seq is None:
            continue
        seq = str(seq).strip().upper()
        if len(seq) < 20 or set(seq) - set('ACGT'):
            continue
        enrich = float(r[11]) if r[11] is not None else 0.0
        sites.append((seq, enrich))
    # assign guide: exact 20-mer match in any window/strand
    best = None
    for g in guide_list:
        hits = sum(1 for s, _ in sites
                   if min(hamming(s[w:w+20], g) for w in range(len(s) - 19)) == 0
                   or min(hamming(rc(s)[w:w+20], g) for w in range(len(s) - 19)) == 0)
        if best is None or hits > best[1]:
            best = (g, hits)
    g, hits = best
    if hits == 0:
        out[target] = dict(n_sites=len(sites), guide=None)
        continue
    mm_enr, in_crisp = [], []
    peac20 = set()
    for s, e in sites:
        cand = [s[w:w+20] for w in range(len(s) - 19)] + [rc(s)[w:w+20] for w in range(len(s) - 19)]
        mm = min(hamming(c, g) for c in cand)
        s20 = min(cand, key=lambda c: hamming(c, g))
        peac20.add(s20)
        mm_enr.append((mm, e))
        in_crisp.append((s20 in guides[g], e))
    cs = guides[g]
    inter = len(peac20 & cs)
    by_mm = {}
    for mm, e in mm_enr:
        by_mm.setdefault(mm, []).append(e)
    enr_in = [e for x, e in in_crisp if x]
    enr_out = [e for x, e in in_crisp if not x]
    mwu = float(mannwhitneyu(enr_in, enr_out)[1]) if enr_in and enr_out else None
    out[target] = dict(n_sites=len(sites), guide=g, guide_hits=hits,
                       crisprsql_tsai_sites=len(cs), overlap=inter,
                       jaccard=round(inter / len(peac20 | cs), 3) if peac20 | cs else 0,
                       mm_curve={str(k): dict(n=len(v), median_enrich=round(float(np.median(v)), 2))
                                 for k, v in sorted(by_mm.items()) if k <= 7},
                       enrich_mwu_crisprsql_vs_not_p=mwu)
json.dump(out, open('results/peacseq_crossmethod.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
