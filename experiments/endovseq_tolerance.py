"""EndoV-seq in-vitro mismatch tolerance (Wienert 2019, Nat Commun
10.1038/s41467-018-07988-z, MOESM4 Supp Fig 1): ABE7.10 deamination vs
Cas9 cleavage on identical mismatched RNA substrates for 3 guides.
Retention relative to each guide's on-target, by mismatch count and
position (pos 1 = PAM-distal)."""
import json
import openpyxl
import numpy as np

wb = openpyxl.load_workbook('data/raw/endovseq/moesm4.xlsx', read_only=True)
ws = wb['Supplementary Figure 1']
rows = []
for r in ws.iter_rows(min_row=6, values_only=True):
    if r[1] is not None:
        rows.append((str(r[1]).strip(), r[2:5], r[5:8]))

on_targets = [x for x in rows if x[0].isupper() and 'U' in x[0]]
per_guide = {}
for ot in on_targets:
    seq = ot[0]
    per_guide[seq] = dict(on_abe=np.mean(ot[1]), on_cas9=np.mean(ot[2]), variants=[])

for label, abe, cas9 in rows:
    if label == 'GFP control' or (label.isupper() and 'U' in label):
        continue
    for ot in on_targets:
        seq = ot[0]
        if len(label) == len(seq) and sum(1 for a, b in zip(label, seq) if a.upper() == b) >= len(seq) - 4:
            mms = [(i + 1, a, b) for i, (a, b) in enumerate(zip(label, seq)) if a.upper() != b]
            # lowercase letters mark the substituted bases
            mms = [(i + 1, label[i], seq[i]) for i in range(len(seq)) if label[i].islower() or label[i].upper() != seq[i]]
            mms = [(i + 1) for i in range(len(seq)) if label[i].upper() != seq[i] or label[i].islower()]
            per_guide[seq]['variants'].append(dict(
                n_mm=len(mms), positions=mms,
                abe=float(np.mean(abe)), cas9=float(np.mean(cas9))))
            break

all_var = [v for g in per_guide.values() for v in g['variants']]
on_abe = np.mean([g['on_abe'] for g in per_guide.values()])
on_cas9 = np.mean([g['on_cas9'] for g in per_guide.values()])

by_nmm = {}
for v in all_var:
    by_nmm.setdefault(v['n_mm'], []).append((v['abe'], v['cas9']))
ret_by_nmm = {k: dict(n=len(vs),
                      abe_retention=round(float(np.mean([a for a, _ in vs])) / on_abe, 3),
                      cas9_retention=round(float(np.mean([c for _, c in vs])) / on_cas9, 3))
              for k, vs in sorted(by_nmm.items())}

pos_abe, pos_cas9 = {}, {}
for v in all_var:
    if v['n_mm'] == 1:
        p = v['positions'][0]
        pos_abe.setdefault(p, []).append(v['abe'])
        pos_cas9.setdefault(p, []).append(v['cas9'])
seed = [p for p in range(13, 21)]
distal = [p for p in range(1, 8)]
out = dict(
    n_guides=len(per_guide), n_variants=len(all_var),
    on_target_mean=dict(abe=round(on_abe, 2), cas9=round(on_cas9, 2)),
    retention_by_mismatch_count=ret_by_nmm,
    single_mm_seed_vs_distal=dict(
        abe_seed=round(float(np.mean([x for p in seed for x in pos_abe.get(p, [])])), 2),
        abe_distal=round(float(np.mean([x for p in distal for x in pos_abe.get(p, [])])), 2),
        cas9_seed=round(float(np.mean([x for p in seed for x in pos_cas9.get(p, [])])), 2),
        cas9_distal=round(float(np.mean([x for p in distal for x in pos_cas9.get(p, [])])), 2)),
)
json.dump(out, open('results/endovseq_tolerance.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
