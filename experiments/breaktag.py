"""BreakTag (Scheller/Nabet? Nat Protoc 2025, 10.1038/s41596-025-01271-4):
(a) MOESM2 BreakTag VEGFA site 1 off-targets (hg38) vs crisprSQL Tsai
GUIDE-seq VEGFA site 1 lifted hg19->hg38: rediscovery rate.
(b) MOESM4: blunt vs staggered scission profile by Cas9 variant."""
import json
import csv
import openpyxl
from collections import defaultdict
from pyliftover import LiftOver

lo = LiftOver('data/hg19ToHg38.over.chain.gz')
cs = []
with open('data/crisprsql/100720.csv') as f:
    for row in csv.DictReader(f):
        if row['study_name'] == 'Tsai' and row['grna_target_sequence'].startswith('GGGTGGGGGGAGTTTGCTCC'):
            cs.append((row['target_chr'].replace('chr', ''), int(float(row['target_start']))))
cs38 = [(c, r[0][1]) for c, s in cs if (r := lo.convert_coordinate('chr' + c, s))]

wb = openpyxl.load_workbook('data/raw/breaktag/moesm2.xlsx', read_only=True)
bt = [(str(r[0]).replace('chr', ''), int(r[1]))
      for r in wb[list(wb.sheetnames)[0]].iter_rows(min_row=2, values_only=True)
      if r[0] and r[5] == 'VEGFA_site_1']
ov = sum(1 for c, s in bt if any(c == c2 and abs(s - s2) <= 50 for c2, s2 in cs38))

wb4 = openpyxl.load_workbook('data/raw/breaktag/moesm4.xlsx', read_only=True)
variants = defaultdict(list)
for r in wb4[list(wb4.sheetnames)[0]].iter_rows(min_row=2, values_only=True):
    if r[0] and r[6]:
        try:
            variants[str(r[6])].append(float(r[5]))
        except (TypeError, ValueError):
            pass

out = dict(
    vegfa1=dict(n_breaktag=len(bt), n_guideseq=len(cs38), overlap=ov,
                guideseq_rediscovered=round(ov / len(cs38), 3),
                jaccard=round(ov / (len(bt) + len(cs38) - ov), 3)),
    variant_blunt={k: dict(n=len(v),
                           frac_blunt=round(sum(1 for x in v if x > 0.5) / len(v), 3))
                   for k, v in variants.items()},
)
json.dump(out, open('results/breaktag.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
