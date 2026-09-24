"""Detect-seq (Lei 2021, Nat Methods 10.1038/s41592-021-01172-w, MOESM5):
CBE off-target site sets per target/cell line. (a) Detect-seq (CBE
chemistry) vs crisprSQL GUIDE-seq (nuclease, Tsai) for VEGFA site 2 /
EMX1 / HEK293 site 4 - coordinate overlap (+-50bp windows).
(b) Same method, two cell lines: HEK293 site 4 in HEK293T vs MCF7.
(c) RUNX1: Cpf1BE vs BE4max site counts."""
import json
import csv
import openpyxl
from collections import defaultdict


def load_sites(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    out = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] and r[1] is not None and r[2] is not None:
            out.append((str(r[0]).replace('chr', ''), int(r[1]), int(r[2])))
    return out


def overlap(a, b, pad=50):
    by_chrom = defaultdict(list)
    for c, s, e in b:
        by_chrom[c].append((s, e))
    n = 0
    for c, s, e in a:
        if any(not (e + pad < bs or s - pad > be) for bs, be in by_chrom.get(c, [])):
            n += 1
    return n

d_v2 = load_sites('data/raw/detectseq/moesm5.xlsx', 'HEK293T.VEGFA')
d_emx1 = load_sites('data/raw/detectseq/moesm5.xlsx', 'HEK293T.EMX1')
d_h4_293 = load_sites('data/raw/detectseq/moesm5.xlsx', 'HEK293T.HEK293 site 4')
d_h4_mcf7 = load_sites('data/raw/detectseq/moesm5.xlsx', 'MCF7.HEK293 site 4')
d_cpf1be = load_sites('data/raw/detectseq/moesm5.xlsx', 'HEK293T.Cpf1BE.RUNX1')
d_be4max = load_sites('data/raw/detectseq/moesm5.xlsx', 'HEK293T.BE4max.RUNX1')

# crisprSQL GUIDE-seq (Tsai) sites by guide, hg19
GS = {'GACCCCCTCCACCCCGCCTC': 'VEGFA2', 'GAGTCCGAGCAGAAGAAGAA': 'EMX1',
      'GGCACTGCGGCTGGAGGTGG': 'HEK4'}
cs = defaultdict(list)
genomes = defaultdict(set)
with open('data/crisprsql/100720.csv') as f:
    for row in csv.DictReader(f):
        if row['study_name'] == 'Tsai' and row['grna_target_sequence'][:20] in GS:
            g = GS[row['grna_target_sequence'][:20]]
            cs[g].append((row['target_chr'].replace('chr', ''), int(float(row['target_start'])), int(float(row['target_end']))))
            genomes[g].add(row['genome'])

out = dict(
    detectseq_counts=dict(VEGFA2=len(d_v2), EMX1=len(d_emx1),
                          HEK4_HEK293T=len(d_h4_293), HEK4_MCF7=len(d_h4_mcf7),
                          RUNX1_Cpf1BE=len(d_cpf1be), RUNX1_BE4max=len(d_be4max)),
    crisprsql_tsai_counts={g: len(v) for g, v in cs.items()},
    crisprsql_genomes={g: sorted(x) for g, x in genomes.items()},
)
for name, dset in (('VEGFA2', d_v2), ('EMX1', d_emx1), ('HEK4', d_h4_293)):
    if cs[name]:
        ov = overlap(dset, cs[name])
        out[f'detectseq_vs_guideseq_{name}'] = dict(n_detect=len(dset), n_guide=len(cs[name]),
                                                    overlap=ov,
                                                    jaccard=round(ov / (len(dset) + len(cs[name]) - ov), 3))
ov_cc = overlap(d_h4_293, d_h4_mcf7)
out['hek4_hek293t_vs_mcf7'] = dict(overlap=ov_cc,
                                   jaccard=round(ov_cc / (len(d_h4_293) + len(d_h4_mcf7) - ov_cc), 3))
json.dump(out, open('results/detectseq_overlap.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
