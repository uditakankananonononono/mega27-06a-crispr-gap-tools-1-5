# mega27-06a: CRISPR/biohacking gap tools 1-5

Part of the user's 27-item computational-biology mega-program. Item 6 first half: 5 of 10
verified CRISPR/biohacking research gaps, each built as a real tool with CNN/GNN scoring,
benchmarked against public leaders (CRISPRon / DeepCRISPR / Rule Set 3 class) on public
datasets. No stubs, no simulations of results; hermetic pytest suites (live dataset calls
only outside CI). Math derivations + proofs go in the per-tool papers.

Public dataset sources (real):
- On-target efficacy: CRISPRon (Xiang et al. 2021, rth.dk), DeepCRISPR (bm2-lab), Rule Set 3 (gpp-rnd.github.io/rs3)
- Off-target: GUIDE-seq, CHANGE-seq, CIRCLE-seq compiled sets (dagrate/public_data_crisprCas9), crisprSQL, CRISPR-Bulge (OrensteinLab)

## Command-line tool

The `crisprgap` CLI exposes the validated scoring and calibration pieces. All output is JSON.

```bash
python -m crisprgap cfd GAGTCCGAGCAGAAGAAGAAGGG GAGTCCGAGCAGAAGAAAAAGGG   # CFD score (Doench 2016)
python -m crisprgap mit GAGTCCGAGCAGAAGAAGAA GAGTCCGAGCAGAAGAAAAA         # MIT specificity (Hsu 2013)
python -m crisprgap calibrate scores.csv --fit platt                      # ECE/Brier + Platt fit (CSV: score,label)
python -m crisprgap seqstats GAGTCCGAGCAGAAGAAGAAGGG                      # GC + dinucleotide composition
```

CFD is defined only for substitution-only ACGT 23-mers; the CLI rejects indels and non-canonical bases with exit code 2 and `"applicable": false`.

## Evidence audit (September 25, 2026)

The paper's 44-tool and 124-dataset headings are provisional inventory counts, not verified passes against the strict research/data-tool and accession gates. See `EVIDENCE_AUDIT.md`; the updated PDF embeds Times New Roman text (math fonts remain Computer Modern). Research-tool and dataset evidence work remains open.
