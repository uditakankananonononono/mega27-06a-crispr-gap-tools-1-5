# mega27-06a: CRISPR/biohacking gap tools 1-5

Part of the user's 27-item computational-biology mega-program. Item 6 first half: 5 of 10
verified CRISPR/biohacking research gaps, each built as a real tool with CNN/GNN scoring,
benchmarked against public leaders (CRISPRon / DeepCRISPR / Rule Set 3 class) on public
datasets. No stubs, no simulations of results; hermetic pytest suites (live dataset calls
only outside CI). Math derivations + proofs go in the per-tool papers.

Public dataset sources (real):
- On-target efficacy: CRISPRon (Xiang et al. 2021, rth.dk), DeepCRISPR (bm2-lab), Rule Set 3 (gpp-rnd.github.io/rs3)
- Off-target: GUIDE-seq, CHANGE-seq, CIRCLE-seq compiled sets (dagrate/public_data_crisprCas9), crisprSQL, CRISPR-Bulge (OrensteinLab)
