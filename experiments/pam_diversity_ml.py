"""Metagenome-mined Cas9 PAM diversity + ML PAM prediction (Nat Commun 2026,
10.1038/s41467-026-69098-5, MOESM8): per-cluster PAM-prediction accuracies by
confidence bin (Fig 2c/2d), and ClinVar pathogenic-variant targetability of
mined PAMs vs SpCas9 NGG (Supp Fig 8a/8b).
Gap-2/4: is ML PAM-prediction confidence calibrated, and how much does mined
PAM diversity expand the targetable pathogenic-variant space?"""
import json
import numpy as np
import openpyxl

wb = openpyxl.load_workbook('data/raw/pam_diversity_ml/moesm8.xlsx', read_only=True)

# Fig 2c: median accuracy by confidence threshold
thresh, med_acc, counts = [], [], []
for i, row in enumerate(wb['Fig. 2c'].iter_rows(values_only=True)):
    if i == 0:
        continue
    if row[0] is not None:
        thresh.append(float(row[0])); med_acc.append(float(row[1])); counts.append(int(row[2]))

# Fig 2d: per-cluster accuracies in confidence bins 0.7-0.8, 0.8-0.9, 0.9-1.0
bins = {'0.7-0.8': [], '0.8-0.9': [], '0.9-1.0': []}
for i, row in enumerate(wb['Fig. 2d'].iter_rows(values_only=True)):
    if i == 0:
        continue
    for k, key in enumerate(bins):
        if row[k] is not None:
            bins[key].append(float(row[k]))
bin_stats = {k: {'n': len(v), 'mean_accuracy': float(np.mean(v))} for k, v in bins.items()}

# Supp Fig 8: ClinVar variant targetability, mined PAMs vs SpCas9 NGG
clinvar = {}
for sheet, key in (('Supplementary Fig. 8a', 'G_to_A'), ('Supplementary Fig. 8b', 'T_to_C')):
    vals = {}
    for i, row in enumerate(wb[sheet].iter_rows(values_only=True)):
        if i == 0 or row[0] is None:
            continue
        label = ' '.join(str(row[0]).split())
        vals[label] = {'count': int(row[1]), 'pct': float(row[2])}
    clinvar[key] = vals

out = {
    'confidence_threshold_calibration': [
        {'min_confidence': t, 'median_accuracy': a, 'n_cas9_clusters': c}
        for t, a, c in zip(thresh, med_acc, counts)],
    'accuracy_by_confidence_bin': bin_stats,
    'clinvar_targetability': clinvar,
    'calibration_gap_at_top_bin': float(1.0 - bin_stats['0.9-1.0']['mean_accuracy']),
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/pam_diversity_ml.json', 'w'), indent=1)
