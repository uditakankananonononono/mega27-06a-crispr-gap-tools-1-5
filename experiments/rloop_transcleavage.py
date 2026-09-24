"""Cas9 trans-cleavage modulation by spacer length and mismatch position
(Nat Commun 2026, 10.1038/s41467-026-68789-3, MOESM6): Fig 2A/B mismatch
series (positions 5/8/13/19, double 5-6, triple 4-5-6; random vs short
dsDNA; triplicate) and Fig 3A spacer-length series (49..20 nt).
Gap-1/4: how do spacer length and mismatch position tune collateral
trans-cleavage?"""
import json
import numpy as np
import openpyxl
from scipy.stats import spearmanr

wb = openpyxl.load_workbook('data/raw/rloop_transcleavage/moesm6.xlsx', read_only=True)

def panel_rows(ws, start_row):
    """3 replicate rows starting at start_row (0-based), cols 1.."""
    rows = list(ws.iter_rows(values_only=True))
    return rows

ws = wb['Fig2']
rows = list(ws.iter_rows(values_only=True))

def grid(row_start, cols):
    out = {}
    for name, c in cols.items():
        vals = []
        for dr in range(3):
            try:
                vals.append(float(rows[row_start + dr][c]))
            except (TypeError, ValueError, IndexError):
                pass
        out[name] = vals
    return out

# Panel A starts row 3 (R1); col pairs: (random, short) per position
posA = {'original': (1, 2), 'mm5': (3, 4), 'mm8': (5, 6), 'mm13': (7, 8), 'mm19': (9, 10),
        'mm5-6': (11, 12), 'mm4-5-6': (13, 14)}
panelA = {}
for name, (cr, cs) in posA.items():
    rv = [float(rows[4 + i][cr]) for i in range(3) if rows[4 + i][cr] is not None]
    sv = [float(rows[4 + i][cs]) for i in range(3) if rows[4 + i][cs] is not None]
    panelA[name] = {'random': float(np.mean(rv)) if rv else None,
                    'short': float(np.mean(sv)) if sv else None}
# Panel B starts row 10 (R1)
panelB = {}
for name, (cr, cs) in posA.items():
    try:
        rv = [float(rows[14 + i][cr]) for i in range(3)]
        sv = [float(rows[14 + i][cs]) for i in range(3)]
        panelB[name] = {'random': float(np.mean(rv)), 'short': float(np.mean(sv))}
    except (TypeError, ValueError, IndexError):
        pass

# Fig 3A spacer lengths
ws3 = wb['Fig3']
rows3 = list(ws3.iter_rows(values_only=True))
lengths = [49, 35, 32, 29, 26, 23, 20]
spacer = {}
for i, L in enumerate(lengths):
    vals = [float(rows3[3 + dr][2 + i]) for dr in range(3)]
    spacer[L] = float(np.mean(vals))
spacer['NT'] = float(np.mean([float(rows3[3 + dr][1]) for dr in range(3)]))
slen = [L for L in lengths]
svals = [spacer[L] for L in slen]
rho_len = float(spearmanr(slen, svals).statistic)

out = {
    'fig2_panelA_by_mismatch': panelA,
    'fig2_panelB_by_mismatch': panelB,
    'fig3_spacer_signal': spacer,
    'spacer_length_spearman': rho_len,
    'spacer_max_over_min': max(svals) / min(svals),
    'triple_mismatch_suppression': panelA['mm4-5-6']['short'] / panelA['mm5']['short'],
}
print(json.dumps(out, indent=1))
json.dump(out, open('results/rloop_transcleavage.json', 'w'), indent=1)
