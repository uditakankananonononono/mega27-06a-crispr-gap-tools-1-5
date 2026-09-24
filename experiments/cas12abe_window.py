"""Cas12a base editors (Nat Commun 2025, 10.1038/s41467-025-59653-x,
MOESM6 Fig.2F): per-cytosine editing % across positions C3-C8 by BE
system - editing-window width (positions with >5% mean activity)."""
import json
import openpyxl
import numpy as np
from collections import defaultdict

wb = openpyxl.load_workbook('data/raw/cas12a_be/moesm6.xlsx', read_only=True)
ws = wb['Fig.2F']
data = defaultdict(lambda: defaultdict(list))
cur_be = None
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is not None and str(r[0]).strip():
        cur_be = str(r[0])
    if cur_be is None or r[4] is None:
        continue
    be, pos = cur_be, str(r[4])
    try:
        vals = [float(x) for x in r[5:8] if x is not None]
    except (TypeError, ValueError):
        continue
    if vals:
        data[be][pos].append(float(np.mean(vals)))

out = {}
for be, d in data.items():
    pos_mean = {p: round(float(np.mean(v)), 2) for p, v in sorted(d.items())}
    if len(pos_mean) >= 4:
        width = sum(1 for x in pos_mean.values() if x > 5)
        peak = max(pos_mean, key=pos_mean.get)
        out[be] = dict(positions=pos_mean, window_width_gt5pct=width, peak=peak)
res = dict(n_be_systems=len(out), windows=out)
json.dump(res, open('results/cas12abe_window.json', 'w'), indent=1)
for be, d in out.items():
    print(be, d['positions'], 'width:', d['window_width_gt5pct'])
