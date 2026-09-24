"""Curve-modeling extras over committed results (no new downloads):
- lmfit Hill fits to the in-vivo LNP dose ladder (ds118) and the spacer
  ladder (ds124);
- kneed knee-point detection on the LNP dose-response (gap-3);
- ruptures change-point detection on the Sdd7/BE4max positional editing
  profiles (ds117; gap-2 window edges);
- dcor distance correlation between editor site-activity vectors (ds117;
  gap-1 nonlinear dependence beyond Spearman).
"""
import json
import numpy as np
from lmfit import Model
from kneed import KneeLocator
import dcor
import ruptures as rpt

out = {}

# --- Hill fit: LNP dose response (ds118 committed JSON) ---
d118 = json.load(open('results/lnp_prime_editing.json'))
doses = [0.5, 1, 2, 4]
edit = [d118['dose_response'][f'{d} mg/kg']['editing'] for d in doses]
ind = [d118['dose_response'][f'{d} mg/kg']['indels'] for d in doses]

def hill(x, emax, ec50, n):
    return emax * x**n / (ec50**n + x**n)

m = Model(hill)
par = m.make_params(emax=60, ec50=1, n=2)
par['n'].set(min=0.3, max=8)
fit_e = m.fit(edit, par, x=np.array(doses))
fit_i = m.fit(ind, par, x=np.array(doses))
out['lnp_hill_editing'] = {k: float(fit_e.params[k].value) for k in ('emax', 'ec50', 'n')}
out['lnp_hill_editing']['rsquared'] = float(fit_e.rsquared)
out['lnp_hill_indels'] = {k: float(fit_i.params[k].value) for k in ('emax', 'ec50', 'n')}
out['lnp_hill_indels']['rsquared'] = float(fit_i.rsquared)

# --- knee of the dose curve (linear-ish x axis in log dose) ---
kl = KneeLocator(np.log2(doses), edit, curve='concave', direction='increasing')
out['lnp_knee_log2dose'] = float(kl.knee) if kl.knee is not None else None

# --- spacer ladder Hill-like monotone fit (ds124) ---
d124 = json.load(open('results/rloop_transcleavage.json'))
L = np.array([49, 35, 32, 29, 26, 23, 20], dtype=float)
sig = np.array([d124['fig3_spacer_signal'][str(int(x))] for x in L])
# signal vs 1/L: linear fit
a, b = np.polyfit(1.0 / L, sig, 1)
pred = a / L + b
ss_res = float(np.sum((sig - pred) ** 2))
ss_tot = float(np.sum((sig - sig.mean()) ** 2))
out['spacer_inverse_length_fit'] = {'slope': float(a), 'intercept': float(b),
                                    'r2': 1 - ss_res / ss_tot}

# --- ruptures change points on positional profiles (ds117) ---
d117 = json.load(open('results/sdd7_cbe_window.json'))
cp = {}
for ed, prof in d117['positional_profiles'].items():
    y = np.array(prof['profile'], dtype=float)
    algo = rpt.Pelt(model='l2').fit(y)
    # penalty tuned to detect the window flanks, not noise
    bkps = algo.predict(pen=3.0)
    cp[ed] = {'changepoints': [int(b) for b in bkps], 'profile': [float(v) for v in y]}
out['window_changepoints'] = cp

# --- dcor between editor site-activity vectors (ds117) ---
# rebuild per-site editor vectors from the experiment source of truth:
import openpyxl
wb = openpyxl.load_workbook('data/raw/sdd7_cbe/moesm8.xlsx', read_only=True)
rows = list(wb['Figure 3b'].iter_rows(values_only=True))
blocks, cur = [], {}
for r in rows[2:]:
    c0 = str(r[0]).strip() if r[0] is not None else ''
    if c0 == 'BE4max' or (not c0 and r[1] == 'BE4max'):
        if cur:
            blocks.append(cur)
        cur = {}
        continue
    if not c0:
        continue
    try:
        cur[c0] = [float(r[k]) for k in (1, 2, 3, 4)]
    except (TypeError, ValueError):
        continue
if cur:
    blocks.append(cur)
primary = blocks[0]
editors = ['BE4max', 'Sdd7', 'Sdd7e1', 'Sdd7e2']
sites = sorted(primary)
vecs = {ed: np.array([primary[s][i] for s in sites]) for i, ed in enumerate(editors)}
dc = {}
for i, a in enumerate(editors):
    for b in editors[i + 1:]:
        dc[f'{a}_vs_{b}'] = float(dcor.distance_correlation(vecs[a], vecs[b]))
out['editor_distance_correlation'] = dc
out['editor_distance_correlation_min'] = float(min(dc.values()))

print(json.dumps(out, indent=1))
json.dump(out, open('results/curve_modeling_extras.json', 'w'), indent=1)
