"""Gap-3 cross-library transfer: train on DeepPE Library 1 (Kim 2021,
43k pegRNAs, HEK293T PE2), test on PRIDICT2 Library 1 (Mathis 2023,
92k pegRNAs), and reverse. Five z-scored design features mapped between
the libraries: PBS length, RT length, PBS GC%, RT GC%, Cas9 score
(DeepSpCas9 <-> deepcas9)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

DEEP = {"PBS length": "pbs", "RT length": "rt",
        "GC contents_1\n(PBS)": "gc_pbs", "GC contents_2\n(RT)": "gc_rt",
        "DeepSpCas9 score": "cas9"}
PRID = {"PBSlength": "pbs", "RTlength": "rt", "PBS_GC_content": "gc_pbs",
        "RT_GC_content": "gc_rt", "deepcas9": "cas9"}


def load_deeppe():
    df = pd.read_excel("data/raw/deeppe/moesm4.xlsx",
                       sheet_name="Library 1 (HT-training, test)", skiprows=1)
    y = pd.to_numeric(df["Measured PE efficiency"], errors="coerce")
    X = df[list(DEEP)].rename(columns=DEEP).apply(pd.to_numeric, errors="coerce")
    ok = y.notna() & X.notna().all(axis=1)
    return X[ok], y[ok]


def load_pridict2():
    df = pd.read_parquet("/tmp/pridict2_l1.parquet")
    y = pd.to_numeric(df["averageedited"], errors="coerce")
    X = df[list(PRID)].rename(columns=PRID).apply(pd.to_numeric, errors="coerce")
    ok = y.notna() & X.notna().all(axis=1)
    return X[ok], y[ok]


def z(X):
    Z = (X - X.mean()) / X.std()
    return Z.fillna(0.0)  # constant features carry no signal


Xd, yd = load_deeppe()
Xp, yp = load_pridict2()
Xd, Xp = z(Xd), z(Xp)

r1 = Ridge(alpha=1.0).fit(Xd, yd)
t1 = float(spearmanr(r1.predict(Xp), yp).statistic)
r2 = Ridge(alpha=1.0).fit(Xp, yp)
t2 = float(spearmanr(r2.predict(Xd), yd).statistic)

out = dict(n_deeppe=int(len(yd)), n_pridict2=int(len(yp)),
           features=list(DEEP.values()),
           deeppe_to_pridict2=round(t1, 3), pridict2_to_deeppe=round(t2, 3),
           deeppe_within_cv=0.669, pridict2_within_cv=0.523,
           note='within-CV references from results/gap3_deeppe.json and results/pridict2_pechromatin.json')
json.dump(out, open("results/pe_cross_library_transfer.json", "w"), indent=1)
print(json.dumps(out, indent=1))
