"""Gap 3: DeepPE Library 2 (Kim 2021 MOESM4 sheet 2; 6,152 pegRNAs,
position/type library). Edit position = index of first lowercase base
in the 3' extension (5'->3'); edit multiplicity = number of lowercase
bases. Question: how do edit position and multiplicity shape PE
efficiency? (Lowercase encodes edited bases; substitution-vs-indel
identity is not resolved from this encoding - stated in paper.)"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    df = pd.read_excel("data/raw/deeppe/moesm4.xlsx",
                       sheet_name="Library 2 (Position, Type)", skiprows=1)
    ext = df["3' extension sequence of pegRNA"].astype(str)
    first = ext.map(lambda s: next((i for i, c in enumerate(s) if c.islower()), np.nan))
    nedit = ext.map(lambda s: sum(c.islower() for c in s))
    y = pd.to_numeric(df["Measured PE efficiency"], errors="coerce")
    ok = y.notna() & first.notna()
    df2 = pd.DataFrame({"first_edit": first[ok].astype(int), "n_edit": nedit[ok].astype(int),
                        "y": y[ok]})
    out = {"n": int(len(df2)),
           "first_edit_range": [int(df2.first_edit.min()), int(df2.first_edit.max())],
           "spearman_position": round(float(spearmanr(df2.first_edit, df2.y).statistic), 3),
           "spearman_nedit": round(float(spearmanr(df2.n_edit, df2.y).statistic), 3),
           "by_position": df2.groupby("first_edit").y.mean().round(2).to_dict(),
           "by_nedit": df2.groupby("n_edit").y.mean().round(2).to_dict()}
    json.dump(out, open("results/gap3_deeppe_lib2.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:900])


if __name__ == "__main__":
    main()
