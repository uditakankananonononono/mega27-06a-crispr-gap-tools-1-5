"""Gap 2: in-vitro -> cellular degradation on matched sites. CHANGE-seq
MOESM3 Table 3 (in-vitro reads) vs Table 7 (rhAmpSeq cellular indel %
at 151 shared sites, 3 RNP replicates vs 3 controls)."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    xl = pd.ExcelFile("data/raw/changeseq/moesm3.xlsx")
    t3 = xl.parse("CHANGE-seq_Supp_Table_3")
    t7 = xl.parse("CHANGE-seq_Supp_Table_7")
    # cellular indel frequency: mean of 3 RNP replicates minus mean control
    indel_rnp = sum(pd.to_numeric(t7[f"Indel_RNP_{i}"], errors="coerce") for i in (1, 2, 3))
    tot_rnp = sum(pd.to_numeric(t7[f"Total_RNP_{i}"], errors="coerce") for i in (1, 2, 3))
    indel_ctl = sum(pd.to_numeric(t7[f"Indel_control_{i}"], errors="coerce") for i in (1, 2, 3))
    tot_ctl = sum(pd.to_numeric(t7[f"Total_control_{i}"], errors="coerce") for i in (1, 2, 3))
    t7["cell_indel_pct"] = (indel_rnp / tot_rnp - indel_ctl / tot_ctl) * 100
    # match on guide + off-target sequence
    key7 = t7["Target Name"].astype(str) + "|" + t7["Off-target Sequence"].astype(str).str.upper()
    key3 = t3["name"].astype(str) + "|" + t3["offtarget_sequence"].astype(str).str.upper()
    rmap = dict(zip(key3, t3["CHANGEseq_reads"]))
    t7["change_reads"] = [rmap.get(k) for k in key7]
    m = t7.dropna(subset=["change_reads", "cell_indel_pct"])
    out = {"n_matched": int(len(m)),
           "spearman_change_vs_cellular": round(float(
               spearmanr(m["change_reads"], m["cell_indel_pct"]).statistic), 3),
           "frac_cell_positive": float((m.cell_indel_pct > 0.1).mean()),
           "cell_indel_median_pos": float(m.loc[m.cell_indel_pct > 0.1, "cell_indel_pct"].median()),
           "change_reads_median": float(m.change_reads.median())}
    json.dump(out, open("results/changeseq_cellular.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
