"""Gap 4: mismatch tolerance of 13 Cas9 variants (Kim 2020 MOESM3
mismatch sheets, Day 4 and Day 7, 3,120 rows each). Per variant: mean
background-subtracted indel frequency by guide-target mismatch count,
and D4-vs-D7 consistency. Background-subtracted values used where
present; raw day-minus-day0 rates otherwise."""
import json

import numpy as np
import pandas as pd

VARIANTS = ["SpCas9", "xCas9", "SpCas9-NG", "eSpCas9(1.1)", "SpCas9-HF1",
            "HypaCas9", "evoCas9", "Sniper-Cas9", "VRQR-HF1", "VQR",
            "VRQR", "VRER", "QQR1"]


def load(sheet, day):
    df = pd.read_excel("data/raw/deepspcas9variants/moesm3.xlsx", sheet_name=sheet)
    guide = df["Guide RNA sequence"].astype(str).str.upper()
    ctx = df["Target context sequence"].astype(str).str.upper()
    mm = []
    for g, c in zip(guide, ctx):
        tgt = c[4:24]  # guide window inside 30-mer context
        mm.append(sum(a != b for a, b in zip(g, tgt)) if len(tgt) == 20 else np.nan)
    df["mm"] = mm
    out = {}
    for v in VARIANTS:
        fcol = f"Background subtracted indel frequencies\n(%, {v}, Day {day})"
        freq = pd.to_numeric(df[fcol], errors="coerce")
        # fallback: raw day4/day7 rate minus day0 rate
        raw = (pd.to_numeric(df[f"Indel read count\n{v}, Day {day}"], errors="coerce") /
               pd.to_numeric(df[f"Total read count\n{v}, Day {day}"], errors="coerce") * 100 -
               pd.to_numeric(df["Indel read count\n(Day 0)"], errors="coerce") /
               pd.to_numeric(df["Total read count (Day 0)"], errors="coerce") * 100)
        freq = freq.fillna(raw)
        df[v] = freq.clip(lower=0)
        by = df.dropna(subset=["mm"]).groupby("mm")[v].mean()
        out[v] = {int(k): round(float(val), 2) for k, val in by.items()}
    return df, out


def main():
    d4, by4 = load("Mismatch target sequences(D4)", 4)
    d7, by7 = load("Mismatch target sequences(D7)", 7)
    out = {"n_rows": int(len(d4)), "day4": by4, "day7": by7,
           "mm_counts": d4.mm.value_counts().sort_index().to_dict()}
    # key gap-4 readout: activity at 1-2 mismatches, variant vs WT
    sel = {}
    for v in VARIANTS:
        m1 = by4[v].get(1, np.nan); m2 = by4[v].get(2, np.nan)
        sel[v] = {"mm1": m1, "mm2": m2}
    out["by_mm_day4"] = sel
    # WT vs HF ratio at 1 mismatch
    out["wt_mm1"] = by4["SpCas9"].get(1)
    out["hf_mm1"] = {v: by4[v].get(1) for v in ["eSpCas9(1.1)", "SpCas9-HF1", "HypaCas9", "evoCas9", "Sniper-Cas9"]}
    json.dump(out, open("results/gap4_mismatch_tolerance.json", "w"), indent=1)
    print("mm counts:", out["mm_counts"])
    print("WT mm1:", out["wt_mm1"], "| HF mm1:", out["hf_mm1"])
    print("WT by mm (d4):", by4["SpCas9"])
    print("xCas9 by mm:", by4["xCas9"])


if __name__ == "__main__":
    main()
