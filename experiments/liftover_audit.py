"""Assembly harmonization audit: lift crisprSQL hg19 loci to hg38 with
pyliftover (UCSC chain). How many survive, and do lifted coordinates still
match the recorded target sequence? Checks: lift rate, sequence identity
at lifted locus vs recorded target_sequence (revcomp-aware).
"""
import json

import numpy as np
import pandas as pd
from pyliftover import LiftOver
from pyfaidx import Fasta

from crisprgap.sequence import reverse_complement


def main():
    lo = LiftOver("data/hg19ToHg38.over.chain.gz")
    fa = Fasta("data/hg38.fa")
    df = pd.read_csv("data/crisprsql/100720.csv", low_memory=False)
    h = df[df.genome == "hg19"].reset_index(drop=True)
    lifted, seq_ok, seq_rc, unmapped, other = 0, 0, 0, 0, 0
    for r in h.itertuples():
        try:
            chrom, s, e = str(r.target_chr), int(r.target_start), int(r.target_end)
        except (ValueError, TypeError):
            other += 1
            continue
        out = lo.convert_coordinate(chrom, s)
        if not out:
            unmapped += 1
            continue
        lifted += 1
        nchrom, ns, nstrand = out[0][0], out[0][1], out[0][2]
        rec = str(r.target_sequence).upper()
        ln = len(rec)
        # crisprSQL intervals cover 22 of the 23 recorded bases; the extra
        # base sits 5' of the interval on + strand and 3' on - strand
        try:
            cands = [fa[nchrom][ns - 1:ns - 1 + ln].seq.upper(),
                     fa[nchrom][ns:ns + ln].seq.upper()]
            cands += [reverse_complement(c) for c in cands]
        except Exception:
            other += 1
            continue
        if rec in cands:
            seq_ok += 1
        else:
            other += 1
    out = {
        "n_hg19_rows": int(len(h)),
        "lifted": int(lifted),
        "unmapped": int(unmapped),
        "lifted_seq_matches": int(seq_ok),
        "lifted_seq_revcomp": int(seq_rc),
        "other_or_mismatch": int(other),
        "lift_rate": lifted / len(h),
        "seq_concordance_of_lifted": seq_ok / lifted if lifted else None,
    }
    json.dump(out, open("results/liftover_audit.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
