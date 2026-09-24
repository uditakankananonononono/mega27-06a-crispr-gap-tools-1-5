"""crisprgap CLI: off-target scoring, calibration, and sequence utilities.

Every subcommand prints a JSON object to stdout so the tool can be scripted.
Exit code 0 on success, 2 on invalid input.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _print(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def cmd_cfd(args: argparse.Namespace) -> int:
    from crisprgap.cfd import cfd_score, cfd_applicable
    wt, off = args.wildtype.upper(), args.offtarget.upper()
    if not cfd_applicable(wt, off):
        _print({
            "error": "CFD is defined only for substitution-only ACGT 23-mers "
                     "(no indels, same length, canonical bases)",
            "wildtype": wt, "offtarget": off, "applicable": False,
        })
        return 2
    _print({
        "score": float(cfd_score(wt, off)),
        "wildtype": wt, "offtarget": off,
        "applicable": True, "method": "CFD (Doench et al. 2016)",
    })
    return 0


def cmd_mit(args: argparse.Namespace) -> int:
    from crisprgap.baselines import mit_score
    g, off = args.guide.upper(), args.offtarget.upper()
    if len(g) != len(off) or not set(g + off) <= set("ACGT"):
        _print({
            "error": "MIT score needs two equal-length ACGT sequences",
            "guide": g, "offtarget": off, "applicable": False,
        })
        return 2
    _print({
        "score": float(mit_score(g, off)),
        "guide": g, "offtarget": off,
        "n_mismatches": int(sum(a != b for a, b in zip(g, off))),
        "applicable": True, "method": "MIT specificity (Hsu et al. 2013)",
    })
    return 0


def _load_scores_labels(path: str) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        try:
            si, li = header.index("score"), header.index("label")
        except ValueError:
            raise SystemExit(f"{path}: CSV needs 'score' and 'label' columns, got {header}")
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) <= max(si, li) or not parts[si]:
                continue
            scores.append(float(parts[si]))
            labels.append(int(float(parts[li])))
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)


def cmd_calibrate(args: argparse.Namespace) -> int:
    from crisprgap.calibration import (
        expected_calibration_error, brier_score, fit_platt, apply_platt,
        fit_isotonic,
    )
    scores, labels = _load_scores_labels(args.csv)
    n_pos = int(labels.sum())
    out: dict = {
        "n": int(len(labels)),
        "n_pos": n_pos,
        "prevalence": float(labels.mean()) if len(labels) else None,
        "raw": {
            "ece": float(expected_calibration_error(scores, labels)),
            "brier": float(brier_score(labels, scores)),
        },
    }
    if len(labels) < 50:
        out["warning"] = ("fewer than 50 samples: calibration fits on slices this "
                          "small are unstable (see paper sec. calibration starvation)")
    if args.fit == "platt":
        a, b = fit_platt(scores, labels)
        cal = apply_platt(scores, a, b)
        out["platt"] = {"a": float(a), "b": float(b),
                        "ece": float(expected_calibration_error(cal, labels)),
                        "brier": float(brier_score(labels, cal))}
    elif args.fit == "isotonic":
        iso = fit_isotonic(scores, labels)
        cal = iso.predict(scores)
        out["isotonic"] = {
            "ece": float(expected_calibration_error(cal, labels)),
            "brier": float(brier_score(labels, cal)),
            "note": "in-sample isotonic ECE is optimistically biased; use held-out folds",
        }
    _print(out)
    return 0


def cmd_seqstats(args: argparse.Namespace) -> int:
    from crisprgap.sequence import gc_content, dinucleotide_features
    seq = args.sequence.upper()
    if not set(seq) <= set("ACGT"):
        _print({"error": "sequence must be ACGT only", "sequence": seq})
        return 2
    dinuc = dinucleotide_features(seq)
    bases = "ACGT"
    labels = [a + b for a in bases for b in bases]
    _print({
        "sequence": seq,
        "length": len(seq),
        "gc_content": float(gc_content(seq)),
        "dinucleotide_frequencies": {k: float(v) for k, v in zip(labels, dinuc)},
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crisprgap",
        description="Verified CRISPR off-target scoring and calibration tools "
                    "(mega27-06a gap set 1-5). All output is JSON.",
    )
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("cfd", help="CFD score between two 23-mers (substitutions only)")
    c.add_argument("wildtype", help="wildtype 23-mer (20nt guide + NGG PAM)")
    c.add_argument("offtarget", help="off-target 23-mer")
    c.set_defaults(func=cmd_cfd)

    m = sub.add_parser("mit", help="MIT specificity score between guide and off-target")
    m.add_argument("guide", help="guide sequence (ACGT)")
    m.add_argument("offtarget", help="off-target sequence (same length)")
    m.set_defaults(func=cmd_mit)

    k = sub.add_parser("calibrate", help="ECE/Brier for a score,label CSV; optional Platt/isotonic fit")
    k.add_argument("csv", help="CSV with columns score,label")
    k.add_argument("--fit", choices=["platt", "isotonic"], default=None,
                   help="also fit a recalibration map in-sample")
    k.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("seqstats", help="GC content and dinucleotide frequencies")
    s.add_argument("sequence", help="ACGT sequence")
    s.set_defaults(func=cmd_seqstats)
    return p


def main(argv: list[str] | None = None) -> int:
    from crisprgap import __version__
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        _print({"crisprgap": __version__})
        return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
