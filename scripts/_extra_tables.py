"""Extra paper tables sourced from committed multiseed/audit JSON artifacts.
Imported by make_paper_assets.py; kept separate so the legacy single-seed
table functions stay untouched."""
import json
import os

R = "results"


def table_ablation5():
    if not os.path.exists(f"{R}/offtarget_ablation_multiseed.json"):
        return ""
    d = json.load(open(f"{R}/offtarget_ablation_multiseed.json"))
    rows = []
    for name in ("seq_only", "pam", "epigen", "pam_epigen"):
        v = d["conditions"][name]
        rows.append("%s & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ \\\\" %
                    (name, v["auroc_mean"], v["auroc_sd"],
                     v["auroc_non_ngg_mean"], v["auroc_non_ngg_sd"]))
    return ("\\begin{table}[h]\\centering\\caption{Ablation: PAM and chromatin feature channels "
            "(gaps 4-5), five fully-seeded guide-grouped splits (mean $\\pm$ sd AUROC). "
            "All conditions are within noise of seq-only; the winner varies by seed "
            "(pam_epigen 3/5, epigen 2/5).}\\label{tab:ablation}"
            "\\begin{tabular}{lcc}\\hline Features & AUROC (5 seeds) & AUROC non-NGG (5 seeds) \\\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n")


def table_calib():
    d = json.load(open(f"{R}/offtarget_multiseed.json"))
    m = d["mean"]
    rows = ["uncalibrated & %.3f" % m["ece_uncal"],
            "isotonic (val slice) & %.3f" % m["ece_iso"],
            "Platt (val slice) & %.3f" % m["ece_platt"]]
    return ("\\begin{table}[h]\\centering\\caption{Calibration on crisprSQL: mean ECE over the "
            "five fully-seeded splits (recalibrators fit on each split's validation slice). "
            "At this sample size, recalibration hurts on average.}\\label{tab:calib}"
            "\\begin{tabular}{lc}\\hline & mean ECE (5 splits) \\\\\\hline\n"
            + " \\\\\n".join(rows + [""]) + "\hline\\end{tabular}\\end{table}\n")


def table_perstudy():
    d = json.load(open(f"{R}/study_routing.json"))
    seed0 = [r for r in d["rows"] + d["thin_rows_excluded"] if r["seed"] == 0]
    order = {"Anderson": 0, "Cameron": 1, "Kim": 2}
    seed0.sort(key=lambda r: order.get(r["study"], 9))
    rows = ["%s & %d & %d / %d & %.3f & %.3f \\\\" %
            (r["study"], r["n_test"], r["n_pos"], r["n_neg"], r["auroc_gnn"], r["auroc_mit"])
            for r in seed0]
    return ("\\begin{table}[h]\\centering\\caption{Per-study held-out AUROC on the canonical "
            "split, with class counts. The Anderson GNN ``win'' rests on \\textbf{2 positives}; "
            "across five seeded splits only 13 (study, split) rows carry $\\ge$5 positives and "
            "$\\ge$5 negatives, and the GNN wins one of them -- per-study rankings on this "
            "corpus are not estimable (Section~6).}\\label{tab:perstudy}"
            "\\begin{tabular}{lcccc}\\hline Study & $n$ test & pos / neg & GNN & MIT \\\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n")
