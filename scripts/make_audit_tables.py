"""Regenerate audit appendix tables from committed result JSONs.

- papers/routing_table.tex: every (study, split) row of the study-routing audit
  (results/study_routing.json), thin rows flagged.
- papers/ablation_perseed_table.tex: gaps 4-5 per-seed condition AUROCs
  (results/offtarget_ablation_multiseed.json).
"""
import json


def routing_table():
    d = json.load(open("results/study_routing.json"))
    rows = []
    for r in d["rows"]:
        rows.append(f"{r['study']} & {r['seed']} & {r['n_test']} & {r['n_pos']}/{r['n_neg']} & "
                    f"{r['prevalence']:.3f} & {r['auroc_gnn']:.3f} & {r['auroc_mit']:.3f} & "
                    f"{'GNN' if r['auroc_gnn'] > r['auroc_mit'] else 'MIT'} \\\\")
    thin = []
    for r in d["thin_rows_excluded"]:
        thin.append(f"{r['study']} & {r['seed']} & {r['n_test']} & {r['n_pos']}/{r['n_neg']} & "
                    f"- & {r['auroc_gnn']:.3f} & {r['auroc_mit']:.3f} & (degenerate) \\\\")
    cap = ("Study-routing audit: every (study, split) row with $\\ge$30 test pairs. "
           "Top block: rows with $\\ge$5 positives and $\\ge$5 negatives (the estimable set; "
           "GNN wins 1 of 13). Bottom block: near-degenerate rows excluded from the analysis - "
           "note the seed-0 Anderson row (2 positives) that produced the apparent GNN win "
           "in the single-split table.")
    return ("\\begin{table}[h]\\centering\\small\\caption{" + cap + "}\\label{tab:routing}\n"
            "\\begin{tabular}{lccccccc}\\hline\n"
            "Study & split & $n$ & pos/neg & prev. & GNN & MIT & winner \\\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\n" + "\n".join(thin) +
            "\n\\hline\\end{tabular}\\end{table}\n")


def ablation_perseed_table():
    d = json.load(open("results/offtarget_ablation_multiseed.json"))
    conds = ["seq_only", "pam", "epigen", "pam_epigen"]
    rows = []
    for c in conds:
        per = d["conditions"][c]["auroc_per_seed"]
        per_s = " & ".join(f"{v:.3f}" for v in per)
        cname = c.replace("_", "\\_")
        rows.append("%s & %s & $%.3f \\pm %.3f$ \\\\" % (cname, per_s, d["conditions"][c]["auroc_mean"], d["conditions"][c]["auroc_sd"]))
    cap = ("Gaps 4--5 ablation, per-seed held-out AUROC (fully seeded guide-grouped splits). "
           "The condition ranking is unstable across seeds; all means lie within one sd of "
           "seq-only except PAM-alone, which is worse.")
    return ("\\begin{table}[h]\\centering\\small\\caption{" + cap + "}\\label{tab:ablperseed}\n"
            "\\begin{tabular}{lcccccc}\\hline\n"
            "Channel & seed 0 & seed 1 & seed 2 & seed 3 & seed 4 & mean $\\pm$ sd \\\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n")


if __name__ == "__main__":
    open("papers/routing_table.tex", "w").write(routing_table())
    open("papers/ablation_perseed_table.tex", "w").write(ablation_perseed_table())
    print("wrote routing_table.tex + ablation_perseed_table.tex")
