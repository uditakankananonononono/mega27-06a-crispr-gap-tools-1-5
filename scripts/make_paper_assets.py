"""Generate paper tables (LaTeX) and figures (PDF) from results/*.json."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, P = "results", "papers"
import sys as _sys
if "scripts" not in _sys.path:
    _sys.path.insert(0, "scripts")
from _extra_tables import table_ablation5, table_calib, table_perstudy


def table_crossdata():
    d = json.load(open(f"{R}/crossdata_ridge_3ds.json"))["summary"]["ridge"]
    rows = []
    for train, v in d.items():
        cross = ", ".join(f"{o}: {s:.3f}" for o, s in v["cross_mean"].items())
        rows.append(f"{train} & {v['within_mean']:.3f} & {cross} & {v['generalization_gap']:.3f} \\\\")
    return ("\\begin{table}[h]\\centering\\caption{Cross-dataset generalization "
            "(ridge, 3 seeds): held-out vs transfer Spearman.}\\label{tab:crossdata}"
            "\\begin{tabular}{lccc}\\hline Train & Within & Cross & Gap \\\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n")


def fig_crossdata():
    d = json.load(open(f"{R}/crossdata_ridge_3ds.json"))["summary"]["ridge"]
    names = list(d.keys())
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    within = [d[n]["within_mean"] for n in names]
    crosses = [sum(v for v in d[n]["cross_mean"].values()) / max(len(d[n]["cross_mean"]), 1) for n in names]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], within, width=0.4, label="within-dataset")
    ax.bar([i + 0.2 for i in x], crosses, width=0.4, label="cross-dataset (mean)")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("Spearman $\\rho$"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{P}/fig_crossdata.pdf"); plt.close(fig)


def table_offtarget():
    d = json.load(open(f"{R}/offtarget_metrics.json"))
    rows = [
        f"GNN (this work) & {d['gnn_test_auroc']:.3f} & {d['gnn_test_auprc']:.3f} \\\\",
        f"MIT / Hsu 2013 & {d['mit_test_auroc']:.3f} & {d['mit_test_auprc']:.3f} \\\\",
        f"CFD / Doench 2016 & {d['cfd_subset_auroc']:.3f} & {d['cfd_subset_auprc']:.3f} \\\\",
    ]
    cal = (f"\\begin{{table}}[h]\\centering\\caption{{Calibration on held-out crisprSQL pairs "
           f"(isotonic/Platt fit on validation slice).}}\\label{{tab:calib}}"
           "\\begin{tabular}{lcc}\\hline & ECE & Brier \\\\\\hline\n"
           f"Uncalibrated & {d['ece_uncalibrated']:.3f} & {d['brier_uncalibrated']:.3f} \\\\\n"
           f"Platt & {d['ece_platt_calibrated']:.3f} & {d['brier_platt_calibrated']:.3f} \\\\\n"
           f"Isotonic & {d['ece_isotonic_calibrated']:.3f} & {d['brier_isotonic_calibrated']:.3f} \\\\\n"
           "\\hline\\end{tabular}\\end{table}\n")
    return ("\\begin{table}[h]\\centering\\caption{Off-target classification on crisprSQL, "
            "guide-grouped held-out split.}\\label{tab:offtarget}"
            "\\begin{tabular}{lcc}\\hline Model & AUROC & AUPRC \\\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n") + cal


def table_ablation():
    if not os.path.exists(f"{R}/offtarget_ablation.json"):
        return ""
    d = json.load(open(f"{R}/offtarget_ablation.json"))["conditions"]
    rows = []
    for name, v in d.items():
        extra = f" & {v['auroc_non_ngg_pam']:.3f}" if "auroc_non_ngg_pam" in v else " & -"
        rows.append(f"{name} & {v['auroc']:.3f} & {v['auprc']:.3f}{extra} \\\\")
    return ("\\begin{table}[h]\\centering\\caption{Ablation: PAM and chromatin feature channels "
            "(gaps 4-5).}\\label{tab:ablation}"
            "\\begin{tabular}{lccc}\\hline Features & AUROC & AUPRC & AUROC non-NGG \\\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n")


def table_pegrna():
    if not os.path.exists(f"{R}/pegrna_metrics_HEK.json"):
        return ""
    d = json.load(open(f"{R}/pegrna_metrics_HEK.json"))["means"]
    rows = "\n".join([
        f"Ridge (numeric only) & {d['ridge_numeric_spearman']:.3f} \\\\",
        f"CNN from scratch & {d['scratch_spearman']:.3f} \\\\",
        f"CNN, Cas9-pretrained trunk & {d['transfer_spearman']:.3f} \\\\",
    ])
    return ("\\begin{table}[h]\\centering\\caption{pegRNA efficiency (HEK), grouped held-out "
            "Spearman (gap 3).}\\label{tab:pegrna}"
            "\\begin{tabular}{lc}\\hline Model & Spearman \\\\\\hline\n"
            + rows + "\n\\hline\\end{tabular}\\end{table}\n")


def _esc(tex: str) -> str:
    return tex.replace("_", "\\_")


def table_crossdata_cnn():
    if not os.path.exists(f"{R}/crossdata_cnn_3ds.json"):
        return ""
    d = json.load(open(f"{R}/crossdata_cnn_3ds.json"))["summary"]["cnn"]
    rows = []
    for train, v in d.items():
        cross = ", ".join(f"{o}: {s2:.3f}" for o, s2 in v["cross_mean"].items())
        rows.append(f"{train} & {v['within_mean']:.3f} & {cross} & {v['generalization_gap']:.3f} " + chr(92)*2)
    return ("\\begin{table}[h]\\centering\\caption{Cross-dataset generalization, CNN "
            "(3 seeds, 6-epoch compute budget; within-dataset numbers are undertrained "
            "relative to Table 1). Negative transfer to DeepHF appears.}\\label{tab:crosscnn}"
            "\\begin{tabular}{lccc}\\hline Train & Within & Cross & Gap \\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n")


def table_pooled():
    if not os.path.exists(f"{R}/crossdata_pooled.json"):
        return ""
    d = json.load(open(f"{R}/crossdata_pooled.json"))
    import numpy as np
    k1 = [v['doench_pooled_to_deephf'] for v in d.values()]
    k2 = [v['all_pooled_to_deephf_heldout'] for v in d.values()]
    k3 = [v['deephf_within'] for v in d.values()]
    m = lambda xs: float(np.mean(xs))
    return ("\\begin{table}[h]\\centering\\caption{Pooling mitigation attempt (gap 1, 3 seeds).}"
            "\\label{tab:pooled}\\begin{tabular}{lc}\\hline Training & DeepHF Spearman \\\\\hline\n"
            + f"Doench-pooled & {m(k1):.3f} " + chr(92)*2 + "\n"
            + f"All-pooled (incl.\\ 80\\% DeepHF) & {m(k2):.3f} " + chr(92)*2 + "\n"
            + f"DeepHF only (within) & {m(k3):.3f} " + chr(92)*2 + "\n\\hline\\end{tabular}\\end{table}\n")


def table_scaling():
    if not os.path.exists(f"{R}/pegrna_transfer_scaling.json"):
        return ""
    d = json.load(open(f"{R}/pegrna_transfer_scaling.json"))
    rows = [f"{n} & {v['scratch']:.3f} & {v['transfer']:.3f} " + chr(92)*2 for n, v in sorted(d.items(), key=lambda kv: int(kv[0]))]
    return ("\\begin{table}[h]\\centering\\caption{pegRNA scarce-regime scaling (HEK, grouped "
            "held-out Spearman). Transfer never beats scratch.}\\label{tab:scaling}"
            "\\begin{tabular}{lcc}\\hline $n$ train & Scratch & Cas9-pretrained \\\\\hline\n"
            + "\n".join(rows) + "\n\\hline\\end{tabular}\\end{table}\n")


if __name__ == "__main__":
    os.makedirs(P, exist_ok=True)
    with open(f"{P}/results_tables.tex", "w") as f:
        f.write(_esc(table_crossdata() + table_crossdata_cnn() + table_pooled() + table_offtarget() + table_calib() + table_ablation5() + table_pegrna() + table_scaling() + table_perstudy()))
    fig_crossdata()
    print("assets written")
