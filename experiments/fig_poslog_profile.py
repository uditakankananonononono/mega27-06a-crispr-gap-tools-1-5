"""Gap-2 figure: per-position mismatch effect profile with cluster-robust CIs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/poslog_inference.json"))
pp = d["per_position"]
x = [r["pos"] for r in pp]
b = [r["coef"] for r in pp]
lo = [r["coef"] - r["ci_lo"] for r in pp]
hi = [r["ci_hi"] - r["coef"] for r in pp]
sig = [r["p"] < 0.05 for r in pp]

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.axhline(0, color="grey", lw=0.8)
ax.errorbar(x, b, yerr=[lo, hi], fmt="none", ecolor="0.6", capsize=2, lw=1)
cols = ["#c0392b" if s else "#7f8c8d" for s in sig]
ax.scatter(x, b, c=cols, s=28, zorder=3)
ax.set_xlabel("protospacer position (1 = PAM-distal, 20 = PAM-proximal)")
ax.set_ylabel("mismatch log-odds (cluster-robust 95% CI)")
ax.set_xticks(range(1, 21))
ax.annotate("pos 15 anomaly", (15, 1.21), xytext=(8.4, 1.35),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
ax.annotate("textbook seed region\n(17-20): null", (18.5, -0.9), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("papers/fig_poslog_profile.pdf")
print("fig saved")
