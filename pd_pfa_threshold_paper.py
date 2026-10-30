"""Paper Fig. 3: P_D and P_FA versus detection threshold (HH and 4-pol detectors).

Same scores as the ROC (report_assets/roc_pol_scores.npz): 3000 rolled real target maps
and the 75 genuine clutter windows. Threshold swept 0..1 in steps of 0.001.
Output: report_assets/pd_pfa_vs_threshold_paper.png / .pdf
"""
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
d = np.load("report_assets/roc_pol_scores.npz")
th = np.linspace(0, 1, 1001)

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.65), sharey=True)
for ax, (tag, title, color) in zip(axes, [("HH", "(a) HH detector", "#1f77b4"),
                                          ("4-pol", "(b) 4-pol detector", "#d62728")]):
    pos, neg = d[tag + "_pos"], d[tag + "_gen"]
    ax.plot(th, [(pos >= t).mean() for t in th], color=color, lw=1.2, label=r"$P_D$")
    ax.plot(th, [(neg >= t).mean() for t in th], "--", color="0.25", lw=1.0, label=r"$P_{FA}$")
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("Detection threshold", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.45, 0.1))
axes[0].set_ylabel("Probability", fontsize=7)
fig.tight_layout(pad=0.3, w_pad=0.6)
for ext in ("png", "pdf"):
    fig.savefig(f"report_assets/pd_pfa_vs_threshold_paper.{ext}", dpi=600, bbox_inches="tight")
print("wrote report_assets/pd_pfa_vs_threshold_paper.*")
