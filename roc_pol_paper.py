"""
Standard-style ROC figure for the paper: two panels, HH detector on the left,
four-polarization detector on the right, one curve each.

Negative class: the GENUINE clutter recording only (75 real, untouched windows),
not the excised/manufactured negatives. Reasoning: excised maps are made with our
own augmentation method, so testing against them is partly circular; the genuine
recording is the one negative set that is unambiguously real and matches the
headline claim ("zero false alarms on the genuine clutter recording").

Reuses the scores already computed by roc_pol.py (report_assets/roc_pol_scores.npz)
-- no need to re-run the models.

Output: report_assets/roc_pol_paper.png / .pdf
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

OUT = "report_assets"
d = np.load(os.path.join(OUT, "roc_pol_scores.npz"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)

panels = [("HH", "HH_pos", "HH_gen", "#1f77b4"),
          ("4-pol", "4-pol_pos", "4-pol_gen", "#d62728")]

results = {}
for ax, (tag, pos_key, neg_key, color) in zip(axes, panels):
    s_pos, s_neg = d[pos_key], d[neg_key]
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    s = np.r_[s_pos, s_neg]
    fpr, tpr, thr = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    results[tag] = {"auc": float(auc), "n_pos": len(s_pos), "n_neg": len(s_neg)}

    ax.plot(fpr, tpr, "-", color=color, lw=2.2)
    ax.plot([0, 1], [0, 1], "--", color="0.65", lw=1)
    ax.fill_between(fpr, tpr, fpr, color=color, alpha=0.08)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Probability of false alarm, $P_{FA}$")
    ax.set_title(f"{tag} detector  (AUC = {auc:.3f})", fontsize=11)
    ax.grid(alpha=0.25)
    ax.text(0.97, 0.05, f"n = {len(s_pos)} targets\nn = {len(s_neg)} clutter",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")

axes[0].set_ylabel(r"Probability of detection, $P_D$")
fig.suptitle("Detection ROC: HH vs. four-polarization input\n"
             "(real held-out targets vs. genuine clutter recording, n=75)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])

for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"roc_pol_paper.{ext}")
    fig.savefig(p, dpi=220, bbox_inches="tight")
    print("wrote", p)

json.dump(results, open(os.path.join(OUT, "roc_pol_paper.json"), "w"), indent=2)
print(results)
