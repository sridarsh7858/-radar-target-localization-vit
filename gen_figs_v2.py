"""
Regenerate the DRDO-report result figures for the no-synthetic (real-data) run.
Outputs *_v2.png into radarLLM/report_assets (originals untouched).
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_localizer, build_detector, N_GATES
from evaluate import tta_localize, tta_presence, build_real_rolled, load_real_targets

A = os.environ.get("REPORT_ASSETS", "report_assets")
os.makedirs(A, exist_ok=True)
OUT_DIR = "dataset_v4"


def save(fig, name):
    p = os.path.join(A, name)
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


# ---- 4.1 metric bars ---------------------------------------------------------
vals = [0.876, 0.902, 0.873]
labs = ["Exact gate\nlocalization", "Within one gate\nlocalization", "Presence detection\n(real clutter, HH)"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar(labs, vals, color=["#1f3b63", "#2e5d8f", "#7ba2c9"], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v*100:.1f}%", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
ax.set_title("Proposed RD ViT performance (real IPIX targets)", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
save(fig, "fig_compare_v2.png")

# ---- convergence figures -----------------------------------------------------
def curves(csv_path, val_col, val_label, title_right, name):
    rows = list(csv.DictReader(open(csv_path)))
    ep = [int(r["epoch"]) for r in rows]
    tl = [float(r["loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    va = [float(r[val_col]) for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.6))
    axs[0].plot(ep, tl, label="Train Loss", color="#1f77b4")
    axs[0].plot(ep, vl, label="Val Loss", color="#ff7f0e")
    axs[0].set_title("Loss"); axs[0].set_xlabel("epoch"); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[1].plot(ep, va, label=val_label, color="#2ca02c")
    axs[1].set_title(title_right); axs[1].set_xlabel("epoch")
    axs[1].set_ylim(0, 1.0); axs[1].legend(); axs[1].grid(alpha=0.3)
    save(fig, name)

curves("results_v4/train_log_loc.csv", "val_aug_loc_acc",
       "Val localization accuracy (14-way)", "Val localization accuracy (14-way)",
       "fig_train_loc_v2.png")
curves("results_hh/train_log_det_hh.csv", "val_real_pres_acc",
       "Val presence accuracy (real targets)", "Val presence accuracy (real targets)",
       "fig_train_det_v2.png")

# ---- transplant illustration (replaces fig_synth) ----------------------------
man = list(csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv"))))
clut_row = next(r for r in man if r["split"] == "train" and r["det"] == "0" and r["is_synth"] == "0")
don_row = next(r for r in man if r["split"] == "train" and r["det"] == "1"
               and r["timestamp"] == "19931107_135603")
clut = np.load(os.path.join(OUT_DIR, clut_row["path"]))[..., 0]
don = np.load(os.path.join(OUT_DIR, don_row["path"]))[..., 0]
dprim = int(don_row["primary"])
g = 4                                                 # transplant destination gate (0-idx)
donr = np.roll(don, (g - dprim) % N_GATES, axis=1)
cols = np.zeros(N_GATES, bool)
for d in (-1, 0, 1):
    cols[(g + d) % N_GATES] = True
trans = clut.copy(); trans[:, cols] = donr[:, cols]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
ext = [1, 14, -256, 256]
for ax, m, t in zip(axs, [clut, don, trans],
                    ["Clutter background", f"Real donor target map (gate {dprim+1})",
                     f"Transplanted map (target moved to gate {g+1})"]):
    ax.imshow(m, aspect="auto", origin="lower", cmap="jet", extent=ext)
    ax.set_title(t, fontsize=10); ax.set_xlabel("range gate")
axs[0].set_ylabel("Doppler")
save(fig, "fig_transplant_v2.png")

# ---- model-based figures (per-gate, samples, ROC) ----------------------------
loc = build_localizer(n_pol=4); loc.load_weights("results_v4/checkpoints/loc.weights.h5")
X, true = build_real_rolled("test", n=3000, seed=0)
soft = tta_localize(loc, X, tag="pergate")
pred = soft.argmax(1)

acc = [float((pred[true == g] == g).mean()) for g in range(N_GATES)]
fig, ax = plt.subplots(figsize=(9.5, 3.4))
ax.bar(np.arange(1, 15), acc, color="#2e5d8f")
m = float((pred == true).mean())
ax.axhline(m, color="red", ls="--", lw=1.2, label=f"mean = {m:.3f}")
ax.set_xlabel("true range gate"); ax.set_ylabel("exact localization accuracy")
ax.set_title("Per-gate exact localization accuracy"); ax.set_ylim(0, 1.05)
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "fig_pergate_v2.png")

rng = np.random.default_rng(3)
fig, axs = plt.subplots(3, 2, figsize=(11, 10))
for i, idx in enumerate(rng.choice(len(X), 3, replace=False)):
    axs[i, 0].imshow(X[idx][..., 0], aspect="auto", origin="lower", cmap="jet",
                     extent=[1, 14, -256, 256])
    axs[i, 0].set_title(f"Input RD map  (true gate {true[idx]+1})", fontsize=10)
    axs[i, 0].set_xlabel("range gate"); axs[i, 0].set_ylabel("Doppler")
    axs[i, 1].bar(np.arange(1, 15), soft[idx], color="firebrick")
    axs[i, 1].axvline(true[idx] + 1, color="green", ls="--", label="true gate")
    axs[i, 1].set_title(f"Localization output  (pred gate {pred[idx]+1})", fontsize=10)
    axs[i, 1].set_xlabel("range gate"); axs[i, 1].set_ylabel("softmax"); axs[i, 1].legend()
save(fig, "fig_samples_v2.png")

# ROC of the HH-only detector on the test split (real positives vs real clutter)
det = build_detector(n_pol=1); det.load_weights("results_hh/checkpoints/det_hh.weights.h5")
rows_t = [r for r in man if r["split"] == "test"]
Xt, _ = load_real_targets("test")
s_pos = tta_presence(det, Xt[..., :1], tag="roc/pos")
neg_rows = [r for r in rows_t if r["det"] == "0"]
Xn = np.stack([np.load(os.path.join(OUT_DIR, r["path"]))[..., :1] for r in neg_rows])
s_neg = tta_presence(det, Xn.astype("float32"), tag="roc/neg")
from sklearn.metrics import roc_curve, roc_auc_score
y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
s = np.r_[s_pos, s_neg]
fpr, tpr, _ = roc_curve(y, s)
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot(fpr, tpr, marker=".", ms=3, lw=1, color="#1f3b63")
ax.set_xlabel("Probability of false alarm (P_FA)")
ax.set_ylabel("Probability of detection (P_D)")
ax.set_title(f"Presence detection ROC (HH), AUC = {roc_auc_score(y, s):.3f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
save(fig, "fig_roc_v2.png")

print("all figures done")
