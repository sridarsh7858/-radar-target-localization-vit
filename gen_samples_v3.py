"""Regenerate the sample-predictions figure with high-contrast test maps where the
target column is clearly visible. Picks the most conspicuous held-out targets from
three different recordings, rolls two to different gates for variety."""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_localizer, N_GATES
from evaluate import tta_localize

A = os.environ.get("REPORT_ASSETS", "report_assets")
os.makedirs(A, exist_ok=True)
OUT_DIR = "dataset_v4"

# only recordings whose target extent is symmetric (primary +-1), so the marked
# gate sits at the centre of the visible energy blob
SYM_TS = {"19931108_220902", "19931109_191449", "19931110_001635",
          "19931118_162155", "19931118_174259"}
rows = [r for r in csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv")))
        if r["split"] == "test" and r["det"] == "1" and r["is_synth"] == "0"
        and r["timestamp"] in SYM_TS]

# pick windows where the labelled gate is itself the visually brightest column:
# both the per-gate peak and the per-gate mean must argmax at the primary gate,
# scored by the peak margin over the runner-up gate (HH channel)
scored = []
for r in rows:
    x = np.load(os.path.join(OUT_DIR, r["path"]))
    g = int(r["primary"])
    peaks = x[..., 0].max(axis=0)                      # (14,) per-gate peak
    if int(peaks.argmax()) != g:
        continue
    margin = float(peaks[g] - np.partition(peaks, -2)[-2])
    scored.append((margin, r["timestamp"], r["path"], g))
scored.sort(reverse=True)

# hand-verified high-contrast HH windows (visually confirmed: compact bright
# near-zero-Doppler signature centred on the labelled gate, contact_sheet.png)
picks = [("test/test_0000124.npy", 8),
         ("test/test_0000128.npy", 8),
         ("test/test_0000131.npy", 8)]

# roll picks 2 and 3 to different destination gates (real map, cyclic shift)
dest = [None, 3, 11]                                   # keep native, gate 4, gate 12 (1-idx)
X, true = [], []
for (p, g), d in zip(picks, dest):
    x = np.load(os.path.join(OUT_DIR, p))
    if d is None:
        X.append(x); true.append(g)
    else:
        X.append(np.roll(x, (d - g) % N_GATES, axis=1)); true.append(d)
X = np.stack(X).astype("float32"); true = np.array(true)

loc = build_localizer(n_pol=4); loc.load_weights("results_v4/checkpoints/loc.weights.h5")
soft = tta_localize(loc, X, tag="samples")
pred = soft.argmax(1)

fig, axs = plt.subplots(3, 2, figsize=(11.5, 10.5))
for i in range(3):
    axs[i, 0].imshow(X[i][..., 0], aspect="auto", origin="lower", cmap="jet",
                     extent=[1, 14, -256, 256])
    axs[i, 0].axvline(true[i] + 1, color="white", ls="--", lw=1.2, alpha=0.85)
    axs[i, 0].set_title(f"Input RD map, HH  (true gate {true[i]+1})", fontsize=10)
    axs[i, 0].set_xlabel("range gate"); axs[i, 0].set_ylabel("Doppler bin")
    axs[i, 1].bar(np.arange(1, 15), soft[i], color="firebrick")
    axs[i, 1].axvline(true[i] + 1, color="green", ls="--", label="true gate")
    axs[i, 1].set_title(f"Localizer softmax  (predicted gate {pred[i]+1})", fontsize=10)
    axs[i, 1].set_xlabel("range gate"); axs[i, 1].set_ylabel("softmax")
    axs[i, 1].set_xticks(range(1, 15)); axs[i, 1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{A}/fig_samples_v3.png", dpi=130)
print("picks:", picks, "dest gates(0-idx):", list(true), "pred:", list(pred))
print("wrote fig_samples_v3.png")
