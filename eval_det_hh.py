"""
Evaluate the HH-ONLY detector (detector alone, no localizer fusion).

Threshold tuned on VAL (balanced rolled real positives vs real clutter),
then applied once to TEST:
  - P_D on native and rolled real targets
  - P_FA on genuine clutter windows and on excised (manufactured) clutter
  - balanced presence accuracy (rolled positives vs sampled negatives, the
    same framing as the main evaluate.py) + AUC
All maps sliced to the HH channel (pol 0).
"""
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from model import build_detector, N_GATES
from evaluate import tta_presence

OUT_DIR = "dataset_v4"
CKPT = "results_hh/checkpoints/det_hh.weights.h5"


def rows_for(split):
    return [r for r in csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv")))
            if r["split"] == split]


def load(rows):
    return np.stack([np.load(os.path.join(OUT_DIR, r["path"]))[..., :1]
                     for r in rows]).astype("float32")


def rolled(X, prim, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for x, p in zip(X, prim):
        for g in range(N_GATES):
            out.append(np.roll(x, (g - p) % N_GATES, axis=1))
    return np.stack(out)


def main():
    det = build_detector(n_pol=1)
    det.load_weights(CKPT)

    # ---- tune threshold on VAL ----------------------------------------------
    v = rows_for("val")
    vt = [r for r in v if r["det"] == "1" and r["is_synth"] == "0"]
    Xv = load(vt)
    pv = np.array([int(r["primary"]) for r in vt])
    rng = np.random.default_rng(7)
    pos = np.stack([np.roll(x, int(rng.integers(N_GATES)), axis=1) for x in Xv])
    neg = load([r for r in v if r["det"] == "0"])
    sp = tta_presence(det, pos, tag="val/pos")
    sn = tta_presence(det, neg, tag="val/neg")
    m = min(len(sp), len(sn))
    y = np.r_[np.ones(m), np.zeros(m)]
    best = (-1, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        a = ((np.r_[sp[:m], sn[:m]] >= t).astype(int) == y).mean()
        if a > best[0]:
            best = (a, float(t))
    thr = best[1]
    print(f"\nval-tuned threshold = {thr:.2f}  (val balanced acc {best[0]:.3f})")

    # ---- TEST ----------------------------------------------------------------
    te = rows_for("test")
    tt = [r for r in te if r["det"] == "1" and r["is_synth"] == "0"]
    Xt = load(tt)
    pt = np.array([int(r["primary"]) for r in tt])
    s_nat = tta_presence(det, Xt, tag="test/native")
    Xr = rolled(Xt, pt)
    sel = np.random.default_rng(0).choice(len(Xr), 3000, replace=False)
    s_rol = tta_presence(det, Xr[sel], tag="test/rolled")
    s_gen = tta_presence(det, load([r for r in te if r["det"] == "0" and r["is_synth"] == "0"]),
                         tag="test/negGenuine")
    s_exc = tta_presence(det, load([r for r in te if r["det"] == "0" and r["is_synth"] == "1"]),
                         tag="test/negExcised")

    print(f"\nHH-ONLY DETECTOR @ thr={thr:.2f}:")
    print(f"  P_D  native positions      = {(s_nat >= thr).mean():.3f}  (n={len(s_nat)})")
    print(f"  P_D  rolled across gates   = {(s_rol >= thr).mean():.3f}  (n={len(s_rol)})")
    print(f"  P_FA genuine clutter       = {(s_gen >= thr).mean():.3f}  (n={len(s_gen)})")
    print(f"  P_FA excised clutter       = {(s_exc >= thr).mean():.3f}  (n={len(s_exc)})")
    s_neg_all = np.r_[s_gen, s_exc]
    nneg = np.random.default_rng(1).choice(len(s_neg_all), len(s_rol), replace=True)
    y2 = np.r_[np.ones(len(s_rol)), np.zeros(len(s_rol))]
    s2 = np.r_[s_rol, s_neg_all[nneg]]
    acc = ((s2 >= thr).astype(int) == y2).mean()
    print(f"  balanced presence accuracy = {acc:.3f}   (protocol of evaluate.py)")
    print(f"  AUC vs genuine clutter     = {roc_auc_score(np.r_[np.ones(len(s_rol)), np.zeros(len(s_gen))], np.r_[s_rol, s_gen]):.4f}")
    print(f"  AUC vs all clutter         = {roc_auc_score(y2, s2):.4f}")


if __name__ == "__main__":
    main()
