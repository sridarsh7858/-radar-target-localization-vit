"""
Supplementary honesty checks for the v4 evaluation (senior-reviewer audit).

 1. UNROLLED localization: accuracy on the test maps exactly as recorded
    (target at its native gate 7/9), same TTA decode as the main eval.
 2. Per-recording unrolled accuracy (is the headline carried by easy files?).
 3. Presence, decomposed: P_D on real positives; P_FA separately on
    (a) GENUINE untouched clutter windows (real negatives) and
    (b) EXCISED (manufactured) clutter maps -- using the SAME val-tuned
    fusion w=0.15 / threshold=0.52 as the main eval. Plus AUC vs genuine only.
"""
import csv
import os

import numpy as np

from model import build_localizer, build_detector, N_GATES
from evaluate import tta_localize, tta_presence, circ_dist

OUT_DIR = "dataset_v4"
RES_DIR = "results_v4"
W, THR = 0.15, 0.52          # from results_v4/test_metrics.json (val-tuned)


def rows_for(split):
    return [r for r in csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv")))
            if r["split"] == split]


def main():
    loc = build_localizer(n_pol=4); loc.load_weights(os.path.join(RES_DIR, "checkpoints/loc.weights.h5"))
    det = build_detector(n_pol=4);  det.load_weights(os.path.join(RES_DIR, "checkpoints/det.weights.h5"))

    rows = rows_for("test")
    tgt_rows = [r for r in rows if r["det"] == "1" and r["is_synth"] == "0"]
    Xt = np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in tgt_rows]).astype("float32")
    prim = np.array([int(r["primary"]) for r in tgt_rows])
    ts = np.array([r["timestamp"] for r in tgt_rows])

    # ---- 1. UNROLLED localization (native recorded positions) ---------------
    soft = tta_localize(loc, Xt, tag="loc/native")
    pred = soft.argmax(1)
    exact = float((pred == prim).mean())
    w1 = float((circ_dist(pred, prim) <= 1).mean())
    print(f"\nUNROLLED (native gate) localization: exact={exact:.3f}  within-1={w1:.3f}  (n={len(Xt)})")

    # ---- 2. per-recording breakdown -----------------------------------------
    print("\nPer-recording unrolled exact accuracy:")
    for t in sorted(set(ts)):
        m = ts == t
        print(f"  {t}: {float((pred[m] == prim[m]).mean()):.3f}  (n={int(m.sum())}, gate {prim[m][0]+1})")

    # ---- 3. presence decomposition ------------------------------------------
    lm_pos = soft.max(1)
    dp_pos = tta_presence(det, Xt, tag="det/pos")
    s_pos = W * lm_pos + (1 - W) * dp_pos

    real_neg_rows = [r for r in rows if r["det"] == "0" and r["is_synth"] == "0"]
    exc_neg_rows = [r for r in rows if r["det"] == "0" and r["is_synth"] == "1"]
    Xr = np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in real_neg_rows]).astype("float32")
    Xe = np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in exc_neg_rows]).astype("float32")
    s_real = W * tta_localize(loc, Xr, tag="loc/negR").max(1) + (1 - W) * tta_presence(det, Xr, tag="det/negR")
    s_exc = W * tta_localize(loc, Xe, tag="loc/negE").max(1) + (1 - W) * tta_presence(det, Xe, tag="det/negE")

    pd_ = float((s_pos >= THR).mean())
    pfa_real = float((s_real >= THR).mean())
    pfa_exc = float((s_exc >= THR).mean())
    print(f"\nPRESENCE @ val-tuned thr={THR}, w={W}:")
    print(f"  P_D  (real targets, native)          = {pd_:.3f}   (n={len(s_pos)})")
    print(f"  P_FA (GENUINE clutter windows)       = {pfa_real:.3f}   (n={len(s_real)})")
    print(f"  P_FA (EXCISED/manufactured clutter)  = {pfa_exc:.3f}   (n={len(s_exc)})")

    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_real))]
    s = np.r_[s_pos, s_real]
    print(f"  AUC vs GENUINE clutter only          = {roc_auc_score(y, s):.4f}")
    y2 = np.r_[np.ones(len(s_pos)), np.zeros(len(s_exc))]
    s2 = np.r_[s_pos, s_exc]
    print(f"  AUC vs excised clutter only          = {roc_auc_score(y2, s2):.4f}")


if __name__ == "__main__":
    main()
