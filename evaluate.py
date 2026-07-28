"""
EVALUATE (HONEST, gate-agnostic) - the metric we trust.

NO synthetic targets. Takes the held-out REAL IPIX target maps (test split, is_synth==0,
det==1) and ROLLS each one to place its real target at every one of the 14 range gates.
The model must track the shift -- so this measures genuine 14-way localization of the REAL
return without a fixed-gate prior. Presence negatives = held-out REAL clutter (det==0).

A max-energy baseline (argmax of per-gate energy) is reported ALONGSIDE the model, because
on a fixed-gate dataset a trivial "loudest gate" rule is a strong localizer; the learned
model is only meaningful if it clearly beats it.

Note on the split: dataset_v2 uses a TIME-BLOCK split with guard gap (same recordings, disjoint
guard-separated time windows), not leave-one-recording-out. So "generalization" here is to an
unseen time window of the same physical target, not a brand-new target -- see Discussion.

Draws BOTH confusion matrices (saved as images):
  - confusion_localization.png : 14x14 (true gate vs predicted gate)
  - confusion_presence.png     : 2x2  (clutter vs target, threshold 0.5)

Prints:
  - localization exact / within-1 accuracy (model) + max-energy baseline
  - map-level presence accuracy
"""
import csv
import functools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model import build_localizer, build_detector, N_GATES

OUT_DIR = "dataset_v4"          # 4-pol dataset (HH baseline was dataset_v2/results_v2)
RES_DIR = "results_v4"
N_POL = 4                       # HH/HV/VH/VV
MANIFEST = os.path.join(OUT_DIR, "labels_balanced.csv")
TEST_BANK = "test"          # held-out clutter split for the honest OOD test
VAL_BANK = "val"            # held-out clutter split for threshold tuning
LOC_CKPT = os.path.join(RES_DIR, "checkpoints", "loc.weights.h5")
DET_CKPT = os.path.join(RES_DIR, "checkpoints", "det.weights.h5")


@functools.lru_cache(maxsize=None)
def load_clutter_bank(split):
    """Assemble the clutter bank for `split` as one (N,512,14,1) array.

    The original code expected a pre-stacked dataset_v2/clutter_bank/<split>.npy,
    but build_dataset.py never produces it -- it writes per-map .npy files plus a
    manifest (labels_balanced.csv) that flags clutter-only maps with det==0. We
    pick those det==0 maps for the requested split and stack them, so the bank is
    real, recording-disjoint clutter with NO target -- exactly what the injection
    pipeline below assumes."""
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"'{MANIFEST}' not found. Run: python build_dataset.py")
    rows = [r for r in csv.DictReader(open(MANIFEST))
            if r["split"] == split and int(float(r["det"])) == 0]
    if not rows:
        raise SystemExit(f"No clutter (det==0) maps for split '{split}' in {MANIFEST}")
    maps = [np.load(os.path.join(OUT_DIR, r["path"])) for r in rows]
    return np.stack(maps).astype("float32")


def circ_dist(a, b, n=N_GATES):
    d = np.abs(a - b)
    return np.minimum(d, n - d)


def tta_localize(model, X, batch=64, tag="loc"):
    """Roll-averaged localization (the model's intended test-time ensemble).

    The backbone uses absolute position embeddings, so it is not exactly range-
    equivariant; averaging the prediction over all 14 range-rolls (rolling the input,
    then un-rolling the output distribution back to the original gate frame) turns
    that into a free ensemble. Returns the averaged (n, 14) soft distribution."""
    acc = np.zeros((len(X), N_GATES), "float64")
    for k in range(N_GATES):
        print(f"\r    TTA[{tag}] {len(X)} maps, roll {k + 1}/{N_GATES}", end="", flush=True)
        s = model.predict(np.roll(X, k, axis=2), batch_size=batch, verbose=0)["loc_soft"]
        acc += np.roll(s, -k, axis=1)                 # un-roll into the original frame
    print()
    return acc / N_GATES


def tta_presence(model, X, batch=64, tag="pres"):
    """Roll-averaged presence score over the 14 range-rolls (same ensemble idea)."""
    acc = np.zeros(len(X), "float64")
    for k in range(N_GATES):
        print(f"\r    TTA[{tag}] {len(X)} maps, roll {k + 1}/{N_GATES}", end="", flush=True)
        acc += model.predict(np.roll(X, k, axis=2),
                             batch_size=batch, verbose=0)["presence"].ravel()
    print()
    return acc / N_GATES


@functools.lru_cache(maxsize=None)
def load_real_targets(split):
    """Held-out REAL (non-synthetic) target maps for `split` + primary gate (0-idx).
    These are genuine IPIX returns -- no synthetic injection."""
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"'{MANIFEST}' not found. Run: python build_dataset.py")
    rows = [r for r in csv.DictReader(open(MANIFEST))
            if r["split"] == split and int(float(r["det"])) == 1 and r["is_synth"] == "0"]
    if not rows:
        raise SystemExit(f"No real target maps (det==1, is_synth==0) for split '{split}'")
    X = np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in rows]).astype("float32")
    primary = np.array([int(np.argmax([float(r[f"loc_g{i}"]) for i in range(N_GATES)]))
                        for r in rows], "int64")
    return X, primary


def build_real_rolled(split, n=None, seed=0):
    """Every held-out REAL target map rolled to place its target at each of the 14 gates.
    Returns X (M,512,14,1) and the true gate per map -> uniform gate coverage, no synthetic.
    If `n` is given, a random subset of the rolled set is returned."""
    Xt, primary = load_real_targets(split)
    Xs, gates = [], []
    for x, p in zip(Xt, primary):
        for g in range(N_GATES):
            Xs.append(np.roll(x, (g - p) % N_GATES, axis=1))    # move real target to gate g
            gates.append(g)
    X = np.stack(Xs).astype("float32")
    gates = np.array(gates, "int64")
    if n is not None and n < len(X):
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X, gates = X[idx], gates[idx]
    return X, gates


def maxenergy_predict(X):
    """Trivial baseline: predict the gate with the strongest PEAK amplitude (max over
    Doppler+pol) -- the canonical naive amplitude localizer, and the strongest of the
    trivial rules on these normalized maps. The learned model must clearly beat it."""
    return X.max(axis=(1, 3)).argmax(1).astype("int64")        # (n,512,14,1) -> (n,)


def build_negatives(split, n, seed):
    """Pure-clutter maps (NO target) from the bank -> presence negatives."""
    bank = load_clutter_bank(split)
    rng = np.random.default_rng(seed)
    return np.clip(bank[rng.integers(0, len(bank), size=n)], 0.0, 1.0).astype("float32")


def presence_scores(loc, det, X):
    """Fused presence evidence = localizer peak-confidence + detector sigmoid.

    The detector alone is a weak presence caller (it keys on the real-sphere
    appearance); the localizer's max softmax (how concentrated its mass is on ONE
    gate) is a stronger 'a target exists somewhere' signal. We return BOTH per-map
    components so the caller can blend them with a val-tuned weight."""
    loc_max = tta_localize(loc, X).max(1)              # peak concentration in [0,1]
    det_p = tta_presence(det, X)                       # detector sigmoid in [0,1]
    return loc_max, det_p


def best_presence_params(loc, det):
    """Tune the fusion weight w and decision threshold that maximise presence
    accuracy on a held-out VAL clutter draw. Score = w*loc_max + (1-w)*det_p."""
    Xp, _ = build_real_rolled(VAL_BANK, n=1500, seed=100)
    lmp, dp = presence_scores(loc, det, Xp)
    lmn, dn = presence_scores(loc, det, build_negatives(VAL_BANK, 1500, 101))
    y = np.concatenate([np.ones(len(lmp)), np.zeros(len(lmn))]).astype(int)
    best = (-1.0, 0.5, 0.5)                            # (acc, w, thr)
    for w in np.linspace(0.0, 1.0, 21):
        s = np.concatenate([w * lmp + (1 - w) * dp, w * lmn + (1 - w) * dn])
        for t in np.linspace(0.05, 0.95, 91):
            acc = ((s >= t).astype(int) == y).mean()
            if acc > best[0]:
                best = (acc, float(w), float(t))
    return best[1], best[2]                            # w, thr


def plot_confusion(cm, classes, title, path):
    """Confusion-matrix heatmap: counts in cells, colour = row-normalized fraction."""
    cm = np.asarray(cm, float)
    row = cm.sum(1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(classes) + 2),
                                    max(4, 0.5 * len(classes) + 2)))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    seed = 0
    loc_model = build_localizer(n_pol=N_POL); loc_model.load_weights(LOC_CKPT)
    det_model = build_detector(n_pol=N_POL);  det_model.load_weights(DET_CKPT)
    print(f"Loaded {LOC_CKPT} + {DET_CKPT}")

    # 14-fold roll-averaging (TTA) means every map is predicted 14x, and GPU is off on
    # native Windows, so we cap the test draw. 3000 rolled real maps is plenty for a
    # stable estimate (set TEST_N=0 in the code for the full 11,550-map sweep).
    TEST_N = 3000

    # tune the presence fusion weight + threshold on a held-out VAL draw
    print("[1/3] tuning presence fusion weight + threshold on VAL ...")
    w, thr = best_presence_params(loc_model, det_model)
    print(f"      fusion w={w:.2f}, threshold={thr:.2f}")

    # positives (held-out REAL targets rolled across all 14 gates) + negatives (real clutter)
    print(f"[2/3] scoring TEST: real targets rolled across 14 gates (cap {TEST_N}) ...")
    X, true = build_real_rolled(TEST_BANK, n=TEST_N or None, seed=seed)
    loc_soft = tta_localize(loc_model, X, tag="loc/pos")
    pred = loc_soft.argmax(1)
    pres_pos = w * loc_soft.max(1) + (1 - w) * tta_presence(det_model, X, tag="det/pos")
    print("[3/3] scoring TEST negatives (real clutter) ...")
    Xneg = build_negatives(TEST_BANK, len(X), seed + 1)
    lmn, dn = presence_scores(loc_model, det_model, Xneg)
    pres_neg = w * lmn + (1 - w) * dn

    # ---- LOCALIZATION: exact + within-1, the 14x14 confusion matrix, + baseline ----
    loc_exact = float((pred == true).mean())
    within1 = float((circ_dist(pred, true) <= 1).mean())
    base_pred = maxenergy_predict(X)                          # trivial loudest-gate rule
    base_exact = float((base_pred == true).mean())
    base_within1 = float((circ_dist(base_pred, true) <= 1).mean())
    cm_loc = confusion_matrix(true, pred, labels=list(range(N_GATES)))
    plot_confusion(cm_loc, list(range(1, N_GATES + 1)),
                   f"Localization confusion (exact={loc_exact:.3f}, baseline={base_exact:.3f})",
                   os.path.join(RES_DIR, "confusion_localization.png"))

    # ---- PRESENCE: map-level accuracy at the tuned threshold, + 2x2 matrix ----
    y_true = np.concatenate([np.ones(len(pres_pos)), np.zeros(len(pres_neg))]).astype(int)
    y_pred = (np.concatenate([pres_pos, pres_neg]) >= thr).astype(int)
    pres_acc = float((y_pred == y_true).mean())
    cm_pres = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plot_confusion(cm_pres, ["clutter", "target"],
                   f"Presence confusion (acc={pres_acc:.3f}, thr={thr:.2f}, w={w:.2f})",
                   os.path.join(RES_DIR, "confusion_presence.png"))

    json.dump({"localization_exact": loc_exact, "localization_within1": within1,
               "baseline_maxenergy_exact": base_exact,
               "baseline_maxenergy_within1": base_within1,
               "presence_acc": pres_acc, "presence_threshold": thr,
               "presence_fusion_w": w, "n": int(len(X)), "eval": "real_rolled"},
              open(os.path.join(RES_DIR, "test_metrics.json"), "w"), indent=2)

    print(f"\n  LOCALIZATION exact     = {loc_exact:.3f}   (max-energy baseline {base_exact:.3f})")
    print(f"  LOCALIZATION within-1  = {within1:.3f}   (+-1 gate; baseline {base_within1:.3f})")
    print(f"  PRESENCE accuracy      = {pres_acc:.3f}   (threshold {thr:.2f}, fusion w={w:.2f})")
    print(f"  eval = {len(X)} held-out REAL targets rolled across 14 gates (no synthetic)")
    if loc_exact <= base_exact + 0.01:
        print("  ** WARNING: model does NOT clearly beat the trivial max-energy baseline. **")
    print("\n  confusion_localization.png   confusion_presence.png")


if __name__ == "__main__":
    main()
