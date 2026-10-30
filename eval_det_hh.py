"""
Evaluate a DETECTOR ALONE (no localizer fusion), for any polarization config.

Threshold tuned on VAL (balanced rolled real positives vs real clutter),
then applied once to TEST:
  - P_D on native and rolled real targets
  - P_FA on genuine clutter windows and on excised (manufactured) clutter
  - balanced presence accuracy (rolled positives vs sampled negatives, the
    same framing as the main evaluate.py) + AUC

--pol selects the input channels sliced from the 4-pol dataset_v4 maps:
  hh | hv | vh | vv  -> single channel, build_detector(n_pol=1)
  all                -> all four channels, build_detector(n_pol=4)
Default hh/results_hh reproduces the original HH-only run.

Writes DETECTOR-ONLY confusion matrices -- results_v4/confusion_presence.png is
the FUSED score w*loc_max+(1-w)*det_p, so it is NOT comparable to the detector
alone:
  - confusion_presence_<pol>.png      balanced draw, evaluate.py protocol
  - confusion_presence_<pol>_raw.png  all test negatives, no resampling
plus test_metrics_<pol>.json with the raw counts behind both.
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from model import build_detector, N_GATES
from evaluate import tta_presence, plot_confusion

OUT_DIR = "dataset_v4"
POL_CH = {"hh": 0, "hv": 1, "vh": 2, "vv": 3}       # channel order in the .npy maps
DEFAULT_CKPT = {p: f"results_hh/checkpoints/det_{p}.weights.h5" for p in POL_CH}
DEFAULT_CKPT["all"] = "results_v4/checkpoints/det.weights.h5"


def rows_for(split):
    return [r for r in csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv")))
            if r["split"] == split]


def load(rows, ch):
    """Load maps and slice the requested channels. ch=None -> keep all 4 pols."""
    X = np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in rows]).astype("float32")
    return X if ch is None else X[..., ch:ch + 1]


def rolled(X, prim, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for x, p in zip(X, prim):
        for g in range(N_GATES):
            out.append(np.roll(x, (g - p) % N_GATES, axis=1))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", choices=["hh", "hv", "vh", "vv", "all"], default="hh",
                    help="input polarization(s): one channel, or all 4")
    ap.add_argument("--ckpt", default=None, help="detector weights (default: per --pol)")
    ap.add_argument("--out", default=None, help="output dir (default: results_hh / results_v4)")
    args = ap.parse_args()

    ch = None if args.pol == "all" else POL_CH[args.pol]
    n_pol = 4 if ch is None else 1
    ckpt = args.ckpt or DEFAULT_CKPT[args.pol]
    if not os.path.exists(ckpt):
        ap.error(f"no checkpoint at {ckpt} -- train it first with "
                 f"`python train_det_hh.py --pol {args.pol}`, or pass --ckpt")
    res_dir = args.out or ("results_v4" if args.pol == "all" else "results_hh")
    os.makedirs(res_dir, exist_ok=True)
    tag = "4pol" if args.pol == "all" else args.pol

    det = build_detector(n_pol=n_pol)
    det.load_weights(ckpt)
    print(f"Loaded {ckpt}  (--pol {args.pol}, n_pol={n_pol}) -> {res_dir}")

    # ---- tune threshold on VAL ----------------------------------------------
    v = rows_for("val")
    vt = [r for r in v if r["det"] == "1" and r["is_synth"] == "0"]
    Xv = load(vt, ch)
    pv = np.array([int(r["primary"]) for r in vt])
    rng = np.random.default_rng(7)
    pos = np.stack([np.roll(x, int(rng.integers(N_GATES)), axis=1) for x in Xv])
    neg = load([r for r in v if r["det"] == "0"], ch)
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
    Xt = load(tt, ch)
    pt = np.array([int(r["primary"]) for r in tt])
    s_nat = tta_presence(det, Xt, tag="test/native")
    Xr = rolled(Xt, pt)
    sel = np.random.default_rng(0).choice(len(Xr), 3000, replace=False)
    s_rol = tta_presence(det, Xr[sel], tag="test/rolled")
    s_gen = tta_presence(det, load([r for r in te if r["det"] == "0" and r["is_synth"] == "0"], ch),
                         tag="test/negGenuine")
    s_exc = tta_presence(det, load([r for r in te if r["det"] == "0" and r["is_synth"] == "1"], ch),
                         tag="test/negExcised")

    label = args.pol.upper() if args.pol != "all" else "4-POL"
    print(f"\n{label} DETECTOR (detector alone) @ thr={thr:.2f}:")
    print(f"  P_D  native positions      = {(s_nat >= thr).mean():.3f}  (n={len(s_nat)})")
    print(f"  P_D  rolled across gates   = {(s_rol >= thr).mean():.3f}  (n={len(s_rol)})")
    print(f"  P_FA genuine clutter       = {(s_gen >= thr).mean():.3f}  (n={len(s_gen)})")
    print(f"  P_FA excised clutter       = {(s_exc >= thr).mean():.3f}  (n={len(s_exc)})")
    s_neg_all = np.r_[s_gen, s_exc]
    nneg = np.random.default_rng(1).choice(len(s_neg_all), len(s_rol), replace=True)
    y2 = np.r_[np.ones(len(s_rol)), np.zeros(len(s_rol))]
    s2 = np.r_[s_rol, s_neg_all[nneg]]
    acc = ((s2 >= thr).astype(int) == y2).mean()
    auc_gen = roc_auc_score(np.r_[np.ones(len(s_rol)), np.zeros(len(s_gen))], np.r_[s_rol, s_gen])
    auc_all = roc_auc_score(y2, s2)
    print(f"  balanced presence accuracy = {acc:.3f}   (protocol of evaluate.py)")
    print(f"  AUC vs genuine clutter     = {auc_gen:.4f}")
    print(f"  AUC vs all clutter         = {auc_all:.4f}")

    # ---- confusion matrices (DETECTOR ALONE -- no localizer fusion) ----------
    # (1) balanced draw: mirrors results_v4/confusion_presence.png's framing so the
    #     two are read side by side -- the only difference is the score being
    #     thresholded (det_p here, fused there).
    cm_bal = confusion_matrix(y2.astype(int), (s2 >= thr).astype(int), labels=[0, 1])
    plot_confusion(cm_bal, ["clutter", "target"],
                   f"{label} detector-only presence (acc={acc:.3f}, thr={thr:.2f})",
                   os.path.join(res_dir, f"confusion_presence_{tag}.png"))
    # (2) raw: every test negative once, no sampling with replacement -- the honest
    #     class ratio, since (1) resamples 3000 negatives from only 825 maps.
    s_raw = np.r_[s_rol, s_neg_all]
    y_raw = np.r_[np.ones(len(s_rol)), np.zeros(len(s_neg_all))].astype(int)
    acc_raw = float(((s_raw >= thr).astype(int) == y_raw).mean())
    cm_raw = confusion_matrix(y_raw, (s_raw >= thr).astype(int), labels=[0, 1])
    plot_confusion(cm_raw, ["clutter", "target"],
                   f"{label} detector-only, all test negatives (acc={acc_raw:.3f}, thr={thr:.2f})",
                   os.path.join(res_dir, f"confusion_presence_{tag}_raw.png"))

    json.dump({"pol": args.pol, "n_pol": n_pol, "ckpt": ckpt, "fusion": False,
               "threshold": thr, "val_balanced_acc": float(best[0]),
               "pd_native": float((s_nat >= thr).mean()), "n_native": int(len(s_nat)),
               "pd_rolled": float((s_rol >= thr).mean()), "n_rolled": int(len(s_rol)),
               "pfa_genuine": float((s_gen >= thr).mean()), "n_genuine": int(len(s_gen)),
               "pfa_excised": float((s_exc >= thr).mean()), "n_excised": int(len(s_exc)),
               "balanced_acc": float(acc), "auc_genuine": float(auc_gen),
               "auc_all": float(auc_all),
               "cm_balanced": {"tn": int(cm_bal[0, 0]), "fp": int(cm_bal[0, 1]),
                               "fn": int(cm_bal[1, 0]), "tp": int(cm_bal[1, 1])},
               "cm_raw": {"tn": int(cm_raw[0, 0]), "fp": int(cm_raw[0, 1]),
                          "fn": int(cm_raw[1, 0]), "tp": int(cm_raw[1, 1])},
               "raw_acc": acc_raw},
              open(os.path.join(res_dir, f"test_metrics_{tag}.json"), "w"), indent=2)
    print(f"\n  {res_dir}/confusion_presence_{tag}.png"
          f"   {res_dir}/confusion_presence_{tag}_raw.png"
          f"   {res_dir}/test_metrics_{tag}.json")


if __name__ == "__main__":
    main()
