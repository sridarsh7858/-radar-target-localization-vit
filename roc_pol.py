"""
ROC curves for the presence detector: HH channel vs. all four polarizations.

Same test protocol as eval_det_hh.py -- held-out real target maps rolled across
the 14 range gates as positives, held-out real clutter windows as negatives, and
14-fold roll-averaging (TTA) on every score. Nothing is thresholded here: the ROC
sweeps the full range of the presence sigmoid, so it shows the separability that
a single operating point hides.

Two negative sets are reported, because they answer different questions:
  genuine  -- the 75 windows of the clutter-only recording (real sea, no target)
  all      -- genuine + the 750 excised maps (target gates overwritten with
              clutter gates of the same map), the full balanced negative class

Outputs (to --out, default report_assets/):
  roc_pol.png / roc_pol.pdf   two panels: linear ROC and log-scale P_FA
  roc_pol.json                AUCs and the sampled curve points

Run:  python roc_pol.py
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from model import build_detector, N_GATES
from evaluate import tta_presence

OUT_DIR = "dataset_v4"
HH_CKPT = "results_hh/checkpoints/det_hh.weights.h5"
POL4_CKPT = "baseline_v4_backup/det.weights.h5"   # the checkpoint the paper reports
N_POS = 3000                                      # rolled positives, as in eval_det_hh.py


def rows_for(split):
    with open(os.path.join(OUT_DIR, "labels_balanced.csv")) as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def load(rows, ch):
    X = np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in rows]).astype("float32")
    return X if ch is None else X[..., ch:ch + 1]


def rolled(X, prim):
    """Every target map rolled so its target visits each of the 14 gates in turn."""
    return np.stack([np.roll(x, (g - p) % N_GATES, axis=1)
                     for x, p in zip(X, prim) for g in range(N_GATES)])


def score_detector(ckpt, ch, n_pol, te, tag):
    """TTA presence scores for the positives and both negative sets."""
    det = build_detector(n_pol=n_pol)
    det.load_weights(ckpt)
    tt = [r for r in te if r["det"] == "1" and r["is_synth"] == "0"]
    Xr = rolled(load(tt, ch), np.array([int(r["primary"]) for r in tt]))
    sel = np.random.default_rng(0).choice(len(Xr), N_POS, replace=False)
    s_pos = tta_presence(det, Xr[sel], tag=f"{tag}/pos")
    s_gen = tta_presence(det, load([r for r in te if r["det"] == "0" and r["is_synth"] == "0"], ch),
                         tag=f"{tag}/negGenuine")
    s_exc = tta_presence(det, load([r for r in te if r["det"] == "0" and r["is_synth"] == "1"], ch),
                         tag=f"{tag}/negExcised")
    return s_pos, s_gen, s_exc


def curve(s_pos, s_neg):
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    s = np.r_[s_pos, s_neg]
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("REPORT_ASSETS", "report_assets"))
    ap.add_argument("--hh-ckpt", default=HH_CKPT)
    ap.add_argument("--pol4-ckpt", default=POL4_CKPT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    te = rows_for("test")
    runs = {}
    for tag, ckpt, ch, n_pol in [("HH", args.hh_ckpt, 0, 1),
                                 ("4-pol", args.pol4_ckpt, None, 4)]:
        print(f"\n[{tag}] {ckpt}")
        s_pos, s_gen, s_exc = score_detector(ckpt, ch, n_pol, te, tag)
        runs[tag] = {"pos": s_pos, "gen": s_gen, "all": np.r_[s_gen, s_exc]}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"HH": "#1f77b4", "4-pol": "#d62728"}
    styles = {"gen": "-", "all": "--"}
    names = {"gen": "genuine clutter recording", "all": "all clutter (genuine + excised)"}
    for tag, s in runs.items():
        for key in ("gen", "all"):
            fpr, tpr, auc = curve(s["pos"], s[key])
            out[f"{tag}_{key}"] = {"auc": auc, "n_pos": len(s["pos"]), "n_neg": len(s[key]),
                                   "pfa_resolution": 1.0 / len(s[key]),
                                   "fpr": fpr.tolist(), "tpr": tpr.tolist()}
            lbl = f"{tag} vs {names[key]} (AUC {auc:.3f}, n={len(s[key])})"
            # below 1/n_neg the curve carries no measurement, so only draw what is measured
            keep = fpr >= 1.0 / len(s[key]) - 1e-12
            axes[0].plot(fpr, tpr, styles[key], color=colors[tag], lw=1.8, label=lbl)
            axes[1].plot(fpr[keep], tpr[keep], styles[key], color=colors[tag], lw=1.8, label=lbl)
            print(f"  AUC {tag:6s} vs {names[key]:34s} = {auc:.4f}")

    # the negative sets are small, so the smallest measurable P_FA is 1/n_neg
    for n_neg, lab in [(75, "1/75 (genuine set)"), (825, "1/825 (all clutter)")]:
        axes[1].axvline(1.0 / n_neg, color="0.55", ls=":", lw=1)
        axes[1].text(1.0 / n_neg, 0.515, f"  {lab}", fontsize=7, color="0.35", rotation=90,
                     va="bottom")

    axes[0].plot([0, 1], [0, 1], ":", color="0.6", lw=1, label="chance")
    axes[0].set_xlim(-0.005, 1); axes[0].set_ylim(0, 1.005)
    axes[0].set_xlabel("Probability of false alarm $P_{FA}$")
    axes[1].set_xscale("log"); axes[1].set_xlim(8e-4, 1); axes[1].set_ylim(0.5, 1.005)
    axes[1].set_xlabel("Probability of false alarm $P_{FA}$ (log scale)")
    for ax, t in zip(axes, ["(a) linear axes",
                            "(b) log false-alarm axis, measured range only"]):
        ax.set_ylabel("Probability of detection $P_D$")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Presence-detector ROC: HH channel vs. four-polarization input "
                 "(3000 held-out rolled real targets)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.out, f"roc_pol.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  wrote {p}")
    json.dump(out, open(os.path.join(args.out, "roc_pol.json"), "w"), indent=2)
    # raw scores, so operating points can be re-derived without re-running the models
    np.savez(os.path.join(args.out, "roc_pol_scores.npz"),
             **{f"{tag}_{k}": v for tag, s in runs.items() for k, v in s.items()})


if __name__ == "__main__":
    main()
