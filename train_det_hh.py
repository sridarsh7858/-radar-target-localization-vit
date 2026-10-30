"""
SINGLE-POLARIZATION detector (matching the report's stated single-pol methodology).

Minimal delta from train.py: identical data, augmentation, loss and checkpoint
selection -- the only change is slicing ONE polarization channel from the 4-pol
dataset_v4 maps, and build_detector(n_pol=1). Short schedule for a deadline run:
18 epochs max, patience 6.

--pol hh|hv|vh|vv selects the channel (default hh, the original run).
Checkpoint -> results_hh/checkpoints/det_<pol>.weights.h5
(hh keeps its original filename det_hh.weights.h5 so existing paths still resolve.)
"""
import argparse
import os

import numpy as np
from tensorflow import keras

from model import build_detector
from train import RDSequence, RealPresVal, make_sched

RES_DIR = "results_hh"
POL_CH = {"hh": 0, "hv": 1, "vh": 2, "vv": 3}       # channel order in the .npy maps


class PolSequence(RDSequence):
    """RDSequence that returns only one polarization channel."""

    def __init__(self, *a, ch=0, **kw):
        self.ch = ch
        super().__init__(*a, **kw)

    def __getitem__(self, b):
        X, y, w = super().__getitem__(b)
        return X[..., self.ch:self.ch + 1], y, w


class PolRealPresVal(RealPresVal):
    """RealPresVal on one polarization channel only."""

    def __init__(self, *a, ch=0, **kw):
        super().__init__(*a, **kw)
        self.X = self.X[..., ch:ch + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", choices=list(POL_CH), default="hh",
                    help="polarization channel to train on")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    ch = POL_CH[args.pol]
    ckpt = os.path.join(RES_DIR, "checkpoints", f"det_{args.pol}.weights.h5")
    os.makedirs(os.path.join(RES_DIR, "checkpoints"), exist_ok=True)

    print(f"\n========== TRAINING DETECTOR ({args.pol.upper()} only, ch {ch}) ==========")
    trd = PolSequence("train", args.batch, augment=True, shuffle=True, task="det", ch=ch)
    vad = PolSequence("val", args.batch, task="det", ch=ch)
    det = build_detector(n_pol=1)
    det.compile(
        optimizer=keras.optimizers.AdamW(make_sched(args.epochs, len(trd), args.lr),
                                         weight_decay=1e-4),
        loss={"presence": keras.losses.BinaryFocalCrossentropy(gamma=2.0)},
        metrics={"presence": [keras.metrics.AUC(name="auc"),
                              keras.metrics.BinaryAccuracy(name="acc")]})
    det.fit(trd, validation_data=vad, epochs=args.epochs, verbose=1, callbacks=[
        PolRealPresVal(ckpt, patience=args.patience, ch=ch),
        keras.callbacks.CSVLogger(os.path.join(RES_DIR, f"train_log_det_{args.pol}.csv"))])
    print(f"{args.pol.upper()} detector best -> {ckpt}")


if __name__ == "__main__":
    main()
