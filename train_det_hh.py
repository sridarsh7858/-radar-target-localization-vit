"""
HH-ONLY detector (single-pol, matching the report's stated methodology).

Minimal delta from train.py: identical data, augmentation, loss and checkpoint
selection -- the only change is slicing polarization channel 0 (HH) from the
4-pol dataset_v4 maps, and build_detector(n_pol=1). Short schedule for a
deadline run: 18 epochs max, patience 6.

Checkpoint -> results_hh/checkpoints/det_hh.weights.h5
"""
import argparse
import os

import numpy as np
from tensorflow import keras

from model import build_detector
from train import RDSequence, RealPresVal, make_sched

RES_DIR = "results_hh"
DET_CKPT = os.path.join(RES_DIR, "checkpoints", "det_hh.weights.h5")


class HHSequence(RDSequence):
    """RDSequence that returns only the HH channel (pol 0)."""

    def __getitem__(self, b):
        X, y, w = super().__getitem__(b)
        return X[..., :1], y, w


class HHRealPresVal(RealPresVal):
    """RealPresVal on the HH channel only."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.X = self.X[..., :1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "checkpoints"), exist_ok=True)

    print("\n========== TRAINING DETECTOR (HH only) ==========")
    trd = HHSequence("train", args.batch, augment=True, shuffle=True, task="det")
    vad = HHSequence("val", args.batch, task="det")
    det = build_detector(n_pol=1)
    det.compile(
        optimizer=keras.optimizers.AdamW(make_sched(args.epochs, len(trd), args.lr),
                                         weight_decay=1e-4),
        loss={"presence": keras.losses.BinaryFocalCrossentropy(gamma=2.0)},
        metrics={"presence": [keras.metrics.AUC(name="auc"),
                              keras.metrics.BinaryAccuracy(name="acc")]})
    det.fit(trd, validation_data=vad, epochs=args.epochs, verbose=1, callbacks=[
        HHRealPresVal(DET_CKPT, patience=args.patience),
        keras.callbacks.CSVLogger(os.path.join(RES_DIR, "train_log_det_hh.csv"))])
    print(f"HH detector best -> {DET_CKPT}")


if __name__ == "__main__":
    main()
