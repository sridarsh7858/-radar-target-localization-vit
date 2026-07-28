"""
Step: TRAIN two INDEPENDENT models on dataset_v2/ -- a localizer and a detector.

- LOCALIZER (build_localizer): categorical-crossentropy on loc_soft, masked to target
  maps. Best checkpoint = max GENUINE 14-way val localization (AugLocVal callback) ->
  results_v2/checkpoints/loc.weights.h5.
- DETECTOR (build_detector): BinaryFocalCrossentropy on presence, on ALL maps. Best
  checkpoint = min val loss -> results_v2/checkpoints/det.weights.h5.
- Shared augmentation (same RDSequence, `task` switches the label returned): REAL IPIX
  target signatures transplanted to every gate (range-roll) + excise-to-clutter for class
  balance + amplitude/noise jitter. NO synthetic targets -- every positive is a real return
  moved in range, so both models see the real target at all 14 gates.
The two models do not share weights, so detection and localization never compete.
"""
import argparse
import csv
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import build_localizer, build_detector, N_GATES
from ipix_targets import target_cols

OUT_DIR = "dataset_v4"          # 4-pol dataset (HH baseline was dataset_v2/results_v2)
RES_DIR = "results_v4"
N_POL = 4                       # HH/HV/VH/VV -- cross-pol is the small-target lever
LOC_CKPT = os.path.join(RES_DIR, "checkpoints", "loc.weights.h5")
DET_CKPT = os.path.join(RES_DIR, "checkpoints", "det.weights.h5")


def load_split(split):
    rows = [r for r in csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv")))
            if r["split"] == split]
    paths = [os.path.join(OUT_DIR, r["path"]) for r in rows]
    # det_vec (det_g*) is no longer loaded: only loc_vec + map-level presence are used.
    loc = np.array([[float(r[f"loc_g{i}"]) for i in range(N_GATES)] for r in rows], "float32")
    pres = np.array([[float(r["det"])] for r in rows], "float32")
    return paths, loc, pres


class RDSequence(keras.utils.Sequence):
    def __init__(self, split, batch=32, augment=False, shuffle=False, task="loc", **kw):
        super().__init__(**kw)
        self.task = task                        # "loc" -> loc_soft ; "det" -> presence
        self.paths, self.loc, self.pres = load_split(split)
        self.batch, self.augment, self.shuffle = batch, augment, shuffle
        self.order = np.arange(len(self.paths))
        self.rng = np.random.default_rng(0)
        # donor pool for target-transplant augmentation: every real/aug-able
        # target map in this split, with its primary gate (extent = primary +-1).
        self.donors = [(p, int(l.argmax()))
                       for p, l, pr in zip(self.paths, self.loc, self.pres)
                       if pr[0] == 1]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __getitem__(self, b):
        ids = self.order[b * self.batch:(b + 1) * self.batch]
        X = np.stack([np.load(self.paths[i]) for i in ids])          # (B,512,14,4)
        loc = self.loc[ids].copy()
        pres = self.pres[ids].copy()
        if self.augment:
            # Present the REAL IPIX target at EVERY gate so the model learns the transferable
            # cue (energy concentrated in one range bin) rather than the gate-7/9 fingerprint.
            # NO synthetic targets: for each clutter map we either
            #   r<0.50  paste a REAL donor target (a genuine IPIX return) at a random gate
            #   else    keep it as clutter
            # and each real target map is excised back to clutter with prob 0.40, keeping
            # presence roughly balanced. The range-roll below then moves it across all gates.
            for j in range(len(ids)):
                r = self.rng.random()
                if pres[j, 0] == 0 and r < 0.50:
                    dpath, dprim = self.donors[
                        int(self.rng.integers(len(self.donors)))]
                    donor = np.load(dpath)
                    g = int(self.rng.integers(0, N_GATES))
                    donor = np.roll(donor, (g - dprim) % N_GATES, axis=1)
                    cols = target_cols(g)               # target extent = g +-1
                    X[j][:, cols, :] = donor[:, cols, :]
                    loc[j] = 0.0
                    loc[j][g] = 1.0
                    pres[j] = 1.0
                elif pres[j, 0] == 1 and r < 0.40:
                    cols = target_cols(int(loc[j].argmax()))   # primary +-1
                    tgt = np.where(cols)[0]
                    clut = np.where(~cols)[0]
                    for gg in tgt:
                        X[j][:, gg, :] = X[j][:, int(self.rng.choice(clut)), :]
                    loc[j] = 0.0
                    pres[j] = 0.0
            for j in range(len(ids)):
                k = int(self.rng.integers(0, N_GATES))
                if k:
                    X[j] = np.roll(X[j], k, axis=1)                  # range axis
                    loc[j] = np.roll(loc[j], k)
            # appearance jitter: scale amplitude + add noise so the localizer keys
            # on the target's *relative* energy concentration, not exact pixels
            # -> genuine, transferable localization instead of memorized values.
            scale = self.rng.uniform(0.8, 1.2, (len(ids), 1, 1, 1)).astype("float32")
            X *= scale
            X += self.rng.normal(0.0, 0.02, X.shape).astype("float32")
            np.clip(X, 0.0, 1.0, out=X)
        # softmax-localization target: one-hot of the primary gate (loc==1.0),
        # built AFTER the roll so it tracks the shifted target. Clutter/synth
        # maps (pres==0) get an all-zero target and are masked out via sample wt.
        B = len(ids)
        loc_soft = np.zeros((B, N_GATES), "float32")
        mask = pres.ravel().astype("float32")                        # 1 real target, else 0
        for j in range(B):
            if mask[j]:
                loc_soft[j, int(loc[j].argmax())] = 1.0
        # return only the label this model needs (two independent models):
        #  - "loc": one-hot primary gate, masked to target maps
        #  - "det": map-level presence on ALL maps
        if self.task == "loc":
            return X, {"loc_soft": loc_soft}, {"loc_soft": mask}
        return X, {"presence": pres}, {"presence": np.ones(B, "float32")}


def load_target_maps(split):
    """Real (non-synthetic) target maps for a split + their primary gate (0-idx).
    Used by AugLocVal to measure genuine 14-way localization on held-out data."""
    rows = [r for r in csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv")))
            if r["split"] == split and r["det"] == "1" and r["is_synth"] == "0"]
    X = np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in rows])
    primary = np.array([int(r["primary"]) for r in rows])
    return X, primary


class AugLocVal(keras.callbacks.Callback):
    """Select the checkpoint on GENUINE localization, not the gate-7/9-only val acc.

    Every epoch: roll the val target maps across all 14 range gates (the same
    honest test evaluate.py reports) and check the predicted gate tracks the
    shift. Saves the best weights on this metric and drives early stopping, so we
    keep the model that genuinely localizes across all gates -- not a lucky early
    epoch on a noisy 2-class metric.
    """

    def __init__(self, X, primary, ckpt, patience=12):
        super().__init__()
        self.X, self.primary, self.ckpt, self.patience = X, primary, ckpt, patience
        self.best, self.wait, self.best_epoch = -1.0, 0, 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        correct = total = 0
        for k in range(N_GATES):
            Xk = np.roll(self.X, k, axis=2)                          # roll range axis
            pk = self.model.predict(Xk, batch_size=128, verbose=0)["loc_soft"].argmax(1)
            exp = (self.primary + k) % N_GATES
            correct += int((pk == exp).sum()); total += len(exp)
        acc = correct / total
        logs["val_aug_loc_acc"] = acc
        msg = f"  val_aug_loc_acc (genuine 14-way) = {acc:.4f}"
        if acc > self.best:
            self.best, self.wait, self.best_epoch = acc, 0, epoch
            self.model.save_weights(self.ckpt)
            print(msg + f"  -> NEW BEST, saved {self.ckpt}")
        else:
            self.wait += 1
            print(msg + f"  (best {self.best:.4f} @ epoch {self.best_epoch}, "
                        f"wait {self.wait}/{self.patience})")
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"  early stop: no val_aug_loc_acc gain in {self.patience} epochs")


def load_clutter_maps(split):
    """Real clutter maps (no target) for a split -> (N,512,14,1). Used as presence
    negatives. Includes excised-real (synthetic-clutter) maps -- those are still clutter."""
    rows = [r for r in csv.DictReader(open(os.path.join(OUT_DIR, "labels_balanced.csv")))
            if r["split"] == split and r["det"] == "0"]
    return np.stack([np.load(os.path.join(OUT_DIR, r["path"])) for r in rows]).astype("float32")


class RealPresVal(keras.callbacks.Callback):
    """Select the DETECTOR checkpoint on REAL-target presence accuracy -- NO synthetic
    targets -- instead of val_loss on the fixed-gate real-sphere val set.

    Positives = real val target maps ROLLED to a random gate each (so presence can't key
    on a fixed position); negatives = held-out val clutter maps. Balanced, mirrors
    AugLocVal for the localizer.
    """

    def __init__(self, ckpt, patience=12, seed=7):
        super().__init__()
        Xt, _ = load_target_maps("val")                    # real val targets + primary gate
        rng = np.random.default_rng(seed)
        pos = np.stack([np.roll(x, int(rng.integers(N_GATES)), axis=1) for x in Xt])
        neg = load_clutter_maps("val")
        m = min(len(pos), len(neg))                        # balance the two classes
        pos = pos[rng.choice(len(pos), m, replace=False)]
        neg = neg[rng.choice(len(neg), m, replace=False)]
        self.X = np.concatenate([pos, neg]).astype("float32")
        self.y = np.concatenate([np.ones(m), np.zeros(m)]).astype(int)
        self.ckpt, self.patience = ckpt, patience
        self.best, self.wait, self.best_epoch = -1.0, 0, 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        s = self.model.predict(self.X, batch_size=128, verbose=0)["presence"].ravel()
        acc = max(((s >= t).astype(int) == self.y).mean()
                  for t in np.linspace(0.05, 0.95, 91))
        logs["val_real_pres_acc"] = acc
        msg = f"  val_real_pres_acc (real targets) = {acc:.4f}"
        if acc > self.best:
            self.best, self.wait, self.best_epoch = acc, 0, epoch
            self.model.save_weights(self.ckpt)
            print(msg + f"  -> NEW BEST, saved {self.ckpt}")
        else:
            self.wait += 1
            print(msg + f"  (best {self.best:.4f} @ epoch {self.best_epoch}, "
                        f"wait {self.wait}/{self.patience})")
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"  early stop: no val_real_pres_acc gain in {self.patience} epochs")


def weighted_bce(pos_weight, smoothing=0.0):
    pw = tf.constant(pos_weight, tf.float32)

    def loss(y_true, y_pred):
        if smoothing > 0:
            y_true = y_true * (1 - smoothing) + 0.5 * smoothing
        eps = 1e-7
        p = tf.clip_by_value(y_pred, eps, 1 - eps)
        l = -(pw * y_true * tf.math.log(p) + (1 - y_true) * tf.math.log(1 - p))
        return tf.reduce_mean(l)
    return loss


def make_sched(epochs, steps_per_epoch, lr):
    steps = epochs * steps_per_epoch
    return keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=steps, warmup_target=lr,
        warmup_steps=max(1, int(0.05 * steps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--phase", choices=["both", "loc", "det"], default="both",
                    help="train both models, only the localizer, or only the detector")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "checkpoints"), exist_ok=True)

    # ==================== 1) LOCALIZER (loc_soft only) ====================
    if args.phase in ("both", "loc"):
        print("\n========== TRAINING LOCALIZER ==========")
        tr = RDSequence("train", args.batch, augment=True, shuffle=True, task="loc")
        va = RDSequence("val", args.batch, task="loc")
        loc = build_localizer(n_pol=N_POL)
        loc.compile(
            optimizer=keras.optimizers.AdamW(make_sched(args.epochs, len(tr), args.lr),
                                             weight_decay=1e-4),
            loss={"loc_soft": keras.losses.CategoricalCrossentropy(label_smoothing=0.02)},
            metrics={"loc_soft": keras.metrics.CategoricalAccuracy(name="acc")})
        Xv, primv = load_target_maps("val")
        if len(primv) > 256:
            sel = np.random.default_rng(0).choice(len(primv), 256, replace=False)
            Xv, primv = Xv[sel], primv[sel]
        print(f"AugLocVal: {len(primv)} val targets x {N_GATES} rolls per epoch")
        loc.fit(tr, validation_data=va, epochs=args.epochs, verbose=1, callbacks=[
            AugLocVal(Xv, primv, LOC_CKPT, patience=args.patience),
            keras.callbacks.CSVLogger(os.path.join(RES_DIR, "train_log_loc.csv"))])
        print(f"Localizer best -> {LOC_CKPT}")

    # ==================== 2) DETECTOR (presence only) =====================
    if args.phase in ("both", "det"):
        print("\n========== TRAINING DETECTOR ==========")
        trd = RDSequence("train", args.batch, augment=True, shuffle=True, task="det")
        vad = RDSequence("val", args.batch, task="det")
        det = build_detector(n_pol=N_POL)
        det.compile(
            optimizer=keras.optimizers.AdamW(make_sched(args.epochs, len(trd), args.lr),
                                             weight_decay=1e-4),
            # focal loss focuses the detector on HARD target/clutter cases -> higher AUC.
            loss={"presence": keras.losses.BinaryFocalCrossentropy(gamma=2.0)},
            metrics={"presence": [keras.metrics.AUC(name="auc"),
                                  keras.metrics.BinaryAccuracy(name="acc")]})
        # Checkpoint on REAL-target presence (RealPresVal), not val_loss: real val
        # positives rolled across gates measure generic 'is a target here' skill without
        # a fixed-position prior. No synthetic targets anywhere in the pipeline.
        det.fit(trd, validation_data=vad, epochs=args.epochs, verbose=1, callbacks=[
            RealPresVal(DET_CKPT, patience=args.patience),
            keras.callbacks.CSVLogger(os.path.join(RES_DIR, "train_log_det.csv"))])
        print(f"Detector best -> {DET_CKPT}")


if __name__ == "__main__":
    main()
