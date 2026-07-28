"""
Step: NORMALIZE + BALANCE the labeled MATLAB RD maps -> dataset_v2/{train,val,test}/*.npy

Pipeline:
  1. Norm stats : per-polarisation 1st/99th dB percentile from TRAIN real maps only.
  2. Write reals: every labeled map -> (512,14,1) float32 in [0,1] as .npy (HH only).
  3. Balance    : per split, synthesize clutter-only maps (excise target gates with
                  random clutter columns from the SAME map) until det0 == det1 (50/50).
                  Synthetic maps: det=0, all-zero labels, primary=-1, is_synth=1,
                  and are EXCLUDED from the norm stats (computed before they exist).
  4. Manifest   : dataset_v2/labels_balanced.csv  +  dataset_v2/meta.json (pos_weight).

MATLAB layout: rd_maps (4,14,512,1023) = (pol, range, doppler, window).
Per-map target : (512,14,1) = (doppler, range, HH)  -> transpose then keep pol 0.
"""
import csv
import glob
import json
import os
import random

import h5py
import numpy as np

from ipix_targets import N_GATES

MATLAB_DIR = "matlab_rd"
OUT_DIR = "dataset_v4"          # 4-pol build (HH baseline stays in dataset_v2)
LABELS_CSV = "dataset_v2/labels.csv"   # reuse the existing per-window labels/splits
N_POL = 4                       # store ALL 4 pols HH/HV/VH/VV -- cross-pol is the
                                # strongest small-target channel (the honest accuracy lever)
P_LO, P_HI = 1.0, 99.0
STAT_WIN_STEP = 8          # sample every 8th train window for percentile estimate
STAT_DOP_STEP = 4          # and every 4th doppler bin
SEED = 1234


def read_timestamp(h5):
    return "".join(chr(c) for c in np.array(h5["timestamp"]).ravel())


def load_file_maps(h5):
    """Read the whole rd_maps array ONCE (contiguous, fast) and reorder to
    (window, doppler, range, pol) = (1023, 512, 14, 4). Per-window slicing on
    the last HDF5 axis is strided and ~100x slower, so we never do that."""
    arr = np.array(h5["rd_maps"])          # (pol, range, doppler, window)
    # keep ALL 4 pols -> (W, 512, 14, 4) = (window, doppler, range, pol)
    return arr.transpose(3, 2, 1, 0).astype("float32")


def load_labels():
    labels_path = LABELS_CSV
    if not os.path.exists(labels_path):
        raise SystemExit(
            f"'{labels_path}' not found. Run step 1 first:\n"
            f"    python build_labels.py")
    rows = list(csv.DictReader(open(labels_path)))
    by_ts = {}
    for r in rows:
        by_ts.setdefault(r["timestamp"], []).append(r)
    return rows, by_ts


def compute_norm_stats(files, by_ts):
    """Per-pol (lo, hi) dB percentiles from a subsample of TRAIN real maps."""
    vals = [[] for _ in range(N_POL)]
    for f in files:
        with h5py.File(f, "r") as h:
            ts = read_timestamp(h)
            if ts not in by_ts:
                continue
            arr = load_file_maps(h)                     # (W,512,14,4)
            train_idx = sorted({int(r["win_idx"]) for r in by_ts[ts]
                                if r["split"] == "train"})
            for w in train_idx[::STAT_WIN_STEP]:
                m = arr[w]                               # (512,14,4)
                for p in range(N_POL):
                    vals[p].append(m[::STAT_DOP_STEP, :, p].ravel())
    lo = np.zeros(N_POL, "float32")
    hi = np.zeros(N_POL, "float32")
    for p in range(N_POL):
        v = np.concatenate(vals[p])
        lo[p], hi[p] = np.percentile(v, [P_LO, P_HI])
    return lo, hi


def normalize(m, lo, hi):
    out = np.empty_like(m)
    for p in range(N_POL):
        out[..., p] = np.clip((m[..., p] - lo[p]) / (hi[p] - lo[p] + 1e-9), 0.0, 1.0)
    return out


def excise(m, det_vec, rng):
    """Make a clutter-only copy: overwrite each target gate with a random clutter gate."""
    tgt = np.where(det_vec > 0)[0]
    clut = np.where(det_vec == 0)[0]
    out = m.copy()
    for g in tgt:
        out[:, g, :] = m[:, int(rng.choice(clut)), :]
    return out


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT_DIR, s), exist_ok=True)

    files = sorted(glob.glob(os.path.join(MATLAB_DIR, "*.mat")))
    _, by_ts = load_labels()

    print("Computing norm stats from train reals ...")
    lo, hi = compute_norm_stats(files, by_ts)
    for p in range(N_POL):
        print(f"  pol{p}: lo={lo[p]:.2f} dB  hi={hi[p]:.2f} dB")

    manifest = []
    counters = {"train": 0, "val": 0, "test": 0}
    # remember real TARGET maps per split for later synthesis: (path, det_vec)
    target_pool = {"train": [], "val": [], "test": []}

    print("\nWriting real maps ...")
    for f in files:
        with h5py.File(f, "r") as h:
            ts = read_timestamp(h)
            if ts not in by_ts:
                continue
            arr = load_file_maps(h)                     # (W,512,14,4)
            for r in by_ts[ts]:
                split = r["split"]
                det_vec = np.array([float(r[f"det_g{i}"]) for i in range(N_GATES)], "float32")
                loc_vec = np.array([float(r[f"loc_g{i}"]) for i in range(N_GATES)], "float32")
                m = normalize(arr[int(r["win_idx"])], lo, hi)

                idx = counters[split]; counters[split] += 1
                rel = f"{split}/{split}_{idx:07d}.npy"
                np.save(os.path.join(OUT_DIR, rel), m)

                present = int(r["present"])
                manifest.append({
                    "path": rel, "split": split, "timestamp": ts,
                    "is_synth": 0, "det": present, "primary": int(r["primary"]),
                    **{f"det_g{i}": det_vec[i] for i in range(N_GATES)},
                    **{f"loc_g{i}": loc_vec[i] for i in range(N_GATES)},
                })
                if present == 1:
                    target_pool[split].append((rel, det_vec))
        print(f"  {ts}: done ({counters})")

    print("\nBalancing (synthesizing clutter to 50/50) ...")
    for split in ("train", "val", "test"):
        n_det1 = sum(1 for m in manifest if m["split"] == split and m["det"] == 1)
        n_det0 = sum(1 for m in manifest if m["split"] == split and m["det"] == 0)
        need = n_det1 - n_det0
        pool = target_pool[split]
        random.shuffle(pool)
        zeros = {f"det_g{i}": 0.0 for i in range(N_GATES)}
        zeros.update({f"loc_g{i}": 0.0 for i in range(N_GATES)})
        for k in range(need):
            src_rel, det_vec = pool[k % len(pool)]
            src = np.load(os.path.join(OUT_DIR, src_rel))
            synth = excise(src, det_vec, rng)
            idx = counters[split]; counters[split] += 1
            rel = f"{split}/{split}_{idx:07d}.npy"
            np.save(os.path.join(OUT_DIR, rel), synth)
            manifest.append({"path": rel, "split": split, "timestamp": "SYNTH",
                             "is_synth": 1, "det": 0, "primary": -1, **zeros})
        print(f"  {split}: real {n_det1}+{n_det0}  -> +{need} synth -> {n_det1}+{n_det1}")

    # ---- manifest + meta ----------------------------------------------------
    cols = (["path", "split", "timestamp", "is_synth", "det", "primary"]
            + [f"det_g{i}" for i in range(N_GATES)]
            + [f"loc_g{i}" for i in range(N_GATES)])
    with open(os.path.join(OUT_DIR, "labels_balanced.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for m in manifest:
            w.writerow({c: m[c] for c in cols})

    # per-cell positive rate (hard det labels) over the whole balanced set -> pos_weight
    det_mat = np.array([[m[f"det_g{i}"] for i in range(N_GATES)] for m in manifest])
    pos_rate = float(det_mat.mean())
    pos_weight = float((1 - pos_rate) / pos_rate)
    meta = {
        "norm_lo": lo.tolist(), "norm_hi": hi.tolist(),
        "p_lo": P_LO, "p_hi": P_HI,
        "counts": {s: sum(1 for m in manifest if m["split"] == s)
                   for s in ("train", "val", "test")},
        "det_counts": {s: {"det1": sum(1 for m in manifest if m["split"] == s and m["det"] == 1),
                           "det0": sum(1 for m in manifest if m["split"] == s and m["det"] == 0)}
                       for s in ("train", "val", "test")},
        "pos_cell_rate": pos_rate, "pos_weight": pos_weight,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nTotal maps: {len(manifest)}  splits={meta['counts']}")
    print(f"pos_cell_rate={pos_rate:.4f}  pos_weight={pos_weight:.2f}")
    print(f"Manifest -> {OUT_DIR}/labels_balanced.csv   Meta -> {OUT_DIR}/meta.json")


if __name__ == "__main__":
    main()
