"""
Step: LABEL the MATLAB-generated RD maps.

Reads matlab_rd/*.mat (each holds 1023 RD maps produced at stride 128 over the
131,072-pulse timeline) and writes dataset_v2/labels.csv with, for every real
map, its contiguous train/val/test split and its DUAL label vectors.

This does NOT yet do normalization or 50/50 map balancing -- it produces the
honest per-map label table those later steps build on.

Split (per file, over the PULSE timeline -- never random):
  train : pulses [0, 70%)          keep ALL stride-128 windows  (overlap, ~2x)
  val   : pulses [70%, 85%)        keep every 2nd window        (stride 256)
  test  : pulses [85%, 100%)       keep every 2nd window        (stride 256)
  A GUARD_PULSES gap after each train/val boundary drops windows that would let
  overlapping train windows leak into val/test.
"""
import csv
import glob
import os

import h5py
import numpy as np

from ipix_targets import IPIX_TARGETS, gate_label_vectors, is_usable, N_GATES

MATLAB_DIR = "matlab_rd"
OUT_DIR = "dataset_v2"
PULSES_PER_FILE = 131072
WINDOW = 256
MAT_STRIDE = 128          # stride MATLAB used for all 1023 windows
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15           # test = remaining 0.15
GUARD_PULSES = 256


def assign_split(start_pulse: int, win_idx: int):
    """Return 'train' / 'val' / 'test' / None (None => dropped: guard or stride)."""
    train_end = int(TRAIN_FRAC * PULSES_PER_FILE)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * PULSES_PER_FILE)
    end_pulse = start_pulse + WINDOW

    # train: every stride-128 window fully inside the train block
    if end_pulse <= train_end:
        return "train"
    # val: stride-256 (even idx), fully inside val block, past the guard gap
    if start_pulse >= train_end + GUARD_PULSES and end_pulse <= val_end:
        return "val" if (win_idx % 2 == 0) else None
    # test: stride-256 (even idx), fully inside test block, past the guard gap
    if start_pulse >= val_end + GUARD_PULSES and end_pulse <= PULSES_PER_FILE:
        return "test" if (win_idx % 2 == 0) else None
    return None   # inside a guard gap (or odd window in val/test)


def read_timestamp(h5) -> str:
    return "".join(chr(c) for c in np.array(h5["timestamp"]).ravel())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(MATLAB_DIR, "*.mat")))

    det_cols = [f"det_g{i}" for i in range(N_GATES)]
    loc_cols = [f"loc_g{i}" for i in range(N_GATES)]
    header = ["timestamp", "win_idx", "start_pulse", "split",
              "present", "det", "primary"] + det_cols + loc_cols

    rows = []
    stats = {}   # timestamp -> {split: count}
    skipped_files = []

    for f in files:
        with h5py.File(f, "r") as h:
            ts = read_timestamp(h)
            n_windows = int(np.array(h["n_windows"])[0, 0])

        if not is_usable(ts):
            skipped_files.append((ts, IPIX_TARGETS.get(ts, {}).get("present")))
            continue

        det_vec, loc_vec = gate_label_vectors(ts)
        present = bool(IPIX_TARGETS[ts]["present"])
        det = int(present)                                  # map-level presence
        primary = (int(np.argmax(loc_vec)) if present else -1)
        stats.setdefault(ts, {"train": 0, "val": 0, "test": 0})

        for wi in range(n_windows):
            start = wi * MAT_STRIDE
            split = assign_split(start, wi)
            if split is None:
                continue
            stats[ts][split] += 1
            rows.append([ts, wi, start, split, int(present), det, primary]
                        + [f"{v:.1f}" for v in det_vec]
                        + [f"{v:.1f}" for v in loc_vec])

    out_csv = os.path.join(OUT_DIR, "labels.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    # ---- report -------------------------------------------------------------
    print(f"Wrote {len(rows)} labeled maps -> {out_csv}\n")
    print(f"{'timestamp':<18}{'present':>8}{'train':>8}{'val':>7}{'test':>7}")
    tot = {"train": 0, "val": 0, "test": 0}
    n_target = n_clutter = 0
    for ts in sorted(stats):
        present = IPIX_TARGETS[ts]["present"]
        n_target += present
        n_clutter += (not present)
        s = stats[ts]
        for k in tot:
            tot[k] += s[k]
        print(f"{ts:<18}{str(present):>8}{s['train']:>8}{s['val']:>7}{s['test']:>7}")
    print(f"{'-'*48}")
    print(f"{'TOTAL':<18}{'':>8}{tot['train']:>8}{tot['val']:>7}{tot['test']:>7}")
    print(f"\nUsable files: {len(stats)}  ({n_target} target, {n_clutter} clutter)")
    if skipped_files:
        print("Skipped (present=None / unknown):")
        for ts, p in skipped_files:
            print(f"  {ts}  present={p}")


if __name__ == "__main__":
    main()
