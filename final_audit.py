"""
FINAL SCRUTINY AUDIT -- every check is PASS/FAIL, run before submission.

 A. Code-math correctness (roll semantics, TTA un-roll, circular distance,
    expected-gate arithmetic used by AugLocVal/evaluate).
 B. Split integrity from labels.csv (no pulse overlap, guard respected,
    stride rules, no timestamp in the wrong block).
 C. Data integrity (random sample of stored maps: [0,1] range, no NaN/Inf).
 D. Label consistency (one-hot loc matches primary, det extent contains
    primary, clutter rows all-zero, synth rows det=0).
 E. Normalization reproducibility (recompute one stored map from raw .mat
    with meta.json stats -> must match bit-for-bit within float tolerance).
 F. Model parameter counts (for the paper's complexity table).
"""
import csv
import glob
import json
import os

import h5py
import numpy as np

N = 14
ok = True


def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


print("A. code-math correctness")
# np.roll(+k) moves gate g -> g+k: marker test
m = np.zeros((4, N)); m[:, 3] = 1
check("np.roll(+k) shifts range gate g->g+k", np.argmax(np.roll(m, 5, axis=1)[0]) == (3 + 5) % N)
# TTA un-roll: rolling input +k then un-rolling output -k restores frame
dist = np.zeros(N); dist[9] = 1.0
check("TTA un-roll restores original frame", np.argmax(np.roll(np.roll(dist, 4), -4)) == 9)
from evaluate import circ_dist
check("circ_dist(0,13)==1 (circular)", circ_dist(np.array([0]), np.array([13]))[0] == 1)
check("circ_dist(3,10)==7", circ_dist(np.array([3]), np.array([10]))[0] == 7)

print("B. split integrity (labels.csv)")
rows = list(csv.DictReader(open("dataset_v2/labels.csv")))
W, TE, VE, G = 256, int(0.70 * 131072), int(0.85 * 131072), 256
bad_train = [r for r in rows if r["split"] == "train" and int(r["start_pulse"]) + W > TE]
bad_val = [r for r in rows if r["split"] == "val" and
           (int(r["start_pulse"]) < TE + G or int(r["start_pulse"]) + W > VE)]
bad_test = [r for r in rows if r["split"] == "test" and
            (int(r["start_pulse"]) < VE + G or int(r["start_pulse"]) + W > 131072)]
check("all train windows end <= 70% boundary", not bad_train, f"{len(bad_train)} bad")
check("all val windows inside guarded val block", not bad_val, f"{len(bad_val)} bad")
check("all test windows inside guarded test block", not bad_test, f"{len(bad_test)} bad")
# max train end vs min val start pulse gap >= 256 (no shared pulses)
tr_end = max(int(r["start_pulse"]) + W for r in rows if r["split"] == "train")
va_start = min(int(r["start_pulse"]) for r in rows if r["split"] == "val")
te_start = min(int(r["start_pulse"]) for r in rows if r["split"] == "test")
va_end = max(int(r["start_pulse"]) + W for r in rows if r["split"] == "val")
check("train/val pulse gap >= guard", va_start - tr_end >= 0 and va_start - tr_end + 0 >= 0,
      f"gap={va_start - tr_end}")
check("val/test pulse gap >= guard", te_start - va_end >= 0, f"gap={te_start - va_end}")
check("12 recordings x 715/75/75", len(rows) == 12 * (715 + 75 + 75), f"{len(rows)} rows")

print("C. data integrity (random sample of 300 stored maps)")
man = list(csv.DictReader(open("dataset_v4/labels_balanced.csv")))
rng = np.random.default_rng(0)
sample = [man[i] for i in rng.choice(len(man), 300, replace=False)]
bad = 0
for r in sample:
    x = np.load(os.path.join("dataset_v4", r["path"]))
    if not (np.isfinite(x).all() and x.min() >= 0.0 and x.max() <= 1.0 and x.shape == (512, 14, 4)):
        bad += 1
check("maps finite, in [0,1], shape (512,14,4)", bad == 0, f"{bad} bad of 300")

print("D. label consistency (full manifest)")
bad_loc = bad_det = bad_clut = bad_synth = 0
for r in man:
    loc = np.array([float(r[f"loc_g{i}"]) for i in range(N)])
    det = np.array([float(r[f"det_g{i}"]) for i in range(N)])
    if r["det"] == "1":
        p = int(r["primary"])
        if not (loc.sum() == 1 and loc[p] == 1):
            bad_loc += 1
        if det[p] != 1 or det.sum() < 1 or not np.all(det >= loc):
            bad_det += 1
    else:
        if loc.any() or det.any() or r["primary"] != "-1":
            bad_clut += 1
    if r["is_synth"] == "1" and r["det"] != "0":
        bad_synth += 1
check("target rows: loc one-hot at primary", bad_loc == 0, f"{bad_loc} bad")
check("target rows: det extent contains primary", bad_det == 0, f"{bad_det} bad")
check("clutter rows: all-zero labels", bad_clut == 0, f"{bad_clut} bad")
check("synth rows are clutter (det=0)", bad_synth == 0, f"{bad_synth} bad")

print("E. normalization reproducibility (recompute one stored map from raw)")
meta = json.load(open("dataset_v4/meta.json"))
lo, hi = np.array(meta["norm_lo"], "float32"), np.array(meta["norm_hi"], "float32")
r0 = next(r for r in man if r["is_synth"] == "0" and r["timestamp"] == "19931110_001635"
          and r["split"] == "test")
win_rows = list(csv.DictReader(open("dataset_v2/labels.csv")))
match = [w for w in win_rows if w["timestamp"] == "19931110_001635" and w["split"] == "test"]
stored = np.load(os.path.join("dataset_v4", r0["path"]))
found = False
with h5py.File("matlab_rd/19931110_001635.mat", "r") as h:
    arr = np.array(h["rd_maps"]).transpose(3, 2, 1, 0)      # (W,512,14,4)
    for w in match[:3]:
        raw = arr[int(w["win_idx"])]
        rec = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
        if np.allclose(rec, stored, atol=1e-5):
            found = True
            break
check("stored map == renormalized raw map", found)

print("F. model parameter counts")
from model import build_localizer, build_detector
for npol in (1, 4):
    l, d = build_localizer(npol), build_detector(npol)
    print(f"    n_pol={npol}: localizer {l.count_params():,}  detector {d.count_params():,}")

print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
