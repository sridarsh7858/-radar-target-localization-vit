# CLAUDE.md — RD-ViT project guide

Read this first. It explains what the project is, how to set it up from scratch, which
files produce which numbers in the paper, and the traps that are easy to fall into.
`README.md` has the public-facing summary; this file is the working handbook.

## What this project is

RD-ViT detects a small floating target in sea clutter and says **which of 14 range gates**
it is in, using the real **IPIX Dartmouth-1993** X-band radar recordings (no synthetic
targets anywhere). Two independent networks, same backbone design, no shared weights:

- **Localizer** — softmax over 14 gates → predicted gate = argmax.
- **Detector** — prepended learnable `[DET]` token → sigmoid presence probability.

Backbone (`model.py`): input RD map `512 Doppler × 14 gates × n_pol` → 4 conv blocks
(16/32/64/128 filters, 3×3, BN, ReLU, max-pool along Doppler only, 512 → 8) → reshape to
14 range tokens (8×128 = 1024-d) → linear to 128-d + learnable position embedding →
4 pre-norm transformer blocks (4 heads, MLP 128→256→128, GELU, dropout 0.15).
**Tokens = range gates**, so attention compares every gate with every other gate.

Paper title: *"RD-ViT: A hybrid range Doppler vision transformer for detection and
localization of small marine targets"* (the manuscript is kept outside this repo).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt                      # tensorflow>=2.16, numpy, scipy, h5py, scikit-learn, matplotlib
```

Python 3.10 and 3.11 both work. A CPU is enough (training is slow but fine; GPU optional).

**Data is NOT in the repo.** Get the IPIX Dartmouth-1993 "stare" recordings from
McMaster (http://soma.ece.mcmaster.ca/ipix/) and put the `.cdf` files in `Dataset/`,
e.g. `Dataset/19931107_135603_starea.cdf`. The 14 timestamps needed are listed in
`ipix_targets.py` (12 are used, 2 are excluded as unverified). Files are matched by the
timestamp stored inside the CDF, so the filename suffix does not matter.

## Running the pipeline (from the repo root, in order)

| # | Command | Output (gitignored unless noted) |
|---|---|---|
| 1 | `python build_rd_maps.py` | `matlab_rd/<ts>.mat` — RD maps, shape (4 pol, 14 gates, 512 Doppler, 1023 windows) |
| 2 | `python build_labels.py` | `dataset_v2/labels.csv` — time-block split + labels |
| 3 | `python build_dataset.py` | `dataset_v4/*.npy` (~2 GB), `labels_balanced.csv`, `meta.json` (tracked) |
| 4 | `python train.py` | `results_v4/checkpoints/{loc,det}.weights.h5` (4-pol) |
| 4b | `python train_det_hh.py` | `results_hh/checkpoints/det_hh.weights.h5` (HH-only detector; `--pol hv/vh/vv` for others) |
| 5 | `python evaluate.py` | `results_v4/test_metrics.json` + confusion PNGs (localizer numbers) |
| 5b | `python eval_det_hh.py --pol hh` | `results_hh/test_metrics_hh.json` (HH detector alone) |
| 5c | `python eval_det_hh.py --pol all --ckpt baseline_v4_backup/det.weights.h5` | `results_v4/test_metrics_4pol.json` (4-pol detector alone) |

Steps 1–3 must run once even if you use the committed weights, because evaluation reads
`dataset_v4/`. Everything is seeded, so results reproduce.

Checks: `python final_audit.py` (PASS/FAIL on split integrity, labels, normalization,
param counts) and `python eval_checks.py` (native-position accuracy, per-recording
accuracy, P_FA on genuine vs excised clutter).

## Which weights are "the paper's" — read this

| Folder | What it is |
|---|---|
| `baseline_v4_backup/` | **The weights the paper reports.** Localizer 87.6% exact / 90.2% ±1, 4-pol detector. Identical to the first commit's `results_v4/checkpoints/`. |
| `results_v4/` | A **newer retrain** (88.2% / 90.7%). Does **not** match the paper. `train.py` writes here, so retraining overwrites it. |
| `results_hh/` | HH-only detector — the one the paper reports (97.3% P_D). |

To reproduce the paper's localizer numbers, evaluate with `baseline_v4_backup/loc.weights.h5`
(copy it over `results_v4/checkpoints/loc.weights.h5`, or point `LOC_CKPT` in
`evaluate.py` at it). Never quote `results_v4/test_metrics.json` as the paper's number
without checking which weights produced it.

## Where each paper number / figure comes from

| Paper item | Source |
|---|---|
| Table III localization 87.6% / 90.2%, baseline 10.1% / 27.9% | `evaluate.py` on `baseline_v4_backup/loc.weights.h5` (n = 3000 rolled test targets) |
| Native-position accuracy 87.5% | `eval_checks.py` |
| HH detector P_D 97.3%, AUC 0.980 | `results_hh/test_metrics_hh.json` (`pd_rolled`, `auc_genuine`) |
| 4-pol detector accuracy 99.3%, 0 false alarms on genuine clutter | `results_v4/test_metrics_4pol.json` (`balanced_acc`, `pfa_genuine`) |
| Table IV 14×14 confusion matrix | `evaluate.py` helpers `build_real_rolled("test", n=3000, seed=0)` + `tta_localize` on the backup weights |
| Table VI parameter counts | `final_audit.py` (section F) |
| Table V ablation, Table VI MFLOPs | **Not in the repo** — the ablation variants and FLOP counting were run separately and the scripts were not kept |
| Fig. 1 architecture | `gen_arch_v2.py` |
| Fig. 2 sample RD maps (Hz axis) | `rd_maps_4pol_fig.py` (clutter `19931107_145028` w880, target `19931107_141630` w984) |
| Fig. 3 P_D / P_FA vs threshold | `roc_pol.py` (writes `report_assets/roc_pol_scores.npz`) → `pd_pfa_threshold_paper.py` |
| ROC figure | `roc_pol.py` → `roc_pol_paper.py` |

Figure scripts write to `report_assets/` (gitignored; override with `REPORT_ASSETS`).

## Key design decisions (why the code looks the way it does)

- **Real targets only sit at gate 7 (9 recordings) or gate 9 (2).** A naive model would
  memorize "gate 7". Hence `train.py` `RDSequence`: paste a real donor target into a
  clutter map at a random gate (p = 0.5), excise targets back to clutter (p = 0.4), then
  cyclically **range-roll** every map with its label, then amplitude ×[0.8, 1.2] + noise σ 0.02.
- **Evaluation rolls each held-out target to all 14 gates** and uses 14-fold roll-averaging
  (TTA) so no gate prior can help. The max-energy baseline is reported alongside.
- **Time-blocked split, not random**: 70/15/15 of each recording's pulse timeline, with a
  256-pulse guard gap; val/test keep every 2nd window (stride 256) so no two evaluation
  windows share a pulse. Windows overlap (256 long, stride 128), so a random split would leak.
- **Normalization** = per-polarization 1st/99th dB percentiles from the **train split only**
  (`dataset_v4/meta.json`).
- **Clutter negatives**: the one genuine clutter-only recording (`19931107_145028`, 75 test
  windows) plus "excised" maps (target gates overwritten with other gates of the same map).
  Always report P_FA on **genuine** clutter separately — excised negatives are easier and
  partly circular. (HH P_FA: 22.7% genuine vs 2.5% excised; 4-pol: 0% genuine.)
- **Labels**: localization = hard one-hot on the primary gate; detection = map-level presence.
  `target_cols(g)` = gate g ± 1 (the target bleeds into neighbours).

## Conventions

- Gates are **1-indexed** in `ipix_targets.py` and the paper, **0-indexed** in arrays.
- Polarization channel order everywhere: `[HH, HV, VH, VV]` = channels 0..3.
- Map array layout on disk: `(512 Doppler, 14 range, n_pol)`; range axis = `axis=1`.
- Doppler axis: 512-pt FFT, PRF 1000 Hz, fftshifted → bin k ↦ (k − 256)·1000/512 Hz.
- `rithick/` and paper drafts are not part of the repo (see `.gitignore`).
