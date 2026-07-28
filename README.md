# RD-ViT — Radar Target Detection & Range-Gate Localization

A Vision-Transformer pipeline for small-target detection and range-gate localization in
sea clutter, trained and evaluated on the **IPIX Dartmouth-1993** radar dataset.

Two independent models (no shared weights) operate on range–Doppler map stacks:

- **Detector** — is a target present in this dwell?
- **Localizer** — which of the 14 range gates contains it?

Everything is trained on **real IPIX returns only** — no synthetic targets — with
time-blocked splits and a guard gap to prevent leakage between train and test.

## Results

Held-out test set (n = 3000), real returns only:

| Task | Metric | RD-ViT | Max-energy baseline |
|---|---|---:|---:|
| Localization | exact gate | **87.6 %** | 10.1 % |
| Localization | within ±1 gate | **90.2 %** | 27.9 % |
| Detection (4-pol) | accuracy | **99.3 %** | — |
| Detection (HH only) | P_D | **97.3 %** | — |

Confusion matrices: `results_v4/confusion_localization.png`, `results_v4/confusion_presence.png`.
Full numbers: `results_v4/test_metrics.json`.

## Requirements

Python 3.11.

```bash
pip install -r requirements.txt
```

You also need the IPIX `.cdf` recordings in a folder named `Dataset/` next to these
scripts — 12 usable stare recordings. See `ipix_targets.py` for the exact list.
The recordings are not redistributed here.

## Pipeline

Run from the repository root, in order:

| # | Command | Produces |
|---|---|---|
| 1 | `python build_rd_maps.py` | `matlab_rd/<ts>.mat` — RD-map stacks (4 polarizations, 1023 windows per recording) |
| 2 | `python build_labels.py` | `dataset_v2/labels.csv` — time-blocked train/val/test labels with guard gap |
| 3 | `python build_dataset.py` | `dataset_v4/` — normalized (train-split stats) and balanced maps, ~2 GB of `.npy` |
| 4 | `python train.py` | `results_v4/checkpoints/` — localizer + detector (4-pol) |
| 5 | `python evaluate.py` | `results_v4/test_metrics.json` + confusion matrix PNGs |

Optional: `python train_det_hh.py` — HH-only detector → `results_hh/`

Everything is seeded; the pipeline regenerates end-to-end from the raw `.cdf` files.

## Pretrained weights

Committed, so you can skip step 4:

| File | Model | Input shape |
|---|---|---|
| `results_v4/checkpoints/loc.weights.h5` | Localizer | 512 × 14 × 4 |
| `results_v4/checkpoints/det.weights.h5` | Detector | 512 × 14 × 4 |
| `results_hh/checkpoints/det_hh.weights.h5` | Detector (HH) | 512 × 14 × 1 |

**Steps 1–3 must still be run once**, because evaluation reads the maps from
`dataset_v4/` and normalization statistics are derived from the training split.
The stats themselves are tracked in `dataset_v4/meta.json` for reference.

## Auditing & figures

| Script | Purpose |
|---|---|
| `eval_checks.py` | Supplementary audit — per-recording accuracy, P_FA split |
| `eval_det_hh.py` | HH-detector evaluation |
| `final_audit.py` | End-to-end sanity checks on splits, labels, and math |
| `gen_arch_v2.py` | Architecture diagram |
| `gen_figs_v2.py` | Result figures |
| `gen_samples_v3.py` | Sample-prediction figure |

The `gen_*` scripts write to `report_assets/` by default; override with the
`REPORT_ASSETS` environment variable.

## Repository layout

```
model.py            localizer + detector architectures
ipix_targets.py     per-recording target manifest (primary/secondary gates)
build_*.py          data pipeline (steps 1-3)
train*.py           training (step 4)
evaluate.py         held-out evaluation (step 5)
eval_*.py           supplementary audits
final_audit.py      end-to-end sanity checks
gen_*.py            figure generation
results_v4/         4-pol baseline: checkpoints, metrics, confusion matrices
results_hh/         HH-only detector
dataset_v4/         normalization stats (maps themselves are gitignored)
```

Raw recordings, generated datasets, and training logs are excluded from version
control — see `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).

The IPIX Dartmouth-1993 recordings are not included and are not covered by this
license; obtain them from their original source.
