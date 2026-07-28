# RD-ViT — Radar Target Detection & Range-Gate Localization

A Vision-Transformer pipeline that finds small floating targets hidden in sea clutter,
trained and evaluated on the **IPIX Dartmouth-1993** X-band radar dataset.

**87.6 % exact range-gate localization, against a 10.1 % classical baseline** — trained
on real radar returns only, with no synthetic targets anywhere in the pipeline.

## The problem

A small object floating on the ocean — a beach ball wrapped in wire mesh, in the IPIX
trials — produces a radar return that is *weaker than the waves around it*. Sea clutter
is spiky, non-Gaussian, and at low grazing angles individual wave crests routinely return
more energy than the target does.

This breaks the obvious approach. Picking the range gate with the most energy is right
only **10.1 %** of the time — barely above the 7.1 % you'd get by guessing at random
among 14 gates. The energy simply isn't where the target is.

RD-ViT gets it right **87.6 %** of the time: **8.7× the classical baseline**. It does that
by learning the *Doppler signature* — a floating target bobs with the swell and produces
a narrow, coherent Doppler line, while clutter energy smears across the spectrum — and by
letting a transformer compare all 14 range gates against each other before deciding.

## Results

Held-out test set (n = 3000), real returns only, no synthetic data:

| Task | Metric | RD-ViT | Classical baseline | Random |
|---|---|---:|---:|---:|
| Localization | exact gate | **87.6 %** | 10.1 % | 7.1 % |
| Localization | within ±1 gate | **90.2 %** | 27.9 % | 21.4 % |
| Detection (4-pol) | presence accuracy | **99.3 %** | — | 50 % |
| Detection (HH only) | P_D | **97.3 %** | — | — |
| Detection (HH only) | AUC vs clutter | **0.980** | — | 0.5 |

The baseline is max-energy gate selection. Test targets are rolled across all 14 gates so
no gate index can be memorized. Detection threshold and polarization-fusion weight are
tuned on the **validation** split only.

Confusion matrices: `results_v4/confusion_localization.png`, `results_v4/confusion_presence.png`.
Full numbers: `results_v4/test_metrics.json`.

## How it works

Two **independent** models — no shared weights, trained separately:

- **Detector** — is a target present anywhere in this dwell? → one sigmoid
- **Localizer** — which of the 14 range gates holds it? → 14 logits

Both read the same input: a range–Doppler map stack of shape `512 × 14 × n_pol`
(Doppler bins × range gates × polarizations).

```
RD map (512 Doppler × 14 range × n_pol)
        │
        ▼
CNN patch-embedder ──────►  14 range tokens, dim 128   (Doppler 512 → 8 → projected)
        │
        ▼
+ learnable position embedding
        │
        ▼
4 × Transformer encoder blocks  (4 heads, FFN 256, dropout 0.15)
        │
        ├─► Localizer:  per-token MLP  ──►  14 gate logits
        └─► Detector:   prepend [DET] token, MLP on it  ──►  presence
```

The key design choice is **tokens = range gates**. Self-attention lets every gate see
every other gate, so the model can judge a candidate return *relative to the clutter
around it* rather than in isolation — which is exactly what a fixed-threshold detector
cannot do.

**Preprocessing** (`build_rd_maps.py`) follows the classical chain: per-window per-gate DC
removal → long-baseline IQ phase-imbalance correction → 256-pulse Hamming taper →
512-point zero-padded FFT → `fftshift` → 20·log₁₀ magnitude in dB.

## Dataset

**IPIX Dartmouth-1993**, X-band coherent radar in stare mode, Osborne Head Gunnery Range,
Nova Scotia. Each recording is 131,072 pulses across 14 range gates and 4 polarizations
(HH, HV, VH, VV). Sliced into 256-pulse windows at stride 128 → **1023 RD maps per
recording per polarization**.

The recordings are **not redistributed here** — obtain them from their original source and
place the `.cdf` files in a folder named `Dataset/`.

### Recordings

14 recordings in the manifest, **12 used** (11 with a verified target, 1 verified
clutter-only) and **2 excluded** because their target presence could not be verified.
`ipix_targets.py` is the single source of truth; gates are 1-indexed below to match the
literature.

| # | Timestamp | Date | Contents | Primary gate | Secondary gates | Used |
|---|---|---|---|:---:|:---:|:---:|
| 1 | `19931107_135603` | Nov 7 | Target | 9 | 8, 10, 11 | ✅ |
| 2 | `19931107_141630` | Nov 7 | Target | 9 | 8, 10, 11 | ✅ |
| 3 | `19931107_145028` | Nov 7 | **Clutter only** | — | — | ✅ |
| 4 | `19931108_213827` | Nov 8 | Unverified | — | — | ❌ |
| 5 | `19931108_220902` | Nov 8 | Target | 7 | 6, 8 | ✅ |
| 6 | `19931109_191449` | Nov 9 | Target | 7 | 6, 8 | ✅ |
| 7 | `19931109_202217` | Nov 9 | Target | 7 | 6, 8, 9 | ✅ |
| 8 | `19931110_001635` | Nov 10 | Target | 7 | 6, 8 | ✅ |
| 9 | `19931111_163625` | Nov 11 | Target | 7 | 6, 8, 9 | ✅ |
| 10 | `19931118_023604` | Nov 18 | Target | 7 | 5, 6, 8 | ✅ |
| 11 | `19931118_035737` | Nov 18 | Unverified | — | — | ❌ |
| 12 | `19931118_162155` | Nov 18 | Target | 7 | 6, 8 | ✅ |
| 13 | `19931118_162658` | Nov 18 | Target | 7 | 6, 8, 9 | ✅ |
| 14 | `19931118_174259` | Nov 18 | Target | 7 | 6, 8 | ✅ |

The **clutter-only** recording (#3) is the honest negative class — genuine sea returns
with no target present, not a synthetic or excised fake.

**Primary** is the gate holding the strongest target return; **secondary** gates are
adjacent gates the target also bleeds energy into. The two are labelled differently on
purpose: detection marks primary + secondary as positive (physically honest — the target
really does affect those gates), while localization uses a **hard one-hot on the primary
alone**, which sharpens the localizer onto the true gate.

### Splits

Splits are **time-blocked within each recording** — 70 % train / 15 % val / 15 % test by
pulse index — with a **256-pulse guard gap** after each boundary. Because windows overlap
at stride 128, a naive random split would put near-identical maps in both train and test;
the guard gap drops every window that straddles a boundary, so no test window shares a
single pulse with a training window.

| Split | Maps | Balance |
|---|---:|---|
| Train | 15,730 | 50 / 50 target / clutter |
| Val | 1,650 | 50 / 50 |
| Test | 1,650 | 50 / 50 |

Normalization is 1st–99th percentile scaling with statistics computed on the **training
split only** and reused verbatim for val and test. Those statistics are committed in
`dataset_v4/meta.json`.

**No synthetic targets.** Class balance comes from transplant/excise augmentation using
only real returns: moving a real target's energy to a different gate, or excising it to
manufacture a negative. `final_audit.py` verifies the splits, labels, and math end to end.

## Requirements

Python 3.11.

```bash
pip install -r requirements.txt
```

Plus the IPIX `.cdf` recordings in `Dataset/` — see the table above.

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
