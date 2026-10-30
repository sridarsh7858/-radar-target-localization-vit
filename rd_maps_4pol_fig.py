"""RD maps of all four polarizations for the two Fig. 2 samples, with physical axes.

Samples (identified by matching the published Fig. 2 pixels):
  clutter : 19931107_145028, window 880
  target  : 19931107_141630, window 984  (target at gate 9)
Doppler axis: 512-point FFT at PRF = 1000 Hz, fftshifted -> bin k is (k - 256) * 1000/512 Hz,
shown from -500 to +500 Hz.
Outputs: report_assets/rd_4pol_hz.png/.pdf, report_assets/rd_fig2_hz.png/.pdf
"""
import h5py
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRF, NFFT = 1000.0, 512
SAMPLES = [("Clutter only", "19931107_145028", 880), ("Target present", "19931107_141630", 984)]
POLS = ["HH", "HV", "VH", "VV"]
VMIN, VMAX = -5, 65
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})

df = PRF / NFFT
extent = [0.5, 14.5, (-NFFT // 2 - 0.5) * df, (NFFT // 2 - 0.5) * df]


def load(ts, w):
    with h5py.File(f"matlab_rd/{ts}.mat", "r") as f:
        return f["rd_maps"][:, :, :, w]          # (4 pol, 14 gate, 512 doppler) dB


def draw(ax, rd):
    im = ax.imshow(rd.T, origin="lower", aspect="auto", cmap="magma", vmin=VMIN, vmax=VMAX,
                   interpolation="nearest", extent=extent)
    ax.set_xticks(range(1, 15))
    ax.set_yticks([-500, -250, 0, 250, 500]); ax.set_ylim(-500, 500)
    ax.tick_params(labelsize=8)
    return im


# ---- figure 1: 2 rows (clutter / target) x 4 polarizations ----------------------
fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), sharex=True, sharey=True)
for r, (name, ts, w) in enumerate(SAMPLES):
    maps = load(ts, w)
    for c, pol in enumerate(POLS):
        im = draw(axes[r, c], maps[c])
        axes[r, c].set_title(f"{name}: {pol}", fontsize=11, fontweight="bold")
        if r == 1:
            axes[r, c].set_xlabel("Range gate", fontsize=10)
    axes[r, 0].set_ylabel("Doppler frequency (Hz)", fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label("FFT magnitude (dB)", fontsize=10)
for ext in ("png", "pdf"):
    fig.savefig(f"report_assets/rd_4pol_hz.{ext}", dpi=300, bbox_inches="tight")

# ---- figure 2: the paper's Fig. 2 (HH), drawn at IEEE column width ---------------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=True)
for ax, (name, ts, w), tag in zip(axes, SAMPLES, "ab"):
    im = draw(ax, load(ts, w)[0])
    ax.set_title(f"({tag}) {name}", fontsize=7.5, fontweight="bold", pad=3)
    ax.set_xlabel("Range gate", fontsize=7, labelpad=1)
    ax.set_xticks([1, 4, 7, 10, 14])
    ax.tick_params(labelsize=6, length=2, pad=1)
axes[0].set_ylabel("Doppler frequency (Hz)", fontsize=7, labelpad=1)
k = 269                                          # target peak bin (fftshifted index)
fd = (k - NFFT // 2) * df                        # = 25.4 Hz
axes[1].annotate(f"Target: gate 9, {fd:.1f} Hz", xy=(9, fd), xytext=(2.6, 330), fontsize=5.5,
                 color="white", arrowprops=dict(arrowstyle="->", color="white", lw=0.6),
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", lw=0.5, alpha=0.7))
fig.subplots_adjust(left=0.13, right=0.86, bottom=0.17, top=0.9, wspace=0.08)
cax = fig.add_axes([0.88, 0.17, 0.025, 0.73])
cb = fig.colorbar(im, cax=cax); cb.set_label("FFT magnitude (dB)", fontsize=6.5, labelpad=2)
cb.ax.tick_params(labelsize=6, length=2, pad=1)
for ext in ("png", "pdf"):
    fig.savefig(f"report_assets/rd_fig2_hz.{ext}", dpi=600)
print("wrote report_assets/rd_4pol_hz.* and report_assets/rd_fig2_hz.*")
