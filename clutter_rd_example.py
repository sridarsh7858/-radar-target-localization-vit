"""Example RD map showing sea clutter occupying all range gates within a narrow Doppler band."""
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
fn = sys.argv[1] if len(sys.argv) > 1 else "matlab_rd/19931107_135603.mat"
w = int(sys.argv[2]) if len(sys.argv) > 2 else 500
with h5py.File(fn, "r") as f:
    rd = f["rd_maps"][0, :, :, w]          # HH, (14 range, 512 doppler) dB
rd = rd - rd.max()
nb = rd.shape[1]
dop = np.arange(nb) - nb // 2               # Doppler bin index, 0 = zero Doppler
# per-gate clutter extent: bins within 10 dB of that gate's peak
widths, centres = [], []
for g in range(rd.shape[0]):
    row = rd[g]; m = row >= row.max() - 10
    widths.append(m.sum()); centres.append(dop[np.argmax(row)])
print("gate  peak_bin  bins_within_10dB")
for g, (c, wd) in enumerate(zip(centres, widths)): print(f"{g+1:4d}  {c:8d}  {wd:6d}")
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(rd.T, aspect="auto", origin="lower", cmap="jet", vmin=-50, vmax=0,
               interpolation="nearest", extent=[0.5, rd.shape[0] + 0.5, dop[0], dop[-1]])
ax.set_xlabel("Range gate"); ax.set_ylabel("Doppler bin (0 = zero Doppler)")
ax.set_xticks(range(1, rd.shape[0] + 1))
ax.set_title(f"HH RD map, {fn.split('/')[-1][:-4]}, window {w}")
plt.colorbar(im, ax=ax, label="dB (rel. peak)")
plt.tight_layout(); out = f"report_assets/clutter_rd_example_w{w}.png"; plt.savefig(out, dpi=150); print(out)
