"""
Step 0: RAW IPIX NetCDF -> matlab_rd/<timestamp>.mat  (Python replacement for the
original MATLAB RD-map generator; the downstream pipeline is unchanged).

Implements the report's preprocessing chain (Sec. 3.2 / Fig. 3.1) exactly:
  1. per-window, per-gate DC removal (zero-Doppler pedestal)
  2. long-baseline IQ phase-imbalance correction of I:
        Icorr = (I - Q sin(beta)) / sqrt(1 - sin^2(beta)),
        sin(beta) = 2 E[I.Q] / E[I^2 + Q^2]   (estimated per gate/pol over the
        WHOLE recording -- the "long baseline")
  3. Hamming taper over the 256-pulse window
  4. 512-point FFT (zero-padded), fftshift so zero Doppler is centred
  5. 20*log10 magnitude (dB)

Windows: 1023 windows of 256 pulses at stride 128 over the 131,072-pulse timeline.

Polarizations (4): the IPIX transmitter alternates H/V; each stored sweep carries
both tx pols and the receiver's like/cross ADC pairs. Channel order written here
matches the pipeline convention pol = [HH, HV, VH, VV]:
  HH = tx H, like    HV = tx H, cross    VH = tx V, cross    VV = tx V, like
ADC samples are unsigned 8-bit stored in int8 -> reinterpret as uint8 first.

Output HDF5 layout (read by build_labels.py / build_dataset.py via h5py):
  rd_maps   : float32 (4, 14, 512, 1023)  = (pol, range, doppler, window)
  timestamp : uint8 ASCII codes of "YYYYMMDD_HHMMSS"
  n_windows : int64 (1,1)
"""
import glob
import os
import re

import h5py
import numpy as np
from scipy.io import netcdf_file

RAW_DIR = "Dataset"
OUT_DIR = "matlab_rd"
N_PULSES = 131072
WINDOW = 256
STRIDE = 128
NFFT = 512
N_GATES = 14
N_WINDOWS = (N_PULSES - WINDOW) // STRIDE + 1          # 1023
EPS = 1e-12


def timestamp_from_name(path):
    m = re.match(r"(\d{8}_\d{6})", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse timestamp from {path}")
    return m.group(1)


def load_iq(path):
    """Return complex64 (4, 14, 131072) = (pol[HH,HV,VH,VV], gate, pulse),
    DC left in (removed per window later), IQ imbalance corrected per gate/pol
    with the long-baseline estimate."""
    f = netcdf_file(path, mmap=False)
    adc = f.variables["adc_data"][:]                    # (nsweep, ntxpol, nrange, nadc) int8
    iI = int(np.array(f.variables["adc_like_I"].data))
    iQ = int(np.array(f.variables["adc_like_Q"].data))
    xI = int(np.array(f.variables["adc_cross_I"].data))
    xQ = int(np.array(f.variables["adc_cross_Q"].data))
    f.close()

    # unsigned 8-bit stored in int8 -> reinterpret, centre roughly at 0
    adc = adc.view(np.uint8).astype(np.float32) - 127.5   # (S, 2, 14, 4)

    out = np.empty((4, N_GATES, N_PULSES), dtype=np.complex64)
    # pol order [HH, HV, VH, VV]: (txpol index, (I,Q) adc indices)
    chans = [(0, iI, iQ), (0, xI, xQ), (1, xI, xQ), (1, iI, iQ)]
    for p, (tx, ci, cq) in enumerate(chans):
        I = adc[:, tx, :, ci].T.copy()                  # (14, S)
        Q = adc[:, tx, :, cq].T.copy()
        # long-baseline stats per gate (mean removed first so sin(beta) is a
        # pure phase-imbalance estimate, not an offset artefact)
        Im = I - I.mean(axis=1, keepdims=True)
        Qm = Q - Q.mean(axis=1, keepdims=True)
        sinb = 2.0 * (Im * Qm).mean(axis=1) / ((Im**2 + Qm**2).mean(axis=1) + EPS)
        sinb = sinb[:, None]
        Icorr = (Im - Qm * sinb) / np.sqrt(1.0 - sinb**2)
        out[p] = (Icorr + 1j * Qm).astype(np.complex64)
    return out


def rd_maps_for_file(iq):
    """(4,14,131072) complex -> (4,14,512,1023) float32 dB RD maps."""
    win = np.hamming(WINDOW).astype(np.float32)
    maps = np.empty((4, N_GATES, NFFT, N_WINDOWS), dtype=np.float32)
    # slice all windows at once: (n_windows, WINDOW) index matrix
    starts = np.arange(N_WINDOWS) * STRIDE
    idx = starts[:, None] + np.arange(WINDOW)[None, :]          # (1023, 256)
    for p in range(4):
        for g in range(N_GATES):
            xw = iq[p, g][idx]                                   # (1023, 256)
            xw = xw - xw.mean(axis=1, keepdims=True)             # per-window DC removal
            spec = np.fft.fft(xw * win, n=NFFT, axis=1)          # (1023, 512)
            spec = np.fft.fftshift(spec, axes=1)                 # zero Doppler centred
            maps[p, g] = (20.0 * np.log10(np.abs(spec) + 1e-6)).T.astype(np.float32)
    return maps


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.cdf")))
    if not files:
        raise SystemExit(f"no .cdf files in {RAW_DIR}/")
    for path in files:
        ts = timestamp_from_name(path)
        out_path = os.path.join(OUT_DIR, f"{ts}.mat")
        if os.path.exists(out_path):
            print(f"{ts}: exists, skipping")
            continue
        print(f"{ts}: reading + correcting IQ ...", flush=True)
        iq = load_iq(path)
        print(f"{ts}: computing {N_WINDOWS} RD maps x 4 pol ...", flush=True)
        maps = rd_maps_for_file(iq)
        del iq
        with h5py.File(out_path, "w") as h:
            h.create_dataset("rd_maps", data=maps, compression=None)
            h.create_dataset("timestamp", data=np.frombuffer(ts.encode(), dtype=np.uint8))
            h.create_dataset("n_windows", data=np.array([[N_WINDOWS]], dtype=np.int64))
        print(f"{ts}: wrote {out_path}  ({maps.nbytes/1e6:.0f} MB)", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
