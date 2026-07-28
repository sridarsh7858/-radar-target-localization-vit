"""
IPIX Dartmouth-1993 ground-truth target database + dual label-vector generator.

This module is the single source of truth for *which range gate holds the target*
in each recording. It is keyed by the CDF internal timestamp (YYYYMMDD_HHMMSS),
which is also embedded in every matlab_rd/*.mat file, so files can be renamed
freely without breaking labels.

Label scheme: DUAL (hard detection + hard localization)
  - detection target  : primary + secondary gates = 1   (physically honest:
                        the floating sphere bleeds energy into +-1/2 gates)
  - localization target: primary = 1, everything else = 0   (HARD one-hot)
                        (a single-gate target sharpens the localizer onto the
                         true primary gate -- guide's request: 0/1 labels only,
                         for better localization accuracy)

Gates are 1-indexed in this table (gate 1..14) to match the literature;
the generated vectors are 0-indexed (g0..g13).
"""
import numpy as np

N_GATES = 14

# present: True / False / None   (None => unverified, dropped from the dataset)
# primary: 1-indexed gate of strongest target return
# secondary: list of 1-indexed adjacent gates also affected
IPIX_TARGETS = {
    # ---- Nov-07 : target in gate 9 -------------------------------------------
    "19931107_135603": {"present": True,  "primary": 9, "secondary": [8, 10, 11]},
    "19931107_141630": {"present": True,  "primary": 9, "secondary": [8, 10, 11]},
    "19931107_145028": {"present": False, "primary": None, "secondary": []},  # genuine clutter
    # ---- Nov-08 .. Nov-18 : target in gate 7 ---------------------------------
    "19931108_213827": {"present": None,  "primary": 7, "secondary": [6, 8]},      # EXCLUDED (unverified)
    "19931108_220902": {"present": True,  "primary": 7, "secondary": [6, 8]},
    "19931109_191449": {"present": True,  "primary": 7, "secondary": [6, 8]},
    "19931109_202217": {"present": True,  "primary": 7, "secondary": [6, 8, 9]},
    "19931110_001635": {"present": True,  "primary": 7, "secondary": [6, 8]},
    "19931111_163625": {"present": True,  "primary": 7, "secondary": [6, 8, 9]},
    "19931118_023604": {"present": True,  "primary": 7, "secondary": [5, 6, 8]},
    "19931118_035737": {"present": None,  "primary": 7, "secondary": [5, 6, 8]},   # EXCLUDED (unverified)
    "19931118_162155": {"present": True,  "primary": 7, "secondary": [6, 8]},
    "19931118_162658": {"present": True,  "primary": 7, "secondary": [6, 8, 9]},
    "19931118_174259": {"present": True,  "primary": 7, "secondary": [6, 8]},
}


def is_usable(timestamp: str) -> bool:
    """True only for files with a verified present flag (True or False)."""
    rec = IPIX_TARGETS.get(timestamp)
    return rec is not None and rec["present"] is not None


def gate_label_vectors(timestamp: str):
    """
    Return (det_vec, loc_vec) for a recording, both length-14, 0-indexed.

      det_vec : float32, 1.0 on primary+secondary gates else 0.0  (detection head)
      loc_vec : float32, 1.0 primary / 0.0 everything else (HARD one-hot, localization head)

    Raises KeyError for unknown / unverified timestamps so callers can't
    silently mislabel a dropped file.
    """
    rec = IPIX_TARGETS[timestamp]
    if rec["present"] is None:
        raise KeyError(f"{timestamp} is unverified (present=None) and must be dropped")

    import numpy as np
    det = np.zeros(N_GATES, dtype="float32")
    loc = np.zeros(N_GATES, dtype="float32")

    if not rec["present"]:                      # genuine clutter file -> all zeros
        return det, loc

    p = rec["primary"] - 1                      # 1-indexed -> 0-indexed
    det[p] = 1.0
    loc[p] = 1.0                                 # HARD one-hot: only the primary
    for s in rec["secondary"]:
        si = s - 1
        det[si] = 1.0                           # detection still marks adjacent gates
        # localization stays 0 on secondary gates (hard 0/1 target)
    return det, loc


def target_cols(primary, n=N_GATES):
    """Boolean length-n mask of a target's range extent: the primary gate +-1
    (circular). REPLACES the old per-gate det_vec for the train.py transplant/excise
    augmentation and the evaluate.py contrast measure. det_vec is no longer a label
    -- only loc_vec (loc_soft) + the map-level presence scalar are. primary 0-indexed.
    """
    cols = np.zeros(n, dtype=bool)
    for d in (-1, 0, 1):
        cols[(int(primary) + d) % n] = True
    return cols


def usable_timestamps():
    """Sorted list of the 12 verified timestamps (11 target + 1 clutter)."""
    return sorted(t for t in IPIX_TARGETS if is_usable(t))
