"""
dsp/mvdr.py

Minimum Variance Distortionless Response (MVDR) beamformer.

Solves the constrained optimisation problem:
    minimise   w^H R w          (minimise output power = suppress interference)
    subject to w^H a_look = 1  (GPS look direction passes through undistorted)

Closed-form solution:
    w = R_dl^{-1} a_look / (a_look^H R_dl^{-1} a_look)

Solved via Cholesky factorisation (not pinv) because R_dl is guaranteed
positive definite after diagonal loading.

Critical design constraint — FIXED look direction:
    GPS signals are ~20-30 dB below the noise floor. MUSIC cannot see them.
    We therefore NEVER estimate the GPS look direction from the data.
    It is fixed to the configured azimuth/elevation (default: zenith, el=90°).
    For a ground-based horizontal array looking up, el=90° means the GPS
    look direction steering vector has equal phase across all elements
    (a_look = [1, 1, 1, 1] · g(90°) = [1, 1, 1, 1]).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from dsp.geometry import steering_vector, GPS_L1_HZ, GPS_L1_D


# ---------------------------------------------------------------------------
# MVDR weight computation
# ---------------------------------------------------------------------------

def mvdr_weights(
    R_dl: np.ndarray,
    look_az_deg: float = 0.0,
    look_el_deg: float = 90.0,
    freq_hz: float = GPS_L1_HZ,
    spacing_m: float = GPS_L1_D,
    cal_offsets_deg: np.ndarray = None,
) -> np.ndarray:
    """
    Compute MVDR beamforming weight vector.

    Parameters
    ----------
    R_dl          : ndarray (4,4) diagonally loaded covariance from diagonal_load()
                    Must be positive definite (guaranteed by diagonal_load).
    look_az_deg   : GPS look direction azimuth in degrees (from config)
    look_el_deg   : GPS look direction elevation in degrees (from config, default 90 = zenith)
    freq_hz       : carrier frequency
    spacing_m     : element spacing
    cal_offsets_deg : per-channel cal offsets to apply to look direction vector

    Returns
    -------
    w : ndarray shape (4,), dtype complex128
        MVDR weight vector satisfying w^H a_look = 1.
    """
    # Build look direction steering vector (include element pattern — GPS at zenith)
    a_look = steering_vector(
        look_az_deg, look_el_deg,
        freq_hz=freq_hz,
        spacing_m=spacing_m,
        cal_offsets_deg=cal_offsets_deg,
        include_pattern=True,
    ).reshape(-1, 1)                              # shape (4, 1)

    # Cholesky solve:  R_dl @ z = a_look  →  z = R_dl^{-1} a_look
    # cho_factor / cho_solve is faster and more numerically stable than np.linalg.inv
    c, low = cho_factor(R_dl)
    z = cho_solve((c, low), a_look)              # shape (4, 1)

    # Distortionless normalisation:  w = z / (a_look^H z)
    denom = (a_look.conj().T @ z).item()         # complex scalar
    w = z / (denom + 1e-12)                      # shape (4, 1)

    return w.ravel().astype(np.complex128)       # shape (4,)


# ---------------------------------------------------------------------------
# Beam pattern (for visualisation)
# ---------------------------------------------------------------------------

def beam_pattern(
    w: np.ndarray,
    az_grid: np.ndarray,
    el_deg: float = 0.0,
    freq_hz: float = GPS_L1_HZ,
    spacing_m: float = GPS_L1_D,
    cal_offsets_deg: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the array gain pattern for a given weight vector.

    G(θ) = 20 log10 |w^H a(θ)|   [dB]

    Normalised so the maximum gain = 0 dB.

    Parameters
    ----------
    w             : ndarray (4,) MVDR weight vector from mvdr_weights()
    az_grid       : 1D array of azimuth angles to evaluate (degrees)
    el_deg        : fixed elevation for the pattern scan (default 0 = horizon)
    freq_hz       : carrier frequency
    spacing_m     : element spacing
    cal_offsets_deg : per-channel cal offsets

    Returns
    -------
    pattern_db : ndarray shape (len(az_grid),)
                 Gain in dB, normalised to 0 dB at maximum.
    """
    gain = np.zeros(len(az_grid))
    for i, az in enumerate(az_grid):
        a = steering_vector(
            az, el_deg,
            freq_hz=freq_hz,
            spacing_m=spacing_m,
            cal_offsets_deg=cal_offsets_deg,
            include_pattern=False,               # pattern excluded for beam scan
        )
        gain[i] = abs(w.conj() @ a)

    gain_db = 20 * np.log10(gain + 1e-15)
    return gain_db - gain_db.max()               # normalise to 0 dB
