"""
dsp/beamform.py

Apply MVDR weight vector to the 4-channel IQ snapshot matrix to produce
a single combined output stream.

Also provides scalar metrics used by the dashboard and test suite:
  - null_depth_db()  : suppression at a specific azimuth
  - passband_gain_db(): gain at the look direction (should be ≈ 0 dB)
  - suppression_db() : peak power reduction before vs after beamforming
"""

import numpy as np
from dsp.geometry import steering_vector, GPS_L1_HZ, GPS_L1_D


# ---------------------------------------------------------------------------
# Core beamforming
# ---------------------------------------------------------------------------

def apply_weights(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Apply weight vector to IQ snapshot matrix.

    y(t) = w^H x(t)   for each time sample t

    Parameters
    ----------
    w : ndarray shape (4,), dtype complex  — MVDR weight vector
    X : ndarray shape (4, N), dtype complex — IQ snapshot matrix

    Returns
    -------
    y : ndarray shape (N,), dtype complex128 — beamformed single-channel output
    """
    if w.shape[0] != X.shape[0]:
        raise ValueError(
            f"Weight length {w.shape[0]} does not match number of channels {X.shape[0]}"
        )
    return (w.conj() @ X).astype(np.complex128)   # shape (N,)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def null_depth_db(
    w: np.ndarray,
    jammer_az_deg: float,
    jammer_el_deg: float = 5.0,
    freq_hz: float = GPS_L1_HZ,
    spacing_m: float = GPS_L1_D,
    cal_offsets_deg: np.ndarray = None,
) -> float:
    """
    Compute array gain at the jammer direction (dB, relative to look direction).

    A deep null means a large negative value (e.g. -40 dB = 10000x suppression).

    Parameters
    ----------
    w             : MVDR weight vector shape (4,)
    jammer_az_deg : jammer azimuth estimated by MUSIC
    jammer_el_deg : jammer elevation (default 5° — low-elevation ground jammer)

    Returns
    -------
    null_db : float, gain at jammer direction relative to look direction peak (dB)
    """
    a_jam = steering_vector(
        jammer_az_deg, jammer_el_deg,
        freq_hz=freq_hz, spacing_m=spacing_m,
        cal_offsets_deg=cal_offsets_deg,
        include_pattern=False,
    )
    a_look = steering_vector(
        0.0, 90.0,
        freq_hz=freq_hz, spacing_m=spacing_m,
        cal_offsets_deg=cal_offsets_deg,
        include_pattern=True,
    )
    gain_jammer = abs(w.conj() @ a_jam)
    gain_look   = abs(w.conj() @ a_look) + 1e-15
    return 20 * np.log10(gain_jammer / gain_look)


def passband_gain_db(
    w: np.ndarray,
    look_az_deg: float = 0.0,
    look_el_deg: float = 90.0,
    freq_hz: float = GPS_L1_HZ,
    spacing_m: float = GPS_L1_D,
    cal_offsets_deg: np.ndarray = None,
) -> float:
    """
    Compute array gain at the GPS look direction (should be ≈ 0 dB).

    MVDR distortionless constraint enforces w^H a_look = 1, so the gain
    at the look direction should be exactly 1 (0 dB). This function
    verifies that constraint is satisfied after weight computation.

    Returns
    -------
    gain_db : float — gain at look direction in dB (ideal = 0.0)
    """
    a_look = steering_vector(
        look_az_deg, look_el_deg,
        freq_hz=freq_hz, spacing_m=spacing_m,
        cal_offsets_deg=cal_offsets_deg,
        include_pattern=True,
    )
    gain = abs(w.conj() @ a_look)
    return 20 * np.log10(gain + 1e-15)


def suppression_db(X_raw: np.ndarray, w: np.ndarray) -> float:
    """
    Measure jammer suppression: peak power before vs after beamforming.

    Uses the FFT peak as a proxy for jammer power. The difference in peak
    power (dBFS before minus dBFS after) is the suppression achieved.

    Parameters
    ----------
    X_raw : ndarray (4, N) — raw 4-channel IQ (before beamforming)
    w     : ndarray (4,)  — MVDR weight vector

    Returns
    -------
    delta_db : float — peak power reduction in dB (positive = suppression)
    """
    # Reference: sum of all channels (coherent combining without nulling)
    x_sum = X_raw.sum(axis=0)
    y_bf  = apply_weights(w, X_raw)

    def peak_power_db(x):
        sp = np.abs(np.fft.fft(x)) ** 2
        return 10 * np.log10(sp.max() + 1e-30)

    p_before = peak_power_db(x_sum)
    p_after  = peak_power_db(y_bf)
    return float(p_before - p_after)
