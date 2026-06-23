"""
dsp/music.py

MUSIC (MUltiple SIgnal Classification) direction-of-arrival estimation
for the 4-element 2×2 URA.

Algorithm:
  1. Eigendecompose R̂ = E Λ E^H  (eigh, ascending order)
  2. Partition: signal subspace = K largest eigenvectors
                noise subspace  = (N-K) smallest eigenvectors  → E_n
  3. Scan: P(θ) = 1 / (‖E_n^H · a(θ)‖² + ε)
  4. Find peaks in P(θ) → estimated jammer bearings

Key design choices:
  - Steering table is precomputed once (build_steering_table) and passed in.
    This avoids allocating steering vectors inside the hot scan loop.
  - Interference threshold gate: if the peak power of the raw spectrum is
    below the configured threshold, MUSIC is skipped and None is returned.
    This prevents phantom nulls on noise when no jammer is present.
  - Minimum peak separation enforces that K distinct sources are resolved,
    not K peaks of the same lobe.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# MUSIC pseudospectrum
# ---------------------------------------------------------------------------

def music_spectrum(
    R: np.ndarray,
    steering_table: np.ndarray,
    n_signals: int = 1,
) -> np.ndarray:
    """
    Compute the MUSIC pseudospectrum across the precomputed steering table.

    Parameters
    ----------
    R             : ndarray shape (N, N), sample covariance matrix (NOT loaded)
                    Use the raw R from compute_covariance(), not R_dl.
                    MUSIC eigendecomposition works on the unmodified R so the
                    signal/noise subspace split is clean.
    steering_table: ndarray shape (n_angles, N), precomputed from build_steering_table()
    n_signals     : number of signal sources (jammers) to resolve

    Returns
    -------
    spectrum_db : ndarray shape (n_angles,), MUSIC pseudospectrum in dB
                  Normalised so peak = 0 dB.
    """
    n = R.shape[0]
    n_noise = n - n_signals

    # Eigendecompose (eigh: Hermitian, ascending eigenvalues)
    _, eigvecs = eigh(R)                          # eigvecs[:, i] = i-th eigenvector

    # Noise subspace: (n_noise) eigenvectors with SMALLEST eigenvalues
    # eigh returns ascending order, so noise subspace = first n_noise columns
    E_n = eigvecs[:, :n_noise]                   # shape (N, n_noise)

    # Projection matrix for noise subspace: P_n = E_n @ E_n^H
    # Instead of forming P_n explicitly, compute the denominator as:
    #   denom[i] = a_i^H @ E_n @ E_n^H @ a_i = ‖E_n^H @ a_i‖²
    # Vectorised over all angles at once using the steering table.
    # steering_table shape: (n_angles, N)
    # E_n shape: (N, n_noise)
    # proj shape: (n_angles, n_noise)
    proj = steering_table @ E_n.conj()            # (n_angles, n_noise)
    denom = np.sum(np.abs(proj) ** 2, axis=1)    # (n_angles,)  = ‖E_n^H a_i‖²

    spectrum = 1.0 / (denom + 1e-12)             # pseudospectrum (linear)

    # Normalise to 0 dB at peak
    spectrum_db = 10 * np.log10(spectrum / spectrum.max())
    return spectrum_db


# ---------------------------------------------------------------------------
# Peak finder
# ---------------------------------------------------------------------------

def find_top_peaks(
    spectrum_db: np.ndarray,
    az_grid: np.ndarray,
    n_signals: int,
    min_sep_deg: float = 15.0,
) -> np.ndarray:
    """
    Return the azimuth angles of the top-N peaks in the MUSIC spectrum.

    Strategy:
      1. scipy.signal.find_peaks with minimum angular separation
      2. Sort by height descending
      3. Greedily pick peaks that are >= min_sep_deg apart
      4. If fewer than n_signals peaks found, pad with the global maximum

    Parameters
    ----------
    spectrum_db  : MUSIC pseudospectrum in dB (from music_spectrum)
    az_grid      : azimuth angles corresponding to spectrum_db entries
    n_signals    : number of peaks to return
    min_sep_deg  : minimum angular separation between peaks in degrees

    Returns
    -------
    peak_azimuths : ndarray shape (n_found,) of azimuth estimates in degrees
                    n_found <= n_signals
    """
    step_deg = float(az_grid[1] - az_grid[0]) if len(az_grid) > 1 else 1.0
    min_dist_samples = max(1, int(min_sep_deg / step_deg))

    all_idx, _ = find_peaks(spectrum_db, distance=min_dist_samples)

    if len(all_idx) == 0:
        # No peaks found — return global maximum
        return az_grid[np.array([np.argmax(spectrum_db)])]

    # Sort by peak height descending
    heights = spectrum_db[all_idx]
    sorted_idx = all_idx[np.argsort(heights)[::-1]]

    # Greedily select peaks separated by >= min_sep_deg
    chosen = []
    for idx in sorted_idx:
        az = az_grid[idx]
        if all(abs(az - az_grid[c]) >= min_sep_deg for c in chosen):
            chosen.append(idx)
        if len(chosen) == n_signals:
            break

    # Pad to n_signals if not enough peaks
    if len(chosen) == 0:
        chosen = [int(np.argmax(spectrum_db))]

    return az_grid[np.array(chosen)]


# ---------------------------------------------------------------------------
# Full MUSIC DOA pipeline
# ---------------------------------------------------------------------------

def music_doa(
    R: np.ndarray,
    steering_table: np.ndarray,
    az_grid: np.ndarray,
    n_signals: int = 1,
    threshold_db: float = None,
    min_sep_deg: float = 15.0,
) -> tuple:
    """
    Full MUSIC DOA estimation pipeline.

    Parameters
    ----------
    R             : ndarray (4,4) raw covariance from compute_covariance()
    steering_table: ndarray (n_angles, 4) from build_steering_table()
    az_grid       : ndarray (n_angles,) azimuth scan angles in degrees
    n_signals     : number of jammers to resolve
    threshold_db  : if the MUSIC peak is below this value (relative to noise floor),
                    return (None, None) — no jammer detected.
                    Pass None to disable the gate.
    min_sep_deg   : minimum angular separation between detected peaks

    Returns
    -------
    spectrum_db   : ndarray (n_angles,) MUSIC pseudospectrum in dB, or None if gated
    peak_azimuths : ndarray of estimated jammer azimuths in degrees, or None if gated
    """
    spectrum_db = music_spectrum(R, steering_table, n_signals)

    # Interference threshold gate
    # The spectrum is normalised so peak = 0 dB. We check the signal-to-noise
    # gap: peak (0 dB by construction) vs the mean of the bottom half of the
    # spectrum. If the gap is too small, there is no real jammer — just noise.
    if threshold_db is not None:
        noise_level = np.median(spectrum_db)          # robust noise floor estimate
        snr_gap = 0.0 - noise_level                   # peak is 0 dB
        if snr_gap < threshold_db:
            return None, None

    peak_azimuths = find_top_peaks(spectrum_db, az_grid, n_signals, min_sep_deg)
    return spectrum_db, peak_azimuths
