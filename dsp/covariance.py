"""
dsp/covariance.py

Sample covariance matrix estimation for the 4-element URA.

Two-step process:
  1. compute_covariance(X)  →  R̂ = (1/N) · X · X^H   (4×4 Hermitian)
  2. diagonal_load(R, K)    →  R_dl = R̂ + δI

DC Removal (critical for real RTL-SDR hardware):
  RTL-SDR and most direct-conversion receivers have a DC spike at 0 Hz
  (center frequency). Without removal, the DC component dominates the largest
  eigenvalue and the "signal subspace" points at DC, not the jammer.
  Fix: subtract the per-channel time-average before computing R.
      X_centered = X - mean(X, axis=1, keepdims=True)

Diagonal Loading:
  With 4 elements and K jammers, the covariance matrix is rank K in the signal
  subspace. When K approaches N-1 (e.g., 3 jammers, 4 elements), R becomes
  near-singular and direct inversion is numerically unstable.
  Solution: add a small scaled identity matrix before inverting.

  Loading formula (from config, diag_load_factor = 10):
      σ̂² = mean of the (N - K) smallest eigenvalues  (noise floor estimate)
      δ   = diag_load_factor × σ̂²
      R_dl = R̂ + δ · I

  Using the mean of the noise eigenvalues (not λ_min or a fixed constant)
  makes the loading level adapt to the actual noise floor — it tightens
  when the array is quiet and relaxes when the noise floor rises.
"""

import numpy as np
from scipy.linalg import eigh


def compute_covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix with DC removal.

    Steps:
      1. Remove DC (per-channel mean) to eliminate RTL-SDR center-frequency spike
      2. R̂ = (1/N) · X_c · X_c^H

    Parameters
    ----------
    X : ndarray shape (n_elements, n_samples), dtype complex
        Raw IQ snapshot matrix from heimdall_source or file_source.
        Rows = antenna channels, columns = time samples.

    Returns
    -------
    R : ndarray shape (n_elements, n_elements), dtype complex128
        Sample covariance matrix. Hermitian, positive semi-definite.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_elements, n_samples), got shape {X.shape}")

    n_elements, n_samples = X.shape

    # DC removal — subtract per-channel time average
    # This is the single most important pre-processing step for RTL-SDR hardware.
    # Without it, a DC spike at center frequency occupies the dominant eigenvalue.
    X_c = X - X.mean(axis=1, keepdims=True)

    # Sample covariance matrix
    # R[m,k] = (1/N) * sum_t  x_m(t) * conj(x_k(t))
    R = (X_c @ X_c.conj().T) / n_samples

    return R.astype(np.complex128)


def diagonal_load(R: np.ndarray, n_signals: int, factor: float = 10.0) -> np.ndarray:
    """
    Apply adaptive diagonal loading to the covariance matrix.

    Loading level:
        σ̂² = mean of the (N - n_signals) smallest eigenvalues
        δ   = factor × σ̂²
        R_dl = R + δ · I

    The noise-subspace mean (not λ_min) is used because a single small
    eigenvalue can be an outlier. The mean gives a more stable noise floor
    estimate across varying SNR conditions.

    Parameters
    ----------
    R        : ndarray shape (N, N), Hermitian covariance matrix
    n_signals: number of signal sources (jammers). Noise subspace has N-n_signals dims.
    factor   : loading multiplier (default 10, from config diag_load_factor)

    Returns
    -------
    R_dl : ndarray shape (N, N), dtype complex128
           Diagonally loaded covariance matrix, safe to invert via Cholesky.
    """
    n = R.shape[0]

    # Eigendecompose to find noise floor (eigh: Hermitian, ascending order)
    eigvals = eigh(R, eigvals_only=True)          # shape (N,), ascending

    # Noise subspace eigenvalues are the (N - n_signals) smallest
    n_noise = n - n_signals
    if n_noise < 1:
        n_noise = 1

    noise_floor = eigvals[:n_noise].mean()        # σ̂²
    delta = factor * max(noise_floor, 0.0)        # clamp to non-negative

    R_dl = R + delta * np.eye(n, dtype=np.complex128)
    return R_dl


def process(
    X: np.ndarray,
    n_signals: int = 1,
    diag_load_factor: float = 10.0,
) -> tuple:
    """
    Full covariance pipeline: DC removal → sample covariance → diagonal loading.

    Convenience wrapper used by pipeline/realtime.py and pipeline/offline.py.

    Parameters
    ----------
    X               : ndarray shape (4, N), raw IQ snapshots
    n_signals       : number of jammers expected
    diag_load_factor: loading multiplier from config

    Returns
    -------
    R    : ndarray (4, 4) — raw sample covariance (after DC removal)
    R_dl : ndarray (4, 4) — diagonally loaded covariance (use for MVDR inversion)
    """
    R = compute_covariance(X)
    R_dl = diagonal_load(R, n_signals, diag_load_factor)
    return R, R_dl
