"""
tests/test_synthetic.py

End-to-end pytest validation of the DSP chain on synthetic IQ data.
No hardware required. All tests must pass before any hardware is connected.

Test coverage:
  T01 — geometry: element 0 phase is zero, cal offset applied correctly
  T02 — geometry: zenith look direction has equal phases across all elements
  T03 — geometry: steering table matches individual steering_vector calls
  T04 — covariance: DC removal — covariance invariant to large DC offset
  T05 — covariance: R is Hermitian and positive semi-definite
  T06 — covariance: diagonal loading makes R positive definite (Cholesky-safe)
  T07 — MUSIC: single jammer bearing within ±3° of truth
  T08 — MUSIC: two jammers both within ±3° of truth
  T09 — MUSIC: threshold gate returns None on noise-only input
  T10 — MUSIC: threshold gate passes through a real jammer
  T11 — MVDR: distortionless constraint |w^H a_look - 1| < 1e-6
  T12 — MVDR: null depth at jammer >= 30 dB below look direction
  T13 — MVDR: two jammers both nulled >= 25 dB
  T14 — beamform: passband gain at look direction ≈ 0 dB (within ±1 dB)
  T15 — beamform: suppression_db > 15 dB for a 30 dB JNR jammer
  T16 — end-to-end: full chain (covariance → MUSIC → MVDR → beamform)
        jammer at known angle, verify bearing + null + passband all pass
"""

import pytest
import numpy as np
from scipy.linalg import eigh, cho_factor

from dsp.geometry import (
    steering_vector, build_steering_table, build_az_grid,
    make_element_positions, load_cal, save_cal, GPS_L1_HZ, GPS_L1_D,
)
from dsp.covariance import compute_covariance, diagonal_load, process
from dsp.music import music_doa, music_spectrum, find_top_peaks
from dsp.mvdr import mvdr_weights, beam_pattern
from dsp.beamform import apply_weights, null_depth_db, passband_gain_db, suppression_db


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(1234)

# Precompute scan grid and steering table once for all tests
AZ_GRID = build_az_grid(-180, 180, 0.5)
STEER_TABLE = build_steering_table(AZ_GRID)


def _make_scene(
    jammer_azimuths_deg,
    jnr_db=30,
    n_samples=512,
    add_dc=False,
):
    """
    Synthetic 4-channel IQ scene.

    GPS is included at zenith at -25 dB relative to noise (below noise floor,
    as in reality). Jammers are at low elevation (5°) at specified azimuths.

    Parameters
    ----------
    jammer_azimuths_deg : list of float — jammer azimuths
    jnr_db              : jammer-to-noise ratio in dB
    n_samples           : IQ samples per channel
    add_dc              : if True, inject a large DC offset on every channel
    """
    n_el = 4
    noise_amp = 1.0
    jammer_amp = noise_amp * 10 ** (jnr_db / 20)
    gps_amp = noise_amp * 10 ** (-25 / 20)       # GPS well below noise floor

    noise = noise_amp * (
        RNG.standard_normal((n_el, n_samples))
        + 1j * RNG.standard_normal((n_el, n_samples))
    ) / np.sqrt(2)

    X = noise.copy()

    # GPS at zenith
    sv_gps = steering_vector(0.0, 90.0, include_pattern=True)
    s_gps = gps_amp * (
        RNG.standard_normal(n_samples) + 1j * RNG.standard_normal(n_samples)
    ) / np.sqrt(2)
    X += np.outer(sv_gps, s_gps)

    # Jammers at low elevation
    for az in jammer_azimuths_deg:
        sv = steering_vector(az, 5.0, include_pattern=False)
        s = jammer_amp * (
            RNG.standard_normal(n_samples) + 1j * RNG.standard_normal(n_samples)
        ) / np.sqrt(2)
        X += np.outer(sv, s)

    if add_dc:
        dc = np.array([500 + 200j, -300 + 100j, 400 - 150j, -250 - 350j])
        X += dc.reshape(4, 1)

    return X


# ---------------------------------------------------------------------------
# T01-T03: Geometry
# ---------------------------------------------------------------------------

def test_T01_element_zero_phase():
    """Element 0 is at origin — its steering phase must always be zero."""
    for az in [-170, -90, 0, 45, 90, 135]:
        a = steering_vector(az, 0.0, include_pattern=False)
        assert abs(np.angle(a[0])) < 1e-10, (
            f"Element 0 phase nonzero at az={az}: {np.angle(a[0]):.2e} rad"
        )


def test_T02_zenith_equal_phases():
    """At zenith (el=90°) the unit vector is [0,0,1]. All elements lie in z=0
    plane, so r_m · u = 0 for all m → all phases equal."""
    a = steering_vector(0.0, 90.0, include_pattern=True)
    assert np.allclose(np.angle(a), 0.0, atol=1e-10), (
        f"Zenith phases not equal: {np.angle(a)}"
    )
    assert abs(abs(a[0]) - 1.0) < 1e-10, (
        f"Zenith element pattern should be sin(90)=1, got {abs(a[0]):.6f}"
    )


def test_T03_steering_table_matches_individual():
    """build_steering_table must produce identical vectors to steering_vector."""
    az_grid = build_az_grid(-90, 90, 5.0)
    table = build_steering_table(az_grid, include_pattern=False)
    for i, az in enumerate(az_grid):
        sv = steering_vector(az, 0.0, include_pattern=False)
        assert np.allclose(table[i], sv, atol=1e-12), (
            f"Table mismatch at az={az:.1f}°"
        )


# ---------------------------------------------------------------------------
# T04-T06: Covariance
# ---------------------------------------------------------------------------

def test_T04_dc_removal():
    """Covariance must be invariant to a large DC offset on every channel."""
    X_clean = _make_scene([45.0])
    X_dc = _make_scene([45.0], add_dc=True)
    R_clean = compute_covariance(X_clean)
    R_dc = compute_covariance(X_dc)
    # Allow small tolerance due to different RNG draws in each call
    # Instead: use same data, just add DC
    X = _make_scene([30.0])
    dc = np.array([800 + 300j, -500 + 200j, 600 - 400j, -300 - 700j])
    X_with_dc = X + dc.reshape(4, 1)
    R1 = compute_covariance(X)
    R2 = compute_covariance(X_with_dc)
    assert np.allclose(R1, R2, atol=1e-10), (
        "DC removal failed: covariance changed with DC offset"
    )


def test_T05_covariance_hermitian_psd():
    """Sample covariance must be Hermitian and positive semi-definite."""
    X = _make_scene([60.0])
    R = compute_covariance(X)
    assert R.shape == (4, 4)
    assert R.dtype == np.complex128
    assert np.allclose(R, R.conj().T, atol=1e-12), "R is not Hermitian"
    eigvals = eigh(R, eigvals_only=True)
    assert np.all(eigvals >= -1e-12), f"R has negative eigenvalues: {eigvals}"


def test_T06_diagonal_loading_positive_definite():
    """After diagonal loading, R must be positive definite (Cholesky succeeds)."""
    X = _make_scene([45.0, -45.0])
    R = compute_covariance(X)
    R_dl = diagonal_load(R, n_signals=2)
    # Will raise LinAlgError if not positive definite
    cho_factor(R_dl)


# ---------------------------------------------------------------------------
# T07-T10: MUSIC
# ---------------------------------------------------------------------------

def test_T07_music_single_jammer():
    """MUSIC must estimate a single jammer bearing within ±3°."""
    true_az = 65.0
    X = _make_scene([true_az], jnr_db=25)
    R = compute_covariance(X)
    spec, peaks = music_doa(R, STEER_TABLE, AZ_GRID, n_signals=1)
    assert peaks is not None, "MUSIC returned None for strong jammer"
    err = abs(peaks[0] - true_az)
    assert err < 3.0, f"Bearing error {err:.2f}° exceeds ±3° for az={true_az}°"


@pytest.mark.parametrize("j1, j2", [
    (-60.0,  30.0),
    ( 10.0, 130.0),
    (-120.0, 45.0),
])
def test_T08_music_two_jammers(j1, j2):
    """MUSIC must resolve two jammers, both within ±3°."""
    X = _make_scene([j1, j2], jnr_db=25)
    R = compute_covariance(X)
    spec, peaks = music_doa(R, STEER_TABLE, AZ_GRID, n_signals=2, min_sep_deg=15)
    assert peaks is not None and len(peaks) == 2, (
        f"Expected 2 peaks, got {peaks}"
    )
    peaks_sorted = np.sort(peaks)
    for true_az, est_az in zip(sorted([j1, j2]), peaks_sorted):
        err = abs(est_az - true_az)
        assert err < 3.0, (
            f"Bearing error {err:.2f}° at true az={true_az}°, estimated {est_az:.2f}°"
        )


def test_T09_music_gate_noise_only():
    """Threshold gate must return None when only noise is present."""
    X = (
        RNG.standard_normal((4, 512)) + 1j * RNG.standard_normal((4, 512))
    ) / np.sqrt(2)
    R = compute_covariance(X)
    spec, peaks = music_doa(R, STEER_TABLE, AZ_GRID, n_signals=1, threshold_db=10.0)
    assert peaks is None, (
        f"Noise-only input should be gated out, got peaks={peaks}"
    )


def test_T10_music_gate_jammer_passes():
    """Threshold gate must NOT suppress a strong jammer."""
    X = _make_scene([120.0], jnr_db=25)
    R = compute_covariance(X)
    spec, peaks = music_doa(R, STEER_TABLE, AZ_GRID, n_signals=1, threshold_db=10.0)
    assert peaks is not None, "Strong jammer was incorrectly gated out"


# ---------------------------------------------------------------------------
# T11-T13: MVDR
# ---------------------------------------------------------------------------

def test_T11_mvdr_distortionless():
    """MVDR distortionless constraint: |w^H a_look - 1| < 1e-6."""
    X = _make_scene([45.0])
    R = compute_covariance(X)
    R_dl = diagonal_load(R, n_signals=1)
    w = mvdr_weights(R_dl, look_az_deg=0.0, look_el_deg=90.0)
    a_look = steering_vector(0.0, 90.0, include_pattern=True)
    err = abs(w.conj() @ a_look - 1.0)
    assert err < 1e-6, f"Distortionless constraint violated: |w^H a - 1| = {err:.2e}"


@pytest.mark.parametrize("jammer_az", [-135, -90, -45, 0, 45, 90, 135])
def test_T12_mvdr_null_depth_single(jammer_az):
    """Single jammer null depth must be >= 30 dB at all azimuths."""
    X = _make_scene([float(jammer_az)], jnr_db=30)
    R = compute_covariance(X)
    R_dl = diagonal_load(R, n_signals=1)
    w = mvdr_weights(R_dl)
    nd = null_depth_db(w, jammer_az_deg=float(jammer_az))
    assert nd < -30.0, (
        f"Null depth {nd:.1f} dB is too shallow at az={jammer_az}° (need < -30 dB)"
    )


def test_T13_mvdr_two_jammers():
    """Two jammers: both must be nulled >= 25 dB."""
    j1, j2 = -50.0, 70.0
    X = _make_scene([j1, j2], jnr_db=30)
    R = compute_covariance(X)
    R_dl = diagonal_load(R, n_signals=2)
    w = mvdr_weights(R_dl)
    for az, label in [(j1, "jammer1"), (j2, "jammer2")]:
        nd = null_depth_db(w, jammer_az_deg=az)
        assert nd < -25.0, (
            f"{label} null depth {nd:.1f} dB too shallow (need < -25 dB)"
        )


# ---------------------------------------------------------------------------
# T14-T15: Beamform
# ---------------------------------------------------------------------------

def test_T14_passband_gain():
    """GPS look direction gain must be within ±1 dB of 0 dB."""
    X = _make_scene([45.0])
    R = compute_covariance(X)
    R_dl = diagonal_load(R, n_signals=1)
    w = mvdr_weights(R_dl)
    pg = passband_gain_db(w)
    assert abs(pg) < 1.0, f"Passband gain {pg:.3f} dB is outside ±1 dB"


def test_T15_suppression_metric():
    """suppression_db must be > 15 dB for a 30 dB JNR jammer."""
    X = _make_scene([45.0], jnr_db=30)
    R = compute_covariance(X)
    R_dl = diagonal_load(R, n_signals=1)
    w = mvdr_weights(R_dl)
    s = suppression_db(X, w)
    assert s > 15.0, f"Suppression {s:.1f} dB is too low (need > 15 dB)"


# ---------------------------------------------------------------------------
# T16: Full end-to-end chain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true_az, n_sig", [
    (45.0, 1),
    (-90.0, 1),
    (150.0, 1),
])
def test_T16_end_to_end(true_az, n_sig):
    """
    Full pipeline: raw IQ → covariance → MUSIC → MVDR → beamform.

    Asserts:
      - MUSIC bearing within ±3°
      - MVDR null depth >= 30 dB at estimated bearing
      - Passband gain within ±1 dB
      - Suppression > 10 dB (FFT-peak metric; null depth is the primary metric above)

    Note: suppression_db uses FFT peak on channel sum as reference.
    At edge azimuths (±90°) the channel sum has less coherent gain so the
    apparent suppression can be lower than the actual null depth. 10 dB is
    the floor here; T12 verifies ≥30 dB null depth more rigorously.
    """
    X = _make_scene([true_az], jnr_db=35)

    # Step 1: covariance
    R, R_dl = process(X, n_signals=n_sig)

    # Step 2: MUSIC
    spec, peaks = music_doa(R, STEER_TABLE, AZ_GRID, n_signals=n_sig)
    assert peaks is not None, f"MUSIC returned None for jammer at {true_az}°"
    bearing_err = abs(peaks[0] - true_az)
    assert bearing_err < 3.0, (
        f"Bearing error {bearing_err:.2f}° at true_az={true_az}°"
    )

    # Step 3: MVDR
    w = mvdr_weights(R_dl)
    nd = null_depth_db(w, jammer_az_deg=peaks[0])
    assert nd < -30.0, f"Null depth {nd:.1f} dB too shallow"

    # Step 4: beamform
    pg = passband_gain_db(w)
    assert abs(pg) < 1.0, f"Passband gain {pg:.3f} dB outside ±1 dB"

    s = suppression_db(X, w)
    assert s > 10.0, f"Suppression {s:.1f} dB too low"
