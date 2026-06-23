"""
dsp/geometry.py

2x2 URA steering vectors for GPS L1 anti-jam beamforming.

Array layout (top view, x = East, y = North):
    [2]=(0,d)   [3]=(d,d)
    [0]=(0,0)   [1]=(d,0)

Conventions (must match across all modules):
  - +j phase convention:  a[m] = exp(+j * 2pi/lam * r_m . u_hat)
  - Element order: (0,0), (1,0), (0,1), (1,1)  — row-major from origin
  - Element pattern: g(el) = sin(el),  el = elevation in radians
    (90 deg = zenith, 0 deg = horizon)
  - Spacing d = lambda/2 at GPS L1 = 0.09515 m (confirm against real CRPA)

Hardware calibration:
  Real KrakenSDR channels have per-channel phase offsets due to cable length
  differences and receiver mismatch. These are measured by calibrate.py and
  stored in cal.yaml as [0.0, dphi1, dphi2, dphi3] in degrees.
  The offset is SUBTRACTED from each element phase in the steering vector so
  the array model matches the physical hardware.
"""

import numpy as np
import yaml
import os

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C = 3e8                          # speed of light, m/s
GPS_L1_HZ = 1575.42e6            # GPS L1 carrier frequency, Hz
GPS_L1_LAM = C / GPS_L1_HZ      # GPS L1 wavelength ≈ 0.19029 m
GPS_L1_D = GPS_L1_LAM / 2       # half-wavelength spacing ≈ 0.09515 m


# ---------------------------------------------------------------------------
# Array element positions
# ---------------------------------------------------------------------------

def make_element_positions(spacing_m: float = GPS_L1_D) -> np.ndarray:
    """
    Return (4, 3) array of element XYZ positions for a 2x2 URA.

    Layout:
        [2]=(0,d,0)   [3]=(d,d,0)
        [0]=(0,0,0)   [1]=(d,0,0)

    All elements lie in the Z=0 plane (array broadside points up, +Z).

    Parameters
    ----------
    spacing_m : element spacing in metres (default lambda/2 at GPS L1)

    Returns
    -------
    pos : ndarray shape (4, 3), dtype float64
    """
    d = spacing_m
    return np.array([
        [0, 0, 0],   # element 0 — origin
        [d, 0, 0],   # element 1 — +X
        [0, d, 0],   # element 2 — +Y
        [d, d, 0],   # element 3 — +X +Y
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Element radiation pattern
# ---------------------------------------------------------------------------

def element_pattern(el_rad: float) -> float:
    """
    Amplitude gain of a single patch element vs elevation.

    GPS patch antennas have a sin(el) amplitude pattern:
        g(el) = sin(el)   for el in [0, pi/2]
        g(el) = 0         for el < 0 (below horizon, blocked by ground plane)

    Note: this differs from a cosine pattern used in some references because
    elevation here is measured from the horizon (0=horizon, 90=zenith),
    not from the zenith (0=zenith, 90=horizon).

    Parameters
    ----------
    el_rad : elevation angle in radians (0 = horizon, pi/2 = zenith)

    Returns
    -------
    g : scalar amplitude gain >= 0
    """
    return float(np.maximum(np.sin(el_rad), 0.0))


# ---------------------------------------------------------------------------
# Core steering vector
# ---------------------------------------------------------------------------

def steering_vector(
    az_deg: float,
    el_deg: float,
    freq_hz: float = GPS_L1_HZ,
    spacing_m: float = GPS_L1_D,
    cal_offsets_deg: np.ndarray = None,
    include_pattern: bool = True,
) -> np.ndarray:
    """
    4-element URA steering vector for a far-field source.

    Phase model (+j convention):
        a[m] = g(el) * exp(+j * 2*pi/lam * (r_m . u_hat))

    where:
        r_m    = position of element m (from make_element_positions)
        u_hat  = unit vector toward the source in 3D space

    Element pattern:
        g(el) = sin(el), with el=90 deg at zenith.
        For MUSIC DOA scanning (jammer at near-horizon), set include_pattern=False.
        The pattern is identical for all scan angles in a fixed-elevation scan so
        it only scales the pseudospectrum uniformly — peaks are unaffected.
        For MVDR look-direction constraint (GPS at zenith), always include it.

    Calibration:
        If cal_offsets_deg is provided, the measured hardware phase offset
        for each channel is subtracted from the steering vector phase:
            a[m] *= exp(-j * cal_offset_rad[m])
        This aligns the steering model with the physical hardware.

    Parameters
    ----------
    az_deg          : source azimuth in degrees (0=East, 90=North, CCW positive)
    el_deg          : source elevation in degrees (0=horizon, 90=zenith)
    freq_hz         : carrier frequency in Hz (default GPS L1)
    spacing_m       : element spacing in metres (default lambda/2 at GPS L1)
    cal_offsets_deg : array shape (4,) of per-channel phase offsets in degrees
                      (measured by calibrate.py, channel 0 is reference = 0.0)
    include_pattern : if True, multiply by element gain g(el).
                      Set False for MUSIC scan tables (faster, avoids zero gain
                      at horizon where sin(0)=0).

    Returns
    -------
    a : ndarray shape (4,), dtype complex128
    """
    lam = C / freq_hz
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    # Unit vector toward source in 3D (az measured from +X axis, CCW)
    u = np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad),
    ])

    elem_pos = make_element_positions(spacing_m)          # (4, 3)

    # Phase delay at each element: phi_m = (2*pi/lam) * (r_m . u)
    phases = (2 * np.pi / lam) * (elem_pos @ u)          # shape (4,)
    a = np.exp(1j * phases)                               # +j convention, no pattern yet

    if include_pattern:
        a *= element_pattern(el_rad)                      # scale by g(el)

    # Subtract hardware calibration offsets if provided
    if cal_offsets_deg is not None:
        cal_rad = np.deg2rad(np.asarray(cal_offsets_deg, dtype=np.float64))
        a *= np.exp(-1j * cal_rad)

    return a.astype(np.complex128)


# ---------------------------------------------------------------------------
# Calibration file I/O
# ---------------------------------------------------------------------------

def load_cal(cal_file: str) -> np.ndarray:
    """
    Load per-channel phase offsets from cal.yaml.

    Returns array shape (4,) in degrees.
    Channel 0 is always 0.0 (reference). Returns zeros if file missing.

    Parameters
    ----------
    cal_file : path to cal.yaml written by calibration/calibrate.py
    """
    if not os.path.exists(cal_file):
        return np.zeros(4, dtype=np.float64)
    with open(cal_file, "r") as f:
        data = yaml.safe_load(f)
    offsets = data.get("phase_offsets_deg", [0.0, 0.0, 0.0, 0.0])
    return np.array(offsets, dtype=np.float64)


def save_cal(offsets_deg: np.ndarray, cal_file: str) -> None:
    """
    Save per-channel phase offsets to cal.yaml.

    Parameters
    ----------
    offsets_deg : array shape (4,) of phase offsets in degrees
    cal_file    : output path
    """
    os.makedirs(os.path.dirname(cal_file) or ".", exist_ok=True)
    data = {
        "phase_offsets_deg": [float(x) for x in offsets_deg],
        "channel_ref": 0,
        "note": "Channel 0 is reference (offset = 0). Measured by calibrate.py.",
    }
    with open(cal_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Steering table (precomputed ROM for fast MUSIC scan)
# ---------------------------------------------------------------------------

def build_steering_table(
    az_grid: np.ndarray,
    el_deg: float = 0.0,
    freq_hz: float = GPS_L1_HZ,
    spacing_m: float = GPS_L1_D,
    cal_offsets_deg: np.ndarray = None,
    include_pattern: bool = False,
) -> np.ndarray:
    """
    Precompute steering vectors for all azimuth scan angles.

    Building the table once before the MUSIC scan loop is ~10x faster than
    calling steering_vector() inside the loop because array allocations are
    avoided per iteration.

    Parameters
    ----------
    az_grid         : 1D array of azimuth angles to scan (degrees)
    el_deg          : fixed elevation for the scan (default 0 = horizon scan)
    freq_hz         : carrier frequency
    spacing_m       : element spacing
    cal_offsets_deg : per-channel calibration offsets from load_cal()
    include_pattern : passed to steering_vector(). Default False for MUSIC scan
                      (element pattern at horizon el=0 is sin(0)=0 which kills
                      the vector — and it doesn't affect MUSIC peak locations anyway)

    Returns
    -------
    table : ndarray shape (len(az_grid), 4), dtype complex128
        table[i] = steering_vector(az_grid[i], el_deg, ...)
    """
    n = len(az_grid)
    table = np.zeros((n, 4), dtype=np.complex128)
    for i, az in enumerate(az_grid):
        table[i] = steering_vector(
            az, el_deg, freq_hz, spacing_m, cal_offsets_deg, include_pattern
        )
    return table


def build_az_grid(
    start_deg: float = -180.0,
    stop_deg: float = 180.0,
    step_deg: float = 0.5,
) -> np.ndarray:
    """
    Return azimuth scan grid in degrees.

    Parameters
    ----------
    start_deg : start of scan (inclusive)
    stop_deg  : end of scan (inclusive)
    step_deg  : angular resolution

    Returns
    -------
    az_grid : 1D ndarray of angles in degrees
    """
    return np.arange(start_deg, stop_deg + step_deg / 2, step_deg)
