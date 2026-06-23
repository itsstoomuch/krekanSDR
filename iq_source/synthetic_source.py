"""
iq_source/synthetic_source.py

Synthetic 4-channel IQ generator that mimics the Heimdall hardware source.
Runs in a background daemon thread, writes to shared_data["rx_buffer"].

The jammer slowly sweeps azimuth so the dashboard panels animate visibly.
Replace this source with heimdall_source.py when KrakenSDR is connected —
the shared_data interface is identical.
"""

import threading
import time
import numpy as np
from dsp.geometry import steering_vector

# Shared state — same dict used by realtime.py and dashboard
shared_data = {
    "rx_buffer":          None,   # (4, N) complex64 — latest IQ frame
    "doa_est":            None,   # float degrees — jammer bearing from MUSIC
    "beam_weights":       None,   # (4,) complex — MVDR weights
    "beam_pattern":       None,   # (n_angles,) dB — beam pattern vs azimuth
    "music_spectrum":     None,   # (n_angles,) dB — MUSIC pseudospectrum
    "spectrum_before":    None,   # (n_fft,) dB — raw channel sum spectrum
    "spectrum_after":     None,   # (n_fft,) dB — beamformed spectrum
    "freq_axis":          None,   # (n_fft,) MHz — FFT frequency axis
    "suppression_db":     0.0,    # scalar — current suppression
    "null_depth_db":      0.0,    # scalar — current null depth
    "latency_ms":         0.0,    # scalar — DSP cycle time
    "jammer_az":          45.0,   # scalar — true jammer az (synthetic only)
    "running":            True,
    "beamforming_active": True,
}

_lock = threading.Lock()


class SyntheticSource:
    """
    Generates synthetic 4-channel IQ frames continuously.

    Parameters
    ----------
    n_samples   : IQ samples per frame (matches n_snapshots in config)
    sample_rate : simulated sample rate (Hz)
    center_freq : simulated center frequency (Hz)
    jnr_db      : jammer-to-noise ratio (dB) — 30 dB is a strong jammer
    sweep_rate  : jammer azimuth sweep speed (degrees per second)
    update_hz   : frames generated per second
    """

    def __init__(
        self,
        n_samples:   int   = 256,
        sample_rate: float = 2.4e6,
        center_freq: float = 1575.42e6,
        jnr_db:      float = 30.0,
        sweep_rate:  float = 8.0,       # deg/s — complete sweep in ~45s
        update_hz:   float = 20.0,      # 20 frames/s
    ):
        self.n_samples   = n_samples
        self.fs          = sample_rate
        self.fc          = center_freq
        self.jnr_db      = jnr_db
        self.sweep_rate  = sweep_rate
        self.update_hz   = update_hz
        self._rng        = np.random.default_rng(42)
        self._jammer_az  = 45.0          # starting azimuth
        self._thread     = None

    def _generate_frame(self) -> np.ndarray:
        """Generate one (4, N) complex IQ frame with jammer + noise."""
        N = self.n_samples
        noise_amp  = 1.0
        jammer_amp = noise_amp * 10 ** (self.jnr_db / 20)
        gps_amp    = noise_amp * 10 ** (-25 / 20)   # GPS well below noise

        # Noise
        X = noise_amp * (
            self._rng.standard_normal((4, N))
            + 1j * self._rng.standard_normal((4, N))
        ) / np.sqrt(2)

        # GPS at zenith (below noise floor — present but invisible to MUSIC)
        sv_gps = steering_vector(0.0, 90.0, include_pattern=True)
        s_gps  = gps_amp * (
            self._rng.standard_normal(N) + 1j * self._rng.standard_normal(N)
        ) / np.sqrt(2)
        X += np.outer(sv_gps, s_gps)

        # Jammer at current azimuth, low elevation
        sv_jam = steering_vector(self._jammer_az, 5.0, include_pattern=False)
        s_jam  = jammer_amp * (
            self._rng.standard_normal(N) + 1j * self._rng.standard_normal(N)
        ) / np.sqrt(2)
        X += np.outer(sv_jam, s_jam)

        return X.astype(np.complex64)

    def _run(self):
        interval = 1.0 / self.update_hz
        az_step  = self.sweep_rate / self.update_hz   # degrees per frame

        while shared_data["running"]:
            t0 = time.perf_counter()

            frame = self._generate_frame()

            with _lock:
                shared_data["rx_buffer"]  = frame
                shared_data["jammer_az"]  = self._jammer_az

            # Sweep azimuth: -180 → +180 → -180 (ping-pong)
            self._jammer_az += az_step
            if self._jammer_az > 180.0:
                self._jammer_az = -180.0

            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        shared_data["running"] = False


def get_lock():
    return _lock
