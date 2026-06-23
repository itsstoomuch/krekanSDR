"""
iq_source/synthetic_source.py

Synthetic 4-channel IQ generator.
Supports 1–2 jammers with configurable type (CW / FMCW / BARRAGE),
azimuth, and JNR — all updatable at runtime via set_jammer_configs().
"""

import threading
import time
import numpy as np
from dsp.geometry import steering_vector

# ── Shared state — written here, read by DSP engine and dashboard ──────────
shared_data = {
    "rx_buffer":          None,   # (4, N) complex64
    "doa_est":            None,
    "beam_weights":       None,
    "beam_pattern":       None,
    "music_spectrum":     None,
    "spectrum_before":    None,
    "spectrum_after":     None,
    "freq_axis":          None,
    "suppression_db":     0.0,
    "null_depth_db":      0.0,
    "latency_ms":         0.0,
    "jammer_az":          45.0,
    "running":            True,
    "beamforming_active": True,
    "source_type":        "synthetic",
}

_lock = threading.Lock()

# ── Jammer config schema ───────────────────────────────────────────────────
# Each jammer is a dict:
#   az_deg  : float  — azimuth angle (-180..+180)
#   el_deg  : float  — elevation angle (0 = horizon, 90 = zenith)
#   jnr_db  : float  — jammer-to-noise ratio in dB
#   type    : str    — "CW" | "FMCW" | "BARRAGE"

DEFAULT_JAMMER_CONFIGS = [
    {"az_deg": 45.0, "el_deg": 5.0, "jnr_db": 30.0, "type": "CW"},
]


class SyntheticSource:
    """
    Generates synthetic 4-channel IQ frames in a daemon thread.

    Parameters
    ----------
    n_samples       : IQ samples per frame
    sample_rate     : simulated sample rate (Hz)
    center_freq     : simulated centre frequency (Hz)
    jnr_db          : backward-compat: sets JNR for the first jammer
    sweep_rate      : backward-compat: if > 0, sweeps jammer 0 azimuth (°/s)
    update_hz       : frames generated per second
    jammer_configs  : list of jammer dicts (overrides jnr_db/sweep_rate)
    """

    def __init__(
        self,
        n_samples:      int   = 256,
        sample_rate:    float = 2.4e6,
        center_freq:    float = 1575.42e6,
        jnr_db:         float = 30.0,
        sweep_rate:     float = 8.0,
        update_hz:      float = 20.0,
        jammer_configs: list  = None,
    ):
        self.n_samples = n_samples
        self.fs        = sample_rate
        self.fc        = center_freq
        self.update_hz = update_hz
        self._rng      = np.random.default_rng(42)
        self._thread   = None
        self._running  = False
        self._cfg_lock = threading.Lock()

        if jammer_configs is not None:
            self._configs    = [dict(c) for c in jammer_configs]
            self._sweep_rate = 0.0
        else:
            self._configs    = [{"az_deg": 45.0, "el_deg": 5.0,
                                  "jnr_db": jnr_db, "type": "CW"}]
            self._sweep_rate = sweep_rate

    # ── Public API ────────────────────────────────────────────────────────

    def set_jammer_configs(self, configs: list):
        """Replace jammer configuration at runtime (thread-safe)."""
        with self._cfg_lock:
            self._configs    = [dict(c) for c in configs]
            self._sweep_rate = 0.0   # disable sweep when explicitly set

    def get_jammer_configs(self) -> list:
        with self._cfg_lock:
            return [dict(c) for c in self._configs]

    # ── Signal generation ─────────────────────────────────────────────────

    def _make_jammer_signal(self, jcfg: dict, N: int) -> np.ndarray:
        """Generate one baseband jammer waveform (length N, complex128)."""
        jtype  = jcfg.get("type", "CW").upper()
        amp    = 10 ** (jcfg.get("jnr_db", 30.0) / 20.0)
        t      = np.arange(N) / self.fs

        if jtype == "CW":
            # Narrow-band tone — 50 kHz offset so it shows as a spike in the spectrum
            f_off = 50e3
            return amp * np.exp(2j * np.pi * f_off * t)

        elif jtype == "FMCW":
            # Linear chirp sweeping ±100 kHz within each frame
            BW  = 200e3
            T   = N / self.fs
            phi = 2 * np.pi * (-BW / 2 * t + (BW / (2 * T)) * t ** 2)
            return amp * np.exp(1j * phi)

        elif jtype == "BARRAGE":
            # Full-band noise — fills the whole receive bandwidth
            return amp * (
                self._rng.standard_normal(N)
                + 1j * self._rng.standard_normal(N)
            ) / np.sqrt(2)

        else:
            raise ValueError(f"Unknown jammer type '{jtype}' — use CW, FMCW, or BARRAGE")

    def _generate_frame(self) -> np.ndarray:
        N         = self.n_samples
        noise_amp = 1.0
        gps_amp   = noise_amp * 10 ** (-25 / 20)

        # Thermal noise on all 4 channels
        X = noise_amp * (
            self._rng.standard_normal((4, N))
            + 1j * self._rng.standard_normal((4, N))
        ) / np.sqrt(2)

        # GPS signal at zenith — well below noise floor; MUSIC never sees it
        sv_gps = steering_vector(0.0, 90.0, include_pattern=True)
        s_gps  = gps_amp * (
            self._rng.standard_normal(N) + 1j * self._rng.standard_normal(N)
        ) / np.sqrt(2)
        X += np.outer(sv_gps, s_gps)

        # Jammers
        with self._cfg_lock:
            cfgs = [dict(c) for c in self._configs]

        for jcfg in cfgs:
            sv = steering_vector(
                jcfg["az_deg"], jcfg.get("el_deg", 5.0),
                include_pattern=False,
            )
            s  = self._make_jammer_signal(jcfg, N)
            X += np.outer(sv, s)

        return X.astype(np.complex64)

    # ── Thread loop ───────────────────────────────────────────────────────

    def _run(self):
        interval = 1.0 / self.update_hz
        az_step  = self._sweep_rate / self.update_hz   # deg per frame

        while shared_data["running"] and self._running:
            t0    = time.perf_counter()
            frame = self._generate_frame()

            with _lock:
                shared_data["rx_buffer"] = frame
                with self._cfg_lock:
                    if self._configs:
                        shared_data["jammer_az"] = self._configs[0]["az_deg"]

            # Backward-compat sweep mode
            if self._sweep_rate > 0.0:
                with self._cfg_lock:
                    if self._configs:
                        self._configs[0]["az_deg"] += az_step
                        if self._configs[0]["az_deg"] > 180.0:
                            self._configs[0]["az_deg"] = -180.0

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True,
                                         name="synthetic-source")
        self._thread.start()

    def stop_source(self):
        """Stop this source thread only — does not shut down the DSP pipeline."""
        self._running = False

    def stop(self):
        """Stop source AND signal the entire pipeline to exit."""
        self._running = False
        shared_data["running"] = False


def get_lock():
    return _lock
