"""
pipeline/realtime.py

Real-time DSP engine — runs in a background daemon thread.

Reads raw IQ from shared_data["rx_buffer"], runs the full chain:
    covariance → MUSIC → MVDR → beamform

Writes results back into shared_data so the dashboard can read them.

Works with any IQ source (synthetic_source, heimdall_source, file_source)
as long as they write (4, N) complex arrays to shared_data["rx_buffer"].
"""

import threading
import time
import numpy as np
import yaml

from dsp.covariance import process as cov_process
from dsp.geometry   import build_az_grid, build_steering_table, load_cal, GPS_L1_HZ, GPS_L1_D
from dsp.music      import music_doa
from dsp.mvdr       import mvdr_weights, beam_pattern
from dsp.beamform   import apply_weights, null_depth_db, passband_gain_db


def _load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _power_spectrum(x: np.ndarray, fs: float, fc_mhz: float):
    """Return (freq_mhz, power_db) for a 1D complex signal."""
    N   = min(len(x), 512)
    win = np.hamming(N)
    sp  = np.fft.fftshift(np.fft.fft(x[:N] * win))
    mag = np.abs(sp) / (win.sum() / 2 + 1e-12)
    db  = 20 * np.log10(mag + 1e-15)
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs)) / 1e6 + fc_mhz
    return freq, db


class RealtimeEngine:
    """
    DSP processing engine thread.

    Parameters
    ----------
    shared_data : dict — shared state written by IQ source, read by dashboard
    lock        : threading.Lock — protects shared_data["rx_buffer"] reads
    config_path : path to config.yaml
    """

    def __init__(self, shared_data: dict, lock: threading.Lock, config_path="config.yaml"):
        self.shared_data = shared_data
        self.lock        = lock
        self.cfg         = _load_config(config_path)
        self._thread     = None
        self._last_buf   = None   # avoid reprocessing the same frame

        # Pre-build scan assets from config
        dsp   = self.cfg["dsp"]
        rf    = self.cfg["rf"]
        array = self.cfg["array"]

        self.az_grid = build_az_grid(
            dsp["az_scan_start_deg"],
            dsp["az_scan_stop_deg"],
            dsp["az_scan_step_deg"],
        )

        cal_offsets = load_cal(self.cfg["calibration"]["cal_file"])
        self.steer_table = build_steering_table(
            self.az_grid,
            el_deg=0.0,
            freq_hz=rf["center_freq_hz"],
            spacing_m=array["spacing_m"],
            cal_offsets_deg=cal_offsets,
            include_pattern=False,
        )

        self.n_signals    = dsp["n_signals"]
        self.load_factor  = dsp["diag_load_factor"]
        self.threshold_db = dsp.get("music_threshold_db", 8.0)
        self.look_az      = self.cfg["look_direction"]["azimuth_deg"]
        self.look_el      = self.cfg["look_direction"]["elevation_deg"]
        self.fc_mhz       = rf["center_freq_hz"] / 1e6
        self.fs           = rf["sample_rate_hz"]
        self.freq_hz      = rf["center_freq_hz"]
        self.spacing_m    = array["spacing_m"]
        self.cal_offsets  = cal_offsets

    def _process(self, X: np.ndarray):
        """Run one full DSP cycle on a (4, N) IQ frame."""
        t0 = time.perf_counter()

        # 1. Covariance
        R, R_dl = cov_process(X, self.n_signals, self.load_factor)

        # 2. Spectrum before beamforming (channel sum — shows raw jammer)
        x_sum = X.sum(axis=0)
        freq, spec_before = _power_spectrum(x_sum, self.fs, self.fc_mhz)

        # 3. MUSIC DOA
        spec_music, peaks = music_doa(
            R, self.steer_table, self.az_grid,
            n_signals=self.n_signals,
            threshold_db=self.threshold_db,
        )

        if peaks is None or not self.shared_data.get("beamforming_active", True):
            # No jammer detected or AJ disabled — pass through raw sum
            _, spec_after = _power_spectrum(x_sum, self.fs, self.fc_mhz)
            latency = (time.perf_counter() - t0) * 1000
            self.shared_data.update({
                "music_spectrum":  spec_music if spec_music is not None
                                   else np.zeros(len(self.az_grid)),
                "beam_pattern":    np.zeros(len(self.az_grid)),
                "spectrum_before": spec_before,
                "spectrum_after":  spec_after,
                "freq_axis":       freq,
                "doa_est":         None,
                "beam_weights":    None,
                "suppression_db":  0.0,
                "null_depth_db":   0.0,
                "latency_ms":      latency,
            })
            return

        doa = float(peaks[0])

        # 4. MVDR weights
        w = mvdr_weights(
            R_dl,
            look_az_deg=self.look_az,
            look_el_deg=self.look_el,
            freq_hz=self.freq_hz,
            spacing_m=self.spacing_m,
            cal_offsets_deg=self.cal_offsets,
        )

        # 5. Beamform
        y_bf = apply_weights(w, X)
        _, spec_after = _power_spectrum(y_bf, self.fs, self.fc_mhz)

        # 6. Beam pattern for visualisation
        pat = beam_pattern(
            w, self.az_grid,
            el_deg=5.0,
            freq_hz=self.freq_hz,
            spacing_m=self.spacing_m,
            cal_offsets_deg=self.cal_offsets,
        )

        # 7. Metrics
        peak_idx    = np.argmax(spec_before)
        suppression = float(spec_before[peak_idx] - spec_after[peak_idx])
        nd          = null_depth_db(w, doa, jammer_el_deg=5.0,
                                    freq_hz=self.freq_hz,
                                    spacing_m=self.spacing_m,
                                    cal_offsets_deg=self.cal_offsets)
        latency = (time.perf_counter() - t0) * 1000

        self.shared_data.update({
            "music_spectrum":  spec_music,
            "beam_pattern":    pat,
            "spectrum_before": spec_before,
            "spectrum_after":  spec_after,
            "freq_axis":       freq,
            "doa_est":         doa,
            "beam_weights":    w,
            "suppression_db":  suppression,
            "null_depth_db":   nd,
            "latency_ms":      latency,
        })

    def _run(self):
        while self.shared_data["running"]:
            with self.lock:
                buf = self.shared_data.get("rx_buffer")

            if buf is None or buf is self._last_buf:
                time.sleep(0.01)
                continue

            self._last_buf = buf
            try:
                self._process(buf.astype(np.complex128))
            except Exception as e:
                import logging
                logging.error(f"DSP engine error: {e}")
                time.sleep(0.05)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="dsp-engine")
        self._thread.start()

    def stop(self):
        self.shared_data["running"] = False
