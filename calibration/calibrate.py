"""
calibration/calibrate.py — KrakenSDR Phase & Amplitude Calibration Tool

PROCEDURE
─────────
1. Connect your RF synthesiser output → 4-way power splitter → all 4 antenna ports.
2. Set the synthesiser to exactly 1575.42 MHz (GPS L1) at a medium power level.
   (Any CW signal works — just make sure it's strong enough to be above the noise.)
3. Run:
       python calibration/calibrate.py               # reads from synthetic (test)
       python calibration/calibrate.py --source heimdall            # real KrakenSDR
       python calibration/calibrate.py --source heimdall --host 192.168.1.10
4. Observe the live display.  When all 4 channels show a stable phase (green tick),
   click  [Save Calibration]  or press  S.

OUTPUT
──────
calibration/cal.yaml — phase offsets in degrees relative to channel 0.
Loaded automatically by dsp/geometry.py at runtime.

WHAT IT MEASURES
────────────────
Phase offset of channels 1-3 relative to channel 0, measured by cross-correlating
each channel with the reference.  A perfect calibration produces phase offsets that
are stable within ±2 ° across frames.
"""

import sys
import os
import argparse
import time
import threading
import logging

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Constants ──────────────────────────────────────────────────────────────
HISTORY_LEN        = 120    # frames of rolling phase history
STABILITY_FRAMES   = 20     # frames used for std-dev check
STABILITY_TOL_DEG  = 2.0    # ° — pass criterion
N_CHANNELS         = 4
CAL_FILE           = os.path.join(os.path.dirname(__file__), "cal.yaml")
UPDATE_MS          = 400    # display refresh interval

# ── Palette ────────────────────────────────────────────────────────────────
BG       = "#0d0d0d"
PANEL    = "#111111"
GRID     = "#1e1e1e"
SPINE    = "#2a2a2a"
TEXT     = "#c8c8c8"
TITLE    = "#e8e8e8"
CH_COLS  = ["#4d9de0", "#e0b84d", "#4caf76", "#d9534f"]   # ch 0-3
PASS_COL = "#4caf76"
FAIL_COL = "#d9534f"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   SPINE,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TITLE,
    "axes.grid":        True,
    "grid.color":       GRID,
    "grid.linewidth":   0.6,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "figure.facecolor": BG,
    "text.color":       TEXT,
})


# ── Phase measurement ──────────────────────────────────────────────────────

def measure_phase_offsets(X: np.ndarray) -> np.ndarray:
    """
    Cross-correlate channels 1-3 against channel 0.
    Returns (3,) array of phase offsets in degrees.

    X : (4, N) complex array — one IQ frame
    """
    ref = X[0]
    offsets = np.zeros(3)
    for i in range(3):
        # Peak of cross-correlation in frequency domain
        cross = ref * np.conj(X[i + 1])
        offsets[i] = np.angle(cross.sum(), deg=True)
    return offsets


def measure_amplitudes(X: np.ndarray) -> np.ndarray:
    """
    RMS amplitude of each channel, normalised to channel 0.
    Returns (4,) array.
    """
    rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=1))
    return rms / (rms[0] + 1e-12)


# ── Source setup ───────────────────────────────────────────────────────────

def _start_synthetic_source():
    from iq_source.synthetic_source import SyntheticSource, shared_data, get_lock
    src = SyntheticSource(
        jnr_db=30.0,
        sweep_rate=0.0,
        jammer_configs=[{"az_deg": 45.0, "el_deg": 5.0, "jnr_db": 30.0, "type": "CW"}],
    )
    src.start()
    return shared_data, get_lock(), src


def _start_heimdall_source(host, port):
    from iq_source.synthetic_source import shared_data, get_lock
    from iq_source.heimdall_source  import HeimdallSource
    src = HeimdallSource(host=host, port=port)
    src.start()
    return shared_data, get_lock(), src


# ── Calibration tool ────────────────────────────────────────────────────────

class CalibrationTool:
    def __init__(self, shared_data: dict, lock, source):
        self.sd       = shared_data
        self.lock     = lock
        self.source   = source
        self._running = True

        # Rolling history: (HISTORY_LEN, 3) — phase offsets of ch1,2,3 vs ch0
        self._phase_hist = np.full((HISTORY_LEN, 3), np.nan)
        self._amp_latest = np.ones(4)
        self._offsets    = np.zeros(3)   # latest measurement
        self._saved      = False

        self._build_figure()
        self._timer = self.fig.canvas.new_timer(interval=UPDATE_MS)
        self._timer.add_callback(self._update)
        self._timer.start()

    # ── Figure layout ──────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 7), facecolor=BG)
        self.fig.canvas.manager.set_window_title("KrakenSDR Calibration Tool")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            hspace=0.48, wspace=0.32,
            left=0.06, right=0.97,
            top=0.88, bottom=0.14,
        )

        # ── Phase offset bars (top-left) ──────────────────────────────────
        self.ax_bar = self.fig.add_subplot(gs[0, 0])
        self.bar_objs = self.ax_bar.bar(
            ["Ch 1", "Ch 2", "Ch 3"],
            [0, 0, 0],
            color=CH_COLS[1:],
            width=0.55,
        )
        self.ax_bar.axhline(0, color=SPINE, lw=1)
        self.ax_bar.set_ylim(-180, 180)
        self.ax_bar.set_ylabel("Phase offset (°)")
        self.ax_bar.set_title("Phase Offsets  (rel. Ch 0)")
        for bar, col in zip(self.bar_objs, CH_COLS[1:]):
            bar.set_color(col)

        # ── Phase stability history (top-centre) ─────────────────────────
        self.ax_hist = self.fig.add_subplot(gs[0, 1])
        self.lines_hist = []
        for i in range(3):
            ln, = self.ax_hist.plot([], [], color=CH_COLS[i + 1], lw=1.2,
                                    label=f"Ch {i + 1}")
            self.lines_hist.append(ln)
        self.ax_hist.set_ylim(-180, 180)
        self.ax_hist.set_xlim(0, HISTORY_LEN)
        self.ax_hist.set_xlabel("Frame")
        self.ax_hist.set_ylabel("Phase (°)")
        self.ax_hist.set_title("Phase History")
        self.ax_hist.legend(loc="upper right", fontsize=7.5)

        # ── Amplitude consistency (top-right) ────────────────────────────
        self.ax_amp = self.fig.add_subplot(gs[0, 2])
        self.amp_bars = self.ax_amp.bar(
            ["Ch 0", "Ch 1", "Ch 2", "Ch 3"],
            [1, 1, 1, 1],
            color=CH_COLS,
            width=0.55,
        )
        self.ax_amp.axhline(1.0, color="#444444", lw=0.8, ls="--")
        self.ax_amp.set_ylim(0, 2.0)
        self.ax_amp.set_ylabel("Amplitude (norm. to Ch 0)")
        self.ax_amp.set_title("Channel Amplitudes")

        # ── Per-channel IQ constellation (bottom row, 4 subplots) ─────────
        self.ax_iq = []
        for ch in range(4):
            ax = self.fig.add_subplot(gs[1, ch % 3 if ch < 3 else 2])
            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"Ch {ch}  IQ", fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
            sc = ax.scatter([], [], s=2, color=CH_COLS[ch], alpha=0.5)
            self.ax_iq.append((ax, sc))
        # Actually do 4 IQ plots in 1 row — reuse bottom row with 4 columns
        # Rebuild: 2 rows, 4 cols
        plt.close(self.fig)
        self.fig = plt.figure(figsize=(14, 7), facecolor=BG)
        self.fig.canvas.manager.set_window_title("KrakenSDR Calibration Tool")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        gs2 = gridspec.GridSpec(
            2, 4,
            figure=self.fig,
            hspace=0.50, wspace=0.30,
            left=0.05, right=0.97,
            top=0.88, bottom=0.12,
        )

        # Phase bars
        self.ax_bar = self.fig.add_subplot(gs2[0, 0])
        self.bar_objs = self.ax_bar.bar(
            ["Ch 1", "Ch 2", "Ch 3"], [0, 0, 0],
            color=CH_COLS[1:], width=0.55,
        )
        self.ax_bar.axhline(0, color=SPINE, lw=1)
        self.ax_bar.set_ylim(-180, 180)
        self.ax_bar.set_ylabel("Phase offset (°)")
        self.ax_bar.set_title("Phase Offsets  (rel. Ch 0)")

        # Phase history
        self.ax_hist = self.fig.add_subplot(gs2[0, 1:3])
        self.lines_hist = []
        for i in range(3):
            ln, = self.ax_hist.plot([], [], color=CH_COLS[i + 1], lw=1.2,
                                    label=f"Ch {i + 1}")
            self.lines_hist.append(ln)
        self.ax_hist.set_ylim(-180, 180)
        self.ax_hist.set_xlim(0, HISTORY_LEN)
        self.ax_hist.set_xlabel("Frame")
        self.ax_hist.set_ylabel("Phase (°)")
        self.ax_hist.set_title("Phase Stability History")
        self.ax_hist.legend(loc="upper right", fontsize=7.5)

        # Amplitude bars
        self.ax_amp = self.fig.add_subplot(gs2[0, 3])
        self.amp_bars = self.ax_amp.bar(
            ["0", "1", "2", "3"], [1, 1, 1, 1],
            color=CH_COLS, width=0.55,
        )
        self.ax_amp.axhline(1.0, color="#444444", lw=0.8, ls="--")
        self.ax_amp.set_ylim(0, 2.0)
        self.ax_amp.set_ylabel("Amplitude (norm.)")
        self.ax_amp.set_title("Ch Amplitudes")

        # IQ constellations
        self.ax_iq = []
        for ch in range(4):
            ax = self.fig.add_subplot(gs2[1, ch])
            ax.set_facecolor(PANEL)
            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"Ch {ch}  IQ scatter", fontsize=8.5)
            ax.axhline(0, color=SPINE, lw=0.6)
            ax.axvline(0, color=SPINE, lw=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            sc = ax.scatter([], [], s=3, color=CH_COLS[ch], alpha=0.5)
            self.ax_iq.append((ax, sc))

        # ── Status bar ─────────────────────────────────────────────────────
        self.status_ax = self.fig.add_axes([0.05, 0.94, 0.70, 0.04],
                                           facecolor="#0a0a0a")
        self.status_ax.set_xticks([])
        self.status_ax.set_yticks([])
        self.status_txt = self.status_ax.text(
            0.01, 0.4, "Waiting for signal …",
            transform=self.status_ax.transAxes,
            color="#888888", fontsize=9.5,
            fontfamily="DejaVu Sans Mono",
        )
        self.pass_txt = self.status_ax.text(
            0.75, 0.3, "",
            transform=self.status_ax.transAxes,
            color=PASS_COL, fontsize=13,
            fontfamily="DejaVu Sans Mono", fontweight="bold",
        )

        # ── Save button ─────────────────────────────────────────────────────
        btn_ax = self.fig.add_axes([0.77, 0.94, 0.20, 0.04])
        self.btn_save = Button(btn_ax, "Save Calibration",
                               color="#1a2a1a", hovercolor="#2a4a2a")
        self.btn_save.label.set_color(PASS_COL)
        self.btn_save.label.set_fontfamily("DejaVu Sans Mono")
        self.btn_save.on_clicked(self._on_save)

    # ── Update ─────────────────────────────────────────────────────────────

    def _update(self):
        buf = self.sd.get("rx_buffer")
        if buf is None:
            return

        X = buf.astype(np.complex128)

        # ── Phase & amplitude ─────────────────────────────────────────────
        offsets = measure_phase_offsets(X)
        amps    = measure_amplitudes(X)

        self._offsets    = offsets
        self._amp_latest = amps

        # Roll phase history
        self._phase_hist = np.roll(self._phase_hist, -1, axis=0)
        self._phase_hist[-1] = offsets

        # ── Phase bar chart ───────────────────────────────────────────────
        for bar, val in zip(self.bar_objs, offsets):
            bar.set_height(val)

        # ── Phase history lines ───────────────────────────────────────────
        xs = np.arange(HISTORY_LEN)
        for i, ln in enumerate(self.lines_hist):
            ln.set_data(xs, self._phase_hist[:, i])

        # ── Amplitude bars ────────────────────────────────────────────────
        for bar, val in zip(self.amp_bars, amps):
            bar.set_height(val)

        # ── IQ scatter ───────────────────────────────────────────────────
        n = min(X.shape[1], 256)
        norm = np.max(np.abs(X)) + 1e-12
        for ch, (ax, sc) in enumerate(self.ax_iq):
            pts = X[ch, :n] / norm
            sc.set_offsets(np.column_stack([pts.real, pts.imag]))

        # ── Stability check ───────────────────────────────────────────────
        recent = self._phase_hist[-STABILITY_FRAMES:]
        valid  = ~np.any(np.isnan(recent), axis=0)
        stds   = np.where(valid, np.std(recent, axis=0), np.inf)

        passes = stds < STABILITY_TOL_DEG
        all_ok = bool(np.all(passes))

        status_parts = []
        for i in range(3):
            tag   = "✓" if passes[i] else "✗"
            color = PASS_COL if passes[i] else FAIL_COL
            status_parts.append(
                f"Ch{i+1}: {offsets[i]:+7.2f}°  σ={stds[i]:.1f}°  {tag}"
            )
        self.status_txt.set_text("   |   ".join(status_parts))
        self.status_txt.set_color(PASS_COL if all_ok else TEXT)

        self.pass_txt.set_text("◉  PASS  —  press S to save" if all_ok else "")

        self.fig.canvas.draw_idle()

    # ── Save ───────────────────────────────────────────────────────────────

    def _save(self):
        # Full 4-element offsets: ch0 = 0, ch1..3 from measurement
        offsets_4ch = [0.0] + list(float(v) for v in self._offsets)
        os.makedirs(os.path.dirname(CAL_FILE), exist_ok=True)
        with open(CAL_FILE, "w") as f:
            yaml.dump(
                {
                    "phase_offsets_deg": offsets_4ch,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "note": "Generated by calibration/calibrate.py",
                },
                f,
                default_flow_style=False,
            )
        logging.info("Calibration saved → %s", CAL_FILE)
        logging.info("  Phase offsets (°): %s", offsets_4ch)
        self._saved = True
        self.pass_txt.set_text("◉  SAVED  ✓")
        self.fig.canvas.draw_idle()

    def _on_save(self, _event=None):
        self._save()

    def _on_key(self, event):
        if event.key in ("s", "S"):
            self._save()

    def run(self):
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KrakenSDR phase calibration tool")
    parser.add_argument(
        "--source", choices=["synthetic", "heimdall"], default="synthetic",
        help="IQ source: 'synthetic' for bench testing, 'heimdall' for real KrakenSDR",
    )
    parser.add_argument("--host", default="localhost",
                        help="Heimdall DAQ host (default: localhost)")
    parser.add_argument("--port", type=int, default=5555,
                        help="Heimdall DAQ port (default: 5555)")
    args = parser.parse_args()

    print("─" * 60)
    print(" KrakenSDR Calibration Tool")
    print(f"  Source  : {args.source}")
    if args.source == "heimdall":
        print(f"  Host    : {args.host}:{args.port}")
    print(f"  Cal file: {CAL_FILE}")
    print("─" * 60)
    print("  Procedure:")
    print("  1. Connect RF synth → 4-way splitter → all 4 antenna ports")
    print("  2. Set synth to 1575.42 MHz")
    print("  3. Watch phase offsets stabilise (σ < 2°)")
    print("  4. Press S or click [Save Calibration]")
    print("─" * 60)

    if args.source == "synthetic":
        sd, lock, src = _start_synthetic_source()
        print("  [NOTE] Using synthetic data — phase offsets will be near 0°")
        print("         Switch to --source heimdall with real hardware")
    else:
        sd, lock, src = _start_heimdall_source(args.host, args.port)

    tool = CalibrationTool(sd, lock, src)
    tool.run()

    src.stop()


if __name__ == "__main__":
    main()
