"""
dashboard/app.py

Live 4-panel anti-jam dashboard.

Panels:
  1 (top-left)  — RF Spectrum: raw (red) vs beamformed (blue)
  2 (top-right) — MUSIC DOA pseudospectrum with jammer bearing marker
  3 (bot-left)  — MVDR beam pattern with null (red) and look dir (green)
  4 (bot-right) — Suppression history + null depth history (rolling 60 s)

Top bar: status label, DOA readout, suppression dB, null depth, latency.

Run modes:
    python dashboard/app.py                  # synthetic sweeping jammer (default)
    python dashboard/app.py --source file --input recordings/jammed.iq  # offline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

from iq_source.synthetic_source import SyntheticSource, shared_data, get_lock
from pipeline.realtime import RealtimeEngine

# Publication-quality plot defaults
plt.rcParams.update({
    "font.size":          11,
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.5,
})

HISTORY_LEN = 200          # rolling history points (~60 s at 3 Hz)
UPDATE_MS   = 300          # dashboard refresh interval


class AntiJamDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GPS L1 Anti-Jam System — Live Dashboard")
        self.root.geometry("1300x820")
        self.root.configure(bg="#1a1a2e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._supp_history  = []
        self._null_history  = []
        self._lat_history   = []

        self._build_ui()
        self._schedule_update()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame",       background="#1a1a2e")
        style.configure("Status.TLabel",     background="#1a1a2e", foreground="#00ff99",
                         font=("Courier", 12, "bold"))
        style.configure("Metric.TLabel",     background="#1a1a2e", foreground="#e0e0e0",
                         font=("Courier", 11))
        style.configure("Title.TLabel",      background="#1a1a2e", foreground="#00ccff",
                         font=("Courier", 14, "bold"))
        style.configure("AJ.TCheckbutton",   background="#1a1a2e", foreground="#00ff99",
                         font=("Courier", 11, "bold"))

        # ── Top bar ───────────────────────────────────────────────────────────
        top = ttk.Frame(self.root, style="Dark.TFrame", padding=(10, 6))
        top.pack(fill="x")

        ttk.Label(top, text="GPS L1 ANTI-JAM", style="Title.TLabel").pack(side="left", padx=10)

        self.status_lbl = ttk.Label(top, text="● INITIALISING", style="Status.TLabel")
        self.status_lbl.pack(side="left", padx=20)

        # Metrics bar
        for attr, text in [
            ("doa_lbl",   "DOA: ---°  "),
            ("supp_lbl",  "Suppression: --- dB  "),
            ("null_lbl",  "Null: --- dB  "),
            ("lat_lbl",   "Latency: --- ms"),
        ]:
            lbl = ttk.Label(top, text=text, style="Metric.TLabel")
            lbl.pack(side="left", padx=6)
            setattr(self, attr, lbl)

        # AJ toggle
        self.aj_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="ENABLE AJ",
            variable=self.aj_var,
            command=self._toggle_aj,
            style="AJ.TCheckbutton",
        ).pack(side="right", padx=20)

        # ── 4-panel figure ────────────────────────────────────────────────────
        self.fig = plt.Figure(figsize=(13, 7.5), facecolor="#0d0d1a")
        gs = self.fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28,
                                   left=0.07, right=0.97, top=0.93, bottom=0.08)

        axes_props = dict(facecolor="#0d0d1a")
        self.ax1 = self.fig.add_subplot(gs[0, 0], **axes_props)  # spectrum
        self.ax2 = self.fig.add_subplot(gs[0, 1], **axes_props)  # MUSIC
        self.ax3 = self.fig.add_subplot(gs[1, 0], **axes_props)  # beam pattern
        self.ax4 = self.fig.add_subplot(gs[1, 1], **axes_props)  # history

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.tick_params(colors="#aaaaaa")
            ax.xaxis.label.set_color("#aaaaaa")
            ax.yaxis.label.set_color("#aaaaaa")
            ax.title.set_color("#00ccff")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            ax.grid(True, color="#1e1e3a", linewidth=0.8)

        # Panel 1: spectrum
        self.ln_before, = self.ax1.plot([], [], color="#ff4444", lw=1.5,
                                         label="Raw (jammer visible)")
        self.ln_after,  = self.ax1.plot([], [], color="#44aaff", lw=1.8,
                                         label="Beamformed (jammer nulled)")
        self.ax1.set_title("Panel 1 — RF Spectrum")
        self.ax1.set_xlabel("Frequency (MHz)")
        self.ax1.set_ylabel("Power (dB)")
        self.ax1.set_ylim(-110, 10)
        self.ax1.legend(loc="upper right", fontsize=8,
                        facecolor="#1a1a2e", edgecolor="#333355",
                        labelcolor="#dddddd")

        # Panel 2: MUSIC
        self.ln_music, = self.ax2.plot([], [], color="#00ffcc", lw=1.5)
        self.vl_doa   = self.ax2.axvline(0, color="#ff4444", lw=2, ls="--",
                                          label="Estimated DOA")
        self.ax2.set_title("Panel 2 — MUSIC DOA Spectrum")
        self.ax2.set_xlabel("Azimuth (°)")
        self.ax2.set_ylabel("Pseudospectrum (dB)")
        self.ax2.set_xlim(-180, 180)
        self.ax2.set_ylim(-65, 5)
        self.ax2.legend(loc="upper right", fontsize=8,
                        facecolor="#1a1a2e", edgecolor="#333355",
                        labelcolor="#dddddd")

        # Panel 3: beam pattern
        self.ln_beam, = self.ax3.plot([], [], color="#ff9900", lw=1.8)
        self.vl_null  = self.ax3.axvline(0, color="#ff4444", lw=2, ls="--",
                                          label="Null (jammer)")
        self.vl_look  = self.ax3.axvline(0, color="#44ff44", lw=2, ls="-.",
                                          label="Look dir (GPS)")
        self.ax3.set_title("Panel 3 — MVDR Beam Pattern")
        self.ax3.set_xlabel("Azimuth (°)")
        self.ax3.set_ylabel("Gain (dB)")
        self.ax3.set_xlim(-180, 180)
        self.ax3.set_ylim(-60, 5)
        self.ax3.legend(loc="upper right", fontsize=8,
                        facecolor="#1a1a2e", edgecolor="#333355",
                        labelcolor="#dddddd")

        # Panel 4: history
        self.ln_supp_h, = self.ax4.plot([], [], color="#44aaff",  lw=1.5,
                                          label="Suppression (dB)")
        self.ln_null_h, = self.ax4.plot([], [], color="#ff9900",  lw=1.5,
                                          label="Null depth (dB)")
        self.ax4.set_title("Panel 4 — Performance History")
        self.ax4.set_xlabel("Frame")
        self.ax4.set_ylabel("dB")
        self.ax4.set_ylim(-10, 80)
        self.ax4.legend(loc="upper right", fontsize=8,
                        facecolor="#1a1a2e", edgecolor="#333355",
                        labelcolor="#dddddd")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------ update

    def _toggle_aj(self):
        shared_data["beamforming_active"] = self.aj_var.get()

    def _schedule_update(self):
        self.root.after(UPDATE_MS, self._update)

    def _update(self):
        if not self.root.winfo_exists():
            return

        sd = shared_data

        # ── Status label ────────────────────────────────────────────────────
        doa  = sd.get("doa_est")
        supp = sd.get("suppression_db", 0.0)
        nd   = sd.get("null_depth_db",  0.0)
        lat  = sd.get("latency_ms",     0.0)

        if doa is not None and sd.get("beamforming_active", True):
            self.status_lbl.config(
                text=f"● AJ ACTIVE — jammer at {doa:.1f}°",
                foreground="#00ff99",
            )
        elif doa is not None:
            self.status_lbl.config(
                text=f"● JAMMER DETECTED — AJ OFF",
                foreground="#ff8800",
            )
        else:
            self.status_lbl.config(
                text="● MONITORING — no jammer detected",
                foreground="#4488ff",
            )

        self.doa_lbl.config( text=f"DOA: {doa:.1f}°  "   if doa else "DOA: ---°  ")
        self.supp_lbl.config(text=f"Suppression: {supp:.1f} dB  ")
        self.null_lbl.config(text=f"Null: {nd:.1f} dB  ")
        self.lat_lbl.config( text=f"Latency: {lat:.1f} ms")

        # ── Panel 1: spectrum ─────────────────────────────────────────────
        freq   = sd.get("freq_axis")
        s_bef  = sd.get("spectrum_before")
        s_aft  = sd.get("spectrum_after")
        if freq is not None and s_bef is not None:
            self.ln_before.set_data(freq, s_bef)
            self.ax1.set_xlim(freq[0], freq[-1])
        if freq is not None and s_aft is not None:
            self.ln_after.set_data(freq, s_aft)
        # Show after line only when AJ is active
        self.ln_after.set_visible(
            s_aft is not None and sd.get("beamforming_active", True) and doa is not None
        )

        # ── Panel 2: MUSIC ────────────────────────────────────────────────
        spec_m = sd.get("music_spectrum")
        az_grid = getattr(self, "_az_grid", None)
        if spec_m is not None:
            if az_grid is None:
                n = len(spec_m)
                az_grid = np.linspace(-180, 180, n)
                self._az_grid = az_grid
            self.ln_music.set_data(az_grid, spec_m)
        if doa is not None:
            self.vl_doa.set_xdata([doa, doa])
            self.vl_doa.set_visible(True)
        else:
            self.vl_doa.set_visible(False)

        # ── Panel 3: beam pattern ─────────────────────────────────────────
        pat = sd.get("beam_pattern")
        if pat is not None and az_grid is not None and np.any(pat != 0):
            self.ln_beam.set_data(az_grid, pat)
        if doa is not None:
            self.vl_null.set_xdata([doa, doa])
            self.vl_null.set_visible(True)
        else:
            self.vl_null.set_visible(False)
        # Look direction marker (GPS zenith = all azimuths equal → show at 0°)
        self.vl_look.set_xdata([0, 0])

        # ── Panel 4: history ──────────────────────────────────────────────
        self._supp_history.append(supp)
        self._null_history.append(abs(nd))
        if len(self._supp_history) > HISTORY_LEN:
            self._supp_history.pop(0)
            self._null_history.pop(0)

        xs = list(range(len(self._supp_history)))
        self.ln_supp_h.set_data(xs, self._supp_history)
        self.ln_null_h.set_data(xs, self._null_history)
        self.ax4.set_xlim(0, max(HISTORY_LEN, len(xs)))

        self.canvas.draw_idle()
        self._schedule_update()

    def _on_close(self):
        shared_data["running"] = False
        self.root.quit()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPS L1 Anti-Jam Dashboard")
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic"],
                        help="IQ source (synthetic | heimdall when SDR connected)")
    parser.add_argument("--jnr",    type=float, default=30.0,
                        help="Jammer-to-noise ratio in dB (synthetic mode)")
    parser.add_argument("--sweep",  type=float, default=8.0,
                        help="Jammer sweep speed in deg/s (synthetic mode)")
    args = parser.parse_args()

    # Start IQ source
    src = SyntheticSource(jnr_db=args.jnr, sweep_rate=args.sweep)
    src.start()
    print(f"[source]  synthetic IQ started — jammer sweeping at {args.sweep} deg/s, JNR={args.jnr} dB")

    # Start DSP engine
    engine = RealtimeEngine(shared_data, get_lock(), config_path="config.yaml")
    engine.start()
    print("[engine]  DSP engine started")

    # Launch dashboard
    print("[dashboard] opening window ...")
    root = tk.Tk()
    app  = AntiJamDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
