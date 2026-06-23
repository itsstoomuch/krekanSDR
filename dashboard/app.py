"""
dashboard/app.py  —  GPS L1 Anti-Jam  ·  Live Monitor

Panels:
  TL — RF Spectrum        : raw vs beamformed
  TR — MUSIC DOA          : pseudospectrum + jammer bearing
  BL — Polar Beam Scanner : radar-style MVDR pattern
  BR — Performance Log    : suppression + null depth history

Right control panel:
  Source selector  — Synthetic / KrakenSDR
  Jammer controls  — count, type, azimuth (synthetic mode only)
  Connection info  — host/port for KrakenSDR mode
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch

from iq_source.synthetic_source import SyntheticSource, shared_data, get_lock
from pipeline.realtime import RealtimeEngine

# ── Palette ────────────────────────────────────────────────────────────────
BG        = "#0d0d0d"
PANEL_BG  = "#111111"
CTRL_BG   = "#0a0a0a"
GRID_COL  = "#1e1e1e"
SPINE_COL = "#2a2a2a"
TEXT_COL  = "#c8c8c8"
TITLE_COL = "#e8e8e8"
ACCENT_R  = "#d9534f"
ACCENT_B  = "#4d9de0"
ACCENT_G  = "#4caf76"
ACCENT_Y  = "#e0b84d"
MUSIC_COL = "#4ec9b0"
HIST1     = "#4d9de0"
HIST2     = "#e0b84d"
ENTRY_BG  = "#1a1a1a"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9.5,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "axes.labelcolor":  TEXT_COL,
    "axes.edgecolor":   SPINE_COL,
    "axes.facecolor":   PANEL_BG,
    "axes.grid":        True,
    "axes.titlecolor":  TITLE_COL,
    "grid.color":       GRID_COL,
    "grid.linewidth":   0.6,
    "xtick.color":      TEXT_COL,
    "ytick.color":      TEXT_COL,
    "figure.facecolor": BG,
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": SPINE_COL,
    "legend.labelcolor":TEXT_COL,
    "legend.fontsize":  8,
})

HISTORY_LEN = 200
UPDATE_MS   = 300
CTRL_W      = 230   # right control panel width in pixels

JAMMER_TYPES = ["CW", "FMCW", "BARRAGE"]


# ── Helpers ────────────────────────────────────────────────────────────────

def _label(parent, text, fg=TEXT_COL, font_size=8, mono=False, **kw):
    ff = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    return tk.Label(parent, text=text, bg=CTRL_BG, fg=fg,
                    font=(ff, font_size), **kw)


def _divider(parent):
    tk.Frame(parent, bg=SPINE_COL, height=1).pack(fill="x", padx=8, pady=6)


def _section(parent, title):
    _label(parent, title, fg="#555555", font_size=7, mono=True).pack(
        anchor="w", padx=10, pady=(8, 2))


# ── Jammer row widget ──────────────────────────────────────────────────────

class JammerRow:
    """One row of controls for a single jammer."""

    def __init__(self, parent, index: int, on_change):
        self.index     = index
        self.on_change = on_change

        self.frame = tk.Frame(parent, bg=CTRL_BG)
        self.frame.pack(fill="x", padx=8, pady=2)

        _label(self.frame, f"J{index + 1}", fg=ACCENT_Y, font_size=8, mono=True
               ).grid(row=0, column=0, sticky="w", padx=(0, 6))

        # Type dropdown
        self.type_var = tk.StringVar(value="CW")
        om = tk.OptionMenu(self.frame, self.type_var, *JAMMER_TYPES,
                           command=lambda _: self.on_change())
        om.config(bg=ENTRY_BG, fg=TEXT_COL, activebackground=ENTRY_BG,
                  activeforeground=ACCENT_Y, highlightthickness=0,
                  relief="flat", font=("DejaVu Sans Mono", 8), width=7,
                  indicatoron=True)
        om["menu"].config(bg=ENTRY_BG, fg=TEXT_COL,
                          activebackground="#2a2a2a", activeforeground=ACCENT_Y,
                          font=("DejaVu Sans Mono", 8))
        om.grid(row=0, column=1, padx=2)

        # Azimuth label + spinbox
        _label(self.frame, "Az", fg=TEXT_COL, font_size=7
               ).grid(row=0, column=2, padx=(6, 1))
        self.az_var = tk.DoubleVar(value=45.0 if index == 0 else -90.0)
        sp = tk.Spinbox(
            self.frame, textvariable=self.az_var,
            from_=-180, to=180, increment=5,
            width=5, bg=ENTRY_BG, fg=TEXT_COL,
            insertbackground=TEXT_COL,
            buttonbackground=ENTRY_BG, relief="flat",
            font=("DejaVu Sans Mono", 8),
            command=self.on_change,
        )
        sp.bind("<Return>", lambda _: self.on_change())
        sp.bind("<FocusOut>", lambda _: self.on_change())
        sp.grid(row=0, column=3, padx=2)
        _label(self.frame, "°", fg=TEXT_COL, font_size=7
               ).grid(row=0, column=4)

    def config_dict(self) -> dict:
        return {
            "az_deg": float(self.az_var.get()),
            "el_deg": 5.0,
            "jnr_db": 30.0,
            "type":   self.type_var.get(),
        }

    def set_visible(self, visible: bool):
        if visible:
            self.frame.pack(fill="x", padx=8, pady=2)
        else:
            self.frame.pack_forget()


# ── Main dashboard ─────────────────────────────────────────────────────────

class AntiJamDashboard:
    def __init__(self, root: tk.Tk, src: SyntheticSource):
        self.root        = root
        self.root.title("GPS L1 Anti-Jam  ·  Live Monitor")
        self.root.geometry("1600x860")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._source     = src          # current IQ source (SyntheticSource or HeimdallSource)
        self._source_type = "synthetic" # "synthetic" | "heimdall"
        self._heimdall   = None         # HeimdallSource instance (lazy create)

        self._supp_hist  = []
        self._null_hist  = []
        self._az_grid    = None
        self._polar_fill = None
        self._null_line  = None
        self._look_line  = None

        self._build_top_bar()
        self._build_main()
        self._schedule_update()

    # ── Top bar ────────────────────────────────────────────────────────────

    def _build_top_bar(self):
        bar = tk.Frame(self.root, bg="#0a0a0a", height=38)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Label(bar, text="GPS L1  ANTI-JAM  PROTOTYPE",
                 bg="#0a0a0a", fg="#888888",
                 font=("DejaVu Sans Mono", 10)).pack(side="left", padx=16, pady=8)

        tk.Frame(bar, bg=SPINE_COL, width=1).pack(side="left", fill="y", pady=6)

        self.status_lbl = tk.Label(bar, text="INITIALISING",
                                   bg="#0a0a0a", fg="#555555",
                                   font=("DejaVu Sans Mono", 10, "bold"))
        self.status_lbl.pack(side="left", padx=16)

        tk.Frame(bar, bg=SPINE_COL, width=1).pack(side="left", fill="y", pady=6)

        self._metric_lbls = {}
        for key, label in [("doa", "DOA"), ("supp", "SUPPRESSION"),
                            ("null", "NULL DEPTH"), ("lat", "LATENCY")]:
            f = tk.Frame(bar, bg="#0a0a0a")
            f.pack(side="left", padx=14, pady=4)
            tk.Label(f, text=label, bg="#0a0a0a", fg="#444444",
                     font=("DejaVu Sans Mono", 7)).pack(anchor="w")
            v = tk.Label(f, text="—", bg="#0a0a0a", fg=TEXT_COL,
                         font=("DejaVu Sans Mono", 10, "bold"))
            v.pack(anchor="w")
            self._metric_lbls[key] = v

        self.aj_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="ENABLE  AJ",
                       variable=self.aj_var,
                       command=self._toggle_aj,
                       bg="#0a0a0a", fg=ACCENT_G,
                       selectcolor="#0a0a0a",
                       activebackground="#0a0a0a",
                       activeforeground=ACCENT_G,
                       font=("DejaVu Sans Mono", 9, "bold"),
                       ).pack(side="right", padx=20)

    # ── Main layout: figure + right control panel ──────────────────────────

    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Left: matplotlib canvas
        canvas_frame = tk.Frame(main, bg=BG)
        canvas_frame.pack(side="left", fill="both", expand=True)
        self._build_figure(canvas_frame)

        # Right: control panel
        ctrl = tk.Frame(main, bg=CTRL_BG, width=CTRL_W)
        ctrl.pack(side="right", fill="y")
        ctrl.pack_propagate(False)
        self._build_control_panel(ctrl)

    # ── 4-panel matplotlib figure ──────────────────────────────────────────

    def _build_figure(self, parent):
        self.fig = plt.Figure(figsize=(13.4, 8.0), facecolor=BG)
        gs = gridspec.GridSpec(
            2, 2,
            figure=self.fig,
            hspace=0.42, wspace=0.30,
            left=0.06, right=0.97,
            top=0.95,  bottom=0.07,
        )

        # TL — RF Spectrum
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ln_raw, = self.ax1.plot([], [], color=ACCENT_R, lw=1.2,
                                      alpha=0.85, label="Raw  (jammer present)")
        self.ln_bf,  = self.ax1.plot([], [], color=ACCENT_B, lw=1.6,
                                      label="Beamformed  (jammer nulled)")
        self.ax1.set_title("RF Spectrum  —  1575.42 MHz")
        self.ax1.set_xlabel("Frequency (MHz)")
        self.ax1.set_ylabel("Power (dBFS)")
        self.ax1.set_ylim(-110, 15)
        self.ax1.legend(loc="lower right")

        # TR — MUSIC DOA
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ln_music, = self.ax2.plot([], [], color=MUSIC_COL, lw=1.4)
        self.vl_doa    = self.ax2.axvline(0, color=ACCENT_R, lw=1.5,
                                           ls="--", alpha=0.9, label="Jammer DOA")
        self.doa_ann   = self.ax2.annotate(
            "", xy=(0, 0), xytext=(0, -15),
            fontsize=8, color=ACCENT_R,
            arrowprops=dict(arrowstyle="->", color=ACCENT_R, lw=1.0),
        )
        self.ax2.set_title("MUSIC Direction-of-Arrival")
        self.ax2.set_xlabel("Azimuth (°)")
        self.ax2.set_ylabel("Pseudospectrum (dB)")
        self.ax2.set_xlim(-180, 180)
        self.ax2.set_ylim(-65, 5)
        self.ax2.set_xticks(range(-180, 181, 45))
        self.ax2.legend(loc="upper left")

        # BL — Polar Beam Scanner
        self.ax3 = self.fig.add_subplot(gs[1, 0], projection="polar")
        self.ax3.set_facecolor(PANEL_BG)
        self.ax3.set_theta_zero_location("N")
        self.ax3.set_theta_direction(-1)
        self.ax3.set_rlim(-60, 5)
        self.ax3.set_rticks([-50, -40, -30, -20, -10, 0])
        self.ax3.set_rlabel_position(100)
        self.ax3.tick_params(colors=TEXT_COL, labelsize=7.5)
        self.ax3.grid(color=GRID_COL, linewidth=0.7)
        for spine in self.ax3.spines.values():
            spine.set_edgecolor(SPINE_COL)
        self.ax3.set_title("MVDR Beam Pattern  (polar)", pad=14, color=TITLE_COL)
        self.ln_polar, = self.ax3.plot([], [], color=ACCENT_Y, lw=1.6)
        self.ax3.legend(
            handles=[
                Patch(color=ACCENT_Y, label="Beam gain"),
                Patch(color=ACCENT_R, label="Null (jammer)"),
                Patch(color=ACCENT_G, label="GPS look dir"),
            ],
            loc="lower right", bbox_to_anchor=(1.18, -0.05), fontsize=7.5,
        )

        # BR — Performance History
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ln_supp_h, = self.ax4.plot([], [], color=HIST1, lw=1.4,
                                          label="Suppression (dB)")
        self.ln_null_h, = self.ax4.plot([], [], color=HIST2, lw=1.4,
                                          label="Null depth (dB)")
        self.ax4.axhline(30, color=SPINE_COL, lw=0.8, ls="--")
        self.ax4.text(2, 31.5, "30 dB target", color="#444444", fontsize=7.5)
        self.ax4.set_title("Performance History")
        self.ax4.set_xlabel("Frame")
        self.ax4.set_ylabel("dB")
        self.ax4.set_ylim(-5, 85)
        self.ax4.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Right control panel ─────────────────────────────────────────────────

    def _build_control_panel(self, parent):
        # Scrollable inner frame
        inner = tk.Frame(parent, bg=CTRL_BG)
        inner.pack(fill="both", expand=True, padx=0, pady=0)

        # ── SOURCE ───────────────────────────────────────────────────────
        _section(inner, "SOURCE")
        self._src_var = tk.StringVar(value="synthetic")

        src_frame = tk.Frame(inner, bg=CTRL_BG)
        src_frame.pack(fill="x", padx=10)

        for val, lbl in [("synthetic", "Synthetic"), ("heimdall", "KrakenSDR")]:
            rb = tk.Radiobutton(
                src_frame, text=lbl, variable=self._src_var, value=val,
                command=self._on_source_change,
                bg=CTRL_BG, fg=TEXT_COL,
                selectcolor=CTRL_BG, activebackground=CTRL_BG,
                activeforeground=ACCENT_B,
                font=("DejaVu Sans", 9),
            )
            rb.pack(anchor="w", pady=1)

        _divider(inner)

        # ── KRAKENSDR CONNECTION ──────────────────────────────────────────
        _section(inner, "KRAKENSDR")
        self._ksdr_frame = tk.Frame(inner, bg=CTRL_BG)
        self._ksdr_frame.pack(fill="x", padx=10, pady=2)

        for row_idx, (lbl, default) in enumerate([("Host", "localhost"),
                                                   ("Port", "5555")]):
            tk.Label(self._ksdr_frame, text=lbl, bg=CTRL_BG, fg=TEXT_COL,
                     font=("DejaVu Sans", 8), width=4, anchor="w"
                     ).grid(row=row_idx, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            tk.Entry(self._ksdr_frame, textvariable=var, width=14,
                     bg=ENTRY_BG, fg=TEXT_COL, insertbackground=TEXT_COL,
                     relief="flat", font=("DejaVu Sans Mono", 8),
                     ).grid(row=row_idx, column=1, padx=(4, 0), pady=2)
            if lbl == "Host":
                self._host_var = var
            else:
                self._port_var = var

        conn_row = tk.Frame(self._ksdr_frame, bg=CTRL_BG)
        conn_row.grid(row=2, column=0, columnspan=2, pady=(4, 0))

        self._conn_btn = tk.Button(
            conn_row, text="Connect",
            command=self._connect_heimdall,
            bg="#1a2a1a", fg=ACCENT_G,
            activebackground="#2a3a2a", activeforeground=ACCENT_G,
            relief="flat", font=("DejaVu Sans Mono", 8, "bold"),
            padx=8, pady=3,
        )
        self._conn_btn.pack(side="left", padx=(0, 4))

        self._disc_btn = tk.Button(
            conn_row, text="Disconnect",
            command=self._disconnect_heimdall,
            bg="#2a1a1a", fg=ACCENT_R,
            activebackground="#3a2a2a", activeforeground=ACCENT_R,
            relief="flat", font=("DejaVu Sans Mono", 8),
            padx=8, pady=3, state="disabled",
        )
        self._disc_btn.pack(side="left")

        self.conn_status_lbl = tk.Label(
            self._ksdr_frame, text="", bg=CTRL_BG, fg="#555555",
            font=("DejaVu Sans Mono", 7),
        )
        self.conn_status_lbl.grid(row=3, column=0, columnspan=2,
                                   sticky="w", pady=(2, 0))

        _divider(inner)

        # ── JAMMER SIMULATION ─────────────────────────────────────────────
        _section(inner, "JAMMER  SIMULATION")

        count_frame = tk.Frame(inner, bg=CTRL_BG)
        count_frame.pack(fill="x", padx=10, pady=(0, 4))
        _label(count_frame, "Count", font_size=8).pack(side="left")
        self._jammer_count = tk.IntVar(value=1)
        for n in (1, 2):
            tk.Radiobutton(
                count_frame, text=str(n),
                variable=self._jammer_count, value=n,
                command=self._on_jammer_count_change,
                bg=CTRL_BG, fg=TEXT_COL,
                selectcolor=CTRL_BG, activebackground=CTRL_BG,
                font=("DejaVu Sans Mono", 9),
            ).pack(side="left", padx=6)

        # Two jammer rows
        self._jammer_rows = [
            JammerRow(inner, 0, self._on_jammer_config_change),
            JammerRow(inner, 1, self._on_jammer_config_change),
        ]
        # Hide jammer 2 initially
        self._jammer_rows[1].set_visible(False)

        _divider(inner)

        # ── INFO ──────────────────────────────────────────────────────────
        _section(inner, "SIGNAL  CHAIN")
        info_lines = [
            "GPS L1  1575.42 MHz",
            "Array   2×2 URA",
            "Spacing λ/2 ≈ 9.5 cm",
            "DOA     MUSIC",
            "Weight  MVDR",
        ]
        for line in info_lines:
            _label(inner, line, fg="#444444", font_size=7, mono=True
                   ).pack(anchor="w", padx=12)

        # Initially hide KrakenSDR section
        self._update_ctrl_visibility()

    # ── Control panel logic ────────────────────────────────────────────────

    def _update_ctrl_visibility(self):
        mode = self._src_var.get()
        # KrakenSDR section always visible (for host/port even in synthetic mode)
        # Jammer section always visible for synthetic reference
        # Just disable jammer rows when in heimdall mode
        for row in self._jammer_rows:
            # Grey out row widgets based on mode
            state = "normal" if mode == "synthetic" else "disabled"
            for child in row.frame.winfo_children():
                try:
                    child.config(state=state)
                except tk.TclError:
                    pass

    def _on_source_change(self):
        mode = self._src_var.get()
        self._update_ctrl_visibility()
        if mode == "synthetic" and self._source_type != "synthetic":
            self._switch_to_synthetic()
        # Switching to heimdall happens via Connect button

    def _on_jammer_count_change(self):
        n = self._jammer_count.get()
        self._jammer_rows[1].set_visible(n >= 2)
        self._on_jammer_config_change()

    def _on_jammer_config_change(self, *_):
        if self._source_type != "synthetic":
            return
        n       = self._jammer_count.get()
        configs = [self._jammer_rows[i].config_dict() for i in range(n)]
        if hasattr(self._source, "set_jammer_configs"):
            self._source.set_jammer_configs(configs)

    def _switch_to_synthetic(self):
        if self._heimdall:
            self._heimdall.stop_source()
        self._source.start() if not self._source._running else None
        self._source_type = "synthetic"
        shared_data["source_type"] = "synthetic"
        self.conn_status_lbl.config(text="")
        self._conn_btn.config(state="normal")
        self._disc_btn.config(state="disabled")

    def _connect_heimdall(self):
        from iq_source.heimdall_source import HeimdallSource

        host = self._host_var.get().strip()
        try:
            port = int(self._port_var.get().strip())
        except ValueError:
            self.conn_status_lbl.config(text="Invalid port", fg=ACCENT_R)
            return

        # Stop current synthetic source
        if hasattr(self._source, "stop_source"):
            self._source.stop_source()

        self._heimdall = HeimdallSource(host=host, port=port)
        self._heimdall.start()
        self._source_type = "heimdall"
        self._src_var.set("heimdall")
        self._update_ctrl_visibility()

        self.conn_status_lbl.config(
            text=f"Connecting to {host}:{port}…", fg="#888888",
        )
        self._conn_btn.config(state="disabled")
        self._disc_btn.config(state="normal")

        # Update status after a brief delay
        self.root.after(2000, self._check_heimdall_status)

    def _check_heimdall_status(self):
        if shared_data.get("source_type") == "heimdall" and \
                shared_data.get("rx_buffer") is not None:
            self.conn_status_lbl.config(text="Connected  ✓", fg=ACCENT_G)
        else:
            self.conn_status_lbl.config(text="No data — check cable/DAQ", fg=ACCENT_R)
            self.root.after(3000, self._check_heimdall_status)

    def _disconnect_heimdall(self):
        if self._heimdall:
            self._heimdall.stop_source()
            self._heimdall = None
        self._source_type = "synthetic"
        self._src_var.set("synthetic")
        self._update_ctrl_visibility()
        self.conn_status_lbl.config(text="Disconnected", fg="#555555")
        self._conn_btn.config(state="normal")
        self._disc_btn.config(state="disabled")
        # Re-start synthetic source
        if hasattr(self._source, "_running") and not self._source._running:
            self._source._running = True
            import threading
            t = threading.Thread(target=self._source._run, daemon=True,
                                 name="synthetic-source")
            t.start()

    # ── AJ toggle ──────────────────────────────────────────────────────────

    def _toggle_aj(self):
        shared_data["beamforming_active"] = self.aj_var.get()

    def _on_close(self):
        shared_data["running"] = False
        self.root.quit()
        self.root.destroy()

    # ── Update scheduler ───────────────────────────────────────────────────

    def _schedule_update(self):
        self.root.after(UPDATE_MS, self._update)

    # ── Main update loop ───────────────────────────────────────────────────

    def _update(self):
        if not self.root.winfo_exists():
            return

        sd   = shared_data
        doa  = sd.get("doa_est")
        supp = sd.get("suppression_db", 0.0)
        nd   = sd.get("null_depth_db",  0.0)
        lat  = sd.get("latency_ms",     0.0)
        aj   = sd.get("beamforming_active", True)

        # Status bar
        if doa is not None and aj:
            self.status_lbl.config(text="AJ  ACTIVE", fg=ACCENT_G)
        elif doa is not None:
            self.status_lbl.config(text="JAMMER  DETECTED  —  AJ  OFF", fg=ACCENT_R)
        else:
            self.status_lbl.config(text="MONITORING", fg="#555555")

        self._metric_lbls["doa"].config(
            text=f"{doa:.1f}°" if doa is not None else "—",
            fg=ACCENT_R if doa is not None else TEXT_COL,
        )
        self._metric_lbls["supp"].config(
            text=f"{supp:.1f} dB",
            fg=ACCENT_B if supp > 15 else TEXT_COL,
        )
        self._metric_lbls["null"].config(
            text=f"{nd:.1f} dB",
            fg=ACCENT_G if nd < -30 else TEXT_COL,
        )
        self._metric_lbls["lat"].config(text=f"{lat:.1f} ms")

        # Panel 1: RF Spectrum
        freq  = sd.get("freq_axis")
        s_bef = sd.get("spectrum_before")
        s_aft = sd.get("spectrum_after")
        if freq is not None and s_bef is not None:
            self.ln_raw.set_data(freq, s_bef)
            self.ax1.set_xlim(freq[0], freq[-1])
        if freq is not None and s_aft is not None:
            self.ln_bf.set_data(freq, s_aft)
        self.ln_bf.set_visible(bool(s_aft is not None and aj and doa is not None))

        # Panel 2: MUSIC
        spec_m = sd.get("music_spectrum")
        if spec_m is not None:
            if self._az_grid is None:
                self._az_grid = np.linspace(-180, 180, len(spec_m))
            self.ln_music.set_data(self._az_grid, spec_m)

        if doa is not None:
            self.vl_doa.set_xdata([doa, doa])
            self.vl_doa.set_visible(True)
            self.doa_ann.set_text(f"{doa:.1f}°")
            peak_y = float(spec_m[np.argmin(np.abs(self._az_grid - doa))]) \
                     if spec_m is not None else 0
            self.doa_ann.xy      = (doa, peak_y)
            self.doa_ann.xyann   = (doa + 8, peak_y - 12)
        else:
            self.vl_doa.set_visible(False)
            self.doa_ann.set_text("")

        # Panel 3: Polar beam scanner
        pat = sd.get("beam_pattern")
        if pat is not None and self._az_grid is not None and np.any(pat != 0):
            theta = np.deg2rad(self._az_grid)
            r     = np.clip(pat, -60, 5)
            self.ln_polar.set_data(theta, r)
            if self._polar_fill is not None:
                self._polar_fill.remove()
            self._polar_fill = self.ax3.fill(theta, r, color=ACCENT_Y, alpha=0.12)[0]

        if doa is not None:
            null_theta = np.deg2rad(doa)
            if self._null_line is not None:
                self._null_line.remove()
            self._null_line = self.ax3.plot(
                [null_theta, null_theta], [-60, 5],
                color=ACCENT_R, lw=2.0, ls="--", alpha=0.9,
            )[0]
        elif self._null_line is not None:
            self._null_line.set_alpha(0.0)

        if self._look_line is not None:
            self._look_line.remove()
        self._look_line = self.ax3.plot(
            [0, 0], [-60, 5],
            color=ACCENT_G, lw=1.8, ls="-.", alpha=0.75,
        )[0]

        # Panel 4: History
        self._supp_hist.append(float(supp))
        self._null_hist.append(float(abs(nd)))
        if len(self._supp_hist) > HISTORY_LEN:
            self._supp_hist.pop(0)
            self._null_hist.pop(0)
        xs = list(range(len(self._supp_hist)))
        self.ln_supp_h.set_data(xs, self._supp_hist)
        self.ln_null_h.set_data(xs, self._null_hist)
        self.ax4.set_xlim(0, max(HISTORY_LEN, len(xs)))

        self.canvas.draw_idle()
        self._schedule_update()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jnr",   type=float, default=30.0)
    parser.add_argument("--sweep", type=float, default=8.0)
    args = parser.parse_args()

    src = SyntheticSource(jnr_db=args.jnr, sweep_rate=args.sweep)
    src.start()

    engine = RealtimeEngine(shared_data, get_lock(), config_path="config.yaml")
    engine.start()

    root = tk.Tk()
    AntiJamDashboard(root, src)
    root.mainloop()


if __name__ == "__main__":
    main()
