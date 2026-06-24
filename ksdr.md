# KrakenSDR Anti-Jam Prototype — Master Build Checklist
# GPS L1 | 4-element 2×2 CRPA | MUSIC DOA + MVDR | Live Dashboard

> **HOW TO USE THIS FILE**
> - Work one checkbox at a time
> - After each ✅ item: save the file, commit the code
> - If Claude hits context limit: start new session, say "read ksdr.md and continue from last unchecked item"
> - Never start a new item until the previous one is committed to disk

---

## CURRENT STATUS
**Last completed:** Phase 1.5 — calibration/calibrate.py + dashboard overhaul
**Next task:** Phase 1.1 — Install software stack (when KrakenSDR arrives)
**Branch/folder:** /home/itsstoomuch/projects/krakensdr_antijam

**Also completed out of order:**
- 1.2 — iq_source/heimdall_source.py (KrakenSDR TCP reader, auto-reconnect)
- 1.5 — calibration/calibrate.py (standalone phase cal tool with matplotlib UI)
- Dashboard redesign — professional dark theme, polar beam scanner panel
- Dashboard v2 — right control panel: source selector, jammer type/angle/count controls, KrakenSDR connect UI

---

## PHASE 0 — DSP Core (No Hardware Required)
> Goal: Port simulation DSP into clean modules. pytest green before any hardware is touched.
> All source math comes from existing: generate_array_data.py, music_spectrum.py, mvdr_beamformer.py

---

### 0.1 — Create new repo folder and config.yaml
- [x] Create `/home/itsstoomuch/projects/krakensdr_antijam/` directory
- [x] Create `config.yaml` with all parameters
- [x] Create empty `__init__.py` files for all packages
- [x] **SAVE CHECKPOINT** — commit message: "init: project structure and config.yaml"

---

### 0.2 — dsp/geometry.py
> Source: generate_array_data.py (steering_vector, element_pattern, elem_pos)
> Key changes vs sim: add cal_offset parameter loaded from cal.yaml

- [x] Port `elem_pos` 2×2 URA layout (4×3 array, element order (0,0)(1,0)(0,1)(1,1))
- [x] Port `steering_vector(az_deg, el_deg)` using +j convention
- [x] Port `element_pattern(el_rad)` → g = sin(elevation)
- [x] Add `cal_offsets_deg` param to steering_vector — subtracts Δφ per element
- [x] Add `load_cal(cal_file)` / `save_cal()` — reads/writes cal.yaml
- [x] Add `build_steering_table(az_grid, el_deg, cal_offsets, include_pattern=False)`
- [x] Add `include_pattern` flag — False for MUSIC scan (sin(0)=0 at horizon)
- [x] All 7 inline checks pass
- [x] **SAVE CHECKPOINT** — commit message: "feat: dsp/geometry.py with cal offset support"

---

### 0.3 — dsp/covariance.py
> Source: music_spectrum.py line `R = (X @ X.conj().T) / n_samples`
> Key changes vs sim: add DC removal (RTL-SDR has DC spike), add diagonal loading formula

- [x] Implement `compute_covariance(X)` — DC removal + sample covariance
- [x] Implement `diagonal_load(R, n_signals, factor)` — adaptive δ = factor × mean(noise eigenvalues)
- [x] Implement `process(X, n_signals, factor)` — convenience wrapper returning R and R_dl
- [x] All 6 inline checks pass (DC invariance, Hermitian, positive definite, near-singular case)
- [x] **SAVE CHECKPOINT** — commit message: "feat: dsp/covariance.py with DC removal and diagonal loading"

---

### 0.4 — dsp/music.py
> Source: music_spectrum.py (eigendecompose, noise subspace, scan loop, find_top_peaks)
> Key changes vs sim: accepts precomputed steering table, returns bearing + spectrum

- [x] Implement `music_spectrum(R, steering_table, n_signals)` — vectorised scan, 0 dB normalised
- [x] Implement `find_top_peaks(spectrum_db, az_grid, n_signals, min_sep_deg)` — greedy separation
- [x] Implement `music_doa()` — full pipeline with threshold gate returning None on noise-only
- [x] All 5 checks pass: single jammer 0.00° error, two jammers exact, gate works both ways
- [x] **SAVE CHECKPOINT** — commit message: "feat: dsp/music.py MUSIC DOA with threshold gate"

---

### 0.5 — dsp/mvdr.py
> Source: mvdr_beamformer.py (diagonal loading + Cholesky solve)
> Key changes vs sim: look direction is FIXED from config (zenith), NOT estimated from data

- [x] Implement `mvdr_weights(R_dl, look_az_deg, look_el_deg, ...)` — Cholesky solve, normalised
- [x] Implement `beam_pattern(w, az_grid, el_deg)` — gain vs azimuth in dB, normalised to 0 dB peak
- [x] Distortionless constraint |w^H a_look - 1| = 2.8e-12 (machine precision)
- [x] Single jammer null = -48.6 dB, two jammers -51.3 / -71.4 dB (target was ≥30 dB)
- [x] **SAVE CHECKPOINT** — commit message: "feat: dsp/mvdr.py fixed look direction MVDR"

---

### 0.6 — dsp/beamform.py
> Source: mvdr_beamformer.py `y = w^H @ X`

- [x] Implement `apply_weights(w, X)` → y = w^H X, shape (N,), dtype complex128
- [x] Implement `null_depth_db(w, jammer_az_deg)` → gain at jammer rel. to look dir (dB)
- [x] Implement `passband_gain_db(w, look_az, look_el)` → gain at GPS look direction (dB)
- [x] Implement `suppression_db(X_raw, w)` → FFT peak power drop before/after (dB)
- [x] All 5 checks pass: passband=0.000 dB, null=-57.9 dB, suppression=33.4 dB
- [x] **SAVE CHECKPOINT** — commit message: "feat: dsp/beamform.py weight application and metrics"

---

### 0.7 — tests/test_synthetic.py
> Source: generate_array_data.py scene + waveforms used as test fixtures

- [x] 26 tests across T01–T16 covering geometry, covariance, MUSIC, MVDR, beamform, end-to-end
- [x] T12 parametrised over 7 azimuths: all null depths < -30 dB
- [x] T08 parametrised over 3 jammer pairs: all bearings within ±3°
- [x] T16 end-to-end at 3 azimuths: all pass
- [x] 26/26 passed in 0.37s
- [x] **SAVE CHECKPOINT** — commit message: "test: synthetic validation suite, 26/26 passing"

---

## PHASE 1 — Hardware Bring-Up
> Goal: KrakenSDR enumerated, Heimdall running, 4 channels coherent, cal.yaml written
> STOP if coherence check fails — do not proceed to Phase 2

---

### 1.1 — Install software stack on Ubuntu host
- [ ] `sudo apt install rtl-sdr librtlsdr-dev`
- [ ] `sudo pip install pyrtlsdr pyyaml`
- [ ] Clone Heimdall: `git clone https://github.com/krakenrf/heimdall_daq_fw.git`
- [ ] Build and verify Heimdall starts without errors
- [ ] Install GNSS-SDR from source (binary packages often outdated for L1 config)
- [ ] **SAVE CHECKPOINT** — commit message: "docs: software stack install verified"

---

### 1.2 — iq_source/heimdall_source.py
> Key lesson from mentor: background thread + lock + shared_data dict

- [x] Implement `HeimdallSource` class — TCP reader for Heimdall DAQ firmware
- [x] Auto-reconnect loop (retries forever by default)
- [x] Sync word search (0x11 0x22 0x33 0x44) for frame alignment
- [x] Parses: frame_idx, fc_hz, fs_hz, n_samp, n_ch from binary header
- [x] int8 IQ payload → complex64 (4, N) array
- [x] `stop_source()` — stops thread only; `stop()` — stops pipeline too
- [x] HEIMDALL_HOST / HEIMDALL_PORT env var overrides
- [x] **SAVE CHECKPOINT** — commit message: "feat: iq_source/heimdall_source.py KrakenSDR TCP reader"

---

### 1.3 — iq_source/file_source.py
- [ ] Implement `FileSource(filepath, n_channels=4, dtype=complex64)`:
  - Reads recorded .iq file in chunks of `n_snapshots`
  - Same interface as `heimdall_source` — yields `(4, N)` arrays
  - Loops file for continuous replay in dashboard test mode
- [ ] **SAVE CHECKPOINT** — commit message: "feat: iq_source/file_source.py offline replay"

---

### 1.4 — iq_source/recorder.py
- [ ] Implement `record(output_path, duration_sec)`:
  - Reads from `shared_data["rx_buffer"]` continuously
  - Writes interleaved complex64 to .iq file with header (n_channels, sample_rate, center_freq)
- [ ] **SAVE CHECKPOINT** — commit message: "feat: iq_source/recorder.py 4-ch IQ capture"

---

### 1.5 — calibration/calibrate.py
> Key lesson from mentor: cross-correlation phase measurement, persist to file

- [x] Standalone tool: `python calibration/calibrate.py [--source synthetic|heimdall] [--host] [--port]`
- [x] `measure_phase_offsets(X)` — cross-correlate ch1,2,3 against ch0 via angle(sum(ch0·conj(chk)))
- [x] `measure_amplitudes(X)` — RMS per channel, normalised to ch0
- [x] Live matplotlib UI: phase bars, phase history, amplitude bars, IQ constellation (4 subplots)
- [x] Rolling stability check: σ < 2° over last 20 frames → PASS (green)
- [x] [Save Calibration] button + S key → writes calibration/cal.yaml
- [x] **SAVE CHECKPOINT** — commit message: "feat: calibration/calibrate.py standalone phase cal tool"

---

### 1.6 — Coherence verification run
- [ ] Connect KrakenSDR, run Heimdall, confirm 4 channels in `lsusb`
- [ ] Inject CW tone via 4-way splitter into all 4 inputs
- [ ] Run `python calibration/calibrate.py` → verify cal.yaml written
- [ ] Confirm Δφ stable within ±2° over 60 seconds
- [ ] **SAVE CHECKPOINT** — commit message: "data: cal.yaml from coherence verification run"

---

## PHASE 2 — GPS Baseline (State A — Clean Sky)
> Goal: Confirm GPS acquisition on raw hardware before any jamming

---

### 2.1 — GNSS-SDR config file
- [ ] Create `gnss/antijam.conf` for GNSS-SDR:
  - Signal source: file (for offline) with TODO for live Heimdall source
  - GPS L1 C/A acquisition + tracking
  - C/N₀ log output path configured
- [ ] **SAVE CHECKPOINT** — commit message: "feat: gnss/antijam.conf GNSS-SDR config"

---

### 2.2 — gnss/parse_cn0.py
- [ ] Implement `parse_cn0_log(log_path)` → returns DataFrame: time, sv_id, cn0_db
- [ ] Implement `live_cn0_tail(log_path)` → generator yielding new rows as GNSS-SDR writes them
- [ ] **SAVE CHECKPOINT** — commit message: "feat: gnss/parse_cn0.py log parser"

---

### 2.3 — GPS baseline acquisition
- [ ] Feed one Heimdall channel into GNSS-SDR (raw, no beamforming)
- [ ] Confirm: ≥4 satellites acquired, C/N₀ ≈ 38–45 dB-Hz, position fix
- [ ] Save baseline C/N₀ log as `recordings/baseline_cn0.log`
- [ ] **SAVE CHECKPOINT** — commit message: "data: baseline GPS acquisition confirmed, C/N0 log saved"

---

## PHASE 3 — Record IQ Clips

---

### 3.1 — Record clean clip
- [ ] Run `python iq_source/recorder.py --output recordings/clean.iq --duration 30`
- [ ] Verify file written, correct shape when loaded
- [ ] **SAVE CHECKPOINT** — commit message: "data: clean.iq recorded (no jammer)"

---

### 3.2 — Record jammed clip
- [ ] Connect conducted CW jammer via combiner + calibrated attenuator (keep below +10 dBm at Kraken port)
- [ ] Verify C/N₀ collapses in GNSS-SDR before recording
- [ ] Run `python iq_source/recorder.py --output recordings/jammed.iq --duration 30`
- [ ] **SAVE CHECKPOINT** — commit message: "data: jammed.iq recorded, C/N0 collapse confirmed"

---

## PHASE 4 — Offline Anti-Jam Pipeline (States B → C)
> Goal: Prove the math works on real recorded hardware data before going real-time

---

### 4.1 — pipeline/offline.py
- [ ] Implement full chain:
  `FileSource → covariance → MUSIC → MVDR → beamform → write combined.iq`
- [ ] Add per-frame logging: doa_est, suppression_db, latency_ms to CSV
- [ ] **SAVE CHECKPOINT** — commit message: "feat: pipeline/offline.py end-to-end processing chain"

---

### 4.2 — Run offline pipeline on recordings
- [ ] `python pipeline/offline.py --input recordings/jammed.iq --output recordings/combined.iq`
- [ ] Feed `jammed.iq` into GNSS-SDR → confirm lock lost (State B)
- [ ] Feed `combined.iq` into GNSS-SDR → confirm lock recovered (State C)
- [ ] **SAVE CHECKPOINT** — commit message: "result: offline anti-jam pipeline validated, GPS recovery confirmed"

---

### 4.3 — plots/before_after.py
- [ ] Generate 3-state C/N₀ plot: clean / jammed / recovered
- [ ] Save at 300 DPI with `bbox_inches='tight'`
- [ ] **SAVE CHECKPOINT** — commit message: "feat: plots/before_after.py A/B/C state comparison"

---

## PHASE 5 — Real-Time DSP Engine
> Build the streaming processing loop, test on recorded data before live hardware

---

### 5.1 — pipeline/realtime.py (DSP engine, no GUI)
- [ ] Implement `dsp_engine()` daemon thread:
  - Reads `shared_data["rx_buffer"]`
  - Runs: covariance → MUSIC → MVDR → beamform
  - Writes back: doa_est, beam_weights, beam_pattern, suppression_db, latency_ms, beamformed_iq
  - Interference threshold gate: skip DSP if no jammer detected
- [ ] Implement `start_engine()` / `stop_engine()`
- [ ] **SAVE CHECKPOINT** — commit message: "feat: pipeline/realtime.py streaming DSP engine"

---

### 5.2 — Test realtime engine on file_source
- [ ] Run realtime engine fed by `FileSource(jammed.iq)` in replay loop
- [ ] Verify `shared_data` fields populate correctly
- [ ] Verify latency < 100ms per cycle on host machine
- [ ] **SAVE CHECKPOINT** — commit message: "test: realtime DSP engine validated on recorded data"

---

## PHASE 6 — Live Dashboard
> Build panels fed from shared_data. Test on file_source first, then swap to heimdall_source

---

### 6.1 — dashboard/spectrum_panel.py (Panel 1)
- [ ] Live plot: raw spectrum (red) + beamformed spectrum (blue) + noise floor (black)
- [ ] Bottom metrics bar: Peak Before, Peak After, Suppression dB, JNR Before, JNR After
- [ ] Interference detected / AJ ACTIVE status label
- [ ] Updates via `root.after(300, update_loop)` — never blocks GUI thread
- [ ] **SAVE CHECKPOINT** — commit message: "feat: dashboard/spectrum_panel.py live spectrum"

---

### 6.2 — dashboard/doa_panel.py (Panel 2)
- [ ] Live MUSIC pseudospectrum plot (teal line)
- [ ] Vertical marker at estimated jammer bearing
- [ ] HPBW shading
- [ ] Annotation: "Jammer DOA: X.X°"
- [ ] **SAVE CHECKPOINT** — commit message: "feat: dashboard/doa_panel.py live MUSIC spectrum"

---

### 6.3 — dashboard/beam_panel.py (Panel 3)
- [ ] Live MVDR beam pattern plot (gain dB vs azimuth)
- [ ] Red dashed line at null direction (jammer)
- [ ] Green dashed line at look direction (GPS / zenith)
- [ ] Null depth annotation: "Null: X.X dB"
- [ ] **SAVE CHECKPOINT** — commit message: "feat: dashboard/beam_panel.py live beam pattern"

---

### 6.4 — dashboard/cn0_panel.py (Panel 4)
- [ ] Rolling 60-second C/N₀ timeline from GNSS-SDR log
- [ ] Per-satellite lines or mean C/N₀
- [ ] Horizontal reference line at baseline (State A)
- [ ] Shade regions: clean / jammed / recovered
- [ ] **SAVE CHECKPOINT** — commit message: "feat: dashboard/cn0_panel.py live C/N0 timeline"

---

### 6.5 — dashboard/app.py (Main window)
- [ ] 2×2 grid layout: 4 panels
- [ ] Top bar: system status, latency readout, Enable AJ checkbox, CAL button
- [ ] Bottom bar: all numeric metrics
- [ ] CAL button → runs `calibrate.py` in background thread, updates cal.yaml
- [ ] Enable AJ checkbox → sets `shared_data["beamforming_active"]`
- [ ] Graceful shutdown: sets `shared_data["running"] = False`, joins threads
- [ ] **SAVE CHECKPOINT** — commit message: "feat: dashboard/app.py main window, all panels integrated"

---

### 6.6 — Test dashboard on file_source (offline mode)
- [ ] Run `python dashboard/app.py --source file --input recordings/jammed.iq`
- [ ] Verify all 4 panels render and update
- [ ] Verify Enable AJ toggle shows suppression in Panel 1
- [ ] **SAVE CHECKPOINT** — commit message: "test: dashboard verified on recorded data offline mode"

---

## PHASE 7 — Live Hardware Integration
> Swap file_source → heimdall_source. Everything else unchanged.

---

### 7.1 — Connect Heimdall to dashboard
- [ ] Run Heimdall in Terminal 1
- [ ] Run `python dashboard/app.py --source heimdall`
- [ ] Verify Panel 1 shows live spectrum
- [ ] Verify Panel 2 updates with real DOA
- [ ] Verify Panel 3 shows real beam pattern
- [ ] **SAVE CHECKPOINT** — commit message: "integration: dashboard live on Heimdall hardware"

---

### 7.2 — Connect GNSS-SDR to beamformed output (Panel 4 goes live)
- [ ] Configure GNSS-SDR to read from beamformed IQ pipe
- [ ] Run `python gnss/feed_gnss_sdr.py`
- [ ] Verify Panel 4 shows live C/N₀
- [ ] **SAVE CHECKPOINT** — commit message: "integration: GNSS-SDR connected to beamformed output"

---

### 7.3 — Full system test: jammer on/off
- [ ] Confirm State A (clean): C/N₀ ≈ 38–45 dB-Hz, lock on ≥4 SVs
- [ ] Inject jammer (conducted): confirm C/N₀ collapse (State B) shown on Panel 4
- [ ] Enable AJ on dashboard: confirm C/N₀ recovers (State C) shown on Panel 4
- [ ] Confirm Panel 2 shows correct jammer bearing
- [ ] Confirm Panel 3 shows null at jammer bearing
- [ ] **SAVE CHECKPOINT** — commit message: "milestone: full end-to-end anti-jam demo working"

---

## PHASE 8 — Characterisation

---

### 8.1 — Null depth vs jammer power sweep
- [ ] Sweep conducted jammer power in 5 dB steps
- [ ] Record suppression_db at each step
- [ ] Plot: null depth (dB) vs J/S (dB)
- [ ] **SAVE CHECKPOINT** — commit message: "result: null depth vs J/S characterisation"

---

### 8.2 — MUSIC bearing accuracy
- [ ] Inject jammer at known angles (phase taper or shielded box)
- [ ] Record estimated vs true bearing
- [ ] Plot: bearing error (°) vs true angle
- [ ] **SAVE CHECKPOINT** — commit message: "result: MUSIC bearing accuracy characterisation"

---

### 8.3 — Jammer type comparison (CW / FMCW / Barrage)
- [ ] Repeat null depth measurement for each waveform type
- [ ] Record suppression_db for each
- [ ] **SAVE CHECKPOINT** — commit message: "result: jammer type comparison complete"

---

### 8.4 — Latency jitter over 60 seconds
- [ ] Log latency_ms for 60s continuous operation
- [ ] Plot: latency jitter timeline
- [ ] Confirm no dropped frames
- [ ] **SAVE CHECKPOINT** — commit message: "result: latency stability characterisation"

---

## ACCEPTANCE CRITERIA (final checklist)

- [ ] `pytest tests/test_synthetic.py` — 100% pass
- [ ] GNSS-SDR loses lock on `jammed.iq` (State B confirmed)
- [ ] GNSS-SDR regains lock on `combined.iq` (State C confirmed offline)
- [ ] Live dashboard opens with `python dashboard/app.py --source heimdall`
- [ ] All 4 panels update in real time with KrakenSDR connected
- [ ] C/N₀ drops and recovers live when jammer toggled (Panel 4)
- [ ] Null depth ≥ 30 dB on single CW jammer
- [ ] Latency < 100 ms per cycle over 60 s continuous operation
- [ ] Phase 8 characterisation plots saved at 300 DPI

---

## HOW TO RESUME AFTER CONTEXT LIMIT

1. Open new Claude Code session in `/home/itsstoomuch/projects/karken`
2. Say: **"read ksdr.md and continue from the last unchecked item"**
3. Claude will read this file, find the first unchecked `- [ ]`, and continue

---

*Last updated: 2026-06-23*
