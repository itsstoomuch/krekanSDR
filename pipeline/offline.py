"""
pipeline/offline.py

Offline anti-jam processing pipeline.

Reads a recorded .iq file, runs the full DSP chain (covariance → MUSIC →
MVDR → beamform) on each frame, and writes the beamformed single-channel
output to a new .iq file suitable for feeding to GNSS-SDR.

Also logs per-frame metrics (DOA, suppression, null depth, latency) to CSV.
"""

import os
import struct
import time
import csv
import numpy as np
import yaml

from dsp.covariance import process as cov_process
from dsp.geometry   import build_az_grid, build_steering_table, load_cal
from dsp.music      import music_doa
from dsp.mvdr       import mvdr_weights
from dsp.beamform   import apply_weights, null_depth_db, suppression_db

HEADER_FMT  = "<I d d 8x"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def run_offline(
    input_file:  str = "recordings/jammed.iq",
    output_file: str = "recordings/combined.iq",
    log_file:    str = "recordings/pipeline_log.csv",
    config_path: str = "config.yaml",
):
    """
    Process a recorded .iq file through the full anti-jam pipeline.

    Parameters
    ----------
    input_file  : path to recorded 4-channel .iq file
    output_file : path to write beamformed 1-channel .iq output
    log_file    : path to write per-frame metrics CSV
    config_path : path to config.yaml
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dsp_cfg = cfg["dsp"]
    rf_cfg  = cfg["rf"]
    arr_cfg = cfg["array"]

    # Build scan assets
    az_grid = build_az_grid(
        dsp_cfg["az_scan_start_deg"],
        dsp_cfg["az_scan_stop_deg"],
        dsp_cfg["az_scan_step_deg"],
    )
    cal_offsets = load_cal(cfg["calibration"]["cal_file"])
    steer_table = build_steering_table(
        az_grid, el_deg=0.0,
        freq_hz=rf_cfg["center_freq_hz"],
        spacing_m=arr_cfg["spacing_m"],
        cal_offsets_deg=cal_offsets,
        include_pattern=False,
    )

    n_signals    = dsp_cfg["n_signals"]
    load_factor  = dsp_cfg["diag_load_factor"]
    threshold_db = dsp_cfg.get("music_threshold_db", 8.0)
    look_az      = cfg["look_direction"]["azimuth_deg"]
    look_el      = cfg["look_direction"]["elevation_deg"]
    freq_hz      = rf_cfg["center_freq_hz"]
    spacing_m    = arr_cfg["spacing_m"]
    n_snapshots  = dsp_cfg["n_snapshots"]

    # Read input header
    with open(input_file, "rb") as fh:
        hdr = fh.read(HEADER_SIZE)
        n_ch, fs, fc = struct.unpack(HEADER_FMT, hdr)
        payload = fh.read()

    samples = np.frombuffer(payload, dtype=np.complex64)
    total_samples = len(samples) // n_ch
    n_frames = total_samples // n_snapshots

    print(f"  Input:  {input_file}")
    print(f"    Channels: {n_ch}, Fs: {fs/1e6:.1f} MHz, Fc: {fc/1e6:.2f} MHz")
    print(f"    Samples: {total_samples}, Frames: {n_frames}")

    # Reshape all data
    usable = n_frames * n_snapshots * n_ch
    data = samples[:usable].reshape(n_frames, n_snapshots, n_ch)
    # Transpose each frame to (n_ch, n_snapshots)

    # Output header (1 channel)
    out_header = struct.pack(HEADER_FMT, 1, fs, fc)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    with open(output_file, "wb") as out_fh, \
         open(log_file, "w", newline="") as csv_fh:

        out_fh.write(out_header)
        writer = csv.writer(csv_fh)
        writer.writerow(["frame", "doa_deg", "suppression_db",
                         "null_depth_db", "latency_ms"])

        n_jammed = 0
        n_clean  = 0

        for i in range(n_frames):
            t0 = time.perf_counter()

            X = data[i].T.astype(np.complex128)   # (n_ch, n_snapshots)

            # Covariance
            R, R_dl = cov_process(X, n_signals, load_factor)

            # MUSIC DOA
            spec, peaks = music_doa(
                R, steer_table, az_grid,
                n_signals=n_signals,
                threshold_db=threshold_db,
            )

            if peaks is None:
                # No jammer — pass through channel sum
                y = X.sum(axis=0).astype(np.complex64)
                doa, supp, nd = None, 0.0, 0.0
                n_clean += 1
            else:
                doa = float(peaks[0])

                # MVDR
                w = mvdr_weights(
                    R_dl, look_az_deg=look_az, look_el_deg=look_el,
                    freq_hz=freq_hz, spacing_m=spacing_m,
                    cal_offsets_deg=cal_offsets,
                )

                # Beamform
                y = apply_weights(w, X).astype(np.complex64)

                # Metrics
                supp = suppression_db(X, w)
                nd = null_depth_db(w, doa, freq_hz=freq_hz,
                                   spacing_m=spacing_m,
                                   cal_offsets_deg=cal_offsets)
                n_jammed += 1

            latency = (time.perf_counter() - t0) * 1000

            # Write beamformed output
            out_fh.write(y.tobytes())

            # Log metrics
            writer.writerow([
                i,
                f"{doa:.2f}" if doa is not None else "",
                f"{supp:.1f}",
                f"{nd:.1f}",
                f"{latency:.2f}",
            ])

    print(f"  Output: {output_file}")
    print(f"    Frames processed: {n_frames}")
    print(f"    Jammer detected:  {n_jammed} frames")
    print(f"    Clean (no jammer):{n_clean} frames")
    print(f"    Log: {log_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline anti-jam pipeline")
    parser.add_argument("--input",  default="recordings/jammed.iq")
    parser.add_argument("--output", default="recordings/combined.iq")
    parser.add_argument("--log",    default="recordings/pipeline_log.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    run_offline(
        input_file=args.input,
        output_file=args.output,
        log_file=args.log,
        config_path=args.config,
    )
