"""
iq_source/recorder.py

Record 4-channel IQ from any running source to a .iq file.

File format matches file_source.py:
  Header (32 bytes): n_channels(u32), sample_rate(f64), center_freq(f64), pad(8)
  Payload: interleaved complex64 — ch0_s0, ch1_s0, ..., ch0_s1, ...
"""

import struct
import time
import numpy as np

from iq_source.synthetic_source import shared_data, _lock

HEADER_FMT = "<I d d 8x"


def record(
    output_path:  str   = "recordings/capture.iq",
    duration_sec: float = 30.0,
    sample_rate:  float = 2.4e6,
    center_freq:  float = 1575.42e6,
    n_channels:   int   = 4,
):
    """
    Record IQ frames from shared_data["rx_buffer"] for the specified duration.

    Must be called while an IQ source (synthetic, heimdall, file) is running
    and writing to shared_data.

    Parameters
    ----------
    output_path  : output .iq file path
    duration_sec : recording duration in seconds
    sample_rate  : sample rate to write in header
    center_freq  : centre frequency to write in header
    n_channels   : number of channels
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    header = struct.pack(HEADER_FMT, n_channels, sample_rate, center_freq)

    frames_written = 0
    last_buf = None
    t_start = time.time()

    with open(output_path, "wb") as fh:
        fh.write(header)

        while time.time() - t_start < duration_sec and shared_data["running"]:
            with _lock:
                buf = shared_data.get("rx_buffer")

            if buf is None or buf is last_buf:
                time.sleep(0.01)
                continue

            last_buf = buf
            X = buf.astype(np.complex64)
            # Transpose to (n_snapshots, n_channels), then flatten interleaved
            fh.write(X.T.tobytes())
            frames_written += 1

    elapsed = time.time() - t_start
    n_samples = frames_written * (buf.shape[1] if buf is not None else 0)
    print(f"  Recorded {frames_written} frames ({n_samples} samples) "
          f"in {elapsed:.1f}s → {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record 4-ch IQ to file")
    parser.add_argument("--output", default="recordings/capture.iq")
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    print("  Waiting for IQ source to start writing to shared_data...")
    print("  (Run this while dashboard or another source is active)")
    record(output_path=args.output, duration_sec=args.duration)
