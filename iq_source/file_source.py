"""
iq_source/file_source.py

Offline IQ replay source — reads a recorded .iq file and yields
(4, N) complex frames at a configurable rate, looping on EOF.

File format:
  Header (32 bytes):
    n_channels : uint32
    sample_rate: float64
    center_freq: float64
    reserved   : 8 bytes zero-pad
  Payload:
    Interleaved complex64 samples: ch0_s0, ch1_s0, ..., ch0_s1, ch1_s1, ...

Also supports raw headerless files (n_channels and dtype passed manually).
"""

import threading
import time
import struct
import numpy as np

from iq_source.synthetic_source import shared_data, _lock

HEADER_FMT  = "<I d d 8x"      # n_ch, fs, fc, padding
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class FileSource:
    """
    Replay recorded IQ from a file, writing frames to shared_data.

    Parameters
    ----------
    filepath    : path to the .iq recording
    n_channels  : number of channels (overrides header if raw file)
    n_snapshots : samples per frame
    dtype       : numpy dtype of the payload samples
    loop        : if True, restart from beginning on EOF
    update_hz   : frames per second (throttle replay speed)
    has_header  : if True, read the 32-byte header for metadata
    """

    def __init__(
        self,
        filepath:    str   = "recordings/jammed.iq",
        n_channels:  int   = 4,
        n_snapshots: int   = 256,
        dtype:       str   = "complex64",
        loop:        bool  = True,
        update_hz:   float = 20.0,
        has_header:  bool  = True,
    ):
        self.filepath    = filepath
        self.n_channels  = n_channels
        self.n_snapshots = n_snapshots
        self.dtype       = np.dtype(dtype)
        self.loop        = loop
        self.update_hz   = update_hz
        self.has_header  = has_header
        self._thread     = None
        self._running    = False

        self.fs = None
        self.fc = None

    def _read_header(self, fh):
        raw = fh.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            return False
        n_ch, fs, fc = struct.unpack(HEADER_FMT, raw)
        self.n_channels = n_ch
        self.fs = fs
        self.fc = fc
        return True

    def _run(self):
        interval    = 1.0 / self.update_hz
        frame_bytes = self.n_channels * self.n_snapshots * self.dtype.itemsize

        while shared_data["running"] and self._running:
            try:
                with open(self.filepath, "rb") as fh:
                    if self.has_header:
                        if not self._read_header(fh):
                            break

                    while shared_data["running"] and self._running:
                        t0  = time.perf_counter()
                        raw = fh.read(frame_bytes)

                        if len(raw) < frame_bytes:
                            if self.loop:
                                fh.seek(HEADER_SIZE if self.has_header else 0)
                                continue
                            else:
                                break

                        samples = np.frombuffer(raw, dtype=self.dtype)
                        # Reshape: interleaved channels → (n_channels, n_snapshots)
                        X = samples.reshape(self.n_snapshots, self.n_channels).T
                        X = X.astype(np.complex64)

                        with _lock:
                            shared_data["rx_buffer"]   = X
                            shared_data["source_type"] = "file"

                        elapsed = time.perf_counter() - t0
                        time.sleep(max(0.0, interval - elapsed))

            except FileNotFoundError:
                import logging
                logging.error(f"FileSource: file not found: {self.filepath}")
                break

            if not self.loop:
                break

    def start(self):
        self._running = True
        shared_data["source_type"] = "file"
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="file-source"
        )
        self._thread.start()

    def stop_source(self):
        self._running = False

    def stop(self):
        self._running = False
        shared_data["running"] = False
