"""
iq_source/heimdall_source.py

KrakenSDR / Heimdall DAQ IQ source.
Reads coherent 5-channel IQ from the Heimdall DAQ firmware over TCP and
exposes the first 4 channels to the DSP pipeline via shared_data.

──────────────────────────────────────────────────────────────────────────
Heimdall DAQ TCP frame format  (heimdall_daq_fw ≥ 1.0, little-endian):

  Offset  Size  Type       Field
  ──────  ────  ─────────  ─────────────────────────────────────────────
     0      4   uint8[4]   Sync word  0x11 0x22 0x33 0x44
     4      4   uint32     Frame index
     8      8   float64    Centre frequency (Hz)
    16      8   float64    Sample rate (Hz)
    24      4   uint32     N samples per channel
    28      4   uint32     N channels  (5 for KrakenSDR)
    32    N×Nch×2  int8    IQ payload — per-channel blocks:
                             ch0: I₀ Q₀ I₁ Q₁ … I_{N-1} Q_{N-1}
                             ch1: …
                             …

Default host/port matches the heimdall_daq_fw default configuration.
If your setup differs, pass host= and port= to the constructor or set
HEIMDALL_HOST / HEIMDALL_PORT environment variables.
──────────────────────────────────────────────────────────────────────────
"""

import os
import socket
import struct
import threading
import logging
import time
import numpy as np

from iq_source.synthetic_source import shared_data, _lock

log = logging.getLogger(__name__)

_SYNC        = b'\x11\x22\x33\x44'
_HEADER_FMT  = "<IddII"          # frame_idx, fc, fs, n_samp, n_ch
_HEADER_BODY = struct.calcsize(_HEADER_FMT)   # 28 bytes after sync
_MAX_SEARCH  = 8192
_RETRY_DELAY = 2.0
_MAX_RETRIES = 0                 # 0 = retry forever


class HeimdallSource:
    """
    Reads coherent 4-ch IQ from a running Heimdall DAQ instance.

    Parameters
    ----------
    host        : Heimdall host  (default: env HEIMDALL_HOST or "localhost")
    port        : Heimdall port  (default: env HEIMDALL_PORT or 5555)
    n_use_ch    : how many channels to pass downstream (max 4 for CRPA)
    timeout_s   : socket read timeout
    """

    def __init__(
        self,
        host:      str   = None,
        port:      int   = None,
        n_use_ch:  int   = 4,
        timeout_s: float = 5.0,
    ):
        self.host      = host or os.getenv("HEIMDALL_HOST", "localhost")
        self.port      = int(port or os.getenv("HEIMDALL_PORT", "5555"))
        self.n_use_ch  = n_use_ch
        self.timeout_s = timeout_s
        self._sock     = None
        self._thread   = None
        self._running  = False

    # ── Socket helpers ─────────────────────────────────────────────────────

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Heimdall socket closed unexpectedly")
            buf += chunk
        return bytes(buf)

    def _connect(self) -> bool:
        attempt = 0
        while shared_data["running"] and self._running:
            attempt += 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout_s)
                sock.connect((self.host, self.port))
                self._sock = sock
                log.info("Heimdall: connected to %s:%d", self.host, self.port)
                shared_data["source_type"] = "heimdall"
                return True
            except OSError as exc:
                log.warning("Heimdall: connect attempt %d failed — %s", attempt, exc)
                if _MAX_RETRIES and attempt >= _MAX_RETRIES:
                    return False
                time.sleep(_RETRY_DELAY)
        return False

    def _close_socket(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    # ── Frame parser ───────────────────────────────────────────────────────

    def _find_sync(self) -> bool:
        """Scan the byte stream until the 4-byte sync word is aligned."""
        window = bytearray(self._recv_exact(4))
        searched = 4
        while bytes(window) != _SYNC:
            window = window[1:] + bytearray(self._recv_exact(1))
            searched += 1
            if searched > _MAX_SEARCH:
                log.warning("Heimdall: sync not found in %d bytes", _MAX_SEARCH)
                return False
        return True

    def _read_frame(self) -> np.ndarray:
        """
        Read one Heimdall frame and return (4, N) complex64 array.
        Raises ConnectionError / struct.error on protocol violations.
        """
        if not self._find_sync():
            raise ConnectionError("Lost sync")

        # Header body (28 bytes after the 4-byte sync word)
        hdr_raw = self._recv_exact(_HEADER_BODY)
        frame_idx, fc_hz, fs_hz, n_samp, n_ch = struct.unpack(_HEADER_FMT, hdr_raw)

        if n_samp < 16 or n_samp > 65536:
            raise ValueError(f"Suspicious n_samp={n_samp} — sync drift?")
        if n_ch < 1 or n_ch > 8:
            raise ValueError(f"Suspicious n_ch={n_ch}")

        # IQ payload
        payload = self._recv_exact(n_samp * n_ch * 2)
        raw = np.frombuffer(payload, dtype=np.int8).astype(np.float32)
        raw /= 128.0                            # normalise to ±1.0

        # Reshape to (n_ch, n_samp, 2) → complex (n_ch, n_samp)
        iq = raw.reshape(n_ch, n_samp, 2)
        X  = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64)

        # Return only the channels the CRPA array uses
        return X[: self.n_use_ch]

    # ── Main thread ────────────────────────────────────────────────────────

    def _run(self):
        if not self._connect():
            log.error("Heimdall: could not connect — source thread exiting")
            return

        while shared_data["running"] and self._running:
            try:
                frame = self._read_frame()
                with _lock:
                    shared_data["rx_buffer"] = frame
            except (ConnectionError, OSError) as exc:
                log.warning("Heimdall: read error — %s  (reconnecting…)", exc)
                self._close_socket()
                if not self._connect():
                    break
            except (struct.error, ValueError) as exc:
                log.warning("Heimdall: protocol error — %s  (resyncing)", exc)
                # Don't reconnect — just re-try sync on next iteration
            except Exception as exc:
                log.error("Heimdall: unexpected error — %s", exc)
                time.sleep(0.1)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True,
                                         name="heimdall-source")
        self._thread.start()

    def stop_source(self):
        """Stop this source thread only — DSP pipeline keeps running."""
        self._running = False
        self._close_socket()

    def stop(self):
        """Stop source AND signal the entire pipeline to exit."""
        self._running = False
        shared_data["running"] = False
        self._close_socket()
