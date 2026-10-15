"""BLE scanner built on Bleak (Windows Bluetooth stack). Passive reception of advertisements only.

The scanner never connects to a device, never reads characteristics, and hashes addresses immediately.
Bleak is imported only when a scan starts, so the rest of the program works without it.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time

from .adparse import observation_from_bleak
from .identifiers import IdMapper


def _import_bleak():
    import bleak
    return bleak


def bluetooth_status() -> tuple[str, str]:
    """Returns (state, message). state: available | missing_library | unknown. Whether the adapter is switched on
    is only known when a scan starts."""
    try:
        _import_bleak()
    except Exception as e:  # ImportError, or a broken Windows runtime
        return "missing_library", f"The Bleak library could not be loaded ({type(e).__name__}). Install it with: pip install bleak"
    return "available", "Bleak is installed. Adapter state is checked when scanning starts."


def explain_error(e: Exception) -> tuple[str, str]:
    text = str(e)
    low = text.lower()
    if "turned off" in low or "radio" in low or "not powered" in low:
        return "bluetooth_off", "Bluetooth is switched off. Turn it on in Windows Settings, then press Start."
    if "access" in low and "denied" in low or "permission" in low or "unauthorized" in low:
        return "permission_denied", "Windows denied Bluetooth access. Open Settings, Privacy and security, Bluetooth, and allow desktop apps."
    if "no bluetooth adapter" in low or "no adapter" in low or "not found" in low:
        return "no_adapter", "No Bluetooth adapter was found."
    return "error", f"Bluetooth scan error: {text[:160]}"


class BleScanner:
    """Runs Bleak in a background thread and puts Observation objects on a queue."""

    label = "Bluetooth LE (real hardware)"
    synthetic = False

    def __init__(self, ids: IdMapper | None = None, max_queue: int = 5000):
        self.ids = ids or IdMapper()
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.state = "stopped"
        self.message = ""
        self.dropped = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()

    def _cb(self, device, adv):
        key = self.ids.key(device.address)
        obs = observation_from_bleak(time.monotonic(), key, adv)
        try:
            self.q.put_nowait(obs)
        except queue.Full:
            self.dropped += 1

    async def _run(self):
        bleak = _import_bleak()
        async with bleak.BleakScanner(detection_callback=self._cb):
            self.state, self.message = "scanning", "Scanning for BLE advertisements."
            self._started.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.2)

    def _thread_main(self):
        try:
            asyncio.run(self._run())
            self.state = "stopped"
        except Exception as e:
            self.state, self.message = explain_error(e)
            self._started.set()

    def start(self, timeout: float = 6.0) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        st, msg = bluetooth_status()
        if st != "available":
            self.state, self.message = st, msg
            return False
        self._stop.clear()
        self._started.clear()
        self.state, self.message = "starting", "Starting scan..."
        self._thread = threading.Thread(target=self._thread_main, daemon=True, name="ble-scan")
        self._thread.start()
        self._started.wait(timeout)
        return self.state == "scanning"

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        if self.state in ("scanning", "starting"):
            self.state, self.message = "stopped", "Scan stopped."

    def read_all(self, limit: int = 2000):
        out = []
        for _ in range(limit):
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out
