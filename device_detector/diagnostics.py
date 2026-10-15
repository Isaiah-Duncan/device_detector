"""Sensor status and an honest list of what the sensors cannot detect."""
from __future__ import annotations

from dataclasses import dataclass, field

DISCLOSURES = [
    "BLE scanning hears only devices that are advertising. Phones and headsets often advertise rarely, or not at all, while connected or idle.",
    "Bluetooth Classic devices (many headsets and speakers in an active audio connection) are not found by a BLE scan.",
    "Wi-Fi, cellular, NFC, ultra-wideband and wired devices are not detected by this program.",
    "Randomized addresses change often, so the count of identifiers is an upper bound on the number of devices.",
    "RSSI depends on transmit power, antennas, bodies, walls and reflections. It is not a distance.",
    "Device types come only from advertised metadata. A manufacturer identifier alone does not identify a phone or headset.",
    "The webcam model recognizes shapes it was trained on (for example 'cell phone' and 'laptop') in view only. It does not see devices in pockets, bags or other rooms, and common models have no 'tablet' class.",
    "Not detecting something is inconclusive. It is never proof of absence.",
]


@dataclass
class SensorStatus:
    bluetooth: str = "unknown"
    bluetooth_msg: str = "Not started."
    scan_permission: str = "unknown"
    camera: str = "off"
    camera_msg: str = "Camera is off. It starts only when you press Start camera."
    model: str = "none"
    model_msg: str = "No object-detection model loaded."
    unsupported: list[str] = field(default_factory=lambda: [
        "Bluetooth Classic discovery", "Reading or decrypting any device communication", "Identifying people or owners",
        "Reliable distance measurement", "Detecting devices that are not advertising"])


def status_lines(s: SensorStatus) -> list[tuple[str, str, str]]:
    """Returns (indicator, name, message). Indicator is 'ok', 'warn' or 'off'."""
    ind = {"scanning": "ok", "available": "ok", "stopped": "off", "unknown": "off", "starting": "warn"}
    return [
        (ind.get(s.bluetooth, "warn"), "Bluetooth LE", s.bluetooth_msg),
        ("warn" if s.scan_permission in ("permission_denied",) else "off", "Scan permission", "Windows reports it when scanning starts." if s.scan_permission == "unknown" else s.scan_permission),
        ("ok" if s.camera == "on" else "warn" if s.camera == "error" else "off", "Webcam", s.camera_msg),
        ("ok" if s.model == "loaded" else "off", "Object model", s.model_msg),
    ]
