"""Settings and constants."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Settings:
    disappear_after_s: float = 30.0   # a device is reported as "not seen" after this many seconds of silence
    rssi_window_s: float = 10.0       # rolling window for averages and variability
    rssi_history: int = 900           # stored RSSI samples per device
    show_names: bool = False          # advertised names can contain personal names, so they stay hidden by default
    save_to_db: bool = False          # default: memory only, nothing written to disk
    db_path: str = "device_detector.db"
    retention_days: int = 0           # 0 = keep until the user clears. Applies to the saved database only.
    camera_index: int = 0
    model_path: str = ""
    camera_conf: float = 0.40
    camera_fps: float = 5.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Settings":
        return cls(**json.loads(text))


CATEGORIES = ["Smartphone", "Wireless audio accessory", "Smartwatch", "Fitness device", "Computer",
              "Other Bluetooth accessory", "Unknown device"]
