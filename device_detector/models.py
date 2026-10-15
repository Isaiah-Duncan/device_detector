"""Plain data classes shared by all modules."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Observation:
    """One received BLE advertisement, reduced to the metadata this program needs.

    The raw payload is never stored. Manufacturer data is reduced to the company identifier.
    `key` is a temporary identifier derived from the address with a random per-session key.
    """
    ts: float
    key: str
    rssi: int | None
    name: str | None = None
    uuids: tuple[str, ...] = ()
    company_ids: tuple[int, ...] = ()
    service_data_uuids: tuple[str, ...] = ()
    tx_power: int | None = None
    appearance: int | None = None
    synthetic: bool = False


@dataclass
class Event:
    ts: float
    kind: str            # discovered | disappeared | rediscovered | reclassified
    key: str
    text: str
    category: str = "Unknown device"
    confidence: str = "none"
    synthetic: bool = False


@dataclass
class Classification:
    category: str = "Unknown device"
    confidence: str = "none"                     # none | low | medium | high
    reasons: list[str] = field(default_factory=list)
