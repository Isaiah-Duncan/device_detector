"""BLE advertisement parsing (raw AD structures and Bleak objects).

Only metadata is kept: flags, service UUIDs, the advertised name, TX power, appearance, the manufacturer
company identifier (not its payload), and the UUIDs of service data (not the data itself).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

BASE_UUID_SUFFIX = "-0000-1000-8000-00805f9b34fb"


@dataclass
class ParsedAd:
    flags: int | None = None
    uuids: list[str] = field(default_factory=list)
    name: str | None = None
    tx_power: int | None = None
    appearance: int | None = None
    company_id: int | None = None
    manufacturer_len: int = 0
    service_data_uuids: list[str] = field(default_factory=list)
    malformed: bool = False


def normalize_uuid(text: str) -> str:
    """'0000180d-0000-1000-8000-00805f9b34fb' -> '180d'. Other UUIDs stay in full, in lower case."""
    t = text.lower()
    if t.endswith(BASE_UUID_SUFFIX) and t.startswith("0000") and len(t) == 36:
        return t[4:8]
    return t


def _uuid16(b: bytes) -> str:
    return f"{int.from_bytes(b, 'little'):04x}"


def _uuid32(b: bytes) -> str:
    return f"{int.from_bytes(b, 'little'):08x}"


def _uuid128(b: bytes) -> str:
    return str(uuid.UUID(bytes=bytes(reversed(b))))


def parse_ad_structures(data: bytes) -> ParsedAd:
    """Parse a raw advertising payload (a sequence of length, type, value structures)."""
    out = ParsedAd()
    i, n = 0, len(data)
    while i < n:
        ln = data[i]
        if ln == 0:                      # padding: the rest is empty
            break
        if i + 1 + ln > n:               # structure runs past the end of the payload
            out.malformed = True
            break
        t, val = data[i + 1], data[i + 2:i + 1 + ln]
        if t == 0x01 and len(val) >= 1:
            out.flags = val[0]
        elif t in (0x02, 0x03):
            out.uuids += [_uuid16(val[k:k + 2]) for k in range(0, len(val) - 1, 2)]
        elif t in (0x04, 0x05):
            out.uuids += [_uuid32(val[k:k + 4]) for k in range(0, len(val) - 3, 4)]
        elif t in (0x06, 0x07):
            out.uuids += [_uuid128(val[k:k + 16]) for k in range(0, len(val) - 15, 16)]
        elif t in (0x08, 0x09):
            out.name = val.decode("utf-8", "replace")
        elif t == 0x0A and len(val) == 1:
            out.tx_power = int.from_bytes(val, "little", signed=True)
        elif t == 0x16 and len(val) >= 2:
            out.service_data_uuids.append(_uuid16(val[:2]))
        elif t == 0x19 and len(val) >= 2:
            out.appearance = int.from_bytes(val[:2], "little")
        elif t == 0xFF and len(val) >= 2:
            out.company_id = int.from_bytes(val[:2], "little")
            out.manufacturer_len = len(val) - 2
        i += 1 + ln
    return out


def observation_from_bleak(ts: float, key: str, adv, synthetic: bool = False):
    """Build an Observation from a Bleak AdvertisementData object. The advertised (local) name is used, not the
    operating system's cached device name."""
    from .models import Observation
    return Observation(
        ts=ts, key=key, rssi=getattr(adv, "rssi", None), name=getattr(adv, "local_name", None) or None,
        uuids=tuple(normalize_uuid(u) for u in (getattr(adv, "service_uuids", None) or [])),
        company_ids=tuple(sorted((getattr(adv, "manufacturer_data", None) or {}).keys())),
        service_data_uuids=tuple(normalize_uuid(u) for u in (getattr(adv, "service_data", None) or {})),
        tx_power=getattr(adv, "tx_power", None), synthetic=synthetic)
