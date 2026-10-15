"""Controlled experiments: record a baseline, introduce a permitted test device, compare.

Findings are statements about observed advertisements only. A new identifier means a device became discoverable;
it does not prove that a device entered the room. A missing identifier is inconclusive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Observation

ROTATING_COMPANIES = {0x004C, 0x0006, 0x0075}

EXPLANATION = (
    "A device becoming discoverable is not the same as a device entering the room. A device can start advertising "
    "when its screen wakes, when a setting changes, or when its address rotates, while it was already in the room. "
    "A device can also stop advertising while it is still present (battery saving, connected to another device). "
    "For these reasons, not seeing a device is inconclusive, and seeing a new identifier is evidence only that "
    "something advertised in this period.")


@dataclass
class DeviceAgg:
    n: int = 0
    rssi_sum: float = 0.0
    rssi_n: int = 0
    rssi_min: int | None = None
    rssi_max: int | None = None
    company_ids: set = field(default_factory=set)

    @property
    def rssi_mean(self):
        return self.rssi_sum / self.rssi_n if self.rssi_n else None


@dataclass
class Segment:
    name: str
    t_start: float
    t_end: float | None = None
    devices: dict = field(default_factory=dict)
    synthetic: bool = False
    categories: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max((self.t_end or self.t_start) - self.t_start, 1e-6)


class ExperimentRecorder:
    def __init__(self):
        self.segments: dict[str, Segment] = {}
        self.active: Segment | None = None

    def start_segment(self, name: str, now: float) -> None:
        if self.active:
            self.stop_segment(now)
        self.active = Segment(name, now)
        self.segments[name] = self.active

    def stop_segment(self, now: float, category_lookup=None) -> Segment | None:
        seg = self.active
        if seg:
            seg.t_end = now
            if category_lookup:
                seg.categories = {k: category_lookup(k) for k in seg.devices}
        self.active = None
        return seg

    def feed(self, o: Observation) -> None:
        seg = self.active
        if not seg:
            return
        seg.synthetic = seg.synthetic or o.synthetic
        d = seg.devices.setdefault(o.key, DeviceAgg())
        d.n += 1
        d.company_ids.update(o.company_ids)
        if o.rssi is not None:
            d.rssi_sum += o.rssi
            d.rssi_n += 1
            d.rssi_min = o.rssi if d.rssi_min is None else min(d.rssi_min, o.rssi)
            d.rssi_max = o.rssi if d.rssi_max is None else max(d.rssi_max, o.rssi)


def compare(a: Segment, b: Segment) -> dict:
    new_keys = sorted(set(b.devices) - set(a.devices))
    gone_keys = sorted(set(a.devices) - set(b.devices))
    both = sorted(set(a.devices) & set(b.devices))
    changed = []
    for k in both:
        da, db = a.devices[k], b.devices[k]
        ra, rb = da.n * 60 / a.duration, db.n * 60 / b.duration
        dm = (db.rssi_mean - da.rssi_mean) if (da.rssi_mean is not None and db.rssi_mean is not None) else None
        changed.append({"key": k, "adv_per_min_a": ra, "adv_per_min_b": rb, "adv_per_min_delta": rb - ra, "rssi_mean_delta": dm})
    rot = [k for k in new_keys if b.devices[k].company_ids & ROTATING_COMPANIES]
    notes = [EXPLANATION]
    if len(rot) >= 2:
        notes.append(f"{len(rot)} new identifiers carry manufacturer ids that often rotate addresses. They may be the same few devices.")
    if gone_keys:
        notes.append(f"{len(gone_keys)} identifiers seen in {a.name} were not seen in {b.name}. This is inconclusive, not proof of absence.")
    if not new_keys:
        notes.append("No new identifiers appeared. This does not prove that no device entered the room.")
    cat_a, cat_b = {}, {}
    for seg, out in ((a, cat_a), (b, cat_b)):
        for k, c in seg.categories.items():
            out[c] = out.get(c, 0) + 1
    return {"a": a.name, "b": b.name, "new_keys": new_keys, "not_seen_in_b": gone_keys, "changed": changed,
            "total_adv_a": sum(d.n for d in a.devices.values()), "total_adv_b": sum(d.n for d in b.devices.values()),
            "adv_per_min_a": sum(d.n for d in a.devices.values()) * 60 / a.duration,
            "adv_per_min_b": sum(d.n for d in b.devices.values()) * 60 / b.duration,
            "categories_a": cat_a, "categories_b": cat_b, "any_synthetic": a.synthetic or b.synthetic, "notes": notes}


def format_comparison(c: dict) -> str:
    lines = []
    if c["any_synthetic"]:
        lines.append("SYNTHETIC SIGNAL: this comparison uses simulated data.")
    lines += [f"Comparison: {c['b']} minus {c['a']}",
              f"Identifiers: {len(c['new_keys'])} new, {len(c['not_seen_in_b'])} not seen again, {len(c['changed'])} in both",
              f"Advertisements per minute: {c['adv_per_min_a']:.1f} -> {c['adv_per_min_b']:.1f}"]
    if c["new_keys"]:
        lines.append("New identifiers: " + ", ".join(c["new_keys"][:12]) + (" ..." if len(c["new_keys"]) > 12 else ""))
    for ch in sorted(c["changed"], key=lambda x: -abs(x["rssi_mean_delta"] or 0))[:6]:
        d = ch["rssi_mean_delta"]
        lines.append(f"  {ch['key']}: adv/min {ch['adv_per_min_a']:.1f} -> {ch['adv_per_min_b']:.1f}, mean RSSI change {'n/a' if d is None else f'{d:+.1f} dB'}")
    lines.append("")
    lines += c["notes"]
    return "\n".join(lines)
