"""Device registry: first/last seen, RSSI history, advertisement rate, disappearance and rediscovery events."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .classifier import classify
from .config import CATEGORIES, Settings
from .models import Classification, Event, Observation
from .rssi import adv_rate, rolling_stats


@dataclass
class DeviceRecord:
    key: str
    first_seen: float
    last_seen: float
    count: int = 0
    rssi: deque = field(default_factory=lambda: deque(maxlen=900))
    times: deque = field(default_factory=lambda: deque(maxlen=600))
    name: str | None = None
    uuids: set = field(default_factory=set)
    company_ids: set = field(default_factory=set)
    service_data: set = field(default_factory=set)
    appearance: int | None = None
    tx_power: int | None = None
    synthetic: bool = False
    present: bool = True
    disappearances: int = 0
    cls: Classification = field(default_factory=Classification)

    def evidence_size(self) -> int:
        return len(self.uuids) + len(self.company_ids) + len(self.service_data) + (self.appearance is not None) + bool(self.name)


class Tracker:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()
        self.devices: dict[str, DeviceRecord] = {}
        self.events: list[Event] = []
        self.new_id_times: deque = deque(maxlen=2000)
        self.last_obs_ts: float | None = None

    # ------------------------------------------------------------------ input
    def feed(self, o: Observation) -> list[Event]:
        ev: list[Event] = []
        self.last_obs_ts = o.ts
        r = self.devices.get(o.key)
        if r is None:
            r = DeviceRecord(o.key, o.ts, o.ts, synthetic=o.synthetic)
            r.rssi = deque(maxlen=self.s.rssi_history)
            self.devices[o.key] = r
            self.new_id_times.append(o.ts)
            fresh = True
        else:
            fresh = False
        before_ev = r.evidence_size()
        if not r.present and not fresh:
            gap = o.ts - r.last_seen
            r.present = True
            ev.append(Event(o.ts, "rediscovered", o.key, f"{o.key} advertised again after {gap:.0f} s of silence", r.cls.category, r.cls.confidence, r.synthetic))
        r.last_seen = o.ts
        r.count += 1
        r.times.append(o.ts)
        if o.rssi is not None:
            r.rssi.append((o.ts, o.rssi))
        if o.name:
            r.name = o.name
        r.uuids.update(o.uuids)
        r.company_ids.update(o.company_ids)
        r.service_data.update(o.service_data_uuids)
        if o.appearance is not None:
            r.appearance = o.appearance
        if o.tx_power is not None:
            r.tx_power = o.tx_power
        if fresh or r.evidence_size() != before_ev:
            old = r.cls
            r.cls = classify(r.name, r.uuids, r.company_ids, r.service_data, r.appearance)
            if fresh:
                ev.append(Event(o.ts, "discovered", o.key, f"{o.key} discovered: {r.cls.category} ({r.cls.confidence} confidence)", r.cls.category, r.cls.confidence, r.synthetic))
            elif (old.category, old.confidence) != (r.cls.category, r.cls.confidence):
                ev.append(Event(o.ts, "reclassified", o.key, f"{o.key} now {r.cls.category} ({r.cls.confidence}); was {old.category} ({old.confidence})", r.cls.category, r.cls.confidence, r.synthetic))
        self.events += ev
        return ev

    def tick(self, now: float) -> list[Event]:
        """Report devices that have been silent too long. Silence is not proof that a device left."""
        ev = []
        for r in self.devices.values():
            if r.present and now - r.last_seen > self.s.disappear_after_s:
                r.present = False
                r.disappearances += 1
                ev.append(Event(now, "disappeared", r.key, f"{r.key} not heard for {now - r.last_seen:.0f} s (inconclusive: it may still be nearby but quiet, or its address may have changed)", r.cls.category, r.cls.confidence, r.synthetic))
        self.events += ev
        return ev

    # ------------------------------------------------------------------ queries
    def present_devices(self) -> list[DeviceRecord]:
        return [r for r in self.devices.values() if r.present]

    def category_counts(self, present_only: bool = True) -> dict[str, int]:
        counts = {c: 0 for c in CATEGORIES}
        for r in self.devices.values():
            if present_only and not r.present:
                continue
            counts[r.cls.category] += 1
        return counts

    def stats(self, key: str, now: float) -> dict:
        r = self.devices[key]
        st = rolling_stats(r.rssi, now, self.s.rssi_window_s)
        st["since_last_s"] = now - r.last_seen
        st["adv_per_s"] = adv_rate(r.times, now, self.s.rssi_window_s, r.first_seen)
        return st

    def churn_per_minute(self, now: float, window_s: float = 60.0) -> float:
        """New identifiers per minute. A high value with few real devices suggests address rotation."""
        n = sum(1 for t in self.new_id_times if now - t <= window_s)
        return n * 60.0 / window_s

    def clear(self) -> None:
        self.devices.clear()
        self.events.clear()
        self.new_id_times.clear()
