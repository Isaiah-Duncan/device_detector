"""Simulated advertisement source for tests and demonstrations. Every item is labeled SYNTHETIC.

Nothing here is a real detection. The schedule is deterministic so that tests can check appearance,
disappearance, rediscovery and address rotation.
"""
from __future__ import annotations

import numpy as np

from .identifiers import IdMapper
from .models import Observation

LABEL = "SYNTHETIC SIGNAL"


class SyntheticSource:
    label = LABEL
    synthetic = True

    def __init__(self, seed: int = 1, ids: IdMapper | None = None):
        self.rng = np.random.default_rng(seed)
        self.ids = ids or IdMapper(b"synthetic-secret")
        self.t0: float | None = None
        self.last = 0.0
        # name, period s, base rssi, kwargs, (visible windows) in seconds since start, rotate_every
        self.items = [
            dict(tag="phone", period=0.7, rssi=-62, comp=(0x004C,), uuids=(), name=None, windows=[(0, 1e9)], rotate=45),
            dict(tag="band", period=1.0, rssi=-70, comp=(), uuids=("180d",), name="SYN Fit Band", windows=[(0, 1e9)], rotate=0),
            dict(tag="buds", period=0.5, rssi=-55, comp=(), uuids=("184e", "1850"), name="SYN Buds", windows=[(20, 50), (80, 1e9)], rotate=0),
            dict(tag="laptop", period=1.2, rssi=-75, comp=(0x0006,), uuids=(), name=None, appearance=0x0080, windows=[(0, 1e9)], rotate=0),
            dict(tag="beacon", period=2.0, rssi=-82, comp=(), uuids=(), sdata=("feaa",), name=None, windows=[(0, 1e9)], rotate=0),
            dict(tag="mystery", period=1.5, rssi=-88, comp=(0x1234,), uuids=(), name=None, windows=[(0, 1e9)], rotate=0),
        ]
        self.next_t = {i["tag"]: 0.0 for i in self.items}
        self.walk = {i["tag"]: 0.0 for i in self.items}

    def poll(self, now: float) -> list[Observation]:
        """Return the advertisements that 'arrived' since the previous poll. `now` is seconds on any clock."""
        if self.t0 is None:
            self.t0 = now
            self.last = now
        rel = now - self.t0
        out: list[Observation] = []
        for it in self.items:
            tag = it["tag"]
            while self.next_t[tag] <= rel:
                t = self.next_t[tag]
                self.next_t[tag] = t + it["period"] * float(self.rng.uniform(0.8, 1.2))
                if not any(a <= t < b for a, b in it["windows"]):
                    continue
                self.walk[tag] = 0.9 * self.walk[tag] + float(self.rng.normal(0, 1.5))
                epoch = int(t // it["rotate"]) if it["rotate"] else 0
                key = self.ids.key(f"SYN-{tag}-{epoch}")
                out.append(Observation(ts=self.t0 + t, key=key, rssi=int(round(it["rssi"] + self.walk[tag])), name=it["name"],
                                       uuids=tuple(it["uuids"]), company_ids=tuple(it["comp"]),
                                       service_data_uuids=tuple(it.get("sdata", ())), appearance=it.get("appearance"),
                                       synthetic=True))
        self.last = now
        return sorted(out, key=lambda o: o.ts)
