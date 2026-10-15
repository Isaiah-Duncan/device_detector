"""Conservative device-category inference from advertisement metadata.

Every category comes with evidence, a confidence level and the reasons. A manufacturer identifier alone never
produces a category. Names are weak evidence because users and vendors choose them freely.
"""
from __future__ import annotations

from .models import Classification

SMARTPHONE, AUDIO, WATCH, FITNESS, COMPUTER, OTHER, UNKNOWN = (
    "Smartphone", "Wireless audio accessory", "Smartwatch", "Fitness device", "Computer",
    "Other Bluetooth accessory", "Unknown device")

# 16-bit GATT service UUIDs from the Bluetooth Assigned Numbers list: (category, weight, meaning)
SERVICE_EVIDENCE = {
    "180d": (FITNESS, 0.5, "Heart Rate service"),
    "1814": (FITNESS, 0.5, "Running Speed and Cadence service"),
    "1816": (FITNESS, 0.5, "Cycling Speed and Cadence service"),
    "1818": (FITNESS, 0.5, "Cycling Power service"),
    "1826": (FITNESS, 0.5, "Fitness Machine service"),
    "184e": (AUDIO, 0.5, "LE Audio Stream Control service"),
    "1850": (AUDIO, 0.5, "LE Audio Published Audio Capabilities service"),
    "1844": (AUDIO, 0.4, "Volume Control service"),
    "1853": (AUDIO, 0.4, "Common Audio service"),
    "1812": (OTHER, 0.5, "Human Interface Device service (keyboard, mouse or similar)"),
    "fe2c": (AUDIO, 0.3, "Google Fast Pair service (mostly earbuds and speakers)"),
    "fd6f": (SMARTPHONE, 0.3, "Exposure Notification service (sent by phones)"),
    "feaa": (OTHER, 0.4, "Eddystone beacon service"),
}
# Appearance category = appearance value >> 6 (Bluetooth Assigned Numbers, GAP appearance)
APPEARANCE_CATEGORY = {1: (SMARTPHONE, "Generic Phone"), 2: (COMPUTER, "Generic Computer"), 3: (WATCH, "Watch"),
                       10: (OTHER, "Media Player"), 13: (FITNESS, "Heart Rate Sensor"),
                       15: (OTHER, "Human Interface Device"), 17: (FITNESS, "Running Walking Sensor"),
                       18: (FITNESS, "Cycling")}
NAME_WORDS = [("airpods", AUDIO), ("earbuds", AUDIO), ("earbud", AUDIO), ("buds", AUDIO), ("headphone", AUDIO),
              ("headset", AUDIO), ("earphone", AUDIO), ("watch", WATCH), ("band", FITNESS), ("tracker", FITNESS),
              ("iphone", SMARTPHONE), ("phone", SMARTPHONE), ("laptop", COMPUTER), ("macbook", COMPUTER),
              ("notebook", COMPUTER)]
WEAK_ONLY_COMPANIES = {0x004C: "Apple", 0x0006: "Microsoft", 0x0075: "Samsung Electronics", 0x00E0: "Google",
                       0x0087: "Garmin", 0x0059: "Nordic Semiconductor"}


def _level(score: float) -> str:
    return "high" if score >= 0.9 else "medium" if score >= 0.5 else "low" if score >= 0.25 else "none"


def classify(name: str | None, uuids, company_ids, service_data_uuids, appearance) -> Classification:
    scores: dict[str, float] = {}
    why: dict[str, list[str]] = {}

    def add(cat, w, text):
        scores[cat] = scores.get(cat, 0.0) + w
        why.setdefault(cat, []).append(text)

    if appearance is not None:
        cat = APPEARANCE_CATEGORY.get(appearance >> 6)
        if cat:
            add(cat[0], 0.6, f"advertised appearance: {cat[1]}")
    for u in sorted(set(uuids) | set(service_data_uuids)):
        ev = SERVICE_EVIDENCE.get(u)
        if ev:
            add(ev[0], ev[1], f"service {u}: {ev[2]}")
    if name:
        low = name.lower()
        for word, cat in NAME_WORDS:
            if word in low:
                add(cat, 0.25, f"advertised name contains '{word}' (weak: names are user or vendor chosen)")
                break
    notes = []
    for c in company_ids:
        if c in WEAK_ONLY_COMPANIES:
            notes.append(f"manufacturer id 0x{c:04X} ({WEAK_ONLY_COMPANIES[c]}) is not enough to name a device type")
    if not scores:
        return Classification(UNKNOWN, "none", ["no advertised evidence of a device type"] + notes)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = ranked[0]
    level = _level(min(s1, 0.95))
    reasons = list(why[best]) + notes
    if len(ranked) > 1 and ranked[1][1] >= 0.5 * s1:
        order = ["none", "low", "medium", "high"]
        level = order[max(0, order.index(level) - 1)]
        reasons.append(f"conflicting evidence also points to {ranked[1][0]}")
    if level == "none":
        return Classification(UNKNOWN, "none", reasons)
    return Classification(best, level, reasons)
