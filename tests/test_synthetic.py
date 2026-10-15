from device_detector.synthetic import LABEL, SyntheticSource
from device_detector.tracker import Tracker


def run(seconds=120, step=0.25, seed=1):
    src, tr, evs = SyntheticSource(seed=seed), Tracker(), []
    t = 1000.0
    while t < 1000.0 + seconds:
        for o in src.poll(t):
            evs += tr.feed(o)
        evs += tr.tick(t)
        t += step
    return src, tr, evs


def test_label_and_flags():
    src, tr, evs = run(10)
    assert LABEL == "SYNTHETIC SIGNAL" and src.synthetic
    assert all(r.synthetic for r in tr.devices.values()) and all(e.synthetic for e in evs)


def test_deterministic():
    _, a, _ = run(60, seed=3)
    _, b, _ = run(60, seed=3)
    assert sorted(a.devices) == sorted(b.devices)


def test_buds_disappear_and_return():
    _, tr, evs = run(120)
    kinds = [e.kind for e in evs]
    assert "disappeared" in kinds and "rediscovered" in kinds


def test_phone_rotation_creates_multiple_identifiers_all_unknown():
    _, tr, _ = run(120)
    apple_like = [r for r in tr.devices.values() if 0x004C in r.company_ids]
    assert len(apple_like) >= 2
    assert all(r.cls.category == "Unknown device" for r in apple_like)


def test_expected_classes():
    _, tr, _ = run(120)
    cats = {r.cls.category for r in tr.devices.values()}
    assert {"Fitness device", "Wireless audio accessory", "Unknown device"} <= cats
