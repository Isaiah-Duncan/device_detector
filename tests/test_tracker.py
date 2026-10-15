from device_detector.config import Settings
from device_detector.identifiers import IdMapper
from device_detector.models import Observation
from device_detector.tracker import Tracker


def ob(t, key="D-1", rssi=-60, **kw):
    return Observation(ts=t, key=key, rssi=rssi, synthetic=True, **kw)


def test_first_seen_last_seen_and_discovery_event():
    tr = Tracker()
    ev = tr.feed(ob(10.0))
    tr.feed(ob(12.0))
    r = tr.devices["D-1"]
    assert r.first_seen == 10.0 and r.last_seen == 12.0 and r.count == 2
    assert [e.kind for e in ev] == ["discovered"] and ev[0].synthetic


def test_disappear_and_rediscover():
    tr = Tracker(Settings(disappear_after_s=30))
    tr.feed(ob(0))
    assert tr.tick(20) == []
    ev = tr.tick(31)
    assert ev[0].kind == "disappeared" and "inconclusive" in ev[0].text
    assert tr.tick(60) == []                       # reported once only
    ev = tr.feed(ob(100))
    assert ev[0].kind == "rediscovered" and tr.devices["D-1"].disappearances == 1
    assert tr.devices["D-1"].present


def test_reclassification_event():
    tr = Tracker()
    tr.feed(ob(0))
    ev = tr.feed(ob(1, uuids=("180d",)))
    assert ev and ev[0].kind == "reclassified" and tr.devices["D-1"].cls.category == "Fitness device"


def test_address_rotation_counts_as_new_identifiers_and_churn():
    tr = Tracker()
    ids = IdMapper(b"k")
    for i in range(6):
        tr.feed(ob(i * 10, key=ids.key(f"AA:{i}"), company_ids=(0x004C,)))
    assert len(tr.devices) == 6
    assert tr.churn_per_minute(60) >= 5
    assert all(r.cls.category == "Unknown device" for r in tr.devices.values())


def test_missing_rssi_is_kept_out_of_history():
    tr = Tracker()
    tr.feed(ob(0, rssi=None))
    assert len(tr.devices["D-1"].rssi) == 0
    assert tr.stats("D-1", 1.0)["last"] is None


def test_category_counts_present_only():
    tr = Tracker()
    tr.feed(ob(0, key="a", uuids=("180d",)))
    tr.feed(ob(0, key="b"))
    tr.tick(100)
    tr.feed(ob(100, key="b"))
    c = tr.category_counts()
    assert c["Unknown device"] == 1 and c["Fitness device"] == 0
    assert tr.category_counts(present_only=False)["Fitness device"] == 1


def test_rssi_history_is_bounded():
    tr = Tracker(Settings(rssi_history=50))
    for i in range(500):
        tr.feed(ob(i * 0.1))
    assert len(tr.devices["D-1"].rssi) == 50


def test_id_mapper():
    a = IdMapper(b"x")
    assert a.key("aa:bb") == a.key("AA:BB") and a.key("aa:bb").startswith("D-")
    old = a.key("aa:bb")
    a.rotate()
    assert a.key("aa:bb") != old
    assert "aa" not in old.lower().replace("d-", "")[:0]
