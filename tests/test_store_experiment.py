import csv
import json
import time

from device_detector.experiment import ExperimentRecorder, compare, format_comparison
from device_detector.models import Event, Observation
from device_detector.store import Store
from device_detector.tracker import Tracker


def make_tracker():
    tr = Tracker()
    tr.feed(Observation(1.0, "D-a", -60, name="Secret Name", uuids=("180d",), synthetic=True))
    tr.feed(Observation(2.0, "D-a", -64, synthetic=True))
    return tr


def test_names_not_saved_by_default(tmp_path):
    st = Store(tmp_path / "x.db")
    sid = st.start_session("s", "synthetic", True)
    st.save_devices(sid, make_tracker())
    assert st.devices(sid)[0]["name"] is None
    st.save_devices(sid, make_tracker(), include_names=True)
    assert st.devices(sid)[0]["name"] == "Secret Name"


def test_exports_roundtrip(tmp_path):
    st = Store()
    sid = st.start_session("s", "synthetic", True)
    st.log_event(sid, Event(1.0, "discovered", "D-a", "hello", synthetic=True))
    st.save_devices(sid, make_tracker())
    p = tmp_path / "e.csv"
    assert st.export_csv(p) == 1
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert rows[0]["kind"] == "discovered" and rows[0]["synthetic"] == "1"
    assert st.export_csv(tmp_path / "d.csv", table="devices") == 1
    j = tmp_path / "e.json"
    st.export_json(j)
    data = json.loads(j.read_text(encoding="utf-8"))
    assert len(data["events"]) == 1 and len(data["devices"]) == 1 and "inconclusive" in data["notice"]


def test_empty_export_has_header(tmp_path):
    st = Store()
    p = tmp_path / "e.csv"
    assert st.export_csv(p) == 0 and "kind" in p.read_text()


def test_purge():
    st = Store()
    sid = st.start_session("old", "x", False)
    st.log_event(sid, Event(1.0, "discovered", "D-a", "t"))
    st.con.execute("UPDATE sessions SET started=?", (time.time() - 10 * 86400,))
    st.con.commit()
    assert st.purge_older_than(5) == 1 and st.events() == []
    sid = st.start_session("new", "x", False)
    st.log_event(sid, Event(1.0, "discovered", "D-a", "t"))
    assert st.purge_older_than(5) == 0 and len(st.events()) == 1
    st.purge_all()
    assert st.events() == []


def seg(rec, name, t0, t1, obs):
    rec.start_segment(name, t0)
    for o in obs:
        rec.feed(o)
    return rec.stop_segment(t1, lambda k: "Unknown device")


def test_compare_new_and_unseen():
    rec = ExperimentRecorder()
    base = seg(rec, "baseline", 0, 60, [Observation(t, "D-1", -60, synthetic=True) for t in range(0, 60, 2)]
               + [Observation(5, "D-gone", -80, synthetic=True)])
    test = seg(rec, "test", 100, 160, [Observation(100 + t, "D-1", -52, synthetic=True) for t in range(0, 60)]
               + [Observation(120, "D-new", -45, company_ids=(0x004C,), synthetic=True)])
    c = compare(base, test)
    assert c["new_keys"] == ["D-new"] and c["not_seen_in_b"] == ["D-gone"]
    ch = c["changed"][0]
    assert ch["key"] == "D-1" and ch["adv_per_min_delta"] > 0 and abs(ch["rssi_mean_delta"] - 8) < 1e-9
    text = format_comparison(c)
    assert "SYNTHETIC SIGNAL" in text and "not proof of absence" in text and "D-new" in text


def test_no_new_identifiers_is_inconclusive():
    rec = ExperimentRecorder()
    a = seg(rec, "a", 0, 10, [Observation(1, "D-1", -60)])
    b = seg(rec, "b", 20, 30, [Observation(21, "D-1", -61)])
    c = compare(a, b)
    assert any("does not prove" in n for n in c["notes"]) and not c["any_synthetic"]


def test_rotation_note():
    rec = ExperimentRecorder()
    a = seg(rec, "a", 0, 10, [Observation(1, "D-1", -60)])
    b = seg(rec, "b", 20, 30, [Observation(21, f"D-n{i}", -60, company_ids=(0x004C,)) for i in range(3)])
    assert any("rotate" in n for n in compare(a, b)["notes"])


def test_feed_without_active_segment_is_ignored():
    rec = ExperimentRecorder()
    rec.feed(Observation(1, "D-1", -60))
    assert rec.segments == {}
