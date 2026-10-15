import builtins
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from device_detector import scanner
from device_detector.diagnostics import DISCLOSURES, SensorStatus, status_lines
from device_detector.vision import decode_yolov8, letterbox


def test_missing_bleak_reports_cleanly(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "bleak":
            raise ImportError("no bleak")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    sc = scanner.BleScanner()
    assert sc.start() is False and sc.state == "missing_library" and "pip install bleak" in sc.message


@pytest.mark.parametrize("text,state", [("The Bluetooth radio is turned off", "bluetooth_off"),
                                        ("Access denied", "permission_denied"),
                                        ("No Bluetooth adapters found", "no_adapter"),
                                        ("weird", "error")])
def test_explain_error(text, state):
    assert scanner.explain_error(RuntimeError(text))[0] == state


def test_bluetooth_off_during_start(monkeypatch):
    class Boom:
        def __init__(self, **k): pass
        async def __aenter__(self): raise OSError("The Bluetooth radio is turned off")
        async def __aexit__(self, *a): pass
    monkeypatch.setattr(scanner, "_import_bleak", lambda: SimpleNamespace(BleakScanner=Boom))
    sc = scanner.BleScanner()
    assert sc.start(timeout=3) is False and sc.state == "bluetooth_off"


def test_scanner_queue_with_fake_bleak(monkeypatch):
    """SYNTHETIC: a fake Bleak object delivers one callback. This tests our wiring only, not Windows."""
    holder = {}

    class Fake:
        def __init__(self, detection_callback): holder["cb"] = detection_callback
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
    monkeypatch.setattr(scanner, "_import_bleak", lambda: SimpleNamespace(BleakScanner=Fake))
    sc = scanner.BleScanner()
    assert sc.start(timeout=3)
    adv = SimpleNamespace(rssi=-61, local_name=None, service_uuids=[], manufacturer_data={0x004C: b"x"}, service_data={}, tx_power=None)
    holder["cb"](SimpleNamespace(address="AA:BB:CC:DD:EE:FF"), adv)
    got = sc.read_all()
    sc.stop()
    assert len(got) == 1 and got[0].key.startswith("D-") and "AA:BB" not in repr(got[0]) and not got[0].synthetic
    assert sc.state == "stopped"


def test_queue_overflow_counts_drops():
    sc = scanner.BleScanner(max_queue=2)
    adv = SimpleNamespace(rssi=-60)
    for _ in range(5):
        sc._cb(SimpleNamespace(address="AA"), adv)
    assert sc.q.qsize() == 2 and sc.dropped == 3


def test_status_and_disclosures():
    lines = status_lines(SensorStatus())
    assert {n for _, n, _ in lines} == {"Bluetooth LE", "Scan permission", "Webcam", "Object model"}
    assert any("Classic" in d for d in DISCLOSURES) and any("inconclusive" in d for d in DISCLOSURES)
    assert "Bluetooth Classic discovery" in SensorStatus().unsupported


def make_tensor(nc=80, n=100):
    t = np.zeros((1, 4 + nc, n), np.float32)
    # one phone (class 67) at the centre of the 640 input, a "person" (class 0), and a weak laptop
    t[0, :4, 0] = [320, 320, 100, 200]; t[0, 4 + 67, 0] = 0.9
    t[0, :4, 1] = [322, 322, 100, 200]; t[0, 4 + 67, 1] = 0.8       # duplicate, removed by NMS
    t[0, :4, 2] = [100, 100, 50, 50];   t[0, 4 + 0, 2] = 0.99
    t[0, :4, 3] = [500, 500, 80, 80];   t[0, 4 + 63, 3] = 0.2
    return t


def test_decode_phone_only_and_nms():
    blob, scale, px, py = letterbox(np.zeros((480, 640, 3), np.uint8))
    assert blob.shape == (1, 3, 640, 640)
    d = decode_yolov8(make_tensor(), 0.4, 0.45, scale, px, py, 640, 480)
    assert len(d) == 1 and "cell phone" in d[0].label and abs(d[0].confidence - 0.9) < 1e-6


def test_decode_empty_and_bad_shape():
    assert decode_yolov8(np.zeros((1, 84, 10), np.float32), 0.4, 0.45, 1, 0, 0, 640, 480) == []
    with pytest.raises(ValueError):
        decode_yolov8(np.zeros((3,), np.float32), 0.4, 0.45, 1, 0, 0, 640, 480)
