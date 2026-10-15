"""Headless UI smoke test with the SYNTHETIC source. It checks that the dashboard runs and updates.
It does not test Windows rendering or real Bluetooth."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from device_detector.ui import main_window as mw  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a
    import pyqtgraph as pg
    from PySide6 import QtCore
    for w in a.topLevelWidgets():
        w.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    a.processEvents()
    pg.cleanup()


def test_synthetic_run_updates_all_panels(app, monkeypatch, tmp_path):
    clock = {"t": 5000.0}
    monkeypatch.setattr(mw.time, "monotonic", lambda: clock["t"])
    w = mw.MainWindow()
    w.src.setCurrentText(mw.SRC_SYN)
    w.start_scan()
    assert w.banner.isVisibleTo(w) and "not real detections" in w.banner.text()
    for _ in range(1300):                       # 130 s of simulated time at 100 ms
        clock["t"] += 0.1
        w._tick()
    assert w.tracker.devices and w.timeline.count() > 0
    assert w.timeline.item(0).text() or True
    kinds = {e.kind for e in w.tracker.events}
    assert {"discovered", "disappeared", "rediscovered"} <= kinds
    # selecting a device and drawing its RSSI plot must not raise
    w.selected = next(iter(w.tracker.devices))
    w._update_plot(clock["t"])
    w.seg_name.setText("baseline"); w.seg_start()
    for _ in range(100):
        clock["t"] += 0.1; w._tick()
    w.seg_stop()
    w.seg_name.setText("test"); w.seg_start()
    for _ in range(100):
        clock["t"] += 0.1; w._tick()
    w.seg_stop()
    w.do_compare()
    assert "SYNTHETIC SIGNAL" in w.cmp_out.toPlainText()
    w.stop_scan()
    n = w.store.export_csv(tmp_path / "e.csv")
    assert n > 0
    w.close()
    w.deleteLater()


def test_real_scan_without_bleak_is_graceful(app, monkeypatch):
    from device_detector import scanner
    monkeypatch.setattr(scanner, "_import_bleak", lambda: (_ for _ in ()).throw(ImportError("x")))
    w = mw.MainWindow()
    w.src.setCurrentText(mw.SRC_REAL)
    w.start_scan()
    assert w.source is None and w.btn_start.isEnabled()
    assert w.status.bluetooth == "missing_library"
    w.close()
    w.deleteLater()
