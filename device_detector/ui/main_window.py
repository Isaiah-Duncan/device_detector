"""Dark dashboard: A devices, B signal strength, C categories, D timeline, E sensor status, plus camera and experiments."""
from __future__ import annotations

import json
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..config import CATEGORIES, Settings
from ..diagnostics import DISCLOSURES, SensorStatus, status_lines
from ..experiment import EXPLANATION, ExperimentRecorder, compare, format_comparison
from ..identifiers import IdMapper
from ..rssi import RSSI_EXPLANATION, estimate_distance
from ..scanner import BleScanner, bluetooth_status
from ..store import Store
from ..synthetic import LABEL, SyntheticSource
from ..tracker import Tracker

STYLE = """
QWidget { background:#1b1d20; color:#e8eaed; font-size:13px; }
QTabWidget::pane { border:1px solid #2c3036; }
QTabBar::tab { background:#23262b; padding:8px 16px; } QTabBar::tab:selected { background:#2c3036; color:#22d3ee; }
QPushButton { background:#2c3036; border:1px solid #3a3f46; padding:6px 12px; border-radius:4px; }
QPushButton:hover { border-color:#22d3ee; } QPushButton:disabled { color:#6b7280; }
QComboBox, QLineEdit, QPlainTextEdit, QTableWidget, QListWidget { background:#15171a; border:1px solid #2c3036; }
QHeaderView::section { background:#23262b; color:#9aa3ad; border:0; padding:4px; }
QLabel#big { font-size:22px; font-weight:600; } QLabel#cap { color:#9aa3ad; font-size:11px; }
QLabel#banner { background:#4a3b12; color:#fbbf24; padding:4px 8px; font-weight:600; }
QLabel#rec { color:#f87171; font-weight:700; }
QCheckBox::indicator { width:14px; height:14px; border:1px solid #6b7280; background:#15171a; border-radius:3px; }
QCheckBox::indicator:checked { background:#22d3ee; border-color:#22d3ee; }
"""
CYAN, EMERALD, AMBER, GREY = "#22d3ee", "#34d399", "#fbbf24", "#6b7280"
SRC_REAL, SRC_SYN = "Bluetooth LE (real hardware)", f"{LABEL}: demo devices"
CONF_COLOR = {"high": EMERALD, "medium": EMERALD, "low": AMBER, "none": GREY}


def make_plot(title, xl, yl):
    w = pg.PlotWidget(title=title)
    w.setBackground("#15171a")
    w.setLabel("bottom", xl)
    w.setLabel("left", yl)
    w.showGrid(x=True, y=True, alpha=0.2)
    return w


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.setWindowTitle("Nearby Device Detection System")
        self.resize(1360, 840)
        self.s = settings or Settings()
        self.ids = IdMapper()
        self.tracker = Tracker(self.s)
        self.store = Store(":memory:")
        self.session_id: int | None = None
        self.source = None
        self.exp = ExperimentRecorder()
        self.status = SensorStatus()
        self.selected: str | None = None
        self.detector = None
        self.camera = None
        self.model_path = ""
        self.cam_dets = []
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)
        self.cam_timer = QtCore.QTimer(self)
        self.cam_timer.timeout.connect(self._cam_tick)
        self._last_refresh = 0.0
        st, msg = bluetooth_status()
        self.status.bluetooth, self.status.bluetooth_msg = st, msg
        self._refresh_status()

    # ------------------------------------------------------------------ layout
    def _build(self):
        c = QtWidgets.QWidget()
        self.setCentralWidget(c)
        root = QtWidgets.QVBoxLayout(c)
        bar = QtWidgets.QHBoxLayout()
        self.src = QtWidgets.QComboBox()
        self.src.addItems([SRC_REAL, SRC_SYN])
        self.btn_start = QtWidgets.QPushButton("Start scan")
        self.btn_stop = QtWidgets.QPushButton("Stop scan")
        self.btn_stop.setEnabled(False)
        self.chk_names = QtWidgets.QCheckBox("Show advertised names")
        self.chk_save = QtWidgets.QCheckBox("Save to database file")
        self.btn_clear = QtWidgets.QPushButton("Clear all data")
        self.btn_csv = QtWidgets.QPushButton("Export events CSV")
        self.btn_json = QtWidgets.QPushButton("Export JSON")
        for w in (QtWidgets.QLabel("Source:"), self.src, self.btn_start, self.btn_stop, self.chk_names, self.chk_save, self.btn_clear, self.btn_csv, self.btn_json):
            bar.addWidget(w)
        bar.addStretch(1)
        root.addLayout(bar)
        self.banner = QtWidgets.QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setVisible(False)
        root.addWidget(self.banner)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._dashboard(), "Dashboard")
        self.tabs.addTab(self._camera_tab(), "Camera (optional)")
        self.tabs.addTab(self._experiment_tab(), "Experiment")
        self.tabs.addTab(self._limits_tab(), "Limits and privacy")
        self.btn_start.clicked.connect(self.start_scan)
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_clear.clicked.connect(self.clear_data)
        self.btn_csv.clicked.connect(lambda: self._export("csv"))
        self.btn_json.clicked.connect(lambda: self._export("json"))
        self.chk_save.toggled.connect(self._toggle_save)
        self.src.currentIndexChanged.connect(self._src_changed)

    def _dashboard(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        # A
        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Category", "Confidence", "RSSI dBm", "Avg dBm", "Adv/s", "Last seen", "State"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._select)
        ga = QtWidgets.QVBoxLayout()
        ga.addWidget(self._cap("A. Nearby devices (temporary identifiers; click a row)"))
        ga.addWidget(self.table)
        # B
        self.plot = make_plot("", "seconds ago", "RSSI (dBm)")
        self.curve = self.plot.plot(pen=pg.mkPen(CYAN, width=1.5), symbol="o", symbolSize=4, symbolBrush=CYAN)
        self.avg_curve = self.plot.plot(pen=pg.mkPen(EMERALD, width=2))
        self.plot.setYRange(-100, -30)
        self.plot.invertX(True)
        self.stat_lbl = QtWidgets.QLabel("Select a device.")
        self.stat_lbl.setWordWrap(True)
        self.dist_lbl = QtWidgets.QLabel("")
        self.dist_lbl.setObjectName("cap")
        self.dist_lbl.setWordWrap(True)
        self.btn_why = QtWidgets.QPushButton("Why RSSI is not a distance")
        self.btn_why.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, "RSSI", RSSI_EXPLANATION))
        gb = QtWidgets.QVBoxLayout()
        gb.addWidget(self._cap("B. Signal strength of the selected device"))
        gb.addWidget(self.plot, 1)
        gb.addWidget(self.stat_lbl)
        gb.addWidget(self.dist_lbl)
        gb.addWidget(self.btn_why)
        # C
        self.cat_table = QtWidgets.QTableWidget(len(CATEGORIES), 2)
        self.cat_table.setHorizontalHeaderLabels(["Category", "Present now"])
        self.cat_table.verticalHeader().setVisible(False)
        self.cat_table.horizontalHeader().setStretchLastSection(True)
        self.total_lbl = QtWidgets.QLabel("")
        self.total_lbl.setObjectName("cap")
        self.total_lbl.setWordWrap(True)
        gc = QtWidgets.QVBoxLayout()
        gc.addWidget(self._cap("C. Device categories"))
        gc.addWidget(self.cat_table)
        gc.addWidget(self.total_lbl)
        # D
        self.timeline = QtWidgets.QListWidget()
        gd = QtWidgets.QVBoxLayout()
        gd.addWidget(self._cap("D. Event timeline"))
        gd.addWidget(self.timeline)
        # E
        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        ge = QtWidgets.QVBoxLayout()
        ge.addWidget(self._cap("E. Sensor status"))
        ge.addWidget(self.status_box)
        g.addLayout(ga, 0, 0)
        g.addLayout(gb, 0, 1)
        g.addLayout(gc, 1, 0)
        g.addLayout(gd, 1, 1)
        g.addLayout(ge, 2, 0, 1, 2)
        g.setRowStretch(0, 3)
        g.setRowStretch(1, 2)
        g.setRowStretch(2, 1)
        g.setColumnStretch(0, 3)
        g.setColumnStretch(1, 2)
        return w

    def _cap(self, text):
        l = QtWidgets.QLabel(text)
        l.setObjectName("cap")
        return l

    def _camera_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("Webcam object detection is off until you press Start camera. Nothing is recorded or saved."))
        row = QtWidgets.QHBoxLayout()
        self.btn_model = QtWidgets.QPushButton("Choose ONNX model...")
        self.chk_consent = QtWidgets.QCheckBox("Everyone in view agrees to this test")
        self.btn_cam_start = QtWidgets.QPushButton("Start camera")
        self.btn_cam_stop = QtWidgets.QPushButton("Stop camera")
        self.btn_cam_stop.setEnabled(False)
        for x in (self.btn_model, self.chk_consent, self.btn_cam_start, self.btn_cam_stop):
            row.addWidget(x)
        row.addStretch(1)
        v.addLayout(row)
        self.cam_label = QtWidgets.QLabel("Camera is off.")
        self.cam_label.setMinimumSize(640, 360)
        self.cam_label.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.cam_label, 1)
        self.cam_info = QtWidgets.QLabel("")
        self.cam_info.setWordWrap(True)
        v.addWidget(self.cam_info)
        from ..vision import EXPORT_HELP
        v.addWidget(self._cap(EXPORT_HELP))
        self.btn_model.clicked.connect(self._choose_model)
        self.btn_cam_start.clicked.connect(self.start_camera)
        self.btn_cam_stop.clicked.connect(self.stop_camera)
        return w

    def _experiment_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        e = QtWidgets.QLabel(EXPLANATION)
        e.setWordWrap(True)
        e.setObjectName("cap")
        v.addWidget(e)
        row = QtWidgets.QHBoxLayout()
        self.seg_name = QtWidgets.QLineEdit("Baseline")
        self.btn_seg_start = QtWidgets.QPushButton("Start segment")
        self.btn_seg_stop = QtWidgets.QPushButton("Stop segment")
        self.btn_seg_stop.setEnabled(False)
        self.seg_rec = QtWidgets.QLabel("")
        self.seg_rec.setObjectName("rec")
        for x in (QtWidgets.QLabel("Segment name:"), self.seg_name, self.btn_seg_start, self.btn_seg_stop, self.seg_rec):
            row.addWidget(x)
        v.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        self.cmp_a, self.cmp_b = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.btn_cmp = QtWidgets.QPushButton("Compare B minus A")
        self.btn_cmp_save = QtWidgets.QPushButton("Save comparison (JSON)")
        for x in (QtWidgets.QLabel("A (baseline):"), self.cmp_a, QtWidgets.QLabel("B (after test device):"), self.cmp_b, self.btn_cmp, self.btn_cmp_save):
            row2.addWidget(x)
        v.addLayout(row2)
        self.cmp_out = QtWidgets.QPlainTextEdit()
        self.cmp_out.setReadOnly(True)
        v.addWidget(self.cmp_out, 1)
        self.last_cmp = None
        self.btn_seg_start.clicked.connect(self.seg_start)
        self.btn_seg_stop.clicked.connect(self.seg_stop)
        self.btn_cmp.clicked.connect(self.do_compare)
        self.btn_cmp_save.clicked.connect(self.save_compare)
        return w

    def _limits_tab(self):
        t = QtWidgets.QPlainTextEdit()
        t.setReadOnly(True)
        priv = ["", "PRIVACY", "- All processing happens on this computer. Nothing is sent anywhere.",
                "- Scanning and camera run only after you press Start, and stop when you press Stop.",
                "- Addresses are hashed with a random key for each run. Real addresses are never shown or stored.",
                "- Advertised names are hidden and not saved unless you tick 'Show advertised names'.",
                "- Raw advertisement payloads are not stored. Only metadata used for analysis is kept.",
                "- Data stays in memory unless you tick 'Save to database file'. 'Clear all data' erases memory and the database.",
                "- The program does not connect to devices, read their data, identify people, or recognize faces."]
        t.setPlainText("WHAT THE SENSORS CANNOT DETECT\n" + "\n".join("- " + d for d in DISCLOSURES) + "\n" + "\n".join(priv) + "\n\nRSSI\n" + RSSI_EXPLANATION)
        return t

    # ------------------------------------------------------------------ scanning
    def _src_changed(self):
        syn = self.src.currentText() == SRC_SYN
        self.banner.setText(f"{LABEL}: simulated devices. These are not real detections." if syn else "")
        self.banner.setVisible(syn)

    def start_scan(self):
        syn = self.src.currentText() == SRC_SYN
        self._src_changed()
        self.ids = IdMapper()
        if syn:
            self.source = SyntheticSource(ids=self.ids)
            self.status.bluetooth, self.status.bluetooth_msg = "scanning", "SYNTHETIC SIGNAL source running. Bluetooth hardware is not used."
        else:
            self.source = BleScanner(self.ids)
            ok = self.source.start()
            self.status.bluetooth, self.status.bluetooth_msg = self.source.state, self.source.message
            self.status.scan_permission = self.source.state if self.source.state == "permission_denied" else "unknown"
            if not ok:
                self.source = None
                self._refresh_status()
                return
        self.session_id = self.store.start_session("scan", self.src.currentText(), syn)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.src.setEnabled(False)
        self._refresh_status()

    def stop_scan(self):
        if self.source is not None and hasattr(self.source, "stop"):
            self.source.stop()
        self.source = None
        if self.session_id is not None:
            self.store.save_devices(self.session_id, self.tracker, self.chk_names.isChecked())
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.src.setEnabled(True)
        self.status.bluetooth, self.status.bluetooth_msg = "stopped", "Scan stopped."
        self._refresh_status()

    def _tick(self):
        now = time.monotonic()
        if self.source is not None:
            obs = self.source.poll(now) if isinstance(self.source, SyntheticSource) else self.source.read_all()
            for o in obs:
                self.exp.feed(o)
                for ev in self.tracker.feed(o):
                    self._log(ev)
            for ev in self.tracker.tick(now):
                self._log(ev)
            if isinstance(self.source, BleScanner) and self.source.state not in ("scanning", "starting"):
                self.status.bluetooth, self.status.bluetooth_msg = self.source.state, self.source.message
                self.stop_scan()
                return
        if now - self._last_refresh > 0.25:
            self._last_refresh = now
            self._refresh(now)

    def _log(self, ev):
        item = QtWidgets.QListWidgetItem(f"{time.strftime('%H:%M:%S')}  {'[SYNTHETIC] ' if ev.synthetic else ''}{ev.text}")
        item.setForeground(QtGui.QColor({"discovered": EMERALD, "rediscovered": CYAN, "disappeared": AMBER, "reclassified": "#c4b5fd"}.get(ev.kind, "#e8eaed")))
        self.timeline.insertItem(0, item)
        while self.timeline.count() > 300:
            self.timeline.takeItem(self.timeline.count() - 1)
        if self.session_id is not None:
            self.store.log_event(self.session_id, ev)

    # ------------------------------------------------------------------ refresh
    def _name(self, r):
        if not r.name:
            return "-"
        return r.name if self.chk_names.isChecked() else "(hidden)"

    def _refresh(self, now):
        recs = sorted(self.tracker.devices.values(), key=lambda r: (not r.present, -(r.rssi[-1][1] if r.rssi else -999)))
        sel_keep = self.selected
        self.table.blockSignals(True)
        self.table.setRowCount(len(recs))
        for row, r in enumerate(recs):
            st = self.tracker.stats(r.key, now)
            vals = [r.key + (" [SYN]" if r.synthetic else ""), self._name(r), r.cls.category, r.cls.confidence,
                    "-" if st["last"] is None else str(st["last"]), "-" if st["mean_window"] is None else f"{st['mean_window']:.0f}",
                    f"{st['adv_per_s']:.1f}", f"{st['since_last_s']:.0f} s ago", "present" if r.present else "not heard"]
            for col, text in enumerate(vals):
                it = self.table.item(row, col)
                is_new = it is None
                if is_new:
                    it = QtWidgets.QTableWidgetItem()
                it.setText(text)
                it.setData(QtCore.Qt.UserRole, r.key)
                if col == 3:
                    it.setForeground(QtGui.QColor(CONF_COLOR.get(r.cls.confidence, GREY)))
                if col == 8:
                    it.setForeground(QtGui.QColor(EMERALD if r.present else GREY))
                it.setToolTip("; ".join(r.cls.reasons) if col in (2, 3) else "")
                if is_new:
                    self.table.setItem(row, col, it)
            if r.key == sel_keep:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        counts = self.tracker.category_counts(True)
        for i, cat in enumerate(CATEGORIES):
            self.cat_table.setItem(i, 0, QtWidgets.QTableWidgetItem(cat))
            self.cat_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(counts[cat])))
        present = len(self.tracker.present_devices())
        self.total_lbl.setText(f"{present} identifiers advertising now, {len(self.tracker.devices)} seen in total, {self.tracker.churn_per_minute(now):.1f} new identifiers per minute. "
                               "Randomized addresses make identifier counts an upper bound on real devices.")
        if self.selected and self.selected in self.tracker.devices:
            self._update_plot(now)
        if self.seg_rec and self.exp.active:
            self.seg_rec.setText(f"RECORDING: {self.exp.active.name} ({now - self.exp.active.t_start:.0f} s)")
        self._refresh_status()

    def _select(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            item = self.table.item(rows[0].row(), 0)
            self.selected = item.data(QtCore.Qt.UserRole) if item else None
            if self.selected:
                self._update_plot(time.monotonic())

    def _update_plot(self, now):
        r = self.tracker.devices[self.selected]
        pts = [(now - t, v) for t, v in r.rssi]
        if pts:
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts], dtype=float)
            self.curve.setData(xs, ys)
            k = 5
            avg = np.convolve(ys, np.ones(k) / k, mode="valid") if len(ys) >= k else ys
            self.avg_curve.setData(xs[k - 1:] if len(ys) >= k else xs, avg)
        st = self.tracker.stats(self.selected, now)
        f = lambda v, u="": "n/a" if v is None else f"{v:.0f}{u}"
        self.stat_lbl.setText(f"{self.selected}{' (SYNTHETIC)' if r.synthetic else ''}: now {f(st['last'], ' dBm')}, min {f(st['min'])}, max {f(st['max'])}, "
                              f"rolling avg {f(st['mean_window'])}, variability (sd) {'n/a' if st['std_window'] is None else format(st['std_window'], '.1f') + ' dB'}, "
                              f"last advertisement {st['since_last_s']:.0f} s ago, {st['adv_per_s']:.1f} adv/s")
        if st["last"] is not None:
            d = estimate_distance(st["last"], r.tx_power if r.tx_power is not None else -59)
            self.dist_lbl.setText(f"EXPERIMENTAL distance guess: about {d['meters']:.1f} m (plausible {d['low']:.1f} to {d['high']:.1f} m). {d['note']}")

    def _refresh_status(self):
        sym = {"ok": "[OK]  ", "warn": "[!]   ", "off": "[ - ]  "}
        lines = [f"{sym[i]}{n}: {m}" for i, n, m in status_lines(self.status)]
        lines.append("[ - ]  Not supported: " + "; ".join(self.status.unsupported))
        self.status_box.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ data control
    def _toggle_save(self, on):
        old = self.store
        if on:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Database file", self.s.db_path, "SQLite (*.db)")
            if not path:
                self.chk_save.blockSignals(True)
                self.chk_save.setChecked(False)
                self.chk_save.blockSignals(False)
                return
            self.s.db_path, self.s.save_to_db = path, True
            self.store = Store(path)
            if self.s.retention_days > 0:
                self.store.purge_older_than(self.s.retention_days)
        else:
            self.s.save_to_db = False
            self.store = Store(":memory:")
        old.close()
        self.session_id = self.store.start_session("scan", self.src.currentText(), self.src.currentText() == SRC_SYN) if self.source else None

    def clear_data(self):
        self.tracker.clear()
        self.timeline.clear()
        self.store.purge_all()
        self.exp = ExperimentRecorder()
        self.cmp_a.clear()
        self.cmp_b.clear()
        self.selected = None
        self.curve.setData([], [])
        self.avg_curve.setData([], [])
        if self.source is not None:
            self.session_id = self.store.start_session("scan", self.src.currentText(), self.src.currentText() == SRC_SYN)

    def _export(self, kind):
        if self.session_id is not None:
            self.store.save_devices(self.session_id, self.tracker, self.chk_names.isChecked())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", f"events.{kind}", f"*.{kind}")
        if path:
            n = self.store.export_csv(path) if kind == "csv" else self.store.export_json(path)
            QtWidgets.QMessageBox.information(self, "Export", f"Wrote {n} events.")

    # ------------------------------------------------------------------ experiments
    def seg_start(self):
        name = self.seg_name.text().strip() or "Segment"
        self.exp.start_segment(name, time.monotonic())
        self.btn_seg_start.setEnabled(False)
        self.btn_seg_stop.setEnabled(True)

    def seg_stop(self):
        seg = self.exp.stop_segment(time.monotonic(), lambda k: self.tracker.devices[k].cls.category if k in self.tracker.devices else "Unknown device")
        self.btn_seg_start.setEnabled(True)
        self.btn_seg_stop.setEnabled(False)
        self.seg_rec.setText("")
        for cb in (self.cmp_a, self.cmp_b):
            cb.clear()
            cb.addItems(list(self.exp.segments))
        if self.cmp_b.count() > 1:
            self.cmp_b.setCurrentIndex(self.cmp_b.count() - 1)
        return seg

    def do_compare(self):
        a, b = self.cmp_a.currentText(), self.cmp_b.currentText()
        if a and b and a in self.exp.segments and b in self.exp.segments and a != b:
            self.last_cmp = compare(self.exp.segments[a], self.exp.segments[b])
            self.cmp_out.setPlainText(format_comparison(self.last_cmp))
        else:
            self.cmp_out.setPlainText("Record at least two segments and choose two different ones.")

    def save_compare(self):
        if not self.last_cmp:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save comparison", "comparison.json", "*.json")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.last_cmp, f, indent=2)

    # ------------------------------------------------------------------ camera
    def _choose_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "ONNX model", "", "ONNX (*.onnx)")
        if not path:
            return
        try:
            from ..vision import OnnxDetector
            self.detector = OnnxDetector(path, self.s.camera_conf)
            self.model_path = path
            self.status.model, self.status.model_msg = "loaded", f"Loaded {path.split('/')[-1].split(chr(92))[-1]}"
        except Exception as e:
            self.detector = None
            self.status.model, self.status.model_msg = "error", f"Model could not be loaded ({type(e).__name__})."
        self._refresh_status()

    def start_camera(self):
        if not self.chk_consent.isChecked():
            self.cam_info.setText("Tick the box to confirm that everyone in view agrees to this test.")
            return
        from ..vision import CameraSession
        self.camera = CameraSession()
        ok, msg = self.camera.open(self.s.camera_index)
        self.status.camera, self.status.camera_msg = ("on" if ok else "error"), msg
        self.cam_info.setText(msg + ("" if self.detector else " No model loaded: the picture is shown without detection."))
        if ok:
            self.btn_cam_start.setEnabled(False)
            self.btn_cam_stop.setEnabled(True)
            self.cam_timer.start(int(1000 / max(self.s.camera_fps, 1)))
        self._refresh_status()

    def stop_camera(self):
        self.cam_timer.stop()
        if self.camera:
            self.camera.close()
        self.camera = None
        self.btn_cam_start.setEnabled(True)
        self.btn_cam_stop.setEnabled(False)
        self.cam_label.setText("Camera is off.")
        self.cam_label.setPixmap(QtGui.QPixmap())
        self.status.camera, self.status.camera_msg = "off", "Camera is off."
        self._refresh_status()

    def _cam_tick(self):
        if not self.camera:
            return
        frame = self.camera.read()
        if frame is None:
            return
        from ..vision import draw_detections
        dets = []
        if self.detector:
            try:
                dets = self.detector.detect(frame)
            except Exception as e:
                self.cam_info.setText(f"Detection error: {type(e).__name__}")
        frame = draw_detections(frame.copy(), dets)
        h, w = frame.shape[:2]
        img = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)
        self.cam_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.cam_label.size(), QtCore.Qt.KeepAspectRatio))
        self.cam_info.setText("; ".join(f"{d.label} {d.confidence:.0%}" for d in dets) if dets else ("No phone or laptop in view (inconclusive)." if self.detector else "No model loaded."))

    def closeEvent(self, e):
        self.stop_camera()
        if self.source is not None:
            self.stop_scan()
        super().closeEvent(e)
