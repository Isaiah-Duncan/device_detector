"""SQLite event and device storage with CSV and JSON export. Default is memory only; a file is used only when the
user turns on saving. Raw payloads are never stored. Advertised names are stored only when the user chose to show them."""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import Event
from .tracker import Tracker

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, started REAL, name TEXT, source TEXT, synthetic INTEGER);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, session_id INTEGER, ts REAL, kind TEXT, device_key TEXT,
  text TEXT, category TEXT, confidence TEXT, synthetic INTEGER);
CREATE TABLE IF NOT EXISTS devices(session_id INTEGER, device_key TEXT, first_seen REAL, last_seen REAL, adv_count INTEGER,
  category TEXT, confidence TEXT, reasons TEXT, rssi_min INTEGER, rssi_max INTEGER, rssi_mean REAL, company_ids TEXT,
  services TEXT, name TEXT, synthetic INTEGER);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def start_session(self, name: str, source: str, synthetic: bool) -> int:
        with self.lock:
            cur = self.con.execute("INSERT INTO sessions(started,name,source,synthetic) VALUES(?,?,?,?)", (time.time(), name, source, int(synthetic)))
            self.con.commit()
            return cur.lastrowid

    def log_event(self, session_id: int, e: Event) -> None:
        with self.lock:
            self.con.execute("INSERT INTO events(session_id,ts,kind,device_key,text,category,confidence,synthetic) VALUES(?,?,?,?,?,?,?,?)",
                             (session_id, e.ts, e.kind, e.key, e.text, e.category, e.confidence, int(e.synthetic)))
            self.con.commit()

    def save_devices(self, session_id: int, tracker: Tracker, include_names: bool = False) -> int:
        n = 0
        with self.lock:
            self.con.execute("DELETE FROM devices WHERE session_id=?", (session_id,))
            for r in tracker.devices.values():
                vals = [v for _, v in r.rssi]
                self.con.execute("INSERT INTO devices VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (session_id, r.key, r.first_seen, r.last_seen, r.count, r.cls.category, r.cls.confidence,
                                  "; ".join(r.cls.reasons), min(vals) if vals else None, max(vals) if vals else None,
                                  sum(vals) / len(vals) if vals else None, ",".join(f"0x{c:04X}" for c in sorted(r.company_ids)),
                                  ",".join(sorted(r.uuids | r.service_data)), r.name if include_names else None, int(r.synthetic)))
                n += 1
            self.con.commit()
        return n

    def events(self, session_id: int | None = None) -> list[dict]:
        q, a = "SELECT * FROM events", ()
        if session_id is not None:
            q, a = q + " WHERE session_id=?", (session_id,)
        with self.lock:
            return [dict(r) for r in self.con.execute(q + " ORDER BY ts", a)]

    def devices(self, session_id: int | None = None) -> list[dict]:
        q, a = "SELECT * FROM devices", ()
        if session_id is not None:
            q, a = q + " WHERE session_id=?", (session_id,)
        with self.lock:
            return [dict(r) for r in self.con.execute(q, a)]

    def export_csv(self, path: str | Path, table: str = "events", session_id: int | None = None) -> int:
        rows = self.events(session_id) if table == "events" else self.devices(session_id)
        cols = list(rows[0].keys()) if rows else (["id", "session_id", "ts", "kind", "device_key", "text", "category", "confidence", "synthetic"] if table == "events" else ["session_id", "device_key"])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    def export_json(self, path: str | Path, session_id: int | None = None) -> int:
        data = {"events": self.events(session_id), "devices": self.devices(session_id),
                "notice": "Rows marked synthetic=1 are simulated. Missing detections are inconclusive, not proof of absence."}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return len(data["events"])

    def purge_older_than(self, days: int) -> int:
        cutoff = time.time() - days * 86400
        with self.lock:
            ids = [r[0] for r in self.con.execute("SELECT id FROM sessions WHERE started < ?", (cutoff,))]
            for sid in ids:
                for t in ("events", "devices"):
                    self.con.execute(f"DELETE FROM {t} WHERE session_id=?", (sid,))
                self.con.execute("DELETE FROM sessions WHERE id=?", (sid,))
            self.con.commit()
        return len(ids)

    def purge_all(self) -> None:
        with self.lock:
            for t in ("events", "devices", "sessions"):
                self.con.execute(f"DELETE FROM {t}")
            self.con.commit()

    def close(self):
        with self.lock:
            self.con.close()
