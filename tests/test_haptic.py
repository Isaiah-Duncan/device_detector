"""Haptic latency test. SYNTHETIC recordings only; no phone, text or network is used."""
import io
import json
import urllib.error

import numpy as np

from device_detector.sensor_lab import cli, haptic, sources

FS = sources.FS


def tone_at(total_s, starts, db=-30.0, dur=0.45, f=180.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 10 ** (-60 / 20), int(total_s * FS)).astype(np.float32)
    t = np.arange(int(dur * FS)) / FS
    b = (10 ** (db / 20) * np.sin(2 * np.pi * f * t)).astype(np.float32)
    for s in starts:
        a = int(s * FS)
        x[a:a + len(b)] += b
    return x


def test_detects_buzz_onsets_within_50ms():
    on, peak = haptic.detect_buzz_onsets(tone_at(20, [3.0, 8.5, 14.0]))
    assert len(on) == 3 and all(abs(o - e) < 0.08 for o, e in zip(on, [3.0, 8.5, 14.0])) and peak > 30


def test_steady_hum_and_short_click_are_not_buzzes():
    rng = np.random.default_rng(1)
    t = np.arange(20 * FS) / FS
    hum = (0.01 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 1e-3, len(t))).astype(np.float32)
    assert haptic.detect_buzz_onsets(hum)[0] == []
    click = tone_at(10, [], seed=2)
    click[5 * FS:5 * FS + 100] += 0.8                      # about 2 ms click
    assert haptic.detect_buzz_onsets(click)[0] == []


def test_silence_and_short_recordings():
    assert haptic.detect_buzz_onsets(np.zeros(FS, np.float32))[0] == []
    assert haptic.detect_buzz_onsets(np.zeros(100, np.float32)) == ([], 0.0)


def test_two_buzzes_closer_than_refractory_merge_into_one():
    on, _ = haptic.detect_buzz_onsets(tone_at(10, [3.0, 3.4]))
    assert len(on) == 1


def test_summary_single_text_latency_and_exact_counts():
    d = {(0, 0): 2.0, (1, 0): 4.0, (2, 0): 3.0}
    rig = haptic.SyntheticRig(delays=lambda b, i: d[(b, i)])
    r = haptic.run_haptic(rig, bursts=3, per_burst=1, auto=True, progress=lambda *a: None)
    assert r["bursts_exact"] == 3 and abs(r["first_latency_s_median"] - 3.0) < 0.1 and abs(r["first_latency_s_max"] - 4.0) < 0.1


def test_burst_merge_and_miss_are_reported_as_fewer():
    d = {(0, 0): 2.0, (0, 1): 2.0, (1, 0): 4.0, (1, 1): 1.5}
    rig = haptic.SyntheticRig(delays=lambda b, i: d.get((b, i), 3.0), drop=[(2, 1)])
    r = haptic.run_haptic(rig, bursts=3, per_burst=2, gap_s=3, auto=True, progress=lambda *a: None)
    assert r["bursts_exact"] == 1 and r["bursts_short"] == 2 and "not reliable" in r["finding"]


def test_nothing_heard_is_inconclusive_not_failure():
    rig = haptic.SyntheticRig(buzz_db=-90.0)
    r = haptic.run_haptic(rig, bursts=2, per_burst=1, auto=True, progress=lambda *a: None)
    assert r["ok"] and r["buzzes_heard"] == 0 and "inconclusive" in r["finding"]


def test_all_sends_failing_is_not_ok():
    class Dead:
        label = "x"
        def run(self, *a, **k): return [[], []], np.zeros(FS, np.float32), 4
    r = haptic.run_haptic(Dead(), bursts=2, per_burst=2, auto=True, progress=lambda *a: None)
    assert r["ok"] is False and "No text could be sent" in r["finding"]


def test_http_sender_cases(monkeypatch):
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    sent = {}
    def ok(req, timeout=0):
        sent.update(json.loads(req.data)); return R(b'{"ok": true}')
    monkeypatch.setattr(haptic.urllib.request, "urlopen", ok)
    assert haptic.http_sender("https://x/api/notify", "pw")(2, 1) == (True, "")
    assert sent == {"pass": "pw", "n": 2, "i": 1, "test": False}
    def bad(req, timeout=0):
        raise urllib.error.HTTPError("u", 401, "x", {}, io.BytesIO(b'{"ok": false, "error": "Wrong passcode."}'))
    monkeypatch.setattr(haptic.urllib.request, "urlopen", bad)
    assert haptic.http_sender("https://x", "pw")(1, 1) == (False, "Wrong passcode.")
    def down(req, timeout=0): raise OSError("no route")
    monkeypatch.setattr(haptic.urllib.request, "urlopen", down)
    ok_, msg = haptic.http_sender("https://x", "pw")(1, 1)
    assert ok_ is False and "Could not reach" in msg


def test_cli_haptic_synthetic_and_not_in_all(tmp_path):
    assert cli.main(["--run", "haptic", "--synthetic", "--bursts", "3", "--out", str(tmp_path)]) == 0
    d = next(tmp_path.iterdir())
    data = json.loads((d / "results.json").read_text(encoding="utf-8"))
    assert [r["test"] for r in data["results"]] == ["haptic"] and "SYNTHETIC SIGNAL" in (d / "summary.md").read_text(encoding="utf-8")
    tmp2 = tmp_path / "all"
    cli.main(["--run", "all", "--synthetic", "--cost-seconds", "1", "--trials", "2", "--idle-minutes", "2", "--out", str(tmp2)])
    names = {r["test"] for r in json.loads((next(tmp2.iterdir()) / "results.json").read_text(encoding="utf-8"))["results"]}
    assert "haptic" not in names


def test_cli_real_haptic_needs_https_address_and_consent(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.AudioSource, "available", staticmethod(lambda: (True, "ok")))
    cli.main(["--run", "haptic", "--out", str(tmp_path / "a")])
    assert "https address" in (next((tmp_path / "a").iterdir()) / "summary.md").read_text(encoding="utf-8")
    monkeypatch.setenv("SENSOR_LAB_PASSCODE", "pw")
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    cli.main(["--run", "haptic", "--notify-url", "https://x/api/notify", "--out", str(tmp_path / "b")])
    assert "Cancelled before any text" in (next((tmp_path / "b").iterdir()) / "summary.md").read_text(encoding="utf-8")


def test_passcode_is_never_written_to_results(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.AudioSource, "available", staticmethod(lambda: (True, "ok")))
    monkeypatch.setenv("SENSOR_LAB_PASSCODE", "SUPERSECRETPASS")
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    cli.main(["--run", "haptic", "--notify-url", "https://x/api/notify", "--out", str(tmp_path)])
    for f in next(tmp_path.iterdir()).iterdir():
        assert "SUPERSECRETPASS" not in f.read_text(encoding="utf-8")
