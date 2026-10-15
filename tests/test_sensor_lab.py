"""Sensor Lab tests. All inputs are SYNTHETIC. They check the maths and the flow, not real hardware."""
import json

import numpy as np
import pytest

from device_detector.sensor_lab import cli, dsp, labtests, report, sources


def test_gcc_phat_recovers_angles_within_resolution():
    for a in (-60, -30, 0, 30, 60):
        x = sources.make_clap_stereo(a, seed=2)
        got = dsp.clap_angle(x, 48000, 0.10)
        assert got is not None and abs(got - a) < 9      # one sample at 48 kHz and 10 cm is about 8 degrees near centre


def test_positive_angle_is_left_channel0():
    x = sources.make_clap_stereo(40, seed=1)
    assert dsp.side_from_angle(dsp.clap_angle(x, 48000, 0.10)) == "left"
    x = sources.make_clap_stereo(-40, seed=1)
    assert dsp.side_from_angle(dsp.clap_angle(x, 48000, 0.10)) == "right"


def test_no_clap_gives_none():
    x = np.random.default_rng(0).normal(0, 1e-4, (48000, 2)).astype(np.float32)
    assert dsp.clap_angle(x, 48000, 0.10) is None and dsp.side_from_angle(None) is None


def test_channels_report_cases():
    s = sources.make_clap_stereo(30)
    assert dsp.channels_report(s)["verdict"] == "different"
    assert dsp.channels_report(np.repeat(s[:, :1], 2, axis=1))["verdict"] == "identical"
    assert dsp.channels_report(np.zeros((1000, 2), np.float32))["verdict"] == "silent"


def test_side_from_x_and_motion_centroid():
    assert dsp.side_from_x(0.1) == "left" and dsp.side_from_x(0.9) == "right" and dsp.side_from_x(0.5) == "center" and dsp.side_from_x(None) is None
    cam = sources.SyntheticCamera(0.15)
    cam.read_gray(); a = cam.read_gray(); b = cam.read_gray()
    cx, frac = dsp.motion_centroid(a, b)
    assert cx is not None and cx < 0.3 and frac > 0.01
    still = np.full((50, 50), 7, np.uint8)
    assert dsp.motion_centroid(still, still)[0] is None


def test_path_loss_fit_recovers_known_model():
    d = [1, 2, 4, 8]
    r = [-58 - 10 * 2.5 * np.log10(x) for x in d]
    f = dsp.fit_path_loss(d, r)
    assert abs(f["n"] - 2.5) < 1e-6 and abs(f["A_1m_dbm"] + 58) < 1e-6 and f["resid_db"] < 1e-6
    assert dsp.fit_path_loss([1, 2], [-60, -66]) is None


def test_error_factor_and_separation():
    assert abs(dsp.distance_error_factor(0, 2.5) - 1.0) < 1e-9
    assert 1.8 < dsp.distance_error_factor(6, 2.0) < 2.1            # 6 dB at n=2 is about a factor 2
    assert dsp.separation(-60, 4, -62, 4) < 1 < dsp.separation(-60, 2, -75, 2)


def test_count_events_baseline_and_refractory():
    v = [-60.0] * 30 + [-30.0] * 3 + [-60.0] * 20 + [-30.0] + [-60.0] * 10
    assert dsp.count_events(v) == 2
    assert dsp.count_events([-60.0] * 10) == 0


def test_cost_test_runs_and_reports():
    r = labtests.run_cost(sources.SyntheticCamera(0.5), seconds=1.0, target_fps=20, sleep=lambda s: None)
    assert r["ok"] and r["frames"] > 3 and r["latency_ms_median"] is not None and "frames per second" in r["finding"]


def test_cost_missing_model_falls_back():
    r = labtests.run_cost(sources.SyntheticCamera(0.5), seconds=0.5, model_path="/nonexistent.onnx", sleep=lambda s: None)
    assert r["ok"] and "could not be loaded" in r["mode"]


class BadCam(sources.SyntheticCamera):
    def open(self):
        return False, "No camera could be opened."


def test_failing_camera_is_reported_not_raised():
    assert labtests.run_cost(BadCam())["ok"] is False
    assert labtests.run_direction(sources.SyntheticAudio(), BadCam(), auto=True)["ok"] is False
    assert labtests.run_idle(sources.SyntheticAudio(), BadCam(), sources.SyntheticBle(), minutes=1)["ok"] is False


def test_stereo_check_detects_merged_microphones():
    r = labtests.run_stereo_check(sources.SyntheticAudio(angle_deg=30, merged=True), auto=True)
    assert r["verdict"] == "identical" and "merges" in r["finding"]
    r = labtests.run_stereo_check(sources.SyntheticAudio(angle_deg=30), auto=True)
    assert r["verdict"] == "different"


def _direction(audio, cam, **kw):
    def driver(side):
        audio.set_angle(labtests.SIDE_ANGLE[side]); cam.set_x(labtests.SIDE_X[side])
    return labtests.run_direction(audio, cam, trials_per_side=3, auto=True, synthetic_driver=driver, **kw)


def test_direction_perfect_on_clean_synthetic_data():
    r = _direction(sources.SyntheticAudio(), sources.SyntheticCamera())
    assert r["sound_correct_pct"] == 100 and r["image_correct_pct"] == 100 and r["agreement_pct"] == 100 and not r["channels_look_swapped"]


def test_direction_merged_mics_is_honestly_poor():
    r = _direction(sources.SyntheticAudio(merged=True), sources.SyntheticCamera())
    assert r["sound_correct_pct"] < 60 and r["image_correct_pct"] == 100


def test_direction_swapped_channels_are_flagged():
    r = _direction(sources.SyntheticAudio(), sources.SyntheticCamera(), swap=True)
    assert r["channels_look_swapped"] is True and r["sound_correct_pct"] < 60


def test_ble_distance_overlap_is_reported():
    class Noisy(sources.SyntheticBle):
        def __init__(self): super().__init__(n=2.0, sigma=12.0)
    r = labtests.run_ble_distance(Noisy(), (1, 2, 4, 8), 60, auto=True)
    assert r["ok"] and r["distance_error_factor_1sigma"] > 2 and "overlap" in r["finding"]
    clean = labtests.run_ble_distance(sources.SyntheticBle(sigma=1.0), (1, 2, 4, 8), 60, auto=True)
    assert clean["distance_error_factor_1sigma"] < 1.3


def test_ble_distance_no_device_and_too_few_samples():
    class Nobody(sources.SyntheticBle):
        def pick(self, ask=input, auto=True): return None
    assert labtests.run_ble_distance(Nobody(), auto=True)["ok"] is False
    class Few(sources.SyntheticBle):
        def collect(self, seconds, distance=1.0): return [-60]
    assert "Too few" in labtests.run_ble_distance(Few(), auto=True)["finding"]


def test_idle_counts_false_alarms_and_identifiers():
    r = labtests.run_idle(sources.SyntheticAudio(clap=False), sources.SyntheticCamera(None), sources.SyntheticBle(), minutes=10)
    assert r["ok"] and r["audio_events"] == 0 and r["motion_events"] == 0 and r["new_ble_identifiers"] >= 6
    class Burst(sources.SyntheticAudio):
        n = 0
        def record(self, seconds):
            Burst.n += 1
            self.clap = Burst.n > 40 and Burst.n % 30 == 0      # quiet baseline, then a loud clap every 30 blocks
            return super().record(seconds)
    noisy = labtests.run_idle(Burst(), sources.SyntheticCamera(0.5), sources.SyntheticBle(), minutes=10)
    assert noisy["audio_events"] >= 3 and noisy["motion_events"] >= 1


def test_microphone_unavailable_message(monkeypatch):
    ok, msg = sources.AudioSource.available()
    if not ok:
        assert "sounddevice" in msg or "microphone" in msg.lower()


def test_cli_all_synthetic_writes_labeled_results(tmp_path, capsys):
    rc = cli.main(["--run", "all", "--synthetic", "--cost-seconds", "1", "--trials", "3", "--idle-minutes", "5", "--out", str(tmp_path)])
    assert rc == 0
    d = next(tmp_path.iterdir())
    data = json.loads((d / "results.json").read_text(encoding="utf-8"))
    assert data["meta"]["synthetic"] is True and {r["test"] for r in data["results"]} == {"cost", "stereo", "direction", "ble_distance", "idle"}
    md = (d / "summary.md").read_text(encoding="utf-8")
    assert "SYNTHETIC SIGNAL" in md and "inconclusive" in md
    csv_text = (d / "runs.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("when,synthetic,test") and len(csv_text.splitlines()) == 6
    assert "—" not in md


def test_cli_nothing_to_run(tmp_path):
    assert cli.main(["--run", "bogus", "--synthetic", "--out", str(tmp_path)]) == 1
