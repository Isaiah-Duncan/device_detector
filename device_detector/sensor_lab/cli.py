"""Sensor Lab command line. Run:  python -m device_detector.sensor_lab"""
from __future__ import annotations

import argparse
import sys

from . import SYNTHETIC_LABEL, haptic, labtests, report
from .sources import (AudioSource, CameraSource, RealBle, SyntheticAudio, SyntheticBle, SyntheticCamera)

INTRO = """Sensor Lab
Tests what this laptop can sense in its own room. Use it only in your own space, with your own devices and with
people who agree. Audio and images are analysed in memory and are never saved. Only numbers are written down.
"""


def build_sources(synthetic: bool, a):
    if synthetic:
        return SyntheticAudio(), SyntheticCamera(), SyntheticBle()
    return AudioSource(), CameraSource(a.camera_index), RealBle()


def run_selected(which: list[str], a, ask=None) -> list[dict]:
    ask = ask or (lambda prompt: input(prompt))      # looked up when called, so it can be replaced in tests
    syn = a.synthetic
    audio, cam, ble = build_sources(syn, a)
    out: list[dict] = []
    auto = syn
    if syn:
        print(f"{SYNTHETIC_LABEL}: simulated sources are used. No hardware is read.")
    for w in which:
        print(f"\n--- test: {w} ---")
        if w == "cost":
            out.append(labtests.run_cost(cam, seconds=a.cost_seconds, model_path=a.model, target_fps=a.fps,
                                         sleep=(lambda s: None) if syn else __import__("time").sleep))
        elif w in ("stereo", "direction"):
            if not syn:
                ok, msg = AudioSource.available()
                if not ok:
                    out.append({"test": w, "ok": False, "finding": msg})
                    print(msg)
                    continue
            if w == "stereo":
                out.append(labtests.run_stereo_check(audio, ask=ask, auto=auto))
            else:
                driver = None
                if syn:
                    def driver(side):
                        audio.set_angle(labtests.SIDE_ANGLE[side])
                        cam.set_x(labtests.SIDE_X[side])
                out.append(labtests.run_direction(audio, cam, a.trials, a.spacing, ask, auto, a.swap, driver))
        elif w == "ble":
            ok, msg = ble.start()
            if not ok:
                out.append({"test": "ble_distance", "ok": False, "finding": msg})
                print(msg)
                continue
            try:
                out.append(labtests.run_ble_distance(ble, a.distances, a.seconds_each, ask, auto))
            finally:
                ble.stop()
        elif w == "haptic":
            if syn:
                rig = haptic.SyntheticRig()
            else:
                ok, msg = AudioSource.available()
                if not ok:
                    out.append({"test": "haptic", "ok": False, "finding": msg})
                    print(msg)
                    continue
                if not a.notify_url.startswith("https://"):
                    msg = "Give the https address of your own notify function with --notify-url (for example https://your-site.vercel.app/api/notify)."
                    out.append({"test": "haptic", "ok": False, "finding": msg})
                    print(msg)
                    continue
                import getpass
                import os
                pw = os.environ.get("SENSOR_LAB_PASSCODE") or getpass.getpass("Passcode for your notify function (not saved): ")
                n_texts = a.bursts * a.per_burst
                if ask(f"This sends {n_texts} real text messages to your own phone and may cost money on your SMS plan. Type y to continue: ").strip().lower() != "y":
                    out.append({"test": "haptic", "ok": False, "finding": "Cancelled before any text was sent."})
                    continue
                rig = haptic.RealRig(haptic.http_sender(a.notify_url, pw))
            out.append(haptic.run_haptic(rig, a.bursts, a.per_burst, a.haptic_gap, a.burst_every, 20.0, ask, auto))
        elif w == "idle":
            if syn:
                audio, cam = SyntheticAudio(clap=False), SyntheticCamera(x_frac=None)
            else:
                ok, msg = AudioSource.available()
                if not ok:
                    out.append({"test": "idle", "ok": False, "finding": msg})
                    print(msg)
                    continue
            ok, msg = ble.start()
            try:
                out.append(labtests.run_idle(audio, cam, ble, a.idle_minutes,
                                             progress=None if syn else (lambda e, t: print(f"\r  {e / 60:.1f} of {t / 60:.0f} min", end="", flush=True))))
            finally:
                ble.stop()
        print(out[-1].get("finding", ""))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sensor_lab", description="Measure what this laptop can sense in its own room.")
    p.add_argument("--run", default="menu", help="menu | all | cost | stereo | direction | ble | idle (comma separated)")
    p.add_argument("--synthetic", action="store_true", help="use simulated sources (labeled SYNTHETIC SIGNAL)")
    p.add_argument("--out", default="sensor_lab_results")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--model", default="", help="optional YOLOv8 .onnx file for the cost test")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--cost-seconds", type=float, default=20.0)
    p.add_argument("--spacing", type=float, default=0.10, help="distance between the two microphones in meters")
    p.add_argument("--swap", action="store_true", help="swap left and right microphone channels")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--distances", type=float, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seconds-each", type=float, default=60.0)
    p.add_argument("--idle-minutes", type=float, default=30.0)
    p.add_argument("--notify-url", default="", help="https address of your own /api/notify function (haptic test)")
    p.add_argument("--bursts", type=int, default=5, help="haptic test: number of bursts")
    p.add_argument("--per-burst", type=int, default=2, help="haptic test: texts in each burst (1 measures delay only)")
    p.add_argument("--haptic-gap", type=float, default=3.0, help="haptic test: seconds between texts inside a burst")
    p.add_argument("--burst-every", type=float, default=30.0, help="haptic test: seconds between bursts")
    a = p.parse_args(argv)
    print(INTRO)
    names = {"1": "cost", "2": "stereo", "3": "direction", "4": "ble", "5": "idle", "6": "haptic"}
    if a.run == "menu":
        print("1 Cost   2 Microphone channels   3 Direction   4 Bluetooth distance   5 Idle false alarms   6 Haptic delay (sends real texts)   A All five tests   Q Quit")
        c = input("Choose: ").strip().lower()
        if c == "q":
            return 0
        which = [v for v in names.values() if v != "haptic"] if c == "a" else [names.get(c, "")]
    else:
        which = [v for v in names.values() if v != "haptic"] if a.run == "all" else [w.strip() for w in a.run.split(",")]
    which = [w for w in which if w in names.values()]
    if not which:
        print("Nothing to run.")
        return 1
    res = run_selected(which, a)
    d = report.write_results(res, a.out, a.synthetic)
    print(f"\nResults written to: {d}\nOpen summary.md to read them, and runs.csv in a spreadsheet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
