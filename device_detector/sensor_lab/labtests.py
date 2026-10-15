"""The five tests. Each function takes sources, returns a plain dict of numbers and a short plain-language finding.

Nothing here saves audio, images or addresses. Only counts, levels, angles and timings leave a test.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import SYNTHETIC_LABEL, dsp
from .sources import FS, wait_ready

SIDE_X = {"left": 0.15, "center": 0.5, "right": 0.85}
SIDE_ANGLE = {"left": 40.0, "center": 0.0, "right": -40.0}


def _pct(v, p):
    return float(np.percentile(v, p)) if len(v) else None


# ---------------------------------------------------------------- test 1: cost
def run_cost(camera, seconds: float = 20.0, model_path: str = "", target_fps: float = 5.0, clock=time.perf_counter,
             sleep=time.sleep) -> dict:
    """Run the camera loop with motion detection (and the ONNX model if given) and measure the cost."""
    import psutil
    proc = psutil.Process()
    detector, note = None, "Motion detection only (no model chosen)."
    if model_path:
        try:
            from ..vision import OnnxDetector
            detector = OnnxDetector(model_path)
            note = "Motion detection plus the ONNX model."
        except Exception as e:
            note = f"Model could not be loaded ({type(e).__name__}). Motion detection only."
    ok, msg = camera.open()
    if not ok:
        return {"test": "cost", "ok": False, "finding": msg}
    proc.cpu_percent(None)
    psutil.cpu_percent(None)
    lat, cpu_p, cpu_s, prev = [], [], [], None
    t0 = clock()
    next_cpu = t0 + 1.0
    frames, period = 0, 1.0 / max(target_fps, 0.5)
    try:
        while clock() - t0 < seconds:
            t = clock()
            g = camera.read_gray()
            if g is not None:
                if prev is not None:
                    dsp.motion_centroid(prev, g)
                prev = g
                if detector is not None:
                    bgr = camera.read_bgr()
                    if bgr is not None:
                        detector.detect(bgr)
                _ = np.abs(np.fft.rfft(np.random.default_rng(0).normal(size=4096)))   # stand-in for one analyzer frame
                frames += 1
                lat.append((clock() - t) * 1000.0)
            if clock() >= next_cpu:
                cpu_p.append(proc.cpu_percent(None))
                cpu_s.append(psutil.cpu_percent(None))
                next_cpu += 1.0
            rest = period - (clock() - t)
            if rest > 0:
                sleep(rest)
    finally:
        camera.close()
    elapsed = max(clock() - t0, 1e-6)
    temp = None
    try:
        tt = psutil.sensors_temperatures()
        vals = [x.current for v in tt.values() for x in v]
        temp = max(vals) if vals else None
    except Exception:
        pass
    res = {"test": "cost", "ok": True, "mode": note, "seconds": round(elapsed, 1), "frames": frames,
           "fps_achieved": round(frames / elapsed, 2), "fps_target": target_fps,
           "latency_ms_median": _pct(lat, 50), "latency_ms_p95": _pct(lat, 95),
           "cpu_process_pct_mean": float(np.mean(cpu_p)) if cpu_p else None,
           "cpu_system_pct_mean": float(np.mean(cpu_s)) if cpu_s else None,
           "memory_mb": round(proc.memory_info().rss / 1e6, 1), "temperature_c_max": temp,
           "cores": psutil.cpu_count(logical=True)}
    keep = res["fps_achieved"] >= 0.9 * target_fps
    res["finding"] = (f"{note} The program kept up with {target_fps:g} frames per second." if keep else
                      f"{note} The program reached only {res['fps_achieved']:.1f} of {target_fps:g} frames per second. "
                      "Lower the frame rate or use a smaller model.")
    return res


# ---------------------------------------------------------------- test 2: stereo check
def run_stereo_check(audio, seconds: float = 3.0, ask=input, auto: bool = False) -> dict:
    wait_ready("Make some steady noise (talk or clap a few times) for 3 seconds after you press Enter.", auto, ask)
    x = audio.record(seconds)
    if x.ndim < 2 or x.shape[1] < 2:
        return {"test": "stereo", "ok": True, "verdict": "mono", "finding": "The microphone delivers one channel only. Direction finding is not possible."}
    rep = dsp.channels_report(x)
    return {"test": "stereo", "ok": True, "verdict": rep["verdict"], "correlation": rep["correlation"],
            "diff_db": rep["diff_db"], "level_db": dsp.rms_db(x), "finding": rep["message"]}


# ---------------------------------------------------------------- test 3: direction trials
def _record_with_camera(audio, camera, seconds: float):
    box = {}
    th = threading.Thread(target=lambda: box.update(x=audio.record(seconds)))
    th.start()
    prev, best, best_x = None, 0.0, None
    t0 = time.perf_counter()
    while th.is_alive() or time.perf_counter() - t0 < 0.2:
        g = camera.read_gray()
        if g is not None:
            if prev is not None:
                cx, frac = dsp.motion_centroid(prev, g)
                if cx is not None and frac > best:
                    best, best_x = frac, cx
            prev = g
        time.sleep(0.03)
        if not th.is_alive() and time.perf_counter() - t0 > seconds:
            break
    th.join()
    return box["x"], best_x


def run_direction(audio, camera, trials_per_side: int = 10, spacing_m: float = 0.10, ask=input, auto: bool = False,
                  swap: bool = False, synthetic_driver=None, mirror_camera: bool = False) -> dict:
    """Clap at left, center and right (with a hand, a phone screen or any moving object in view). Compare the sound
    direction, the image position and the truth. `synthetic_driver(side)` sets up simulated sources."""
    ok, msg = camera.open()
    if not ok:
        return {"test": "direction", "ok": False, "finding": msg}
    rows = []
    try:
        for side in ("left", "right", "center"):
            wait_ready(f"Stand or hold your object at the {side.upper()} of the camera view (as the camera sees it). "
                       f"You will clap {trials_per_side} times, one clap for each prompt.", auto, ask)
            for i in range(trials_per_side):
                if synthetic_driver:
                    synthetic_driver(side)
                elif not auto:
                    ask(f"  Clap {i + 1} of {trials_per_side} at the {side}: press Enter, then clap within 1 second... ")
                x, cx = _record_with_camera(audio, camera, 1.6)
                if swap:
                    x = x[:, ::-1]
                ang = dsp.clap_angle(x, FS, spacing_m)
                img_side = dsp.side_from_x(1.0 - cx if (mirror_camera and cx is not None) else cx)
                rows.append({"truth": side, "angle": ang, "sound_side": dsp.side_from_angle(ang), "image_side": img_side})
    finally:
        camera.close()
    n = len(rows)
    sound_ok = sum(r["sound_side"] == r["truth"] for r in rows)
    img_ok = sum(r["image_side"] == r["truth"] for r in rows)
    agree = sum(r["sound_side"] is not None and r["sound_side"] == r["image_side"] for r in rows)
    none_sound = sum(r["sound_side"] is None for r in rows)
    none_img = sum(r["image_side"] is None for r in rows)
    la = [r["angle"] for r in rows if r["truth"] == "left" and r["angle"] is not None]
    ra = [r["angle"] for r in rows if r["truth"] == "right" and r["angle"] is not None]
    swapped = bool(la and ra and np.mean(la) < np.mean(ra))
    res = {"test": "direction", "ok": True, "trials": n, "sound_correct_pct": round(100 * sound_ok / n, 1),
           "image_correct_pct": round(100 * img_ok / n, 1), "agreement_pct": round(100 * agree / n, 1),
           "sound_missed": none_sound, "image_missed": none_img,
           "mean_angle_left_deg": float(np.mean(la)) if la else None, "mean_angle_right_deg": float(np.mean(ra)) if ra else None,
           "channels_look_swapped": swapped, "mic_spacing_m": spacing_m}
    chance = 100 / 3
    res["finding"] = (f"Sound direction was right {res['sound_correct_pct']}% of the time (guessing gives about {chance:.0f}%). "
                      f"The camera position was right {res['image_correct_pct']}%. They agreed with each other {res['agreement_pct']}%."
                      + (" The left and right channels look swapped: run again with the swap option." if swapped else ""))
    return res


# ---------------------------------------------------------------- test 4: bluetooth distance
def run_ble_distance(collector, distances=(1, 2, 4, 8), seconds_each: float = 60.0, ask=input, auto: bool = False) -> dict:
    """`collector.pick(ask)` chooses the target device. `collector.collect(seconds, distance)` returns RSSI samples."""
    target = collector.pick(ask, auto)
    if target is None:
        return {"test": "ble_distance", "ok": False, "finding": "No device was chosen or no device was heard."}
    rows = []
    for d in distances:
        wait_ready(f"Place YOUR OWN device {d} m from the laptop, in clear view of it.", auto, ask)
        v = np.asarray(collector.collect(seconds_each, d), dtype=float)
        if len(v) < 3:
            rows.append({"distance_m": d, "n": int(len(v)), "mean": None, "std": None})
            continue
        rows.append({"distance_m": d, "n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                     "min": float(v.min()), "max": float(v.max()), "adv_per_s": float(len(v) / seconds_each)})
    good = [r for r in rows if r["mean"] is not None]
    fit = dsp.fit_path_loss([r["distance_m"] for r in good], [r["mean"] for r in good]) if len(good) >= 3 else None
    seps = []
    for a, b in zip(good, good[1:]):
        seps.append({"from_m": a["distance_m"], "to_m": b["distance_m"], "separation": dsp.separation(a["mean"], a["std"], b["mean"], b["std"])})
    res = {"test": "ble_distance", "ok": True, "rows": rows, "fit": fit, "adjacent_separation": seps}
    if fit:
        within = float(np.sqrt(np.mean([r["std"] ** 2 for r in good])))
        sigma = float(np.sqrt(fit["resid_db"] ** 2 + within ** 2))   # error of ONE reading: model misfit plus reading-to-reading scatter
        fac = dsp.distance_error_factor(sigma, fit["n"])
        res["single_reading_sigma_db"] = sigma
        res["distance_error_factor_1sigma"] = fac
        bad = [s for s in seps if s["separation"] < 1.0]
        res["finding"] = (f"Fitted path-loss exponent n = {fit['n']:.2f} (free space is 2). One reading is uncertain by about {sigma:.1f} dB "
                          f"({fit['resid_db']:.1f} dB model misfit plus {within:.1f} dB reading-to-reading scatter), so a single distance estimate is typically off by a factor of about {fac:.1f} "
                          f"(for example, 4 m may read as {4 / fac:.1f} to {4 * fac:.1f} m)."
                          + (f" {len(bad)} of {len(seps)} neighbouring distance pairs overlap heavily, so those distances cannot be told apart." if bad else " Neighbouring distances separated in this run."))
    else:
        res["finding"] = "Too few distances with enough samples to fit a model. Increase the time at each distance."
    return res


# ---------------------------------------------------------------- test 5: idle false alarms
def run_idle(audio, camera, ble, minutes: float = 30.0, block_s: float = 1.0, clock=time.monotonic, sleep=time.sleep,
             progress=None) -> dict:
    """Leave the room quiet and still. Count how many events the detectors report with no real cause."""
    ok, msg = camera.open()
    if not ok:
        return {"test": "idle", "ok": False, "finding": msg}
    levels, motion_events, prev = [], 0, None
    ids = set()
    virtual = bool(getattr(audio, "synthetic", False))
    vt = 0.0
    t0 = clock()
    total = minutes * 60.0
    cool = 0
    try:
        while (vt if virtual else clock() - t0) < total:
            x = audio.record(block_s)
            levels.append(dsp.rms_db(x))
            g = camera.read_gray()
            if g is not None:
                if prev is not None:
                    cx, frac = dsp.motion_centroid(prev, g)
                    if cool > 0:
                        cool -= 1
                    elif cx is not None and frac > 0.01:
                        motion_events += 1
                        cool = 3
                prev = g
            for o in ble.poll():
                ids.add(o.key)
            vt += block_s
            if progress:
                progress(vt if virtual else clock() - t0, total)
    finally:
        camera.close()
    hours = max(total / 3600.0, 1e-9)
    a_ev = dsp.count_events(levels)
    res = {"test": "idle", "ok": True, "minutes": minutes, "audio_events": a_ev, "motion_events": motion_events,
           "new_ble_identifiers": len(ids), "audio_events_per_hour": a_ev / hours, "motion_events_per_hour": motion_events / hours,
           "new_ble_identifiers_per_hour": len(ids) / hours,
           "audio_level_db_median": float(np.median(levels)) if levels else None,
           "audio_level_db_p95": _pct(levels, 95)}
    res["finding"] = (f"In {minutes:g} minutes of a quiet, still room the program counted {a_ev} sound events, {motion_events} motion events "
                      f"and {len(ids)} Bluetooth identifiers. Every sound or motion event here is a false alarm unless something really happened. "
                      "Identifiers are not false alarms: they show how many addresses nearby devices used.")
    return res
