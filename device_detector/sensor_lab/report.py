"""Write results as JSON, a one-row-per-run CSV (for a spreadsheet) and a plain-language summary."""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

from . import SYNTHETIC_LABEL

HONEST = ("These numbers describe this laptop, in this room, on this day. They do not describe other laptops or rooms. "
          "A test that finds nothing is inconclusive, not proof that nothing is there.")


def machine_info() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(), "processor": platform.processor()}
    try:
        import psutil
        info["cores"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return info


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k in ("rows", "adjacent_separation", "finding"):
            continue
        key = prefix + k
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, float):
            out[key] = round(v, 3)
        elif v is None or isinstance(v, (int, str, bool)):
            out[key] = v
    return out


def write_results(results: list[dict], out_dir: str | Path, synthetic: bool) -> Path:
    out = Path(out_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "synthetic": synthetic, "machine": machine_info(), "notice": HONEST}
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=str), encoding="utf-8")
    rows = [{"when": meta["when"], "synthetic": int(synthetic), **_flat(r)} for r in results]
    lead = ["when", "synthetic", "test"]
    cols = lead + sorted({k for r in rows for k in r} - set(lead))
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text(summary(results, meta), encoding="utf-8")
    return out


TITLES = {"cost": "1. What the program costs", "stereo": "2. Does the microphone give two channels?",
          "direction": "3. Sound direction against camera position", "ble_distance": "4. Bluetooth signal against distance",
          "idle": "5. False alarms in a quiet room", "haptic": "6. Haptic delay (text to buzz)"}


def summary(results: list[dict], meta: dict) -> str:
    L = ["# Sensor Lab summary", ""]
    if meta["synthetic"]:
        L += [f"**{SYNTHETIC_LABEL}: every number below comes from simulated sources. None of it describes real hardware.**", ""]
    L += [f"Date: {meta['when']}", f"Computer: {meta['machine'].get('os', '?')}, {meta['machine'].get('cores', '?')} logical cores, "
          f"{meta['machine'].get('ram_gb', '?')} GB RAM", ""]
    for r in results:
        L += [f"## {TITLES.get(r.get('test'), r.get('test'))}", ""]
        if not r.get("ok", False):
            L += [f"Not completed: {r.get('finding', 'unknown reason')}", ""]
            continue
        L += [r.get("finding", ""), ""]
        t = r["test"]
        if t == "cost":
            L += [f"- Frames per second: {r['fps_achieved']} (target {r['fps_target']:g})",
                  f"- Time per frame: median {r['latency_ms_median']:.1f} ms, slowest 5% above {r['latency_ms_p95']:.1f} ms",
                  f"- Program CPU: {r['cpu_process_pct_mean']:.0f}% of one core on average" if r.get("cpu_process_pct_mean") is not None else "- Program CPU: not measured",
                  f"- Memory: {r['memory_mb']} MB",
                  f"- Hottest sensor: {r['temperature_c_max']} C" if r.get("temperature_c_max") is not None else "- Temperature: Windows does not give this to normal programs"]
        elif t == "ble_distance":
            L += ["| Distance (m) | Samples | Mean RSSI (dBm) | Spread (dB) |", "|---|---|---|---|"]
            for x in r["rows"]:
                L.append(f"| {x['distance_m']} | {x['n']} | " + (f"{x['mean']:.1f} | {x['std']:.1f} |" if x["mean"] is not None else "n/a | n/a |"))
        elif t == "direction":
            L += [f"- Trials: {r['trials']}; sound found no clap in {r['sound_missed']}; camera saw no motion in {r['image_missed']}",
                  f"- Microphone spacing assumed: {r['mic_spacing_m'] * 100:.0f} cm (a wrong value scales every angle)"]
        elif t == "haptic":
            L += ["| Burst | Texts sent | Buzzes heard | First buzz after (s) |", "|---|---|---|---|"]
            for i, x in enumerate(r["rows"], 1):
                L.append(f"| {i} | {x['sent']} | {x['heard']} | " + (f"{x['first_latency_s']:.1f} |" if x["first_latency_s"] is not None else "none |"))
            L += ["", f"- Strongest buzz above room noise: {r['buzz_peak_over_noise_db']:.1f} dB",
                  "- The text itself is not visible to a laptop. Only the buzz heard by the microphone is measured."]
        elif t == "idle":
            L += [f"- Sound events per hour: {r['audio_events_per_hour']:.1f}", f"- Motion events per hour: {r['motion_events_per_hour']:.1f}",
                  f"- New Bluetooth identifiers per hour: {r['new_ble_identifiers_per_hour']:.0f}"]
        L.append("")
    L += ["## What these results can and cannot say", "", HONEST, ""]
    return "\n".join(L)
