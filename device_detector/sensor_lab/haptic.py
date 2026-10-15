"""Haptic latency test: send texts to your own phone and listen for each buzz with the laptop microphone.

Texts are sent in separate bursts, with a long pause between bursts, so each buzz can only belong to one burst. A burst of
one text measures the delay. A burst of two or more texts, sent at the gap the app uses, shows whether the buzzes stay
separate. The test measures: the delay from "request sent" to "buzz heard", how much it varies, whether a burst gives
the right number of buzzes, and whether the buzz is loud enough to tell from room noise.
What it cannot measure: the text message itself (it travels over the phone's cellular radio, which a laptop cannot
see), or a buzz that the microphone does not hear (phone in a pocket, in a hand, on a soft surface).
The phone must lie on a hard surface touching the same table as the laptop, and Vibrate on Silent must be on.
Audio is analysed in memory only and is never saved. The passcode is never written to any file.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import numpy as np

from . import dsp
from .sources import FS


# ---------------------------------------------------------------- detection (pure functions)
def band_envelope_db(x: np.ndarray, fs: int, band=(100.0, 400.0), win_s: float = 0.05, hop_s: float = 0.01):
    """Level in dB of the `band` frequencies, one value per hop. Returns (levels_db, hop_s)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    w, h = int(win_s * fs), int(hop_s * fs)
    if len(x) < w:
        return np.array([]), hop_s
    win = np.hanning(w)
    freqs = np.fft.rfftfreq(w, 1.0 / fs)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    out = []
    for s in range(0, len(x) - w + 1, h):
        spec = np.abs(np.fft.rfft(x[s:s + w] * win)) / w
        out.append(20.0 * np.log10(max(float(np.sqrt(np.mean(spec[sel] ** 2))), 1e-9)))
    return np.asarray(out), hop_s


def detect_buzz_onsets(x: np.ndarray, fs: int = FS, over_db: float = 10.0, min_len_s: float = 0.08, refractory_s: float = 0.6,
                       band=(100.0, 400.0)):
    """Times (seconds from the start of the recording) when the low-frequency band rises well above the room noise
    and stays up for at least `min_len_s`. Also returns the strongest rise over the noise in dB."""
    lv, hop = band_envelope_db(x, fs, band)
    if lv.size < 10:
        return [], 0.0
    base = float(np.median(lv))
    peak = float(np.max(lv) - base)
    up = lv > base + over_db
    need = max(1, int(round(min_len_s / hop)))
    onsets, i, last = [], 0, -1e9
    while i < len(up):
        if up[i]:
            j = i
            while j < len(up) and up[j]:
                j += 1
            t = i * hop
            if (j - i) >= need and t - last >= refractory_s:
                onsets.append(t)
                last = t
            i = j
        else:
            i += 1
    return onsets, peak


def summarize(bursts_sent, onsets, peak_db: float, gap_s: float, per_burst: int, burst_every_s: float, send_errors: int = 0) -> dict:
    """`bursts_sent` is a list of lists of send times (seconds from the start of the recording). Each burst owns the
    time from its first send until the next burst's first send."""
    starts = [b[0] for b in bursts_sent if b]
    rows = []
    for k, b in enumerate(bursts_sent):
        if not b:
            continue
        end = bursts_sent[k + 1][0] if k + 1 < len(bursts_sent) and bursts_sent[k + 1] else float("inf")
        ons = [o for o in onsets if b[0] <= o < end]
        rows.append({"sent": len(b), "heard": len(ons), "first_latency_s": (ons[0] - b[0]) if ons else None,
                     "min_spacing_s": float(min(np.diff(ons))) if len(ons) > 1 else None,
                     "last_latency_s": (ons[-1] - b[-1]) if ons else None})
    sent_total = sum(r["sent"] for r in rows)
    heard_total = sum(r["heard"] for r in rows)
    first = [r["first_latency_s"] for r in rows if r["first_latency_s"] is not None]
    spacings = [r["min_spacing_s"] for r in rows if r["min_spacing_s"] is not None]
    exact = sum(1 for r in rows if r["heard"] == r["sent"])
    short = sum(1 for r in rows if r["heard"] < r["sent"])
    over = sum(1 for r in rows if r["heard"] > r["sent"])
    res = {"test": "haptic", "ok": True, "bursts": len(rows), "per_burst": per_burst, "gap_s": gap_s, "texts_sent": sent_total,
           "send_errors": send_errors, "buzzes_heard": heard_total, "bursts_exact": exact, "bursts_short": short, "bursts_extra": over,
           "buzz_peak_over_noise_db": peak_db,
           "first_latency_s_median": float(np.median(first)) if first else None,
           "first_latency_s_min": float(min(first)) if first else None, "first_latency_s_max": float(max(first)) if first else None,
           "min_buzz_spacing_s": float(min(spacings)) if spacings else None, "rows": rows}
    if sent_total == 0:
        res["ok"] = False
        res["finding"] = "No text could be sent. Check the web address and the passcode, and read the error shown during the run."
        return res
    if heard_total == 0:
        res["finding"] = (f"No buzz was heard. The strongest rise over room noise was {peak_db:.1f} dB. This is inconclusive: the phone may have buzzed in a way "
                          "the microphone could not hear (soft surface, pocket, hand), or the texts did not arrive. Put the phone flat on a hard table that touches the "
                          "laptop's table and try again.")
        return res
    parts = []
    if first:
        spread = max(first) - min(first)
        parts.append(f"The first buzz of a burst came a median of {np.median(first):.1f} s after sending (range {min(first):.1f} to {max(first):.1f} s, spread {spread:.1f} s).")
    if per_burst > 1:
        parts.append(f"{exact} of {len(rows)} bursts gave exactly {per_burst} buzzes; {short} gave fewer (buzzes merged or missed) and {over} gave more.")
        if spacings:
            parts.append(f"The closest two buzzes in a burst were {min(spacings):.1f} s apart, for texts sent {gap_s:g} s apart.")
        if short:
            parts.append(f"Fewer buzzes than texts is inconclusive on its own (a quiet buzz can be missed), but it means a count of {per_burst} is not reliable at a {gap_s:g} s gap. Try a longer gap.")
    parts.append("Bursts where nothing was heard are inconclusive, not proof that nothing arrived.")
    res["finding"] = " ".join(parts)
    return res


# ---------------------------------------------------------------- rigs
def http_sender(url: str, passcode: str, timeout: float = 15.0):
    """Returns send(n, i) -> (ok, message). Posts to the person's own notify function."""
    def send(n: int, i: int, test: bool = False):
        body = json.dumps({"pass": passcode, "n": n, "i": i, "test": test}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.loads(r.read().decode() or "{}")
                return bool(j.get("ok")), j.get("error", "")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode()).get("error", f"HTTP {e.code}")
            except Exception:
                msg = f"HTTP {e.code}"
            return False, msg
        except Exception as e:
            return False, f"Could not reach the server ({type(e).__name__})."
    return send


class RealRig:
    label = "Phone and microphone (real hardware)"
    synthetic = False

    def __init__(self, send, clock=time.monotonic, sleep=time.sleep):
        self.send, self.clock, self.sleep = send, clock, sleep

    def run(self, bursts: int, per_burst: int, gap_s: float, burst_every_s: float, tail_s: float, progress=print):
        import sounddevice as sd
        total = 2.0 + (bursts - 1) * burst_every_s + (per_burst - 1) * gap_s + tail_s
        t_rec = self.clock()
        buf = sd.rec(int(total * FS), samplerate=FS, channels=1, dtype="float32")
        out, errors = [], 0
        for b in range(bursts):
            times = []
            for i in range(per_burst):
                target = t_rec + 2.0 + b * burst_every_s + i * gap_s
                while self.clock() < target:
                    self.sleep(0.01)
                t_send = self.clock() - t_rec
                ok, msg = self.send(per_burst, i + 1)
                if ok:
                    times.append(t_send)
                else:
                    errors += 1
                    progress(f"  burst {b + 1}, text {i + 1} failed: {msg}")
            out.append(times)
            progress(f"  burst {b + 1} of {bursts} sent ({len(times)} texts)")
        self.sleep(max(0.0, t_rec + total - self.clock()))
        sd.wait()
        return out, buf[:, 0].copy(), errors


class SyntheticRig:
    """SYNTHETIC: builds a recording with buzz bursts placed at send time plus a simulated, variable delivery delay.
    `delays` is a function (burst, index) -> seconds. `drop` is a set of (burst, index) that never buzz."""
    label = "SYNTHETIC SIGNAL: simulated phone"
    synthetic = True

    def __init__(self, delays=None, drop=(), seed=3, buzz_db=-30.0):
        self.rng = np.random.default_rng(seed)
        self.delays = delays or (lambda b, i: float(self.rng.uniform(1.5, 6.0)))
        self.drop, self.buzz_db = set(drop), buzz_db

    def run(self, bursts, per_burst, gap_s, burst_every_s, tail_s, progress=None):
        total = 2.0 + (bursts - 1) * burst_every_s + (per_burst - 1) * gap_s + tail_s
        x = self.rng.normal(0, 10 ** (-60 / 20), int(total * FS)).astype(np.float32)
        t = np.arange(int(0.45 * FS)) / FS
        buzz = ((10 ** (self.buzz_db / 20)) * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 9 * t))).astype(np.float32)
        out = []
        for b in range(bursts):
            times = []
            for i in range(per_burst):
                s = 2.0 + b * burst_every_s + i * gap_s
                times.append(s)
                if (b, i) in self.drop:
                    continue
                a = int((s + self.delays(b, i)) * FS)
                if a + len(buzz) < len(x):
                    x[a:a + len(buzz)] += buzz
            out.append(times)
        return out, x, 0


def run_haptic(rig, bursts: int = 5, per_burst: int = 2, gap_s: float = 3.0, burst_every_s: float = 30.0, tail_s: float = 20.0,
               ask=input, auto: bool = False, progress=print) -> dict:
    if not auto:
        ask("Put YOUR OWN phone flat on the same hard table as the laptop, with Vibrate on Silent ON. Press Enter to start... ")
    progress(f"{getattr(rig, 'label', '')}: {bursts} bursts of {per_burst} text(s), {gap_s:g} s apart inside a burst, one burst every {burst_every_s:g} s.")
    sent, mono, errors = rig.run(bursts, per_burst, gap_s, burst_every_s, tail_s, progress)
    onsets, peak = detect_buzz_onsets(mono, FS)
    return summarize(sent, onsets, peak, gap_s, per_burst, burst_every_s, errors)
