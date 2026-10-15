"""Input sources for the tests: real hardware or SYNTHETIC SIGNAL. Imports of hardware libraries are lazy."""
from __future__ import annotations

import time

import numpy as np

from . import dsp

FS = 48000


def make_clap_stereo(angle_deg: float, fs: int = FS, spacing_m: float = 0.10, seconds: float = 1.0, at: float = 0.4,
                     noise_db: float = -60.0, seed: int = 0) -> np.ndarray:
    """SYNTHETIC two-channel recording with one clap. Positive angle = source toward channel 0 (left)."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    x = rng.normal(0, 10 ** (noise_db / 20), (n, 2))
    burst = rng.normal(0, 1, int(fs * 0.004)) * np.hanning(int(fs * 0.004)) * 0.6
    tau = spacing_m * np.sin(np.radians(angle_deg)) / dsp.SPEED_OF_SOUND
    d = int(round(abs(tau) * fs))
    s = int(at * fs)
    first, second = (0, 1) if angle_deg >= 0 else (1, 0)
    x[s:s + len(burst), first] += burst
    x[s + d:s + d + len(burst), second] += burst
    return x.astype(np.float32)


class AudioSource:
    label = "Microphone (real hardware)"
    synthetic = False

    def __init__(self, device=None, channels: int = 2):
        self.device, self.channels = device, channels

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import sounddevice as sd
            sd.query_devices(kind="input")
            return True, "Microphone found."
        except Exception as e:
            return False, f"No usable microphone interface ({type(e).__name__}). Install with: pip install sounddevice"

    def record(self, seconds: float) -> np.ndarray:
        import sounddevice as sd
        data = sd.rec(int(seconds * FS), samplerate=FS, channels=self.channels, dtype="float32", device=self.device)
        sd.wait()
        return data


class SyntheticAudio:
    label = "SYNTHETIC SIGNAL: simulated microphone"
    synthetic = True

    def __init__(self, angle_deg: float = 0.0, merged: bool = False, spacing_m: float = 0.10, seed: int = 1, clap: bool = True):
        self.angle, self.merged, self.spacing, self.seed, self.clap = angle_deg, merged, spacing_m, seed, clap

    def set_angle(self, a: float) -> None:
        self.angle = a

    def record(self, seconds: float) -> np.ndarray:
        self.seed += 1
        x = make_clap_stereo(self.angle, FS, self.spacing, seconds, at=min(0.4, seconds / 2), seed=self.seed)
        if not self.clap:
            x = np.random.default_rng(self.seed).normal(0, 10 ** (-60 / 20), x.shape).astype(np.float32)
        if self.merged:
            m = x.mean(axis=1, keepdims=True)
            x = np.repeat(m, 2, axis=1)
        return x


class CameraSource:
    label = "Webcam (real hardware)"
    synthetic = False

    def __init__(self, index: int = 0):
        self.index, self.cam = index, None

    def open(self) -> tuple[bool, str]:
        from ..vision import CameraSession
        self.cam = CameraSession()
        return self.cam.open(self.index)

    def read_gray(self):
        import cv2
        f = self.cam.read() if self.cam else None
        return None if f is None else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    def read_bgr(self):
        return self.cam.read() if self.cam else None

    def close(self):
        if self.cam:
            self.cam.close()


class SyntheticCamera:
    """SYNTHETIC frames: a bright block that sits at a chosen horizontal position (0 left .. 1 right) and flickers."""
    label = "SYNTHETIC SIGNAL: simulated camera"
    synthetic = True

    def __init__(self, x_frac: float | None = 0.5, w: int = 320, h: int = 240):
        self.x, self.w, self.h, self.t = x_frac, w, h, 0

    def set_x(self, x):
        self.x = x

    def open(self):
        return True, "Simulated camera."

    def read_gray(self):
        self.t += 1
        g = np.full((self.h, self.w), 40, np.uint8)
        if self.x is not None and self.t % 2 == 0:
            cx = int(self.x * self.w)
            g[self.h // 3: 2 * self.h // 3, max(0, cx - 20): cx + 20] = 220
        return g

    def read_bgr(self):
        g = self.read_gray()
        return np.stack([g, g, g], axis=2)

    def close(self):
        pass


def wait_ready(prompt: str, auto: bool, input_fn=input) -> None:
    """Pause until the person presses Enter. In simulated runs there is no pause."""
    if not auto:
        input_fn(prompt + "  Press Enter when ready... ")


# ---------------------------------------------------------------- Bluetooth sources
class RealBle:
    label = "Bluetooth LE (real hardware)"
    synthetic = False

    def __init__(self):
        from ..scanner import BleScanner
        self.sc = BleScanner()
        self.started = False

    def start(self) -> tuple[bool, str]:
        ok = self.sc.start()
        self.started = ok
        return ok, self.sc.message

    def stop(self):
        self.sc.stop()

    def poll(self):
        return self.sc.read_all()

    def pick(self, ask=input, auto=False):
        """Listen for 8 seconds, list the strongest devices, and let the person choose their own device."""
        from ..tracker import Tracker
        tr, t_end = Tracker(), time.monotonic() + 8.0
        print("Listening for 8 seconds. Switch your own device on or put it in pairing mode now.")
        while time.monotonic() < t_end:
            for o in self.poll():
                tr.feed(o)
            time.sleep(0.2)
        devs = sorted(tr.devices.values(), key=lambda r: -r.count)[:8]
        if not devs:
            return None
        for i, r in enumerate(devs, 1):
            last = r.rssi[-1][1] if r.rssi else "n/a"
            print(f"  {i}. {r.key}  {r.cls.category} ({r.cls.confidence})  heard {r.count} times  last RSSI {last} dBm")
        k = ask("Choose the number of YOUR OWN device (it must advertise often, for example earbuds in pairing mode): ")
        try:
            self.target = devs[int(k) - 1].key
            return self.target
        except Exception:
            return None

    def collect(self, seconds: float, distance=None):
        key = getattr(self, "target", None)
        out, t_end = [], time.monotonic() + seconds
        while time.monotonic() < t_end:
            for o in self.poll():
                if o.key == key and o.rssi is not None:
                    out.append(o.rssi)
            time.sleep(0.2)
        return out


class SyntheticBle:
    """SYNTHETIC: RSSI follows a log-distance model with shadowing, so the analysis can be checked against a known answer."""
    label = "SYNTHETIC SIGNAL: simulated Bluetooth"
    synthetic = True

    def __init__(self, n: float = 2.6, a_1m: float = -58.0, sigma: float = 4.0, seed: int = 5):
        from ..synthetic import SyntheticSource
        self.n, self.a, self.sigma = n, a_1m, sigma
        self.rng = np.random.default_rng(seed)
        self.src, self.t = SyntheticSource(seed=seed), 0.0

    def start(self):
        return True, "Simulated Bluetooth."

    def stop(self):
        pass

    def poll(self):
        self.t += 1.0
        return self.src.poll(1000.0 + self.t)

    def pick(self, ask=input, auto=True):
        self.target = "D-synthetic"
        return self.target

    def collect(self, seconds: float, distance=1.0):
        n = int(seconds * 1.0)
        mu = self.a - 10 * self.n * np.log10(max(distance, 0.1))
        return list(np.round(mu + self.rng.normal(0, self.sigma, n)))
