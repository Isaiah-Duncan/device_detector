"""Signal processing used by the tests. All functions are pure and take NumPy arrays."""
from __future__ import annotations

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at room temperature


def rms_db(x: np.ndarray) -> float:
    """Level in dB relative to full scale (1.0). Silence returns -120."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(x ** 2)))
    return 20.0 * np.log10(max(r, 1e-6))


def channels_report(stereo: np.ndarray) -> dict:
    """Decide whether two microphone channels carry different information.

    If Windows merges the microphones before the program sees them, both channels are identical and a
    direction estimate is not possible."""
    a, b = stereo[:, 0].astype(np.float64), stereo[:, 1].astype(np.float64)
    ra, rb = float(np.sqrt(np.mean(a ** 2))), float(np.sqrt(np.mean(b ** 2)))
    if max(ra, rb) < 1e-5:
        return {"verdict": "silent", "correlation": None, "diff_db": None,
                "message": "Both channels are silent. Check that the microphone is not muted."}
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 1.0
    diff = rms_db(a - b) - rms_db(a)
    if corr > 0.99995 and diff < -60:
        v, m = "identical", "The two channels are identical. Windows or the driver merges the microphones, so direction finding is not possible here."
    else:
        v, m = "different", "The two channels differ. Direction finding may be possible."
    return {"verdict": v, "correlation": corr, "diff_db": float(diff), "message": m}


def gcc_phat(a: np.ndarray, b: np.ndarray, fs: int, max_tau: float) -> float:
    """Time difference of arrival in seconds, by generalized cross-correlation with phase transform.

    Positive result: the sound reached channel `a` EARLIER than channel `b` (the source is nearer to `a`).
    A parabolic fit gives sub-sample resolution."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a) + len(b)
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    R = np.conj(A) * B
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    m = max(1, int(fs * max_tau))
    cc = np.concatenate((cc[-m:], cc[:m + 1]))
    k = int(np.argmax(cc))
    shift = float(k - m)
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            shift += 0.5 * (y0 - y2) / den
    return shift / fs


def angle_from_tau(tau: float, spacing_m: float) -> float:
    """Angle in degrees from straight ahead. Positive = toward channel 0 (left). Clipped to +/- 90."""
    s = np.clip(SPEED_OF_SOUND * tau / max(spacing_m, 1e-6), -1.0, 1.0)
    return float(np.degrees(np.arcsin(s)))


def find_onset(x: np.ndarray, fs: int, thresh_db: float = -35.0, win_ms: float = 25.0):
    """Index of the first loud sample (a clap), or None."""
    x = np.asarray(x, dtype=np.float64)
    th = 10 ** (thresh_db / 20.0)
    idx = np.flatnonzero(np.abs(x) > th)
    return int(idx[0]) if idx.size else None


def clap_angle(stereo: np.ndarray, fs: int, spacing_m: float, win_ms: float = 25.0):
    """Angle of a clap in a 2-channel recording, or None if no clap was found."""
    mono = stereo.mean(axis=1)
    on = find_onset(mono, fs)
    if on is None:
        return None
    w = int(fs * win_ms / 1000.0)
    s = max(0, on - w // 4)
    seg = stereo[s:s + w]
    if len(seg) < w // 2:
        return None
    tau = gcc_phat(seg[:, 0], seg[:, 1], fs, spacing_m / SPEED_OF_SOUND * 1.2)
    return angle_from_tau(tau, spacing_m)


def side_from_angle(angle: float | None, dead_zone_deg: float = 15.0) -> str | None:
    if angle is None:
        return None
    return "left" if angle > dead_zone_deg else "right" if angle < -dead_zone_deg else "center"


def side_from_x(x_frac: float | None, band: float = 0.2) -> str | None:
    """Image side from a horizontal position 0..1 (0 = left edge of the image as the viewer sees it)."""
    if x_frac is None:
        return None
    return "left" if x_frac < 0.5 - band / 2 else "right" if x_frac > 0.5 + band / 2 else "center"


def motion_centroid(prev_gray: np.ndarray, gray: np.ndarray, thr: int = 25, min_frac: float = 0.002):
    """Horizontal centre (0..1) of the changed pixels between two grey frames, or None if too little changed."""
    d = np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)) > thr
    frac = float(d.mean())
    if frac < min_frac:
        return None, frac
    xs = np.flatnonzero(d.any(axis=0))
    w = d.sum(axis=0).astype(np.float64)
    cx = float((np.arange(d.shape[1]) * w).sum() / w.sum()) / d.shape[1]
    return cx, frac


def fit_path_loss(distances_m, rssi_means):
    """Least-squares fit rssi = A - 10 n log10(d). Returns A (dBm at 1 m), n, and the residual std in dB."""
    d = np.asarray(distances_m, dtype=np.float64)
    r = np.asarray(rssi_means, dtype=np.float64)
    if len(d) < 3 or np.ptp(d) == 0:
        return None
    x = -10.0 * np.log10(d)
    n, A = np.polyfit(x, r, 1)
    res = r - (A + n * x)
    return {"A_1m_dbm": float(A), "n": float(n), "resid_db": float(np.std(res, ddof=1)) if len(res) > 2 else float(np.std(res))}


def distance_error_factor(sigma_db: float, n: float) -> float:
    """One-sigma multiplicative error in distance caused by sigma_db of RSSI scatter: 10 ** (sigma / (10 n))."""
    return float(10 ** (sigma_db / (10.0 * max(abs(n), 0.5))))


def separation(m1, s1, m2, s2) -> float:
    """How well two RSSI groups separate: |difference of means| / pooled standard deviation. Below about 1 the
    groups overlap heavily."""
    p = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
    return float(abs(m1 - m2) / p) if p > 0 else float("inf")


def count_events(levels_db, baseline_n: int = 20, over_db: float = 12.0, refractory: int = 5) -> int:
    """Count loud events in a level series. The baseline is the median of the first `baseline_n` values."""
    v = np.asarray(levels_db, dtype=np.float64)
    if len(v) <= baseline_n:
        return 0
    base = float(np.median(v[:baseline_n]))
    n, cool = 0, 0
    for x in v[baseline_n:]:
        if cool > 0:
            cool -= 1
            continue
        if x > base + over_db:
            n += 1
            cool = refractory
    return n
