"""Signal-strength analysis. RSSI is a relative indicator, not a distance measurement."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

RSSI_EXPLANATION = (
    "Why RSSI is not a reliable distance measurement: the received signal strength depends on the transmit "
    "power of the device (which varies and is often unknown), the antenna design and orientation of both devices, "
    "the body of the person holding the device, walls and furniture, reflections that add or cancel the signal "
    "(multipath), interference on the 2.4 GHz band, and how the Bluetooth adapter and driver measure and report "
    "the value. Two identical devices at the same distance can differ by 10 dB or more, and one device can change "
    "by 10 dB when it is turned in the hand. Use RSSI trends of one device in one place, not absolute numbers.")


def rolling_stats(samples: deque | list, now: float, window_s: float) -> dict:
    """Statistics over (timestamp, rssi) samples. Empty results are None, never zero."""
    vals_all = [r for _, r in samples if r is not None]
    vals = [r for t, r in samples if r is not None and now - t <= window_s]
    out = {"count_all": len(vals_all), "count_window": len(vals), "last": vals_all[-1] if vals_all else None,
           "min": min(vals_all) if vals_all else None, "max": max(vals_all) if vals_all else None,
           "mean_window": float(np.mean(vals)) if vals else None,
           "std_window": float(np.std(vals)) if len(vals) >= 2 else None}
    return out


def estimate_distance(rssi: float, tx_power_1m: float = -59.0) -> dict:
    """EXPERIMENTAL estimate with a log-distance model. The range is deliberately wide and still can be wrong.

    d = 10 ** ((P1m - rssi) / (10 * n)). Typical indoor n is 2.0 (open space) to 4.0 (cluttered). A shadowing
    allowance of +/- 6 dB widens the range further.
    """
    def d(r, n):
        return 10 ** ((tx_power_1m - r) / (10.0 * n))
    near = d(rssi + 6, 4.0)
    far = d(rssi - 6, 2.0)
    mid = d(rssi, 3.0)
    return {"meters": mid, "low": min(near, far), "high": max(near, far), "experimental": True,
            "note": "Experimental estimate. The true distance can lie outside this range."}


def adv_rate(times: deque | list, now: float, window_s: float, first_seen: float) -> float:
    """Advertisements per second over the window, not inflated for devices seen only briefly."""
    span = min(window_s, max(now - first_seen, 1e-6))
    n = sum(1 for t in times if now - t <= window_s)
    return n / max(span, 1.0)
