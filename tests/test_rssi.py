import math

from device_detector.rssi import RSSI_EXPLANATION, adv_rate, estimate_distance, rolling_stats


def test_stats_values():
    s = [(0, -60), (1, -70), (2, -80)]
    r = rolling_stats(s, 2.0, 10)
    assert r["min"] == -80 and r["max"] == -60 and r["last"] == -80
    assert math.isclose(r["mean_window"], -70) and math.isclose(r["std_window"], math.sqrt(200 / 3))


def test_window_excludes_old_but_min_max_keep_all():
    r = rolling_stats([(0, -50), (100, -90)], 100.0, 10)
    assert r["count_window"] == 1 and r["min"] == -90 and r["max"] == -50 and r["mean_window"] == -90
    assert r["std_window"] is None


def test_empty_is_none_not_zero():
    r = rolling_stats([], 5.0, 10)
    assert r["last"] is None and r["mean_window"] is None and r["min"] is None


def test_distance_is_experimental_and_wide():
    d = estimate_distance(-70)
    assert d["experimental"] is True and d["low"] < d["meters"] < d["high"]
    assert d["high"] / d["low"] > 2
    assert estimate_distance(-50)["meters"] < estimate_distance(-80)["meters"]


def test_adv_rate():
    times = [i * 0.5 for i in range(20)]
    assert 1.5 <= adv_rate(times, 10.0, 10.0, 0.0) <= 2.1


def test_explanation_mentions_key_causes():
    for w in ("transmit power", "multipath", "body", "not"):
        assert w in RSSI_EXPLANATION
