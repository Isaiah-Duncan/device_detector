from device_detector.classifier import classify


def test_manufacturer_alone_is_unknown():
    for cid in (0x004C, 0x0006, 0x0075, 0x00E0):
        c = classify(None, (), (cid,), (), None)
        assert c.category == "Unknown device" and c.confidence == "none"
        assert any("not enough" in r for r in c.reasons)


def test_heart_rate_service_is_fitness():
    c = classify(None, ("180d",), (), (), None)
    assert c.category == "Fitness device" and c.confidence == "medium" and c.reasons


def test_name_alone_is_low_confidence_only():
    c = classify("My Earbuds", (), (), (), None)
    assert c.category == "Wireless audio accessory" and c.confidence == "low"


def test_phone_name_plus_apple_id_is_still_low():
    c = classify("iPhone", (), (0x004C,), (), None)
    assert c.confidence == "low"


def test_appearance_watch():
    c = classify(None, (), (), (), 0xC0)       # category 3 = watch
    assert c.category == "Smartwatch" and c.confidence == "medium"


def test_two_sources_raise_confidence():
    c = classify("Buds", ("184e", "1850"), (), (), None)
    assert c.category == "Wireless audio accessory" and c.confidence == "high"


def test_conflict_downgrades():
    a = classify(None, ("180d",), (), (), None)
    b = classify(None, ("180d", "184e"), (), (), None)
    order = ["none", "low", "medium", "high"]
    assert b.category in ("Fitness device", "Wireless audio accessory")
    assert order.index(b.confidence) < order.index("high")
    assert any("conflicting" in r for r in b.reasons)
    assert a.confidence == "medium"


def test_unknown_service_is_unknown():
    assert classify(None, ("abcd",), (), (), None).category == "Unknown device"
