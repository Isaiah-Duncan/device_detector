"""Advertisement parsing. Inputs are hand-built byte strings (SYNTHETIC), not captured traffic."""
from types import SimpleNamespace

from device_detector.adparse import normalize_uuid, observation_from_bleak, parse_ad_structures


def s(t, val):
    return bytes([len(val) + 1, t]) + val


def test_flags_name_txpower():
    p = parse_ad_structures(s(1, b"\x06") + s(9, b"Fit Band") + s(0x0A, bytes([0xF4])))
    assert p.flags == 6 and p.name == "Fit Band" and p.tx_power == -12 and not p.malformed


def test_uuid16_32_128():
    u128 = bytes(reversed(bytes.fromhex("0000180d00001000800000805f9b34fb")))
    p = parse_ad_structures(s(3, b"\x0d\x18\x0f\x18") + s(5, b"\x78\x56\x34\x12") + s(7, u128))
    assert p.uuids[:2] == ["180d", "180f"]
    assert p.uuids[2] == "12345678"
    assert p.uuids[3] == "0000180d-0000-1000-8000-00805f9b34fb"


def test_manufacturer_keeps_id_only():
    p = parse_ad_structures(s(0xFF, b"\x4c\x00" + b"\x01" * 10))
    assert p.company_id == 0x004C and p.manufacturer_len == 10


def test_appearance_and_service_data_uuid():
    p = parse_ad_structures(s(0x19, b"\xc1\x00") + s(0x16, b"\xaa\xfe\x01\x02"))
    assert p.appearance == 0xC1 and p.service_data_uuids == ["feaa"]


def test_malformed_length_is_flagged_not_crashing():
    p = parse_ad_structures(s(9, b"ok") + b"\x20\x09abc")
    assert p.name == "ok" and p.malformed


def test_padding_stops_parsing():
    assert parse_ad_structures(s(9, b"x") + b"\x00\x00\x00").name == "x"


def test_empty():
    p = parse_ad_structures(b"")
    assert p.name is None and not p.uuids and not p.malformed


def test_normalize_uuid():
    assert normalize_uuid("0000180D-0000-1000-8000-00805F9B34FB") == "180d"
    assert normalize_uuid("ABCDEF00-1111-2222-3333-444455556666") == "abcdef00-1111-2222-3333-444455556666"


def test_bleak_adapter_drops_manufacturer_payload():
    adv = SimpleNamespace(rssi=-60, local_name="X", service_uuids=["0000180d-0000-1000-8000-00805f9b34fb"],
                          manufacturer_data={0x004C: b"secret-bytes"}, service_data={"0000feaa-0000-1000-8000-00805f9b34fb": b"zz"},
                          tx_power=None)
    o = observation_from_bleak(1.0, "D-1", adv)
    assert o.company_ids == (0x004C,) and o.uuids == ("180d",) and o.service_data_uuids == ("feaa",)
    assert "secret" not in repr(o)


def test_bleak_adapter_missing_fields():
    o = observation_from_bleak(1.0, "D-1", SimpleNamespace(rssi=None))
    assert o.rssi is None and o.name is None and o.uuids == ()
