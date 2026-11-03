import pytest

from waveform_compression_tools.stations import is_real_station


@pytest.mark.parametrize(
    ("name", "real"),
    [
        ("6c5SYSL", False),
        ("22008d3", False),
        ("S000000", False),
        ("KIKS", True),
        ("OPZ", True),
        (b"OPZ\x00\x00\x00\x00", True),
        (" KIKS ", True),
    ],
)
def test_is_real_station(name: str | bytes, real: bool) -> None:
    assert is_real_station(name) is real
