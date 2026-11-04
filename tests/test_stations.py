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
        pytest.param(b"6c5SYSL", False, id="a grid name as bytes"),
        pytest.param("6c5SYSL\x00", False, id="a grid name padded with a NUL"),
        # A name only counts after both kinds of padding are gone, in any order.
        pytest.param("\x00 6c5SYSL\t\x00", False, id="NULs and whitespace mixed"),
        pytest.param("6c5SYSLX", True, id="8 characters"),
        pytest.param("6c5 SYS", False, id="a space inside is kept"),
        pytest.param("", True, id="no name at all"),
    ],
)
def test_is_real_station(name: str | bytes, real: bool) -> None:
    assert is_real_station(name) is real
