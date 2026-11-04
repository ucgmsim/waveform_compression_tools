import pytest

from waveform_compression_tools.quantise import full_scale


def test_full_scale() -> None:
    assert (full_scale(32), full_scale(24), full_scale(17), full_scale(8)) == (
        2_147_483_647,
        8_388_607,
        65_535,
        127,
    )


@pytest.mark.parametrize("bits", [7, 33])
def test_bits_out_of_range(bits: int) -> None:
    with pytest.raises(ValueError):
        full_scale(bits)
