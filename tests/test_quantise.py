import numpy as np
import pytest

from waveform_compression_tools.quantise import (
    full_scale,
    quantise,
)


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


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        quantise(np.array([0.0, 1.0, bad], dtype=np.float32), 1.0)
