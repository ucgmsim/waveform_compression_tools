from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from waveform_compression_tools.codec import (
    decode_block,
    encode_block,
    max_relative_error,
    within_one_step,
)

from helpers import within_one_step_of_the_setting

# Total FLAC bytes of each set of real traces at each setting, as measured with an
# independent implementation of this encoder (level 5, float64 gains).
REAL_TRACE_FLAC_BYTES = {
    "lf": {32: 722_764, 24: 276_475, 17: 162_256},
    "hf": {32: 881_307, 24: 514_114, 17: 253_797},
    "bb": {32: 1_264_501, 24: 615_835, 17: 297_263},
}

# The largest error over the peak that each setting can report: one step, which is
# 1 / full_scale(bits) of the peak, rounded up.
RELATIVE_ERROR_LIMIT = {32: 4.7e-10, 24: 1.2e-7, 17: 1.6e-5}


def pack(
    streams: list[bytes], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay the streams end to end; return the bytes and each stream's start and size."""
    nbytes = np.array([len(s) for s in streams], dtype=np.int64)
    starts = np.cumsum(nbytes) - nbytes
    compressed = np.frombuffer(b"".join(streams), dtype=np.uint8)
    return compressed, starts.reshape(shape), nbytes.reshape(shape)


def round_trip(w: np.ndarray, bits: int) -> np.ndarray:
    """Encode a block, lay its streams end to end, and decode it."""
    block = encode_block(w, bits)
    compressed, starts, nbytes = pack(block.streams, block.gains.shape)
    offsets = np.zeros_like(block.gains)
    return decode_block(compressed, starts, nbytes, block.gains, offsets, w.shape[2])


def random_walks(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """Return float32 traces that wander like seismograms, each on its own scale."""
    rng = np.random.default_rng(seed)
    walks = np.cumsum(rng.normal(size=shape), axis=2)
    scales = 10.0 ** rng.uniform(-3, 2, shape[:2])
    return (walks * scales[..., None]).astype(np.float32)


def hard_traces(nt: int) -> dict[str, np.ndarray]:
    """Return the traces most likely to come back wrong, each of nt float32 samples."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=nt))
    walk /= np.abs(walk).max()  # A peak of 1.
    spike = np.zeros(nt)
    spike[nt // 2] = -7.0
    # Peaks that no other sample of their trace matches, of either sign.
    peak_first, peak_last = walk.copy(), walk.copy()
    peak_first[0] = -2.0
    peak_last[-1] = 2.0
    traces = {
        "all zeros": np.zeros(nt),
        "a positive constant": np.full(nt, 2.5),
        "a negative constant": np.full(nt, -3.5),
        "a single spike": spike,
        "a peak at the first sample": peak_first,
        "a peak at the last sample": peak_last,
        "a peak of 3.4e38": 3.4e38 * walk,
        "about 1e-39, below float32's normal numbers": 1e-39 * walk,
        "subnormal multiples of 1.4e-45": rng.integers(-5, 6, nt) * 1.4e-45,
        "1e30 beside 1e-30": np.where(np.arange(nt) % 2 == 0, 1e30, 1e-30),
    }
    return {name: trace.astype(np.float32) for name, trace in traces.items()}


@pytest.mark.parametrize("name", REAL_TRACE_FLAC_BYTES)
@pytest.mark.parametrize("bits", [32, 24, 17])
def test_real_trace_sizes_match_the_reference(
    real_traces: Callable[[str], np.ndarray], name: str, bits: int
) -> None:
    # The only check of compressed size: a change that made files bigger, such as to
    # pyflac or to the encoder's settings, fails here and nowhere else.
    block = encode_block(real_traces(name), bits)
    assert sum(map(len, block.streams)) == pytest.approx(
        REAL_TRACE_FLAC_BYTES[name][bits], rel=0.01
    )


@pytest.mark.parametrize("bits", [32, 24, 17])
def test_real_traces_round_trip_within_one_step(
    real_traces: Callable[[str], np.ndarray], bits: int
) -> None:
    w = real_traces("bb")
    back = round_trip(w, bits)
    assert within_one_step(w, back, encode_block(w, bits).gains)
    assert max_relative_error(w, back).max() <= RELATIVE_ERROR_LIMIT[bits]


@pytest.mark.parametrize("bits", [32, 24, 17])
@pytest.mark.parametrize("nt", [1, 2, 4095, 4096, 4097, 10_001, 147_499])
def test_round_trip_at_any_length_within_one_step(bits: int, nt: int) -> None:
    # Long traces make streams over 8 KiB, which libFLAC's wrapper refuses if it is
    # handed a numpy slice instead of bytes.
    w = random_walks((3, 2, nt), seed=nt)
    back = round_trip(w, bits)
    assert back.dtype == np.float32 and back.shape == w.shape
    assert within_one_step(w, back, encode_block(w, bits).gains)
    assert max_relative_error(w, back).max() <= RELATIVE_ERROR_LIMIT[bits]


@pytest.mark.parametrize("bits", range(8, 33))
def test_hard_traces_come_back_within_one_step(bits: int) -> None:
    # Each station holds one of the hard traces, in all three of its components.
    traces = hard_traces(nt=100)
    block = np.stack([np.stack(list(traces.values()))] * 3)
    ok = within_one_step_of_the_setting(block, round_trip(block, bits), bits)
    assert ok.all(), [name for name, good in zip(traces, ok.all(axis=0)) if not good]


@given(
    block=arrays(
        np.float32,
        st.tuples(st.just(3), st.integers(1, 3), st.integers(1, 300)),
        elements=st.floats(width=32, allow_nan=False, allow_infinity=False),
    ),
    bits=st.integers(8, 32),
)
def test_any_finite_block_comes_back_within_one_step(
    block: np.ndarray, bits: int
) -> None:
    # Any finite float32, subnormals included, at any setting.
    assert within_one_step_of_the_setting(block, round_trip(block, bits), bits).all()


@pytest.mark.parametrize("off", [1.0, -0.5, 1e-30])
def test_a_trace_of_zeros_has_to_come_back_as_exact_zeros(off: float) -> None:
    # A trace of zeros has a gain of 1, so one step would let it decode to anything
    # within 1 of zero.
    original = random_walks((3, 2, 50))
    original[:, 0] = 0.0
    gains = encode_block(original).gains
    assert np.all(gains[:, 0] == 1) and within_one_step(original, original, gains)
    decoded = original.copy()
    decoded[1, 0, 25] = off
    assert not within_one_step(original, decoded, gains)


@pytest.mark.parametrize("bad", [0.0, -2.0, np.inf, -np.inf, np.nan])
def test_within_one_step_fails_with_a_gain_that_is_not_finite_and_positive(
    bad: float,
) -> None:
    # Even a perfect decode fails: no right gain is zero, negative or infinite, and a
    # zero gain would make a step infinitely long.
    original = random_walks((3, 2, 50))
    gains = encode_block(original).gains.copy()
    gains[2, 1] = bad
    assert not within_one_step(original, original.copy(), gains)
