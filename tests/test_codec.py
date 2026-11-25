import hashlib

import numpy as np
import pyflac
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from waveform_compression_tools.codec import (
    decode_block,
    encode_block,
    max_relative_error,
    within_carried_limits,
    within_one_step,
)
from waveform_compression_tools.errors import Damaged, NonFiniteSamples
from waveform_compression_tools.stream import decode_stream, encode_stream

from helpers import (
    RealTraces,
    hard_traces,
    within_carried_limits_of_the_setting,
    within_one_step_of_the_setting,
)

# Total FLAC bytes of each set of real traces at each setting, as measured with an
# independent implementation of this encoder (level 5, float64 gains).
REAL_TRACE_FLAC_BYTES = {
    "lf": {32: 722_764, 24: 276_475, 17: 162_256},
    "hf": {32: 881_307, 24: 514_114, 17: 253_797},
    "bb": {32: 1_264_501, 24: 615_835, 17: 297_263},
}

# The SHA-256 of the integers and then the gains that 1.2.0 made of each set of real
# traces at 24 bits, as little-endian int32 and float64.
PLAIN_AT_24_BITS_SHA256 = {
    "lf": "50cd4a16cc36409fd98e7311be0e2713cd780acc0eb16bc799bf6e6887ca753f",
    "hf": "5e152f298c96fc1b47d74a060672142580a979ba7093c0f8b5d82a21c1641a84",
    "bb": "37aa965ea432dded02dcce205519ca7f524a5f8ed4751a13ac5c106ad1590e6a",
}

# How much bigger the real traces' streams are at 24 bits with the rounding error
# carried, in percent.
CARRIED_PERCENT_BIGGER = {"lf": 3.7, "hf": 0.1, "bb": 2.2}

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


def round_trip(
    w: np.ndarray, bits: int, rounding: str = "nearest"
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a block, lay its streams end to end, and decode it, with its gains."""
    block = encode_block(w, bits, rounding=rounding)
    compressed, starts, nbytes = pack(block.streams, block.gains.shape)
    offsets = np.zeros_like(block.gains)
    decoded = decode_block(compressed, starts, nbytes, block.gains, offsets, w.shape[2])
    return decoded, block.gains


def random_walks(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """Return float32 traces that wander like seismograms, each on its own scale."""
    rng = np.random.default_rng(seed)
    walks = np.cumsum(rng.normal(size=shape), axis=2)
    scales = 10.0 ** rng.uniform(-3, 2, shape[:2])
    return (walks * scales[..., None]).astype(np.float32)


@pytest.mark.parametrize("name", REAL_TRACE_FLAC_BYTES)
@pytest.mark.parametrize("bits", [32, 24, 17])
def test_real_trace_sizes_match_the_reference(
    real_traces: RealTraces, name: str, bits: int
) -> None:
    # The only check of the streams' size: a change that made them bigger, such as to
    # pyflac or to the encoder's settings, fails here and nowhere else. test_files
    # checks what a compressed file holds besides them.
    block = encode_block(real_traces(name), bits, rounding="nearest")
    assert sum(map(len, block.streams)) == pytest.approx(
        REAL_TRACE_FLAC_BYTES[name][bits], rel=0.01
    )


@pytest.mark.parametrize("name", PLAIN_AT_24_BITS_SHA256)
@pytest.mark.parametrize("level", [5, 8])
def test_plain_streams_are_as_1_2_0_wrote_them(
    real_traces: RealTraces, name: str, level: int
) -> None:
    # The integers and the gains by their hash, and each stream byte for byte what
    # libFLAC makes of its integers here. The streams' own hash can't be pinned, as
    # libFLAC's arithmetic differs from one processor to another.
    block = encode_block(real_traces(name), 24, level, "nearest")
    q = np.stack([decode_stream(stream) for stream in block.streams])
    held = q.astype("<i4").tobytes() + block.gains.astype("<f8").tobytes()
    assert hashlib.sha256(held).hexdigest() == PLAIN_AT_24_BITS_SHA256[name]
    assert block.streams == [encode_stream(row, level) for row in q]


@pytest.mark.parametrize("name", CARRIED_PERCENT_BIGGER)
def test_carried_sizes(real_traces: RealTraces, name: str) -> None:
    w = real_traces(name)
    plain, carried = (
        sum(map(len, encode_block(w, 24, rounding=rounding).streams))
        for rounding in ("nearest", "carried")
    )
    bigger = 100 * (carried / plain - 1)
    assert bigger == pytest.approx(CARRIED_PERCENT_BIGGER[name], abs=0.5)


@pytest.mark.parametrize("name", REAL_TRACE_FLAC_BYTES)
@pytest.mark.parametrize("bits", [24, 17, 8])
def test_carried_real_traces_keep_both_limits(
    real_traces: RealTraces, name: str, bits: int
) -> None:
    w = real_traces(name)
    back, gains = round_trip(w, bits, "carried")
    assert within_carried_limits(w, back, gains, bits)
    assert within_carried_limits_of_the_setting(w, back, bits).all()


def test_carried_holds_the_running_total(real_traces: RealTraces) -> None:
    # What it is for: the largest running total of the errors, over the traces, in
    # steps.
    w = real_traces("bb")
    step = np.abs(w).max(axis=2, keepdims=True) / (2.0**23 - 1)
    largest = {
        rounding: np.abs(
            np.cumsum(w.astype(np.float64) - round_trip(w, 24, rounding)[0], axis=2)
            / step
        ).max()
        for rounding in ("nearest", "carried")
    }
    assert largest["carried"] < 1 and largest["nearest"] > 50 * largest["carried"]


@pytest.mark.parametrize("bits", range(8, 25))
def test_hard_traces_come_back_within_the_carried_limits(bits: int) -> None:
    # Each station holds one of the hard traces, in all three of its components. The
    # check's own rules agree with the helper's, made from the README's words.
    traces = hard_traces(nt=300, bits=bits)
    block = np.stack([np.stack(list(traces.values()))] * 3)
    back, gains = round_trip(block, bits, "carried")
    ok = within_carried_limits_of_the_setting(block, back, bits)
    assert ok.all(), [name for name, good in zip(traces, ok.all(axis=0)) if not good]
    assert within_carried_limits(block, back, gains, bits)


@pytest.mark.parametrize("bits", [24, 8])
def test_long_traces_pass_the_carried_check(bits: int) -> None:
    # What the check adds up along a trace is exact enough for its slack.
    block = np.stack(list(hard_traces(400_000, bits).values()))[np.newaxis]
    back, gains = round_trip(block, bits, "carried")
    assert within_carried_limits(block, back, gains, bits)


def test_carried_traces_don_t_depend_on_those_beside_them(
    real_traces: RealTraces,
) -> None:
    # In a block of their own or not, and wherever a block's memory is.
    w = real_traces("bb")[..., :3000]
    whole = encode_block(w, 16, rounding="carried")
    for s in range(w.shape[1]):
        one = encode_block(w[:, s : s + 1], 16, rounding="carried")
        assert one.streams == whole.streams[s :: w.shape[1]]
    fortran = encode_block(np.asfortranarray(w), 16, rounding="carried")
    assert fortran.streams == whole.streams


def carrying_the_float64_value(trace: np.ndarray, gain: float) -> np.ndarray:
    # What carried rounding would be if it left float32 out of what it carries.
    owed, back = 0.0, []
    for x in trace.tolist():
        q = np.rint((x + owed) * gain)
        owed += x - q / gain
        back.append(q / gain)
    return np.array(back, np.float32)


CARRIED_FAULTS = [
    "a sample three steps off",
    "a drift of 0.002 steps a sample",
    "a drift of -0.002 steps a sample",
    "a step twice what the setting promises",
    "a doubled gain",
    "a sample of a trace of zeros",
    "a NaN",
    "the float64 value carried",
    "rounded to nearest",
]


@pytest.mark.parametrize("fault", CARRIED_FAULTS)
def test_within_carried_limits_fails(real_traces: RealTraces, fault: str) -> None:
    w = real_traces("bb").copy()
    w[:, 4] = 0.0
    bits = 8 if fault == "the float64 value carried" else 24
    if "drift" in fault:
        bits = 16
    back, gains = round_trip(w, bits, "carried")
    assert within_carried_limits(w, back, gains, bits)
    trace, step = w[1, 2].astype(np.float64), 1 / gains[1, 2]
    if fault == "a sample three steps off":
        # At a sample too small for float32 to round the three steps away.
        back[1, 2, np.abs(trace).argmin()] += 3 * step
    elif "drift" in fault:
        # Only of samples whose error is small, so that each one keeps its own limit.
        error = np.abs(trace - back[1, 2])
        drift = -0.002 if "-" in fault else 0.002
        back[1, 2, error < 0.5 * step] += drift * step
        assert np.abs(trace - back[1, 2]).max() <= max(error.max(), 0.51 * step)
    elif fault == "a step twice what the setting promises":
        # A gain of half the setting's, which the samples were rounded with.
        back, gains = round_trip(w, 23, "carried")
    elif fault == "a doubled gain":
        gains = gains.copy()
        gains[1, 2] *= 2
    elif fault == "a sample of a trace of zeros":
        back[1, 4, 100] = 1e-30
    elif fault == "a NaN":
        back[1, 2, 100] = np.nan
    elif fault == "the float64 value carried":
        # At 8 bits it wanders just past its limit, and only in some of the traces.
        back[0, 3] = carrying_the_float64_value(w[0, 3], gains[0, 3])
    else:
        back, _ = round_trip(w, 24)
    assert not within_carried_limits(w, back, gains, bits)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(("steps", "keeps"), [(0.999, True), (1.001, False)])
def test_the_limit_of_one_step(steps: float, keeps: bool, sign: int) -> None:
    # A trace with a peak of 1, then zeros of which one decodes to the error.
    original = np.zeros((1, 1, 10), np.float32)
    original[0, 0, 0] = 1
    decoded = original.copy()
    decoded[0, 0, 5] = sign * steps * 2.0**-23
    assert within_one_step(original, decoded, np.full((1, 1), 2.0**23)) == keeps


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    ("errors", "gain", "keeps"),
    [
        # A step at 24 bits, of a peak of 1, and the spacing of float32 values at it are
        # both 2 ** -23, within 2e-7 of each. So a sample's limit is 2 steps, and the
        # running total's 1.
        pytest.param([0.999, -1.998, 1.998], 1, True, id="both limits, nearly"),
        pytest.param([0.999, -2.003], 1, False, id="past both"),
        pytest.param([0.5, 0.505], 1, False, id="half a percent past the total's"),
        pytest.param([0.5, 0.505, -0.9], 1, False, id="past it, and back"),
        # With twice the setting's gain, a sample's limit is 1.5 steps, half a step of
        # its own and the spacing. The total's is still the setting's.
        pytest.param([0.95, -1.498], 2, True, id="twice the gain, nearly"),
        pytest.param([0.95, -1.508], 2, False, id="past its own gain's"),
        # With half the setting's gain, they are the setting's.
        pytest.param([0.999, -1.998], 0.5, True, id="half the gain, nearly"),
        pytest.param([0.999, -2.003], 0.5, False, id="past the setting's"),
    ],
)
def test_the_carried_limits(
    errors: list[float], gain: float, keeps: bool, sign: int
) -> None:
    # A trace with a peak of 1, then zeros that decode to the errors, in steps.
    original = np.zeros((1, 1, 10), np.float32)
    original[0, 0, 0] = sign
    decoded = original.copy()
    decoded[0, 0, 1 : 1 + len(errors)] = -sign * np.array(errors) * 2.0**-23
    gains = np.full((1, 1), gain * (2.0**23 - 1))
    assert within_carried_limits(original, decoded, gains, 24) == keeps


@pytest.mark.parametrize("bad", [0.0, -2.0, np.inf, -np.inf, np.nan])
def test_within_carried_limits_fails_with_a_gain_that_is_not_finite_and_positive(
    bad: float,
) -> None:
    original = random_walks((3, 2, 50))
    gains = encode_block(original, 24).gains.copy()
    assert within_carried_limits(original, original.copy(), gains, 24)
    gains[2, 1] = bad
    assert not within_carried_limits(original, original.copy(), gains, 24)


@pytest.mark.parametrize("rounding", ["nearest", "carried"])
@pytest.mark.parametrize(
    "case",
    ["a NaN", "an infinity", "float64", "two dimensions", "no samples", "7 bits"],
)
def test_what_encode_block_refuses(rounding: str, case: str) -> None:
    w, bits, error, message = random_walks((3, 2, 600)), 24, ValueError, "float32 array"
    if case in ("a NaN", "an infinity"):
        w[2, 1, 400] = np.nan if case == "a NaN" else np.inf
        error, message = NonFiniteSamples, "trace has 1 non-finite samples"
    elif case == "float64":
        w = w.astype(np.float64)
    elif case == "two dimensions":
        w = w[0]
    elif case == "no samples":
        w = w[:, :, :0]
    else:
        bits, message = 7, "bits must be from 8 to 32, got 7"
    with pytest.raises(error, match=message):
        encode_block(w, bits, rounding=rounding)


def test_encode_block_carries_at_24_bits_unless_asked() -> None:
    w = random_walks((3, 2, 50))
    default, asked = encode_block(w), encode_block(w, 24, 5, "carried")
    assert default.streams == asked.streams
    np.testing.assert_array_equal(default.gains, asked.gains)


@pytest.mark.parametrize(
    ("rounding", "bits", "message"),
    [("carried", 25, "24 bits or fewer, got 25"), ("other", 24, "got 'other'")],
)
def test_encode_block_refuses_a_rounding(
    rounding: str, bits: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        encode_block(random_walks((3, 2, 50)), bits, rounding=rounding)


@pytest.mark.parametrize("bits", [32, 24, 17])
def test_real_traces_round_trip_within_one_step(
    real_traces: RealTraces, bits: int
) -> None:
    w = real_traces("bb")
    back, gains = round_trip(w, bits)
    assert within_one_step(w, back, gains)
    assert max_relative_error(w, back).max() <= RELATIVE_ERROR_LIMIT[bits]


@pytest.mark.parametrize("bits", [32, 24, 17])
@pytest.mark.parametrize("nt", [1, 2, 4095, 4096, 4097, 10_001, 147_499])
def test_round_trip_at_any_length_within_one_step(bits: int, nt: int) -> None:
    # Long traces make streams over 8 KiB, which libFLAC's wrapper refuses if it is
    # handed a numpy slice instead of bytes.
    w = random_walks((3, 2, nt), seed=nt)
    back, gains = round_trip(w, bits)
    assert back.dtype == np.float32 and back.shape == w.shape
    assert within_one_step(w, back, gains)
    assert max_relative_error(w, back).max() <= RELATIVE_ERROR_LIMIT[bits]


@pytest.mark.parametrize("bits", range(8, 33))
def test_hard_traces_come_back_within_one_step(bits: int) -> None:
    # Each station holds one of the hard traces, in all three of its components.
    traces = hard_traces(nt=100)
    block = np.stack([np.stack(list(traces.values()))] * 3)
    ok = within_one_step_of_the_setting(block, round_trip(block, bits)[0], bits)
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
    assert within_one_step_of_the_setting(block, round_trip(block, bits)[0], bits).all()


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


def test_a_block_decodes_into_the_array_given() -> None:
    # Into a slice of a larger array, such as a whole waveform, in place.
    w = random_walks((3, 4, 300), seed=1)
    block = encode_block(w, 24)
    compressed, starts, nbytes = pack(block.streams, block.gains.shape)
    offsets = np.zeros_like(block.gains)
    args = (compressed, starts, nbytes, block.gains, offsets, 300)
    whole = np.zeros((3, 10, 300), np.float32)
    out = decode_block(*args, out=whole[:, 5:9])
    assert np.shares_memory(out, whole)
    np.testing.assert_array_equal(whole[:, 5:9], decode_block(*args))
    assert not whole[:, :5].any() and not whole[:, 9:].any()
    for wrong in (np.zeros((3, 4, 299), np.float32), np.zeros((3, 4, 300))):
        with pytest.raises(ValueError, match="out must be float32 of shape"):
            decode_block(*args, out=wrong)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("a flipped byte", "a FLAC stream is damaged"),
        ("cut to nothing", "the stream decodes to 0 samples, expected 300"),
    ],
)
def test_a_damaged_stream_is_named_by_its_trace_in_the_file(
    damage: str, message: str
) -> None:
    # Components 2 and 0 of stations 7 and 8 of a file, in that order: the second
    # stream of the first row is component 2 of station 8. The error isn't libFLAC's
    # own: where libFLAC finds the damage, its error is the cause. The file isn't known
    # here, so none is named.
    block = encode_block(random_walks((2, 2, 300), seed=2), 24)
    compressed, starts, nbytes = pack(block.streams, block.gains.shape)
    compressed = compressed.copy()
    if damage == "a flipped byte":
        compressed[starts[0, 1] + nbytes[0, 1] // 2] ^= 0xFF
    else:
        nbytes[0, 1] = 0
    offsets = np.zeros_like(block.gains)
    args = (compressed, starts, nbytes, block.gains, offsets, 300)
    with pytest.raises(Damaged, match=f"component 2, station 8: {message}") as bad:
        decode_block(*args, first_station=7, components=np.array([2, 0]))
    assert bad.value.filename is None
    libflac = (pyflac.DecoderInitException, pyflac.DecoderProcessException)
    assert isinstance(bad.value.__cause__, libflac) == (damage == "a flipped byte")
