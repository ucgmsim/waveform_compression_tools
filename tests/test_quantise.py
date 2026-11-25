import warnings

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from waveform_compression_tools.errors import NonFiniteSamples
from waveform_compression_tools.quantise import (
    check_rounding,
    dequantise,
    float32_spacing,
    full_scale,
    gain_for,
    quantise_carried,
)

from helpers import hard_traces, within_carried_limits_of_the_setting


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


@pytest.mark.parametrize(
    ("rounding", "bits"), [("nearest", 32), ("nearest", 8), ("carried", 24)]
)
def test_a_rounding_and_the_bits_it_takes(rounding: str, bits: int) -> None:
    check_rounding(rounding, bits)


@pytest.mark.parametrize(
    ("rounding", "bits", "message"),
    [
        ("carried", 25, "carried rounding takes 24 bits or fewer, got 25"),
        ("carried", 32, "carried rounding takes 24 bits or fewer, got 32"),
        ("other", 24, "rounding must be 'nearest' or 'carried', got 'other'"),
        ("", 24, "rounding must be 'nearest' or 'carried', got ''"),
    ],
)
def test_a_rounding_that_is_refused(rounding: str, bits: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_rounding(rounding, bits)


def test_the_spacing_of_float32_values_at_a_peak() -> None:
    largest = float(np.finfo(np.float32).max)
    peaks = np.array([1.0, 1.5, 2.0, 1e-30, 1e30, 1e-38, largest, 1e-40, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # np.spacing overflows at the largest value.
        spacing = float32_spacing(peaks)
    assert spacing.dtype == np.float64
    np.testing.assert_array_equal(
        spacing[:6], np.spacing(peaks[:6].astype(np.float32)).astype(np.float64)
    )
    assert spacing[6] == 2.0**104 and spacing[7] == 2.0**-149 and spacing[8] == 0.0


def gains_of(traces: np.ndarray, bits: int) -> np.ndarray:
    return np.array([gain_for(peak, bits) for peak in np.abs(traces).max(axis=1)])


def carried_one_at_a_time(trace: np.ndarray, gain: float, bits: int) -> list[int]:
    # The definition, in Python's own numbers, a sample at a time.
    top, owed, q = 2 ** (bits - 1) - 1, 0.0, []
    for x in trace.tolist():
        q.append(min(max(round((x + owed) * gain), -top), top))
        owed += x - float(np.float32(q[-1] / gain))
    return q


@pytest.mark.parametrize("bits", [8, 16, 24])
@pytest.mark.parametrize("nt", [1, 2, 255, 256, 257, 1000])
def test_carried_matches_its_definition(bits: int, nt: int) -> None:
    # A sample at a time and a trace at a time, so neither the traces beside one nor
    # the stretches of time that are worked together can matter.
    hard = hard_traces(nt, bits)
    traces = np.stack([*hard.values(), *(-trace for trace in hard.values())])
    gains = gains_of(traces, bits)
    q = quantise_carried(traces, gains, bits)
    assert q.dtype == np.int32 and q.shape == traces.shape
    for name, trace, gain, row in zip([*hard, *hard], traces, gains, q, strict=True):
        assert row.tolist() == carried_one_at_a_time(trace, gain, bits), name


def decoded(q: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.stack([dequantise(row, gain) for row, gain in zip(q, gains, strict=True)])


@pytest.mark.parametrize("bits", range(8, 25))
def test_hard_traces_keep_both_limits(bits: int) -> None:
    traces = hard_traces(1000, bits)
    block = np.stack(list(traces.values()))
    gains = gains_of(block, bits)
    back = decoded(quantise_carried(block, gains, bits), gains)
    ok = within_carried_limits_of_the_setting(block, back, bits)
    assert ok.all(), [name for name, good in zip(traces, ok) if not good]


def test_long_traces_keep_both_limits() -> None:
    # Nothing builds up along a trace.
    block = np.stack(list(hard_traces(400_000).values()))
    gains = gains_of(block, 24)
    back = decoded(quantise_carried(block, gains, 24), gains)
    assert within_carried_limits_of_the_setting(block, back, 24).all()


@given(
    traces=arrays(
        np.float32,
        st.tuples(st.integers(1, 3), st.integers(1, 300)),
        elements=st.floats(width=32, allow_nan=False, allow_infinity=False),
    ),
    bits=st.integers(8, 24),
)
def test_any_finite_trace_keeps_both_limits(traces: np.ndarray, bits: int) -> None:
    # Any finite float32, subnormals included, at any setting.
    gains = gains_of(traces, bits)
    back = decoded(quantise_carried(traces, gains, bits), gains)
    assert within_carried_limits_of_the_setting(traces, back, bits).all()


def test_zeros_stay_zeros() -> None:
    # A trace of zeros, and the zeros before a signal starts, beside a trace that has
    # something to carry.
    rng = np.random.default_rng(1)
    traces = rng.normal(size=(3, 1000)).astype(np.float32)
    traces[0] = 0.0
    traces[1, :500] = 0.0
    q = quantise_carried(traces, gains_of(traces, 8), 8)
    assert not q[0].any() and not q[1, :500].any() and q[2, :500].any()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_carried_refuses_samples_that_are_not_finite(bad: float) -> None:
    traces = np.ones((2, 600), np.float32)
    traces[1, 400:402] = bad
    with pytest.raises(NonFiniteSamples, match="trace has 2 non-finite samples"):
        quantise_carried(traces, np.full(2, 127.0), 8)
