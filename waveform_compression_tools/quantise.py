"""Quantisation of float waveforms to integers, and back.

A trace is scaled by a gain so that its largest absolute sample lands on the
largest value a signed integer of ``bits`` bits can hold, then rounded to
integers. The gain is ``full_scale(bits) / peak`` in float64, and decoding is
``q / gain + offset`` in float64, cast to float32.

There are two ways to round (`ROUNDINGS`). ``nearest`` takes the nearest integer to
each sample, so a float32 trace comes back with every sample within one quantisation
step, ``1 / gain``, but its errors needn't cancel. ``carried`` adds what a sample's
rounding left out to the next sample before rounding that. With `float32_spacing` at
the trace's peak added to the step, every sample comes back within that sum, and the
running total of the errors stays within half of it, so the slow part of a signal is
kept far better. It takes `CARRIED_MAX_BITS` bits or fewer. Beyond that, float32's own
spacing is wider than a step.
"""

import numpy as np

from .errors import NonFiniteSamples

# How a trace can be rounded.
ROUNDINGS = ("nearest", "carried")

# The most bits that carried rounding takes: float32's own significant bits.
CARRIED_MAX_BITS = 24

# How many samples of every trace carried rounding works on at once. It only sets how
# much memory that takes and how fast it goes.
_STRETCH = 256


def check_rounding(rounding: str, bits: int) -> None:
    """Check that a rounding is one of `ROUNDINGS`, and takes the given bits.

    Parameters
    ----------
    rounding : str
        How to round: ``nearest`` or ``carried``.
    bits : int
        Bits per sample.

    Raises
    ------
    ValueError
        If `rounding` is neither, or is ``carried`` with `bits` above 24.
    """
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be 'nearest' or 'carried', got {rounding!r}")
    if rounding == "carried" and bits > CARRIED_MAX_BITS:
        raise ValueError(
            f"carried rounding takes {CARRIED_MAX_BITS} bits or fewer, got {bits}"
        )


def full_scale(bits: int) -> int:
    """Return the largest value of a signed integer with the given bits.

    Parameters
    ----------
    bits : int
        Bits per sample, from 8 to 32.

    Returns
    -------
    int
        The largest value, ``2 ** (bits - 1) - 1``.

    Raises
    ------
    ValueError
        If `bits` is outside 8 to 32.
    """
    if not 8 <= bits <= 32:
        raise ValueError(f"bits must be from 8 to 32, got {bits}")
    return 2 ** (bits - 1) - 1


def gain_for(peak: float | np.floating, bits: int) -> float:
    """Return the gain that takes a trace with the given peak to full scale.

    Parameters
    ----------
    peak : float | np.floating
        The largest absolute sample of the trace. A numpy scalar, such as the
        maximum of a float32 trace, is converted to a float first, so the gain
        is always computed in float64.
    bits : int
        Bits per sample, from 8 to 32.

    Returns
    -------
    float
        The gain, ``full_scale(bits) / peak`` in float64, or 1.0 when the
        peak is zero.

    Raises
    ------
    ValueError
        If `bits` is outside 8 to 32.
    """
    scale = full_scale(bits)  # Checks bits even when the peak is zero.
    if peak == 0:
        return 1.0
    # float() so that a float32 peak can't make the division float32.
    return scale / float(peak)


def float32_spacing(peaks: np.ndarray) -> np.ndarray:
    """Return the spacing of float32 values at each trace's peak.

    It is ``2 ** (e - 23)`` for a peak in ``[2 ** e, 2 ** (e + 1))``, and no less than
    ``2 ** -149``, the spacing of float32's subnormal numbers. ``numpy.spacing`` gives
    an infinity for float32's largest value.

    Parameters
    ----------
    peaks : np.ndarray
        The largest absolute sample of each trace.

    Returns
    -------
    np.ndarray
        The spacing at each peak, as float64, or 0 where the peak is zero.
    """
    _, exponents = np.frexp(peaks)  # frexp's exponent is one more than e.
    return np.where(peaks == 0, 0.0, np.ldexp(1.0, np.maximum(exponents - 24, -149)))


def _check_finite(trace: np.ndarray) -> None:
    """Raise NonFiniteSamples if a trace holds a NaN or an infinity.

    Parameters
    ----------
    trace : np.ndarray
        The trace to check.

    Raises
    ------
    NonFiniteSamples
        If it does. The message says how many there are.
    """
    finite = np.isfinite(trace)
    if not finite.all():
        raise NonFiniteSamples(
            f"trace has {int((~finite).sum())} non-finite samples (NaN or infinity)"
        )


def quantise(trace: np.ndarray, gain: float) -> np.ndarray:
    """Scale a trace by a gain and round it to the nearest integer.

    Parameters
    ----------
    trace : np.ndarray
        The samples to quantise.
    gain : float
        The factor to scale by, normally from `gain_for`.

    Returns
    -------
    np.ndarray
        The quantised samples, as int32.

    Raises
    ------
    NonFiniteSamples
        If the trace holds a NaN or an infinity.
    """
    _check_finite(trace)
    return np.rint(trace.astype(np.float64) * gain).astype(np.int32)


def quantise_carried(traces: np.ndarray, gains: np.ndarray, bits: int) -> np.ndarray:
    """Scale traces by their gains and round them, carrying the rounding error along.

    What a sample's rounding left out, which is the sample less the float32 value that
    `dequantise` gives back for it, is added to the next sample of its trace before
    that is rounded. A trace starts with nothing to carry, so it comes out the same
    whatever traces are beside it.

    Parameters
    ----------
    traces : np.ndarray
        The traces to quantise: a float32 array of shape (n, nt).
    gains : np.ndarray
        The factor to scale each trace by, from `gain_for`: float64, of shape (n,).
    bits : int
        Bits per sample, from 8 to 24.

    Returns
    -------
    np.ndarray
        The quantised samples, as int32, of the shape of `traces`.

    Raises
    ------
    NonFiniteSamples
        If a trace holds a NaN or an infinity.
    ValueError
        If `bits` is outside 8 to 24.
    """
    top = full_scale(bits)
    check_rounding("carried", bits)
    for trace in traces:
        _check_finite(trace)
    q = np.empty(traces.shape, dtype=np.int32)
    n, nt = traces.shape
    # What each trace is owed: the sum of its samples so far less their decoded values.
    owed = np.zeros(n)
    scaled = np.empty(n)
    decoded = np.empty(n, dtype=np.float32)
    # A sample needs the one before it, so the traces go side by side, one sample of
    # each at a time. They are turned around a stretch at a time, as a whole block in
    # float64 would be twice its size.
    for start in range(0, nt, _STRETCH):
        stretch = slice(start, start + _STRETCH)
        x = np.ascontiguousarray(traces[:, stretch].T, dtype=np.float64)
        rounded = np.empty(x.shape, dtype=np.int32)
        for i in range(len(x)):
            np.add(x[i], owed, out=scaled)
            scaled *= gains
            np.rint(scaled, out=scaled)
            # Only where the trace is at its peak and is owed more. What this leaves
            # out is owed like any other error.
            np.clip(scaled, -top, top, out=scaled)
            rounded[i] = scaled
            scaled /= gains
            decoded[:] = scaled
            np.subtract(x[i], decoded, out=scaled)
            owed += scaled
        q[:, stretch] = rounded.T
    return q


def dequantise(q: np.ndarray, gain: float, offset: float = 0.0) -> np.ndarray:
    """Turn quantised samples back into floats.

    Parameters
    ----------
    q : np.ndarray
        The quantised samples.
    gain : float
        The gain the samples were quantised with.
    offset : float, default 0.0
        The value added after dividing by the gain.

    Returns
    -------
    np.ndarray
        The samples as float32, with ``q / gain + offset`` computed in float64.
    """
    return (q.astype(np.float64) / gain + offset).astype(np.float32)
