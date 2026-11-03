"""Quantisation of float waveforms to integers, and back.

A trace is scaled by a gain so that its largest absolute sample lands on the
largest value a signed integer of ``bits`` bits can hold, then rounded to the
nearest integer. The gain is ``full_scale(bits) / peak`` in float64. For a
float32 trace, decoding (``q / gain + offset`` in float64, cast to float32)
gives every sample back within one quantisation step, ``1 / gain``.
"""

import numpy as np


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
    ValueError
        If the trace holds a NaN or an infinity.
    """
    finite = np.isfinite(trace)
    if not finite.all():
        raise ValueError(
            f"trace has {int((~finite).sum())} non-finite samples (NaN or infinity)"
        )
    return np.rint(trace.astype(np.float64) * gain).astype(np.int32)


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
