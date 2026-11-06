"""Compress and decode a block of stations, one FLAC stream per trace.

A block is a float32 array of shape (3, n, nt): n stations, each with three
components (such as 000, 090 and ver) of nt time samples. Each of its 3n traces is
quantised on its own (see `waveform_compression_tools.quantise`) and written as one FLAC stream
(see `waveform_compression_tools.stream`). The streams are in row-major (component, station)
order: the first component of every station, then the second, then the third.

A stream carries no sample count, so a stream that has been cut can decode without
an error to fewer samples than it held. `decode_block`, which is told how many
samples to expect, counts them.

`check_standard` says whether a dataset holds its waveform in the layout this
package handles. `max_relative_error` and `within_one_step` compare a decoded block
with the original.
"""

from dataclasses import dataclass

import numpy as np
import pyflac
import xarray as xr

from .errors import NotStandardLayout
from .quantise import dequantise, gain_for, quantise
from .stream import decode_stream, encode_stream

STANDARD_DIMS = ("component", "station", "time")
# The components the BB.bin converter writes, in its order. A file's own component
# coordinate can name them otherwise: some files use x, y and z.
STANDARD_COMPONENTS = ("000", "090", "ver")

# How far past one quantisation step, relative to the step, a decoded sample may be
# without failing the check. It allows for rounding in float64.
STEP_SLACK = 1e-9


def check_standard(ds: xr.Dataset) -> None:
    """Check that a dataset holds its waveform in the layout this package handles.

    The layout is a data variable ``waveform`` of float32 with the dimensions
    ``("component", "station", "time")``, in that order. Whatever else the dataset
    holds does not matter, its ``component`` coordinate included: each trace is
    compressed on its own, and the coordinate is kept as it was (some files
    name their components x, y and z, where BB.bin's are 000, 090 and ver).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to check. The waveform itself is not read.

    Raises
    ------
    NotStandardLayout
        If the dataset is not laid out that way. The message says what differs.
    """
    if "waveform" not in ds.data_vars:
        raise NotStandardLayout(
            f"no data variable 'waveform' (the data variables are {list(ds.data_vars)})"
        )
    waveform = ds["waveform"]
    if waveform.dtype != np.float32:
        raise NotStandardLayout(f"'waveform' is {waveform.dtype}, not float32")
    if waveform.dims != STANDARD_DIMS:
        raise NotStandardLayout(
            f"'waveform' has dimensions {waveform.dims}, not {STANDARD_DIMS}"
        )


@dataclass(frozen=True)
class EncodedBlock:
    """A block of stations, compressed.

    Attributes
    ----------
    streams : list[bytes]
        One FLAC stream per trace, in row-major (component, station) order.
    nbytes : np.ndarray
        The length of each stream in bytes: int64, of shape (3, n).
    gains : np.ndarray
        The gain each trace was quantised with: float64, of shape (3, n).
    """

    streams: list[bytes]
    nbytes: np.ndarray
    gains: np.ndarray


def encode_block(waveform: np.ndarray, bits: int = 32, level: int = 5) -> EncodedBlock:
    """Compress a block of stations, one FLAC stream per trace.

    Each trace is scaled by its own gain, so that its largest absolute sample lands
    on full scale for `bits`, and rounded to the nearest integer. A trace of zeros
    has a gain of 1.

    Parameters
    ----------
    waveform : np.ndarray
        The block: a float32 array of shape (3, n, nt), with nt at least 1.
    bits : int, default 32
        The quantisation setting, from 8 to 32.
    level : int, default 5
        The FLAC compression level, from 0 (fastest) to 8 (smallest).

    Returns
    -------
    EncodedBlock
        The streams, their sizes and the gains.

    Raises
    ------
    NonFiniteSamples
        If a trace holds a NaN or an infinity.
    ValueError
        If `waveform` is not a float32 array of three dimensions with at least one
        sample per trace, or if `bits` is outside 8 to 32.
    pyflac.EncoderInitException
        If libFLAC can't start a trace's stream.
    pyflac.EncoderProcessException
        If libFLAC fails to encode a trace, or to finish its stream.
    """
    if waveform.dtype != np.float32 or waveform.ndim != 3 or waveform.shape[2] == 0:
        raise ValueError(
            "waveform must be a float32 array of shape (component, station, time) "
            f"with at least one sample per trace, got {waveform.dtype} with "
            f"shape {waveform.shape}"
        )
    streams: list[bytes] = []
    gains = np.empty(waveform.shape[:2], dtype=np.float64)
    # Each trace's largest absolute sample, exactly: a NaN or an infinity stays.
    peaks = np.maximum(-waveform.min(axis=2), waveform.max(axis=2))
    for c, s in np.ndindex(*gains.shape):
        gains[c, s] = gain_for(peaks[c, s], bits)
        streams.append(encode_stream(quantise(waveform[c, s], gains[c, s]), level))
    nbytes = np.array([len(stream) for stream in streams], dtype=np.int64)
    return EncodedBlock(streams, nbytes.reshape(gains.shape), gains)


def decode_block(
    compressed: np.ndarray,
    starts: np.ndarray,
    nbytes: np.ndarray,
    gains: np.ndarray,
    offsets: np.ndarray,
    nt: int,
    first_station: int = 0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a block of stations.

    Parameters
    ----------
    compressed : np.ndarray
        The bytes of the streams: a 1-D uint8 array.
    starts : np.ndarray
        Where each stream starts in `compressed`: int64, of shape (3, n).
    nbytes : np.ndarray
        The length of each stream in bytes, of shape (3, n).
    gains : np.ndarray
        The gain each trace was quantised with, of shape (3, n).
    offsets : np.ndarray
        The value added to each trace after dividing by its gain, of shape (3, n).
    nt : int
        The number of samples in every trace.
    first_station : int, default 0
        The block's first station in its file, from which an error's message counts
        the stations.
    out : np.ndarray, optional
        A float32 array of shape (3, n, nt) to decode into, such as a slice of a
        whole waveform. By default a new one is made.

    Returns
    -------
    np.ndarray
        The block, as a float32 array of shape (3, n, nt): `out`, if it is given.

    Raises
    ------
    ValueError
        If `starts`, `nbytes`, `gains` and `offsets` do not all have the same shape,
        or `out` isn't float32 of the block's shape. If a stream is damaged, so that
        libFLAC can't decode it, or decodes to some other number of samples than
        `nt`, as one that was cut short does, or cut to nothing. The message names the
        stream's component and station, counting the stations from `first_station`.
    """
    shapes = [a.shape for a in (starts, nbytes, gains, offsets)]
    if len(set(shapes)) != 1:
        raise ValueError(
            "starts, nbytes, gains and offsets must have the same shape, got "
            f"{shapes[0]}, {shapes[1]}, {shapes[2]} and {shapes[3]}"
        )
    shape = (*starts.shape, nt)
    if out is None:
        block = np.empty(shape, dtype=np.float32)
    elif out.dtype != np.float32 or out.shape != shape:
        raise ValueError(
            f"out must be float32 of shape {shape}, got {out.dtype} of shape "
            f"{out.shape}"
        )
    else:
        block = out
    for c, s in np.ndindex(*starts.shape):
        start, size = int(starts[c, s]), int(nbytes[c, s])
        trace = f"component {c}, station {s + first_station}"
        try:
            # The decoder needs bytes: it fails on a numpy slice of more than about
            # 8 KB.
            q = decode_stream(compressed[start : start + size].tobytes())
        except (pyflac.DecoderInitException, pyflac.DecoderProcessException) as e:
            raise ValueError(f"{trace}: a FLAC stream is damaged ({e})") from e
        if q.size != nt:
            raise ValueError(
                f"{trace}: the stream decodes to {q.size} samples, expected {nt}"
            )
        block[c, s] = dequantise(q, gains[c, s], offsets[c, s])
    return block


def _check_same_block(original: np.ndarray, decoded: np.ndarray) -> None:
    """Raise a ValueError unless the arrays are blocks of one shape.

    Parameters
    ----------
    original : np.ndarray
        The block before compression: it has to have three dimensions.
    decoded : np.ndarray
        The block after decoding: it has to have the shape of `original`.

    Raises
    ------
    ValueError
        If `original` does not have three dimensions, or `decoded` has another shape.
    """
    if original.ndim != 3:
        raise ValueError(
            "expected an array of shape (component, station, time), "
            f"got shape {original.shape}"
        )
    if decoded.shape != original.shape:
        raise ValueError(
            f"decoded has shape {decoded.shape}, but the original has shape "
            f"{original.shape}"
        )


def max_relative_error(original: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    """Return, for each trace, its largest error as a fraction of its peak.

    Parameters
    ----------
    original : np.ndarray
        The block before compression, of shape (3, n, nt).
    decoded : np.ndarray
        The block after decoding, of the same shape.

    Returns
    -------
    np.ndarray
        A float64 array of shape (3, n): for each trace, the largest absolute
        difference between `decoded` and `original` over the largest absolute
        sample of `original`, or 0 for a trace of zeros.

    Raises
    ------
    ValueError
        If the arrays are not blocks of one shape.
    """
    _check_same_block(original, decoded)
    error = np.zeros(original.shape[:2], dtype=np.float64)
    for c, s in np.ndindex(*error.shape):
        trace = original[c, s].astype(np.float64)
        peak = np.abs(trace).max()
        if peak != 0:
            error[c, s] = np.abs(trace - decoded[c, s]).max() / peak
    return error


def within_one_step(
    original: np.ndarray, decoded: np.ndarray, gains: np.ndarray
) -> bool:
    """Check that every decoded sample is within one quantisation step of its original.

    A step is ``1 / gain``, and the check allows a relative slack of 1e-9 on top of
    it for rounding in float64. A trace whose original is all zeros has to decode to
    exact zeros: its gain of 1 says nothing about how far its samples may move. The
    check fails if any gain is not finite and positive, as every gain `encode_block`
    gives is. A NaN never passes.

    Parameters
    ----------
    original : np.ndarray
        The block before compression, of shape (3, n, nt).
    decoded : np.ndarray
        The block after decoding, of the same shape.
    gains : np.ndarray
        The gain each trace was quantised with, of shape (3, n).

    Returns
    -------
    bool
        True if every gain is finite and positive, and every sample of every trace
        is within one step of its original, or exactly zero where the original
        trace is all zeros.

    Raises
    ------
    ValueError
        If the arrays are not blocks of one shape, or `gains` does not have a gain
        for each trace.
    """
    _check_same_block(original, decoded)
    if gains.shape != original.shape[:2]:
        raise ValueError(
            f"gains has shape {gains.shape}, but the original has "
            f"{original.shape[:2]} traces"
        )
    if not np.all(np.isfinite(gains) & (gains > 0)):
        return False
    for c, s in np.ndindex(*gains.shape):
        trace = original[c, s].astype(np.float64)
        step = 1.0 / gains[c, s] if np.any(trace != 0) else 0.0
        if not np.all(np.abs(trace - decoded[c, s]) <= step * (1 + STEP_SLACK)):
            return False
    return True
