"""Compress a netCDF file of waveforms to FLAC, decompress it, and verify it.

A compressed file is the original netCDF file without its ``waveform`` variable:
every other variable, coordinate and attribute is written again as it was, by
xarray with the h5netcdf engine. In the waveform's place, the HDF5 group
``_flac_compressed_waveform`` holds:

- ``compressed``: every trace's FLAC stream, one after another, in one 1-D uint8
  dataset;
- ``stream_starts`` and ``stream_bytes``: where each trace's stream starts in
  ``compressed``, and its length in bytes, as int64 arrays of shape
  (component, station). ``stream_starts`` has the attribute ``stream_size``, the
  number of samples in every trace;
- ``stream_gains`` and ``stream_offsets``: each trace's gain and offset, as float64
  arrays of the same shape. A decoded value is the stream's sample divided by the
  gain, plus the offset, which is always zero;
- ``exact_stations`` and ``exact_waveform``: the real stations (see
  `waveform_compression_tools.stations`), and their traces exactly as they were.
  ``exact_stations`` holds their indices in increasing order, as int64, and
  ``exact_waveform`` their traces, as float32 of shape (component, real station,
  time). A real station has FLAC streams like any other, and decoding puts its exact
  traces in their place.

The group's attributes are ``format_version`` (1), ``dims``
(``["component", "station", "time"]``), ``dtype`` (``"float32"``),
``waveform_attrs`` (the waveform's attributes, as JSON), the optional
``waveform_encoding`` (its ``_FillValue`` and ``missing_value``, as JSON) and
``waveform_attr_types`` (the dtype of each numpy-typed attribute), ``bits`` (the quantisation
setting), ``encoder`` (``"waveform_compression_tools"`` and the package's version) and
``exact_rule`` (which stations have exact copies, in words).

A file whose stations are all real is not compressed, so it stays exact as it is.

A waveform can run to a hundred gigabytes or more, so it is read, compressed and
verified a block of stations at a time. A compressed file appears at its path only
when it is complete: it is written to the path with ``.partial`` added, then renamed.
`compress_file` checks what it writes as it goes, reading the original only once;
`verify_file` checks a compressed file against its original on its own, reading both.
"""

import gc
import json
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyflac
import xarray as xr

from . import __version__
from .codec import (
    STANDARD_DIMS,
    STEP_SLACK,
    check_standard,
    decode_block,
    encode_block,
    max_relative_error,
    within_one_step,
)
from .errors import (
    AllStationsReal,
    AlreadyCompressed,
    CheckFailed,
    HasGroups,
    NotStandardLayout,
    OutputExists,
    SameFile,
)
from .quantise import full_scale
from .stations import REAL_STATION_RULE, is_real_station

GROUP = "_flac_compressed_waveform"
# The waveform read at once: a block holds about this many bytes of it as float32.
BLOCK_BYTES = 128 * 2**20
# What every compressed file records as its encoder: this package and its version.
ENCODER = f"waveform_compression_tools {__version__}"

# The version of the layout described above. A reader refuses any other.
_FORMAT_VERSION = 1
# The compressed bytes are stored in chunks of this size, so that their dataset can
# grow a block at a time.
_CHUNK_BYTES = 4 * 2**20


@dataclass
class CompressResult:
    """What compressing a file made, and how long it took.

    Attributes
    ----------
    bytes_in : int
        The size of the original file.
    bytes_out : int
        The size of the compressed file.
    n_traces : int
        The number of traces compressed, three per station.
    n_exact : int
        The number of real stations, whose traces were also copied exactly.
    bits : int
        The quantisation setting.
    checked : bool
        Whether the compressed file was checked as it was written.
    max_relative_error : float or None
        The largest error the check found in a trace that isn't kept exactly, as a
        fraction of that trace's peak: 0.0 if every such trace came back exactly or is
        all zeros. None if the file wasn't checked.
    read_s : float
        Wall time spent opening the original and reading its waveform, in seconds.
    encode_s : float
        Wall time spent quantising and encoding, in seconds.
    write_s : float
        Wall time spent writing the compressed file, in seconds.
    check_s : float
        Wall time spent checking the compressed file, in seconds: 0.0 if it wasn't.
    read_cpu_s : float
        CPU time spent opening the original and reading its waveform, in seconds:
        the process's, as `time.process_time` counts it, in all its threads.
    encode_cpu_s : float
        CPU time spent quantising and encoding, in seconds, counted the same way.
    write_cpu_s : float
        CPU time spent writing the compressed file, in seconds, counted the same way.
    check_cpu_s : float
        CPU time spent checking the compressed file, in seconds, counted the same way.
    """

    bytes_in: int
    bytes_out: int
    n_traces: int
    n_exact: int
    bits: int
    checked: bool
    max_relative_error: float | None
    read_s: float
    encode_s: float
    write_s: float
    check_s: float
    read_cpu_s: float
    encode_cpu_s: float
    write_cpu_s: float
    check_cpu_s: float


@dataclass
class VerifyResult:
    """How a compressed file compares with its original.

    Attributes
    ----------
    max_relative_error : float
        The largest error of any trace as the file decompresses, with the exact
        copies put back, as a fraction of that trace's peak: 0 if every trace came
        back exactly, and NaN if the original holds a NaN.
    passed : bool
        True if every trace's FLAC stream, a real station's too, decodes within the
        bound the file's setting promises, ``1 / full_scale(bits)`` of the trace's
        peak with a relative slack of 1e-9, and within one quantisation step of
        every original sample, to exact zeros where the original trace is all zeros;
        the file holds exact copies of the real stations and of no others, and every
        real station comes back bit for bit; and the waveform's attributes, and
        everything in the file besides the waveform, are identical to the
        original's.
    n_exact : int
        The number of real stations in the original.
    verify_s : float
        Wall time the check took, in seconds.
    """

    max_relative_error: float
    passed: bool
    n_exact: int
    verify_s: float


class _Stopwatch:
    """Split wall and CPU time between stages.

    Each lap goes to the stage named as it ends.
    """

    def __init__(self) -> None:  # numpydoc ignore=GL08
        self.seconds = {"read": 0.0, "encode": 0.0, "write": 0.0, "check": 0.0}
        self.cpu_seconds = {"read": 0.0, "encode": 0.0, "write": 0.0, "check": 0.0}
        self._last = time.perf_counter()
        self._last_cpu = time.process_time()

    def lap(self, stage: str) -> None:
        """Add the wall and CPU time since the last lap, or since the start, to a stage.

        Parameters
        ----------
        stage : str
            ``"read"``, ``"encode"``, ``"write"`` or ``"check"``.
        """
        now, now_cpu = time.perf_counter(), time.process_time()
        self.seconds[stage] += now - self._last
        self.cpu_seconds[stage] += now_cpu - self._last_cpu
        self._last, self._last_cpu = now, now_cpu


def block_stations_for(
    n_component: int,
    nt: int,
    station_chunk: int | None,
    block_bytes: int = BLOCK_BYTES,
) -> int:
    """Choose how many stations to read at once.

    The budget gives the number of stations whose float32 samples fit in
    `block_bytes`. For a chunked waveform the block is whole station chunks, so no
    chunk is read twice, unless that would exceed the budget by more than a quarter.

    Parameters
    ----------
    n_component : int
        The number of components.
    nt : int
        The number of samples in a trace.
    station_chunk : int or None
        The waveform's chunk along the station axis, or None if it is contiguous.
    block_bytes : int, default BLOCK_BYTES
        The memory budget for a block of the waveform.

    Returns
    -------
    int
        The number of stations in a block, at least 1.
    """
    stations = max(1, block_bytes // (4 * n_component * nt))
    if station_chunk is None:
        return stations
    block = max(1, round(stations / station_chunk)) * station_chunk
    return block if block <= stations + stations // 4 else stations


def _station_chunk(waveform: xr.DataArray) -> int | None:
    """Return the waveform's chunk along the station axis.

    Parameters
    ----------
    waveform : xr.DataArray
        The waveform, as opened from a file, with dimensions (component, station,
        time).

    Returns
    -------
    int or None
        The chunk's length in stations, or None if the waveform is contiguous.
    """
    chunks = waveform.encoding.get("chunksizes")
    return None if chunks is None else int(chunks[1])


def _blocks(n_station: int, block_stations: int) -> list[tuple[int, int]]:
    """Split the stations into blocks.

    Parameters
    ----------
    n_station : int
        The number of stations.
    block_stations : int
        The most stations in a block.

    Returns
    -------
    list[tuple[int, int]]
        For each block in order, its first station and the station after its last.
        Together the blocks hold every station once.

    Raises
    ------
    ValueError
        If `block_stations` is less than 1.
    """
    if block_stations < 1:
        raise ValueError(f"block_stations must be at least 1, got {block_stations}")
    return [
        (s0, min(s0 + block_stations, n_station))
        for s0 in range(0, n_station, block_stations)
    ]


def _plain(value: object) -> object:
    """Turn a numpy value into plain Python, for `json.dumps`.

    Parameters
    ----------
    value : object
        An attribute value that `json.dumps` cannot write as it is.

    Returns
    -------
    object
        The value as a Python scalar, or as lists for an array.

    Raises
    ------
    NotStandardLayout
        If `value` is not a numpy array or scalar. `json.dumps` passes it on.
    """
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise NotStandardLayout(
        f"an attribute of type {type(value).__name__} can't go in JSON"
    )


def _attr_types(attrs: dict) -> dict[str, str]:
    """Name the numpy dtype of each attribute that is a numpy scalar or array.

    Parameters
    ----------
    attrs : dict
        The waveform's attributes.

    Returns
    -------
    dict[str, str]
        Each such attribute's name and dtype string, such as ``"float32"``.

    Raises
    ------
    NotStandardLayout
        If an attribute's dtype can't be named by its string and read back, as a
        compound (structured) one can't.
    """
    types = {}
    for name, value in attrs.items():
        if isinstance(value, np.ndarray | np.generic):
            dtype = value.dtype
            try:
                restorable = np.dtype(str(dtype)) == dtype
            except TypeError:
                restorable = False
            if not restorable:
                raise NotStandardLayout(
                    f"attribute {name!r} of type {dtype} can't be kept"
                )
            types[name] = str(dtype)
    return types


def _written_dtypes(attrs: dict) -> dict[str, str]:
    """Name the dtype each attribute is written with when the dataset is saved.

    Parameters
    ----------
    attrs : dict
        The waveform's attributes, as xarray holds them or as JSON gives them.

    Returns
    -------
    dict[str, str]
        The dtype string of each numeric, boolean or array attribute: a numpy value's
        own, float64 for a Python float, int64 for a Python int and what numpy makes
        of a list. Strings are left out.
    """
    out = {}
    for name, value in attrs.items():
        if isinstance(value, np.ndarray | np.generic):
            out[name] = str(value.dtype)
        elif isinstance(value, bool | int | float | list | tuple):
            try:
                out[name] = str(np.asarray(value).dtype)
            except ValueError:  # A ragged list is no array.
                continue
    return out


def _refuse_fill_in_range(
    block: np.ndarray, fills: list[float], bits: int, encoding_name: str
) -> None:
    """Refuse a block with a trace that reaches a finite fill value.

    A lossy sample within one step of a fill value could decode to exactly it, and
    then read back as missing.

    Parameters
    ----------
    block : np.ndarray
        Traces of shape (component, station, time).
    fills : list[float]
        The finite ``_FillValue`` and ``missing_value`` of the waveform.
    bits : int
        The quantisation setting.
    encoding_name : str
        Names the waveform in the message.

    Raises
    ------
    NotStandardLayout
        If a fill value lies within one quantisation step of a trace's range.
    """
    if not fills:
        return
    low = block.min(axis=2).astype(np.float64)
    high = block.max(axis=2).astype(np.float64)
    step = np.abs(block).max(axis=2) / full_scale(bits)
    for fill in fills:
        if np.any((low - step <= fill) & (fill <= high + step)):
            raise NotStandardLayout(
                f"{encoding_name} has the finite fill value {fill} inside the data's "
                "range, which can't survive lossy compression"
            )


def _encoding_json(encoding: dict) -> str:
    """Write the waveform's ``_FillValue`` and ``missing_value`` as strict JSON.

    Parameters
    ----------
    encoding : dict
        The waveform's netCDF encoding, as xarray read it.

    Returns
    -------
    str
        A JSON object. A float that isn't finite is the string ``"NaN"``,
        ``"Infinity"`` or ``"-Infinity"``. A waveform with no ``_FillValue`` has
        ``{"_FillValue": null}``, so it is written back without one.
    """
    out: dict[str, object] = {"_FillValue": None}
    for name in ("_FillValue", "missing_value"):
        if name not in encoding:
            continue
        value = encoding[name]
        if value is not None and not np.isfinite(value):
            value = str(float(value)).replace("inf", "Infinity").replace("nan", "NaN")
        elif isinstance(value, np.generic):
            value = value.item()
        out[name] = value
    return json.dumps(out)


def _encoding_back(text: str) -> dict:
    """Read what `_encoding_json` wrote, as values for the waveform's encoding.

    Parameters
    ----------
    text : str
        The JSON object.

    Returns
    -------
    dict
        Each value as a float32, the waveform's dtype, or ``None``.
    """
    return {
        name: None if value is None else np.float32(value)
        for name, value in json.loads(text).items()
    }


def _real_stations(ds: xr.Dataset) -> np.ndarray:
    """Find the real stations of a dataset, by their names.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset in the standard layout. Its waveform is not read.

    Returns
    -------
    np.ndarray
        The indices of the real stations along the station dimension (see
        `waveform_compression_tools.stations.is_real_station`), in increasing order, as int64.

    Raises
    ------
    NotStandardLayout
        If the dataset has no ``station`` coordinate along the station dimension, so
        nothing names its stations, or a name is bytes that are not ASCII.
    """
    if "station" not in ds.coords or ds["station"].dims != ("station",):
        raise NotStandardLayout(
            "no 'station' coordinate along the station dimension, so the real "
            "stations can't be told from a synthetic grid's"
        )
    try:
        real = [is_real_station(name) for name in ds["station"].values]
    except UnicodeDecodeError as e:
        raise NotStandardLayout(
            f"a station's name is bytes that aren't ASCII ({e})"
        ) from e
    return np.flatnonzero(real).astype(np.int64)


def _in_block(stations: np.ndarray, s0: int, s1: int) -> slice:
    """Find which of some stations are in a block.

    Parameters
    ----------
    stations : np.ndarray
        Station indices, in increasing order.
    s0 : int
        The block's first station.
    s1 : int
        The station after its last.

    Returns
    -------
    slice
        The part of `stations` from `s0` up to, but not including, `s1`.
    """
    k0, k1 = np.searchsorted(stations, (s0, s1))
    return slice(int(k0), int(k1))


def _refuse_subgroups(path: Path) -> None:
    """Refuse an HDF5 file that holds a group besides its root.

    xarray reads only the root group, so a compressed file would go without it.

    Parameters
    ----------
    path : Path
        The file.

    Raises
    ------
    HasGroups
        If the root group holds a group.
    """
    with h5py.File(path, "r") as f:
        groups = [name for name, item in f.items() if isinstance(item, h5py.Group)]
    if groups:
        raise HasGroups(
            f"{path} holds HDF5 groups besides its root ({', '.join(groups)}), "
            "which a compressed file would go without"
        )


def check_compressible(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Refuse a file that can't be compressed as it is, before any sample is read.

    These are the refusals that need no samples, made in this order: `dst` is `src`, a
    file is at `dst` already, `src` is compressed already, it holds an HDF5 group, it
    isn't in the standard layout, and every station is real. Only metadata and station
    names are read, so a refused file costs no full read. `compress_file` calls this
    first. The refusals that need the samples, of a NaN or an infinity and of a fill
    value inside a trace's range, come during its pass.

    Parameters
    ----------
    src : Path
        The netCDF file to compress.
    dst : Path
        Where the compressed file is to go.
    overwrite : bool, default False
        Whether a file already at `dst` may be replaced.

    Raises
    ------
    SameFile
        If `dst`, or `dst` with ``.partial`` added, resolves to `src`, whatever
        `overwrite` says.
    OutputExists
        If something is at `dst` already and `overwrite` is false.
    AlreadyCompressed
        If `src` holds a compressed waveform already (see `is_compressed`).
    HasGroups
        If `src` holds an HDF5 group besides its root.
    NotStandardLayout
        If `src` is not in the standard layout (see
        `waveform_compression_tools.codec.check_standard`), has no ``station``
        coordinate along the station dimension, has a station whose name is bytes that
        aren't ASCII, has no stations, or has a waveform attribute that can't be
        written as JSON or whose type can't be kept, as a complex or a compound one
        can't.
    AllStationsReal
        If every station of `src` is real, so it stays exact as it is.
    """
    partial = Path(f"{dst}.partial")
    # The partial file is overwritten, and deleted if anything fails, and the rename
    # replaces whatever is at dst: either would destroy the source.
    if src.resolve() in (dst.resolve(), partial.resolve()):
        raise SameFile(f"compressing {src} to {dst} would overwrite it")
    if not overwrite and os.path.lexists(dst):
        raise OutputExists(f"{dst} exists already, and is replaced only if asked")
    if is_compressed(src):
        raise AlreadyCompressed(f"{src} is compressed already")
    _refuse_subgroups(src)
    with xr.open_dataset(src, engine="h5netcdf") as ds:
        check_standard(ds)
        exact = _real_stations(ds)
        waveform = ds["waveform"]
        n_station = waveform.shape[1]
        if n_station == 0:
            raise NotStandardLayout(
                f"{src} has no stations, so there is nothing to compress"
            )
        # Each refuses an attribute that the compressed file can't keep.
        json.dumps(dict(waveform.attrs), default=_plain)
        _attr_types(waveform.attrs)
        if exact.size == n_station:
            raise AllStationsReal(
                f"all {n_station} stations of {src} are real, so it isn't compressed"
            )


def compress_file(
    src: Path,
    dst: Path,
    bits: int = 32,
    level: int = 5,
    block_stations: int | None = None,
    check: bool = True,
    overwrite: bool = False,
) -> CompressResult:
    """Compress a netCDF file in the standard layout, one FLAC stream per trace.

    The file is first refused if it can't be compressed as it is (see
    `check_compressible`). The waveform is then read and encoded a block of stations
    at a time, so memory holds one block and its streams, never the whole waveform.
    The real stations' traces (see `waveform_compression_tools.stations`) are also
    copied exactly, a block at a time. The compressed file is written to `dst` with
    ``.partial`` added, and renamed to `dst` when it is complete. If anything goes
    wrong, the partial file is deleted and the exception raised again, so nothing
    appears at `dst`, and a file already there stays as it was.

    The check proves the compressed file in the same pass, so the original is read
    only once. As soon as a block is written, its streams are read back and decoded as
    a reader would decode them, and its exact copies are read back. The block has to
    pass `verify_file`'s rules against the block still in memory: each trace within one
    step of its own gain, and within the bound the setting promises, of its peak; a
    trace of zeros as exact zeros; and each exact copy bit for bit. When the pass is
    done, the closed partial file is reopened, and its group has to hold what the pass
    wrote, every stream with the CRC-32 it had when it was decoded. Only then is the
    file renamed to `dst`.

    Parameters
    ----------
    src : Path
        The netCDF file to compress, in the standard layout (see
        `waveform_compression_tools.codec.check_standard`), with a ``station``
        coordinate that names its stations, and no HDF5 group besides its root.
    dst : Path
        Where to write the compressed file.
    bits : int, default 32
        The quantisation setting, from 8 to 32.
    level : int, default 5
        The FLAC compression level, from 0 (fastest) to 8 (smallest).
    block_stations : int, optional
        The most stations read and encoded at once. By default a block holds about
        `BLOCK_BYTES` of waveform, in whole chunks of stations if the waveform is
        chunked (see `block_stations_for`).
    check : bool, default True
        Whether to check the compressed file as it is written.
    overwrite : bool, default False
        Whether to replace a file already at `dst`. `src` itself is never replaced.

    Returns
    -------
    CompressResult
        The sizes of both files, the numbers of traces and of real stations, the
        setting, the largest error the check found, and the wall and CPU time spent
        reading, encoding, writing and checking.

    Raises
    ------
    SameFile, OutputExists, AlreadyCompressed, HasGroups, AllStationsReal
        As `check_compressible` raises them, before any sample is read.
    NotStandardLayout
        As `check_compressible` raises it, or if `src` has a finite ``_FillValue`` or
        ``missing_value`` within one quantisation step of a trace's range.
    NonFiniteSamples
        If a trace holds a NaN or an infinity.
    CheckFailed
        If `check` is true and the compressed file fails the check, a stream that
        doesn't decode included.
    ValueError
        If `bits` is outside 8 to 32, which is checked first, or if `block_stations`
        is less than 1.
    pyflac.EncoderInitException
        If libFLAC can't start a trace's stream.
    pyflac.EncoderProcessException
        If libFLAC fails to encode a trace, or to finish its stream.
    """
    full_scale(bits)  # Refuses a setting outside 8 to 32 before the file is opened.
    stopwatch = _Stopwatch()
    check_compressible(src, dst, overwrite)
    partial = Path(f"{dst}.partial")
    with xr.open_dataset(src, engine="h5netcdf") as ds:
        exact = _real_stations(ds)
        waveform = ds["waveform"]
        n_component, n_station, nt = waveform.shape
        if block_stations is None:
            block_stations = block_stations_for(
                n_component, nt, _station_chunk(waveform), BLOCK_BYTES
            )
        blocks = _blocks(n_station, block_stations)
        waveform_encoding = _encoding_json(waveform.encoding)
        attrs: dict[str, object] = {
            "format_version": _FORMAT_VERSION,
            "dims": list(STANDARD_DIMS),
            "dtype": "float32",
            "waveform_attrs": json.dumps(dict(waveform.attrs), default=_plain),
            "waveform_encoding": waveform_encoding,
            "waveform_attr_types": json.dumps(_attr_types(waveform.attrs)),
            "bits": bits,
            "encoder": ENCODER,
            "exact_rule": REAL_STATION_RULE,
        }
        fills = [
            float(v)
            for name in ("_FillValue", "missing_value")
            for v in np.atleast_1d(waveform.encoding.get(name, np.nan))
            if v is not None and np.isfinite(v)
        ]
        stopwatch.lap("read")
        try:
            rest = ds.drop_vars("waveform")
            # A dimension that only the waveform had, as time is in a file without a
            # time coordinate, is gone, so it can't be declared unlimited any more.
            unlimited = ds.encoding.get("unlimited_dims", ())
            rest.to_netcdf(
                partial,
                engine="h5netcdf",
                unlimited_dims=[name for name in unlimited if name in rest.dims],
            )
            with h5py.File(partial, "a") as f:
                group = f.create_group(GROUP)
                compressed = group.create_dataset(
                    "compressed",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.uint8,
                    chunks=(_CHUNK_BYTES,),
                )
                exact_waveform = group.create_dataset(
                    "exact_waveform",
                    shape=(n_component, exact.size, nt),
                    dtype=np.float32,
                )
                traces = (n_component, n_station)
                starts, nbytes = np.zeros(traces, np.int64), np.zeros(traces, np.int64)
                gains, offsets = np.zeros(traces), np.zeros(traces)
                # The file as a reader sees it, each block as soon as it is written.
                written = _Streams(
                    compressed=compressed,
                    starts=starts,
                    nbytes=nbytes,
                    gains=gains,
                    offsets=offsets,
                    nt=nt,
                    bits=bits,
                    waveform_attrs=dict(waveform.attrs),
                    encoding=_encoding_back(waveform_encoding),
                    exact_stations=exact,
                    exact_waveform=exact_waveform,
                )
                crcs = np.zeros(traces, np.uint32)
                worst = 0.0
                stopwatch.lap("write")
                for s0, s1 in blocks:
                    block = waveform[:, s0:s1, :].values
                    stopwatch.lap("read")
                    _refuse_fill_in_range(block, fills, bits, str(src))
                    encoded = encode_block(block, bits, level)
                    stopwatch.lap("encode")
                    # The block's streams go after those already written, in the
                    # order encode_block gives them.
                    end = compressed.shape[0]
                    sizes = encoded.nbytes.ravel()
                    block_starts = end + np.cumsum(sizes) - sizes
                    starts[:, s0:s1] = block_starts.reshape(encoded.nbytes.shape)
                    nbytes[:, s0:s1] = encoded.nbytes
                    gains[:, s0:s1] = encoded.gains
                    compressed.resize((end + int(sizes.sum()),))
                    compressed[end:] = np.frombuffer(
                        b"".join(encoded.streams), dtype=np.uint8
                    )
                    here = _in_block(exact, s0, s1)
                    exact_waveform[:, here] = block[:, exact[here] - s0]
                    stopwatch.lap("write")
                    if check:
                        error, crcs[:, s0:s1] = _check_block(
                            written, block, s0, s1, partial
                        )
                        worst = max(worst, error)
                        stopwatch.lap("check")
                    # pyflac's encoders keep their output in reference cycles, which
                    # would otherwise wait for the next collection and add up.
                    gc.collect()
                datasets = {
                    "stream_starts": starts,
                    "stream_bytes": nbytes,
                    "stream_gains": gains,
                    "stream_offsets": offsets,
                    "exact_stations": exact,
                }
                for name, data in datasets.items():
                    group.create_dataset(name, data=data)
                group["stream_starts"].attrs["stream_size"] = nt
                group.attrs.update(attrs)
            stopwatch.lap("write")
            if check:
                _check_structure(partial, datasets, attrs, nt, crcs, blocks)
                stopwatch.lap("check")
            os.replace(partial, dst)
            stopwatch.lap("write")
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    return CompressResult(
        bytes_in=src.stat().st_size,
        bytes_out=dst.stat().st_size,
        n_traces=n_component * n_station,
        n_exact=int(exact.size),
        bits=bits,
        checked=check,
        max_relative_error=worst if check else None,
        read_s=stopwatch.seconds["read"],
        encode_s=stopwatch.seconds["encode"],
        write_s=stopwatch.seconds["write"],
        check_s=stopwatch.seconds["check"],
        read_cpu_s=stopwatch.cpu_seconds["read"],
        encode_cpu_s=stopwatch.cpu_seconds["encode"],
        write_cpu_s=stopwatch.cpu_seconds["write"],
        check_cpu_s=stopwatch.cpu_seconds["check"],
    )


@dataclass(frozen=True)
class _Streams:
    """The streams of a compressed file, with what it takes to decode them.

    It reads from the file, so it can be used only while the file is open.

    Attributes
    ----------
    compressed : h5py.Dataset
        The bytes of every stream. Only the bytes a block needs are read.
    starts : np.ndarray
        Where each trace's stream starts in `compressed`, of shape
        (component, station).
    nbytes : np.ndarray
        The length of each trace's stream in bytes, of the same shape.
    gains : np.ndarray
        Each trace's gain, of the same shape.
    offsets : np.ndarray
        Each trace's offset, of the same shape.
    nt : int
        The number of samples in every trace.
    bits : int
        The quantisation setting the file declares, from 8 to 32.
    waveform_attrs : dict
        The original waveform's attributes, each numpy value in its own type.
    encoding : dict
        The waveform's ``_FillValue`` and ``missing_value`` to put back. A file
        without ``waveform_encoding`` gets what such files always decoded to, a NaN
        ``_FillValue``.
    exact_stations : np.ndarray
        The real stations, in increasing order.
    exact_waveform : h5py.Dataset
        Their exact traces, of shape (component, real station, nt). Only the traces
        a block needs are read.
    """

    compressed: h5py.Dataset
    starts: np.ndarray
    nbytes: np.ndarray
    gains: np.ndarray
    offsets: np.ndarray
    nt: int
    bits: int
    waveform_attrs: dict
    encoding: dict
    exact_stations: np.ndarray
    exact_waveform: h5py.Dataset

    def decode(self, s0: int, s1: int) -> np.ndarray:
        """Decode a block of stations from their FLAC streams alone.

        Parameters
        ----------
        s0 : int
            The block's first station.
        s1 : int
            The station after its last, greater than `s0`.

        Returns
        -------
        np.ndarray
            The block, as a float32 array of shape (component, s1 - s0, nt).
        """
        starts, nbytes = self.starts[:, s0:s1], self.nbytes[:, s0:s1]
        first, end = int(starts.min()), int((starts + nbytes).max())
        return decode_block(
            self.compressed[first:end],
            starts - first,
            nbytes,
            self.gains[:, s0:s1],
            self.offsets[:, s0:s1],
            self.nt,
        )

    def put_back_exact(self, block: np.ndarray, s0: int, s1: int) -> np.ndarray:
        """Put the exact traces of a decoded block's real stations in place.

        Parameters
        ----------
        block : np.ndarray
            The block from `decode`, changed in place.
        s0 : int
            The block's first station.
        s1 : int
            The station after its last.

        Returns
        -------
        np.ndarray
            The stations whose traces were put back, counted from `s0`.
        """
        here = _in_block(self.exact_stations, s0, s1)
        stations = self.exact_stations[here] - s0
        block[:, stations] = self.exact_waveform[:, here]
        return stations


def _open_streams(f: h5py.File, path: Path) -> _Streams:
    """Read where a compressed file's streams are, and how to decode them.

    Parameters
    ----------
    f : h5py.File
        The compressed file, open for reading.
    path : Path
        The file's path, for error messages.

    Returns
    -------
    _Streams
        The streams, with what it takes to decode them.

    Raises
    ------
    ValueError
        If the file has no group ``_flac_compressed_waveform``, the group is in a
        format version other than 1, its ``bits`` is outside 8 to 32, its
        ``stream_bytes``, ``stream_gains`` or ``stream_offsets`` differs in shape
        from its ``stream_starts``, its ``exact_stations`` are not stations of the
        file in increasing order, or its ``exact_waveform`` is not float32 with each
        of their traces.
    """
    if GROUP not in f:
        raise ValueError(f"{path} has no group {GROUP}, so it is not compressed")
    group = f[GROUP]
    version = int(group.attrs["format_version"])
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"{path} is in format version {version}, "
            f"and only version {_FORMAT_VERSION} can be read"
        )
    bits = int(group.attrs["bits"])
    full_scale(bits)  # Refuses a setting outside 8 to 32, which no file can have.
    starts = group["stream_starts"][...]
    # A block reads only its own stations, so it would never see a value too many.
    for name in ("stream_bytes", "stream_gains", "stream_offsets"):
        if group[name].shape != starts.shape:
            raise ValueError(
                f"{path}: {name} has shape {group[name].shape}, but stream_starts "
                f"has shape {starts.shape}"
            )
    n_component, n_station = starts.shape
    nt = int(group["stream_starts"].attrs["stream_size"])
    # Each exact trace has to go back to a station of the file, and only one.
    exact_stations = group["exact_stations"][...]
    if not (
        exact_stations.ndim == 1
        and np.issubdtype(exact_stations.dtype, np.integer)
        and np.all(exact_stations[1:] > exact_stations[:-1])
        and np.all((exact_stations >= 0) & (exact_stations < n_station))
    ):
        raise ValueError(
            f"{path}: exact_stations has to hold stations from 0 to {n_station - 1}, "
            "in increasing order"
        )
    exact_waveform = group["exact_waveform"]
    shape = (n_component, exact_stations.size, nt)
    if exact_waveform.dtype != np.float32 or exact_waveform.shape != shape:
        raise ValueError(
            f"{path}: exact_waveform is {exact_waveform.dtype} of shape "
            f"{exact_waveform.shape}, not float32 of shape {shape}"
        )
    waveform_attrs = json.loads(group.attrs["waveform_attrs"])
    for name, dtype in json.loads(group.attrs.get("waveform_attr_types", "{}")).items():
        value = waveform_attrs[name]
        waveform_attrs[name] = (
            np.asarray(value, dtype)
            if isinstance(value, list)
            else np.dtype(dtype).type(value)
        )
    encoding = group.attrs.get("waveform_encoding", '{"_FillValue": "NaN"}')
    return _Streams(
        compressed=group["compressed"],
        starts=starts,
        nbytes=group["stream_bytes"][...],
        gains=group["stream_gains"][...],
        offsets=group["stream_offsets"][...],
        nt=nt,
        bits=bits,
        waveform_attrs=waveform_attrs,
        encoding=_encoding_back(encoding),
        exact_stations=exact_stations,
        exact_waveform=exact_waveform,
    )


def _compare_block(
    streams: _Streams, block: np.ndarray, real: np.ndarray, s0: int, s1: int
) -> tuple[float, bool]:
    """Decode a block of stations from a compressed file, and judge it by verify's rules.

    Every trace's FLAC stream is decoded, a real station's too, since a reader that
    ignores the exact copies decodes it. It has to be within the bound the file's
    setting promises, and within one step of its own gain (see `verify_file`). Then,
    with the exact copies put back, every real station has to come back bit for bit.

    Parameters
    ----------
    streams : _Streams
        The compressed file's streams.
    block : np.ndarray
        The original's traces of the block's stations, of shape
        (component, s1 - s0, nt).
    real : np.ndarray
        The original's real stations, in increasing order.
    s0 : int
        The block's first station.
    s1 : int
        The station after its last.

    Returns
    -------
    error : float
        The largest error of a trace as the file decompresses, with the exact copies
        put back, as a fraction of that trace's peak: NaN if the block holds a NaN.
    passed : bool
        Whether the block passes.

    Raises
    ------
    ValueError
        If a stream decodes to the wrong number of samples, or holds no frames.
    pyflac.DecoderProcessException
        If a stream is damaged.
    """
    decoded = streams.decode(s0, s1)
    error = max_relative_error(block, decoded)
    # Each check covers what the others can't. The bound holds every stream to what the
    # file's setting promises, whatever its gains say, so it fails an encoder that wrote
    # wrong gains consistently. One step holds a trace of zeros to exact zeros, where
    # the bound sees no error at all. With the exact copies back, a real station has to
    # come back bit for bit, though neither of the others would see a sample of it
    # moved by less than a step.
    bound = (1 + STEP_SLACK) / full_scale(streams.bits)
    passed = bool(np.all(error <= bound)) and within_one_step(
        block, decoded, streams.gains[:, s0:s1]
    )
    put_back = streams.put_back_exact(decoded, s0, s1)
    # The error reported is of the file as it decompresses.
    error[:, put_back] = max_relative_error(block[:, put_back], decoded[:, put_back])
    here = real[_in_block(real, s0, s1)] - s0  # Counted from the block's first.
    passed = passed and np.array_equal(
        block[:, here].view(np.uint32), decoded[:, here].view(np.uint32)
    )
    return float(error.max()), passed


def _crc32s(
    compressed: h5py.Dataset, starts: np.ndarray, nbytes: np.ndarray
) -> np.ndarray:
    """Compute the CRC-32 of each of a block's streams, as a file holds it.

    Parameters
    ----------
    compressed : h5py.Dataset
        The bytes of every stream. Only the block's are read.
    starts : np.ndarray
        Where each of the block's streams starts in `compressed`, of shape
        (component, station).
    nbytes : np.ndarray
        The length of each in bytes, of the same shape.

    Returns
    -------
    np.ndarray
        Each stream's CRC-32, as uint32 of the same shape.
    """
    first, end = int(starts.min()), int((starts + nbytes).max())
    data = compressed[first:end]
    crcs = [
        zlib.crc32(data[start - first : start - first + size])
        for start, size in zip(starts.flat, nbytes.flat)
    ]
    return np.array(crcs, np.uint32).reshape(starts.shape)


def _check_block(
    streams: _Streams, block: np.ndarray, s0: int, s1: int, path: Path
) -> tuple[float, np.ndarray]:
    """Check a block of stations as soon as it is written, against the original's.

    The block's streams and exact copies are read back from the file, and the block
    has to pass verify's rules (see `_compare_block`). An error while decoding counts
    as a failed check.

    Parameters
    ----------
    streams : _Streams
        The file's streams, as written so far.
    block : np.ndarray
        The original's traces of the block's stations, of shape
        (component, s1 - s0, nt).
    s0 : int
        The block's first station.
    s1 : int
        The station after its last.
    path : Path
        The file, for error messages.

    Returns
    -------
    error : float
        The block's largest error as the file decompresses, as a fraction of a
        trace's peak.
    crcs : np.ndarray
        Each stream's CRC-32 as it was read back: uint32 of shape
        (component, s1 - s0).

    Raises
    ------
    CheckFailed
        If the block fails, or a stream doesn't decode.
    """
    starts, nbytes = streams.starts[:, s0:s1], streams.nbytes[:, s0:s1]
    crcs = _crc32s(streams.compressed, starts, nbytes)
    try:
        error, passed = _compare_block(streams, block, streams.exact_stations, s0, s1)
    except (
        ValueError,
        pyflac.DecoderInitException,
        pyflac.DecoderProcessException,
    ) as e:
        raise CheckFailed(
            f"{path}: stations {s0} to {s1 - 1} don't decode ({e})"
        ) from e
    if not passed:
        raise CheckFailed(
            f"{path}: stations {s0} to {s1 - 1} don't come back as they should at "
            f"{streams.bits} bits"
        )
    return error, crcs


def _check_structure(
    path: Path,
    datasets: dict[str, np.ndarray],
    attrs: dict[str, object],
    nt: int,
    crcs: np.ndarray,
    blocks: list[tuple[int, int]],
) -> None:
    """Check that a closed compressed file holds what `compress_file` wrote and checked.

    The file is reopened for reading. Its group has to hold the datasets and the
    attributes written, each as it was written; the streams and the exact copies
    have to have their types and shapes; and every stream has to have the CRC-32 it
    had when the pass decoded it.

    Parameters
    ----------
    path : Path
        The file, closed since it was written.
    datasets : dict[str, np.ndarray]
        The datasets written when the pass was done, by name.
    attrs : dict[str, object]
        The group's attributes, by name.
    nt : int
        The number of samples in every trace.
    crcs : np.ndarray
        Each trace's stream's CRC-32, as the pass read it back, of shape
        (component, station).
    blocks : list[tuple[int, int]]
        The blocks the pass wrote, each its first station and the station after its
        last.

    Raises
    ------
    CheckFailed
        If the file differs in any of these.
    """
    starts, nbytes = datasets["stream_starts"], datasets["stream_bytes"]
    n_exact = datasets["exact_stations"].size
    shapes = {
        "compressed": (np.dtype(np.uint8), (int(nbytes.sum()),)),
        "exact_waveform": (np.dtype(np.float32), (starts.shape[0], n_exact, nt)),
    }
    with h5py.File(path, "r") as f:
        group = f.get(GROUP)
        if not isinstance(group, h5py.Group) or set(group) != {*datasets, *shapes}:
            raise CheckFailed(f"{path} doesn't hold the datasets that were written")
        if set(group.attrs) != set(attrs) or not all(
            np.array_equal(group.attrs[name], value) for name, value in attrs.items()
        ):
            raise CheckFailed(
                f"{path}: the attributes of {GROUP} differ from those written"
            )
        for name, data in datasets.items():
            if group[name].dtype != data.dtype or not np.array_equal(
                group[name][...], data
            ):
                raise CheckFailed(f"{path}: {name} differs from what was written")
        for name, (dtype, shape) in shapes.items():
            if (group[name].dtype, group[name].shape) != (dtype, shape):
                raise CheckFailed(
                    f"{path}: {name} is {group[name].dtype} of shape "
                    f"{group[name].shape}, not {dtype} of shape {shape}"
                )
        if dict(group["stream_starts"].attrs) != {"stream_size": nt}:
            raise CheckFailed(f"{path}: stream_size isn't {nt}, the samples in a trace")
        for s0, s1 in blocks:
            here = np.s_[:, s0:s1]
            if not np.array_equal(
                _crc32s(group["compressed"], starts[here], nbytes[here]), crcs[here]
            ):
                raise CheckFailed(
                    f"{path}: a stream of stations {s0} to {s1 - 1} differs from the "
                    "one checked"
                )


def decompress_file(path: Path) -> xr.Dataset:
    """Decompress a file written by `compress_file`.

    Parameters
    ----------
    path : Path
        The compressed file.

    Returns
    -------
    xr.Dataset
        The original dataset, loaded into memory: every variable, coordinate and
        attribute of the original, and its ``waveform`` as float32 with the
        dimensions (component, station, time) and its original attributes. The real
        stations' traces are the exact copies the file holds; every other sample is
        within one quantisation step of the original. The whole waveform is decoded,
        so it has to fit in memory.

    Raises
    ------
    ValueError
        If `path` is not a compressed file in format version 1 with a setting from 8
        to 32 and arrays that agree with each other, or a stream decodes to the
        wrong number of samples.
    pyflac.DecoderProcessException
        If a stream is damaged.
    """
    with h5py.File(path, "r") as f:
        streams = _open_streams(f, path)
        n_component, n_station = streams.starts.shape
        waveform = np.empty((n_component, n_station, streams.nt), np.float32)
        block_stations = block_stations_for(n_component, streams.nt, None, BLOCK_BYTES)
        for s0, s1 in _blocks(n_station, block_stations):
            block = streams.decode(s0, s1)
            streams.put_back_exact(block, s0, s1)
            waveform[:, s0:s1] = block
    ds = xr.load_dataset(path, engine="h5netcdf")
    ds["waveform"] = (STANDARD_DIMS, waveform, streams.waveform_attrs)
    ds["waveform"].encoding.update(streams.encoding)
    return ds


def verify_file(
    original: Path, compressed: Path, block_stations: int | None = None
) -> VerifyResult:
    """Check a compressed file against its original, a block of stations at a time.

    The compressed file is decoded as it was written, from its own streams, gains
    and offsets, and each block is compared with the same stations of the original.
    Every trace's FLAC stream is checked, a real station's too, since a reader that
    ignores the exact copies decodes it. Each trace's largest error has to be within
    the bound the file's setting promises: ``1 / full_scale(bits)`` of the trace's
    peak, with the same relative slack of 1e-9 as a step, whatever gains the file
    holds. Every sample also has to be within one quantisation step of the original
    (see `waveform_compression_tools.codec.within_one_step`), so a trace of zeros has to come back
    as exact zeros. Then, with the exact copies put back, every real station (see
    `waveform_compression_tools.stations`) has to come back bit for bit, and the file has to hold
    exact copies of the real stations and of no others. The waveform's attributes,
    and everything besides the waveform (every other variable, coordinate and
    attribute), have to be identical to the original's.

    Parameters
    ----------
    original : Path
        The netCDF file that was compressed, in the standard layout.
    compressed : Path
        The file `compress_file` wrote from it.
    block_stations : int, optional
        The most stations read and decoded at once. By default a block holds about
        `BLOCK_BYTES` of waveform, in whole chunks of stations if the original is
        chunked (see `block_stations_for`). It need not be the number the
        file was compressed with.

    Returns
    -------
    VerifyResult
        The largest error relative to a trace's peak, whether the file passed, the
        number of real stations, and the time the check took.

    Raises
    ------
    NotStandardLayout
        If `original` is not in the standard layout, has no ``station`` coordinate,
        or has a station whose name is bytes that are not ASCII.
    ValueError
        If `compressed` is not a compressed file in format version 1 with a setting
        from 8 to 32 and arrays that agree with each other, if it decodes to a
        waveform of another shape than the original's, if a stream decodes to the
        wrong number of samples, or if `block_stations` is less than 1.
    pyflac.DecoderProcessException
        If a stream is damaged.
    """
    start = time.perf_counter()
    worst = np.float64(0.0)
    with (
        xr.open_dataset(original, engine="h5netcdf") as ds,
        xr.open_dataset(compressed, engine="h5netcdf") as rest,
        h5py.File(compressed, "r") as f,
    ):
        check_standard(ds)
        real = _real_stations(ds)
        waveform = ds["waveform"]
        streams = _open_streams(f, compressed)
        shape = (*streams.starts.shape, streams.nt)
        if shape != waveform.shape:
            raise ValueError(
                f"{compressed} decodes to a waveform of shape {shape}, but the "
                f"original {original} holds one of shape {waveform.shape}"
            )
        # `identical` compares attribute values and not their types, so the types
        # and the encoding are compared on their own.
        same_attrs = xr.Dataset(attrs=dict(waveform.attrs)).identical(
            xr.Dataset(attrs=streams.waveform_attrs)
        ) and _written_dtypes(waveform.attrs) == _written_dtypes(streams.waveform_attrs)
        # The encoding as JSON, so a NaN fill value equals a NaN one.
        same_encoding = json.loads(_encoding_json(waveform.encoding)) == json.loads(
            _encoding_json(streams.encoding)
        )
        passed = (
            ds.drop_vars("waveform").identical(rest)
            and same_attrs
            and same_encoding
            and np.array_equal(streams.exact_stations, real)
        )
        if block_stations is None:
            block_stations = block_stations_for(
                shape[0], shape[2], _station_chunk(waveform), BLOCK_BYTES
            )
        for s0, s1 in _blocks(waveform.shape[1], block_stations):
            block = waveform[:, s0:s1, :].values
            error, block_passed = _compare_block(streams, block, real, s0, s1)
            passed = passed and block_passed
            worst = np.maximum(worst, error)  # A NaN stays.
            gc.collect()
    return VerifyResult(
        max_relative_error=float(worst),
        passed=passed,
        n_exact=int(real.size),
        verify_s=time.perf_counter() - start,
    )


def is_compressed(path: Path) -> bool:
    """Say whether a file was written by `compress_file`.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    bool
        True if `path` is an HDF5 file with the group ``_flac_compressed_waveform``,
        and False otherwise, as when it does not exist.
    """
    if not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as f:
        return isinstance(f.get(GROUP), h5py.Group)
