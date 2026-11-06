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
``waveform_attrs`` (the waveform's attributes, as JSON),
``waveform_encoding`` (its ``_FillValue`` and ``missing_value``, as JSON) and
``waveform_attr_types`` (the dtype of each numpy-typed attribute), ``bits`` (the quantisation
setting), ``encoder`` (``"waveform_compression_tools"`` and the package's version) and
``exact_rule`` (which stations have exact copies, in words).

A file whose stations are all real is not compressed, so it stays exact as it is.

A waveform can run to a hundred gigabytes or more, so it is read, compressed and
verified a block of stations at a time. A compressed file appears at its path only
when it is complete: it is written to the path with ``.partial`` added, then renamed.
`compress_file` checks the file it writes as `verify_file` would, but reads the
original's samples only once; `verify_file` checks a compressed file against its
original on its own, reading both again.
"""

import gc
import json
import os
import time
import zlib
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from . import __version__
from .codec import (
    STANDARD_DIMS,
    STEP_SLACK,
    EncodedBlock,
    check_standard,
    decode_block,
    encode_block,
    max_relative_error,
    within_one_step,
)
from .content_hash import content_hash
from .destinations import (
    HDF5Output,
    check_destination,
    naming,
    write_netcdf,
    writing,
)
from .errors import (
    AllStationsReal,
    AlreadyCompressed,
    CheckFailed,
    HasGroups,
    NotCompressed,
    NotStandardLayout,
    UnknownFormat,
    about,
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
# grow a block at a time, or of the waveform's own size if that is smaller, so that a
# small file doesn't take 4 MiB.
_CHUNK_BYTES = 4 * 2**20
# The waveform's encoding that a compressed file keeps.
_FILL_NAMES = ("_FillValue", "missing_value")


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
    max_relative_error : float or None
        The largest error the check found in a trace that isn't kept exactly, as a
        fraction of that trace's peak: 0.0 if every such trace came back exactly or is
        all zeros. None if the file wasn't checked (see `compress_file`).
    output_hash : str
        The compressed file's content hash (see
        `waveform_compression_tools.content_hash`), taken before it was renamed into
        place.
    read_s : float
        Wall time spent opening the original and reading its waveform, in seconds.
    encode_s : float
        Wall time spent quantising and encoding, in seconds.
    write_s : float
        Wall time spent writing the compressed file, in seconds.
    check_s : float
        Wall time spent checking the compressed file, in seconds: 0.0 if it wasn't.
    hash_s : float
        Wall time spent hashing the compressed file, in seconds.
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
    max_relative_error: float | None
    output_hash: str
    read_s: float
    encode_s: float
    write_s: float
    check_s: float
    hash_s: float
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
        True if the file passes every rule of `verify_file`.
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
        stages = ("read", "encode", "write", "check", "hash")
        self.seconds = dict.fromkeys(stages, 0.0)
        self.cpu_seconds = dict.fromkeys(stages, 0.0)
        self._last = time.perf_counter()
        self._last_cpu = time.process_time()

    def lap(self, stage: str) -> None:
        """Add the wall and CPU time since the last lap, or since the start, to a stage.

        Parameters
        ----------
        stage : str
            ``"read"``, ``"encode"``, ``"write"``, ``"check"`` or ``"hash"``.
        """
        now, now_cpu = time.perf_counter(), time.process_time()
        self.seconds[stage] += now - self._last
        self.cpu_seconds[stage] += now_cpu - self._last_cpu
        self._last, self._last_cpu = now, now_cpu


def block_stations_for(n_component: int, nt: int, station_chunk: int | None) -> int:
    """Choose how many stations to read at once.

    The budget gives the number of stations whose float32 samples fit in
    `BLOCK_BYTES`, as it is when this is called. For a chunked waveform the block is
    whole station chunks, so no chunk is read twice, unless that would exceed the
    budget by more than a quarter.

    Parameters
    ----------
    n_component : int
        The number of components.
    nt : int
        The number of samples in a trace.
    station_chunk : int or None
        The waveform's chunk along the station axis, or None if it is contiguous.

    Returns
    -------
    int
        The number of stations in a block, at least 1.
    """
    stations = max(1, BLOCK_BYTES // (4 * n_component * nt))
    if station_chunk is None:
        return stations
    block = max(1, round(stations / station_chunk)) * station_chunk
    return block if block <= stations + stations // 4 else stations


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


def _station_blocks(
    shape: tuple[int, ...],
    chunks: Sequence[int] | None,
    block_stations: int | None = None,
) -> list[tuple[int, int]]:
    """Split a waveform's stations into the blocks it is read in.

    Parameters
    ----------
    shape : tuple of int
        The waveform's shape: (component, station, time).
    chunks : sequence of int or None
        The waveform's chunks in its file, or None if it is contiguous.
    block_stations : int, optional
        The most stations in a block. By default a block holds about `BLOCK_BYTES` of
        the waveform, in whole chunks of stations if it is chunked (see
        `block_stations_for`).

    Returns
    -------
    list[tuple[int, int]]
        For each block in order, its first station and the station after its last.

    Raises
    ------
    ValueError
        If `block_stations` is less than 1.
    """
    n_component, n_station, nt = shape
    if block_stations is None:
        station_chunk = None if chunks is None else int(chunks[1])
        block_stations = block_stations_for(n_component, nt, station_chunk)
    return _blocks(n_station, block_stations)


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


def _json_value(value: object, name: str, typed: bool) -> object:
    """Make an attribute's value one that strict JSON holds.

    A numpy value becomes plain Python, an array lists. Each float that isn't finite,
    on its own or in a list, becomes its name: ``"NaN"``, ``"Infinity"`` or
    ``"-Infinity"``.

    Parameters
    ----------
    value : object
        The attribute's value.
    name : str
        The attribute's name, for the message.
    typed : bool
        Whether the attribute has a numpy type, which brings a number back from its
        name when the file is read.

    Returns
    -------
    object
        The value as None, a string, a number, or lists of them.

    Raises
    ------
    NotStandardLayout
        If the value is of a type JSON can't hold, as bytes and a complex number are,
        or a float isn't finite and the attribute has no numpy type, so its name would
        come back as a string.
    """
    if isinstance(value, np.ndarray | np.generic):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [_json_value(v, name, typed) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        if not typed:
            raise NotStandardLayout(
                f"attribute {name!r} holds {value}, which strict JSON can't hold and "
                "which has no numpy type to come back as"
            )
        # JSON's own names for them: NaN, Infinity and -Infinity.
        return json.dumps(value)
    if value is None or isinstance(value, str | int | float):
        return value
    raise NotStandardLayout(
        f"an attribute of type {type(value).__name__} can't go in JSON"
    )


def _attrs_json(attrs: dict) -> tuple[str, dict[str, str]]:
    """Write the waveform's attributes as strict JSON, and name their numpy types.

    A float that isn't finite, on its own or in a list, is the string ``"NaN"``,
    ``"Infinity"`` or ``"-Infinity"``, as in ``waveform_encoding``. Its type, which
    ``waveform_attr_types`` records (see `_attr_types`), makes a number of it again
    when the file is read.

    Parameters
    ----------
    attrs : dict
        The waveform's attributes.

    Returns
    -------
    text : str
        A JSON object of the attributes.
    types : dict[str, str]
        The dtype of each numpy-typed attribute (see `_attr_types`).

    Raises
    ------
    NotStandardLayout
        If an attribute's type can't be kept (see `_attr_types`), or its value can't go
        in JSON (see `_json_value`).
    """
    types = _attr_types(attrs)
    out = {
        name: _json_value(value, name, name in types) for name, value in attrs.items()
    }
    return json.dumps(out, allow_nan=False), types


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
    block: np.ndarray, fills: list[float], bits: int, src: Path
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
    src : Path
        The file the block is read from, for the message.

    Raises
    ------
    NotStandardLayout
        If a fill value lies within one quantisation step of a trace's range.
    """
    if not fills:
        return
    lowest, highest = block.min(axis=2), block.max(axis=2)
    low, high = lowest.astype(np.float64), highest.astype(np.float64)
    # A trace's peak is the larger of -lowest and highest, with no copy of the block.
    step = np.maximum(-lowest, highest) / full_scale(bits)
    for fill in fills:
        if np.any((low - step <= fill) & (fill <= high + step)):
            raise NotStandardLayout(
                f"{src} has the finite fill value {fill} inside the data's range, "
                "which can't survive lossy compression",
                src,
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

    Raises
    ------
    NotStandardLayout
        If a value is of a type JSON can't hold (see `_json_value`).
    """
    out: dict[str, object] = {"_FillValue": None}
    for name in _FILL_NAMES:
        if name in encoding:
            # Read back as the waveform's type, which brings a number back from its name.
            out[name] = _json_value(encoding[name], name, typed=True)
    return json.dumps(out, allow_nan=False)


def _comparable_fills(encoding: dict) -> dict:
    """Give the waveform's ``_FillValue`` and ``missing_value`` in a form to compare.

    Parameters
    ----------
    encoding : dict
        The waveform's netCDF encoding.

    Returns
    -------
    dict
        Each of them as `_encoding_json` writes it, read back: a NaN is the string
        ``"NaN"``, so two NaNs compare equal.

    Raises
    ------
    NotStandardLayout
        If a value is of a type JSON can't hold (see `_json_value`).
    """
    return json.loads(_encoding_json(encoding))


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


def _refuse_unkeepable_fill_values(encoding: dict, src: Path) -> None:
    """Refuse a waveform whose ``_FillValue`` and ``missing_value`` couldn't be kept.

    Each has to be one value. If both are there, they have to be the same, a NaN the
    same as a NaN, as the check compares them (see `_comparable_fills`): xarray can't write a dataset whose two
    differ, so the file would compress and verify, but the dataset it decompresses to
    couldn't be written to a file.

    Parameters
    ----------
    encoding : dict
        The waveform's netCDF encoding, as xarray read it.
    src : Path
        The file, for the message.

    Raises
    ------
    NotStandardLayout
        If either holds more values than one, or none, or both are there and differ.
    """
    given = {
        name: encoding[name] for name in _FILL_NAMES if encoding.get(name) is not None
    }
    for name, value in given.items():
        if np.ndim(value) != 0:
            raise NotStandardLayout(
                f"{src} has a {name} of {np.size(value)} values, not one, which a "
                "compressed file can't keep",
                src,
            )
    if len(given) == 2:
        fills = _comparable_fills(encoding)
        if fills["_FillValue"] != fills["missing_value"]:
            raise NotStandardLayout(
                f"{src} has a _FillValue of {fills['_FillValue']} and a missing_value "
                f"of {fills['missing_value']}, which differ, so the file it "
                "decompresses to couldn't be written",
                src,
            )


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


def _in_block(stations: np.ndarray, s0: int, s1: int) -> tuple[slice, np.ndarray]:
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
    here : slice
        The part of `stations` from `s0` up to, but not including, `s1`.
    local : np.ndarray
        Those stations, counted from `s0`.
    """
    k0, k1 = np.searchsorted(stations, (s0, s1))
    here = slice(int(k0), int(k1))
    return here, stations[here] - s0


# What it means for a file that isn't HDF5, by the refusal made of it.
_NOT_HDF5: dict[type[NotStandardLayout] | type[NotCompressed], str] = {
    NotStandardLayout: "isn't netCDF in the standard layout",
    NotCompressed: "is not compressed",
}


def _refuse_unless_hdf5(
    path: Path, refusal: type[NotStandardLayout] | type[NotCompressed]
) -> None:
    """Refuse a file that is there but isn't HDF5: it isn't the kind of file asked for.

    A file that isn't there is left to fail as it would, and so is one that can't be
    read, whose check raises `PermissionError`, and an HDF5 file that was cut short,
    which keeps its signature and fails when it is opened. Those aren't refusals: the
    file may be the right kind, and the run went wrong.

    Parameters
    ----------
    path : Path
        The file.
    refusal : type[NotStandardLayout] or type[NotCompressed]
        What to raise. Its message ends with what that means for the file: that it
        isn't netCDF in the standard layout, or is not compressed.

    Raises
    ------
    NotStandardLayout or NotCompressed
        `refusal`, if `path` is a regular file that `h5py.is_hdf5` says isn't HDF5.
    """
    if path.is_file() and not h5py.is_hdf5(path):
        raise refusal(f"{path} isn't an HDF5 file, so it {_NOT_HDF5[refusal]}", path)


def _refuse_subgroups(path: Path) -> None:
    """Refuse an HDF5 file that holds a group besides its root.

    A compressed waveform's group means the file is compressed already. Any other is
    refused because xarray reads only the root group, so a compressed file would go
    without it.

    Parameters
    ----------
    path : Path
        The file.

    Raises
    ------
    AlreadyCompressed
        If the root group holds a compressed waveform's group (see `is_compressed`).
    HasGroups
        If the root group holds any other group.
    """
    with h5py.File(path, "r") as f:
        groups = [name for name, item in f.items() if isinstance(item, h5py.Group)]
    if GROUP in groups:
        raise AlreadyCompressed(f"{path} is compressed already", path)
    if groups:
        raise HasGroups(
            f"{path} holds HDF5 groups besides its root ({', '.join(groups)}), "
            "which a compressed file would go without",
            path,
        )


def _check_dataset(src: Path, ds: xr.Dataset) -> tuple[np.ndarray, dict[str, str]]:
    """Make `check_compressible`'s refusals of the dataset, and find its real stations.

    Parameters
    ----------
    src : Path
        The file the dataset is opened from, for the messages.
    ds : xr.Dataset
        The dataset, open. Its waveform is not read.

    Returns
    -------
    exact : np.ndarray
        The real stations (see `_real_stations`).
    kept : dict[str, str]
        The group's attributes that keep the waveform's own, in the group's order:
        ``waveform_attrs``, its attributes as JSON; ``waveform_encoding``, its
        ``_FillValue`` and ``missing_value`` as JSON; and ``waveform_attr_types``, the
        dtype of each numpy-typed attribute, as JSON.
    """
    check_standard(ds)
    exact = _real_stations(ds)
    waveform = ds["waveform"]
    n_station = waveform.shape[1]
    if n_station == 0:
        raise NotStandardLayout(
            f"{src} has no stations, so there is nothing to compress", src
        )
    # Each refuses an attribute that the compressed file can't keep.
    attrs_json, attr_types = _attrs_json(waveform.attrs)
    _refuse_unkeepable_fill_values(waveform.encoding, src)
    encoding_json = _encoding_json(waveform.encoding)
    if exact.size == n_station:
        raise AllStationsReal(
            f"all {n_station} stations of {src} are real, so it isn't compressed", src
        )
    kept = {
        "waveform_attrs": attrs_json,
        "waveform_encoding": encoding_json,
        "waveform_attr_types": json.dumps(attr_types),
    }
    return exact, kept


@contextmanager
def _compressible(
    src: Path, dst: Path, overwrite: bool
) -> Generator[tuple[xr.Dataset, np.ndarray, dict[str, str]]]:
    """Make `check_compressible`'s refusals, in its order, and keep `src` open.

    Parameters
    ----------
    src : Path
        The netCDF file to compress.
    dst : Path
        Where the compressed file is to go.
    overwrite : bool
        Whether a file already at `dst` may be replaced.

    Yields
    ------
    ds : xr.Dataset
        The dataset in `src`, open for the block. Its waveform has not been read.
    exact : np.ndarray
        Its real stations (see `_real_stations`).
    kept : dict[str, str]
        The group's attributes made from the waveform's attributes and encoding (see
        `_check_dataset`).
    """
    check_destination(dst, src, overwrite=overwrite)
    _refuse_unless_hdf5(src, NotStandardLayout)
    _refuse_subgroups(src)
    with xr.open_dataset(src, engine="h5netcdf") as ds:
        exact, kept = _check_dataset(src, ds)
        yield ds, exact, kept


def check_compressible(
    src: str | os.PathLike[str], dst: str | os.PathLike[str], overwrite: bool = False
) -> None:
    """Refuse a file that can't be compressed as it is, before any sample is read.

    These are the refusals that need no samples, made in this order: `dst` is `src`, a
    file is at `dst` already, and a failure, there is no folder to write `dst` in (see
    `waveform_compression_tools.destinations.check_destination`); then `src` isn't an
    HDF5 file, it is compressed already, it holds an HDF5 group, it isn't in the
    standard layout, and every station is real. Only metadata and station names are
    read, so a refused file costs no full read. A refusal of `src` names it as its
    ``filename``. `compress_file` makes these first. The refusals that need the
    samples, of a NaN or an infinity and of a fill value inside a trace's range, come
    during its pass.

    Parameters
    ----------
    src : str or os.PathLike
        The netCDF file to compress.
    dst : str or os.PathLike
        Where the compressed file is to go.
    overwrite : bool, default False
        Whether a file already at `dst` may be replaced.

    Raises
    ------
    SameFile
        If `dst`, or `dst` with ``.partial`` added, is `src`, by its name, through a
        link, or as a hard link to it, whatever `overwrite` says (see
        `waveform_compression_tools.destinations`).
    OutputExists
        If something is at `dst` already and `overwrite` is false.
    AlreadyCompressed
        If `src` holds a compressed waveform already (see `is_compressed`).
    HasGroups
        If `src` holds an HDF5 group besides its root.
    NotStandardLayout
        If `src` is a file that isn't HDF5, is not in the standard layout (see
        `waveform_compression_tools.codec.check_standard`), has no ``station``
        coordinate along the station dimension, has a station whose name is bytes that
        aren't ASCII, has no stations, has a waveform attribute that can't be written
        as JSON or whose type can't be kept, as a complex or a compound one can't, or
        has a ``_FillValue`` or ``missing_value`` that isn't one value, that JSON can't
        hold, or both, and they differ.
    AllStationsReal
        If every station of `src` is real, so it stays exact as it is.
    FileNotFoundError
        If there is no folder to write `dst` in, naming `dst`, before `src` is read.
    """
    src = Path(src)
    with about(src), _compressible(src, Path(dst), overwrite):
        pass


def compress_file(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
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
    copied exactly, a block at a time. The compressed file is written under a partial
    name, and renamed to `dst` when it is complete (see
    `waveform_compression_tools.destinations.writing`). If anything goes wrong, the
    partial file is deleted and the exception raised again, so nothing appears at `dst`,
    and a file already there stays as it was. The file is written through a
    `waveform_compression_tools.destinations.HDF5Output`, so a write that fails, as on
    a full disk, stops the run at the end of its block with its own error, naming `dst`.

    With `check`, the compressed file is checked as it is written, against what was
    read, so the original's samples are read only once:

    - During the pass, as soon as a block is written, the bytes of its streams are read
      back from the file, once, and decoded with the starts, sizes and gains recorded
      for them, and its exact copies are read back. The block has to pass
      `verify_file`'s rules against the block still in memory: each trace within one
      step of its own gain, and within the bound the setting promises, of its peak; a
      trace of zeros as exact zeros; and each exact copy bit for bit. A CRC-32 is taken
      of each stream, from the bytes decoded, and of each exact copy, from the bytes
      compared.
    - When the pass is done, the closed partial file is reopened read-only. Its group
      has to hold just what the pass wrote: the datasets and attributes written, each
      with the value written, and every stream and every exact copy with the CRC-32 it
      had when it was judged, so the file holds what was judged. The file also has to
      agree with the original in everything but the waveform's samples, as
      `verify_file` compares them: everything besides the waveform identical, the
      waveform's attributes and encoding as a reader gets them back, and exact copies
      of the real stations and of no others.

    Then the closed partial file is hashed (see
    `waveform_compression_tools.content_hash`), with or without the check, and only
    then is it renamed to `dst`, so a file that can't be hashed leaves nothing there. A
    file that passes the check passes `verify_file`, unless it was changed after it was
    judged in a way that its CRC-32s miss.

    Parameters
    ----------
    src : str or os.PathLike
        The netCDF file to compress, in the standard layout (see
        `waveform_compression_tools.codec.check_standard`), with a ``station``
        coordinate that names its stations, and no HDF5 group besides its root.
    dst : str or os.PathLike
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
        setting, the largest error the check found, the compressed file's content
        hash, and the wall time spent reading, encoding, writing, checking and hashing,
        and the CPU time of the first four.

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
    FileNotFoundError
        As `check_compressible` raises it, before `src` is read.
    OSError
        If `src` can't be read, naming it, or the compressed file can't be written, as
        on a full disk, or read back to be hashed, naming `dst`.
    pyflac.EncoderInitException
        If libFLAC can't start a trace's stream.
    pyflac.EncoderProcessException
        If libFLAC fails to encode a trace, or to finish its stream.
    """
    full_scale(bits)  # Refuses a setting outside 8 to 32 before the file is opened.
    src, dst = Path(src), Path(dst)
    stopwatch = _Stopwatch()
    with (
        about(src),
        _compressible(src, dst, overwrite) as (ds, exact, kept),
        writing(dst) as partial,
    ):
        waveform = ds["waveform"]
        n_component, n_station, nt = waveform.shape
        chunks = waveform.encoding.get("chunksizes")
        blocks = _station_blocks(waveform.shape, chunks, block_stations)
        attrs: dict[str, object] = {
            "format_version": _FORMAT_VERSION,
            "dims": list(STANDARD_DIMS),
            "dtype": "float32",
            **kept,
            "bits": bits,
            "encoder": ENCODER,
            "exact_rule": REAL_STATION_RULE,
        }
        # Each is one value, if it is there (see check_compressible).
        fills = [
            float(v)
            for v in map(waveform.encoding.get, _FILL_NAMES)
            if v is not None and np.isfinite(v)
        ]
        # An OSError as src is read names src, and one as the file is written, dst.
        # Everything but the waveform is small, and is read once, here.
        with naming(src):
            rest = ds.drop_vars("waveform").load()
        stopwatch.lap("read")
        # The file is written through out, which never tells HDF5 that a write failed.
        with HDF5Output(partial) as out:
            # A dimension that only the waveform had, as time is in a file without a
            # time coordinate, is gone, so it can't be declared unlimited any more.
            unlimited = ds.encoding.get("unlimited_dims", ())
            rest.to_netcdf(
                out,
                engine="h5netcdf",
                unlimited_dims=[name for name in unlimited if name in rest.dims],
            )
            out.raise_pending()
            with h5py.File(out, "r+") as f:
                group, written = _new_group(
                    f, waveform, exact, bits, kept["waveform_encoding"]
                )
                stream_crcs = np.zeros((n_component, n_station), np.uint32)
                exact_crcs = np.zeros((n_component, exact.size), np.uint32)
                worst = 0.0
                stopwatch.lap("write")
                for s0, s1 in blocks:
                    with naming(src):
                        block = waveform[:, s0:s1, :].values
                    stopwatch.lap("read")
                    _refuse_fill_in_range(block, fills, bits, src)
                    encoded = encode_block(block, bits, level)
                    stopwatch.lap("encode")
                    _append_streams(written, encoded, s0, s1)
                    # The streams are in the file now, and the check reads them back.
                    del encoded
                    here, local = _in_block(exact, s0, s1)
                    written.exact_waveform[:, here] = block[:, local]
                    # A write that failed, or a Ctrl-C, stops the run here, before what
                    # HDF5 writes from then on, held in memory, adds up.
                    out.raise_pending()
                    stopwatch.lap("write")
                    if check:
                        error, stream_crcs[:, s0:s1], exact_crcs[:, here] = (
                            _check_block(written, block, s0, s1, dst)
                        )
                        worst = max(worst, error)
                        stopwatch.lap("check")
                    # pyflac's encoders keep their output in reference cycles, which
                    # would otherwise wait for the next collection and add up. They are
                    # all young, so a young collection frees them, without walking the
                    # whole heap.
                    gc.collect(1)
                datasets = {
                    "stream_starts": written.starts,
                    "stream_bytes": written.nbytes,
                    "stream_gains": written.gains,
                    "stream_offsets": written.offsets,
                    "exact_stations": exact,
                }
                for name, data in datasets.items():
                    group.create_dataset(name, data=data)
                group["stream_starts"].attrs["stream_size"] = nt
                group.attrs.update(attrs)
        stopwatch.lap("write")
        if check:
            record = _Record(datasets, attrs, nt, stream_crcs, exact_crcs, blocks)
            _check_reopened(partial, rest, waveform, record, dst)
            stopwatch.lap("check")
        # Before the rename, so that a file that can't be read back isn't put there.
        output_hash = content_hash(partial)
        stopwatch.lap("hash")
    stopwatch.lap("write")  # The rename.
    return CompressResult(
        bytes_in=src.stat().st_size,
        bytes_out=dst.stat().st_size,
        n_traces=n_component * n_station,
        n_exact=int(exact.size),
        bits=bits,
        max_relative_error=worst if check else None,
        output_hash=output_hash,
        read_s=stopwatch.seconds["read"],
        encode_s=stopwatch.seconds["encode"],
        write_s=stopwatch.seconds["write"],
        check_s=stopwatch.seconds["check"],
        hash_s=stopwatch.seconds["hash"],
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
        The waveform's ``_FillValue`` and ``missing_value`` to put back.
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

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the waveform the streams decode to.

        Returns
        -------
        tuple[int, int, int]
            (component, station, nt).
        """
        n_component, n_station = self.starts.shape
        return n_component, n_station, self.nt

    def read(self, s0: int, s1: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the bytes of a block of stations' streams from the file.

        Parameters
        ----------
        s0 : int
            The block's first station.
        s1 : int
            The station after its last, greater than `s0`.

        Returns
        -------
        data : np.ndarray
            The bytes the block's streams span, as uint8.
        starts : np.ndarray
            Where each of the block's streams starts in `data`, of shape
            (component, s1 - s0).
        """
        starts, nbytes = self.starts[:, s0:s1], self.nbytes[:, s0:s1]
        first, end = int(starts.min()), int((starts + nbytes).max())
        return self.compressed[first:end], starts - first

    def decode(
        self,
        s0: int,
        s1: int,
        read: tuple[np.ndarray, np.ndarray] | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode a block of stations from their FLAC streams alone.

        Parameters
        ----------
        s0 : int
            The block's first station.
        s1 : int
            The station after its last, greater than `s0`.
        read : tuple[np.ndarray, np.ndarray], optional
            The block's streams as `read` gave them, if they have been read already.
            By default they are read from the file.
        out : np.ndarray, optional
            A float32 array of shape (component, s1 - s0, nt) to decode into. By
            default a new one is made.

        Returns
        -------
        np.ndarray
            The block, as a float32 array of shape (component, s1 - s0, nt): `out`, if
            it is given.

        Raises
        ------
        ValueError
            If a stream is damaged, or decodes to the wrong number of samples (see
            `waveform_compression_tools.codec.decode_block`).
        """
        data, starts = self.read(s0, s1) if read is None else read
        return decode_block(
            data,
            starts,
            self.nbytes[:, s0:s1],
            self.gains[:, s0:s1],
            self.offsets[:, s0:s1],
            self.nt,
            first_station=s0,
            out=out,
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
        here, stations = _in_block(self.exact_stations, s0, s1)
        block[:, stations] = self.exact_waveform[:, here]
        return stations


def _new_group(
    f: h5py.File,
    waveform: xr.DataArray,
    exact: np.ndarray,
    bits: int,
    waveform_encoding: str,
) -> tuple[h5py.Group, _Streams]:
    """Make the compressed waveform's group, with its two datasets that a pass fills.

    ``compressed`` is made empty, to grow a block at a time, and then
    ``exact_waveform``, in that order. The arrays of where each stream starts, its
    length, gain and offset are made in memory, as zeros, to be written once the pass
    is done.

    Parameters
    ----------
    f : h5py.File
        The compressed file, open for writing.
    waveform : xr.DataArray
        The original's waveform. Its samples aren't read.
    exact : np.ndarray
        The real stations, in increasing order.
    bits : int
        The quantisation setting.
    waveform_encoding : str
        The waveform's ``_FillValue`` and ``missing_value``, as `_encoding_json` wrote
        them.

    Returns
    -------
    group : h5py.Group
        The group.
    written : _Streams
        The file as a reader sees it, each block as soon as it is written.
    """
    n_component, n_station, nt = waveform.shape
    group = f.create_group(GROUP)
    chunk = min(_CHUNK_BYTES, max(1, 4 * n_component * n_station * nt))
    compressed = group.create_dataset(
        "compressed", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=(chunk,)
    )
    exact_waveform = group.create_dataset(
        "exact_waveform", shape=(n_component, exact.size, nt), dtype=np.float32
    )
    traces = (n_component, n_station)
    written = _Streams(
        compressed=compressed,
        starts=np.zeros(traces, np.int64),
        nbytes=np.zeros(traces, np.int64),
        gains=np.zeros(traces),
        offsets=np.zeros(traces),
        nt=nt,
        bits=bits,
        waveform_attrs=dict(waveform.attrs),
        encoding=_encoding_back(waveform_encoding),
        exact_stations=exact,
        exact_waveform=exact_waveform,
    )
    return group, written


def _append_streams(written: _Streams, encoded: EncodedBlock, s0: int, s1: int) -> None:
    """Write a block's streams after those already written, and record where they are.

    Parameters
    ----------
    written : _Streams
        The file's streams, as written so far. Its starts, lengths and gains for the
        block's stations are set.
    encoded : EncodedBlock
        The block's streams, which go in the order `encode_block` gives them.
    s0 : int
        The block's first station.
    s1 : int
        The station after its last.
    """
    end = written.compressed.shape[0]
    sizes = encoded.nbytes.ravel()
    block_starts = end + np.cumsum(sizes) - sizes
    written.starts[:, s0:s1] = block_starts.reshape(encoded.nbytes.shape)
    written.nbytes[:, s0:s1] = encoded.nbytes
    written.gains[:, s0:s1] = encoded.gains
    written.compressed.resize((end + int(sizes.sum()),))
    written.compressed[end:] = np.concatenate(
        [np.frombuffer(stream, np.uint8) for stream in encoded.streams]
    )


def _compressed_group(f: h5py.File) -> h5py.Group | None:
    """Return a file's compressed waveform, if it holds one.

    Parameters
    ----------
    f : h5py.File
        The file, open.

    Returns
    -------
    h5py.Group or None
        The group ``_flac_compressed_waveform``, or None if the file has no group by
        that name, as when something else has the name.
    """
    group = f.get(GROUP)
    return group if isinstance(group, h5py.Group) else None


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
    NotCompressed
        If the file has no group ``_flac_compressed_waveform``.
    UnknownFormat
        If the group is in a format version other than 1, or its ``bits`` is outside 8
        to 32, which no file can have.
    ValueError
        If the group's ``stream_bytes``, ``stream_gains`` or ``stream_offsets`` differs
        in shape from its ``stream_starts``, its ``exact_stations`` are not stations of
        the file in increasing order, its ``exact_waveform`` is not float32 with each of
        their traces, or its ``waveform_attr_types`` names an attribute that its
        ``waveform_attrs`` lacks.
    """
    group = _compressed_group(f)
    if group is None:
        raise NotCompressed(
            f"{path} has no group {GROUP}, so it is not compressed", path
        )
    version = int(group.attrs["format_version"])
    if version != _FORMAT_VERSION:
        raise UnknownFormat(
            f"{path} is in format version {version}, "
            f"and only version {_FORMAT_VERSION} can be read",
            path,
        )
    bits = int(group.attrs["bits"])
    try:
        full_scale(bits)
    except ValueError as e:
        raise UnknownFormat(
            f"{path} has a setting that can't be read: {e}", path
        ) from e
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
    for name, dtype in json.loads(group.attrs["waveform_attr_types"]).items():
        if name not in waveform_attrs:
            raise ValueError(
                f"{path}: waveform_attr_types names the attribute {name!r}, which "
                "waveform_attrs lacks"
            )
        value = waveform_attrs[name]
        waveform_attrs[name] = (
            np.asarray(value, dtype)
            if isinstance(value, list)
            else np.dtype(dtype).type(value)
        )
    return _Streams(
        compressed=group["compressed"],
        starts=starts,
        nbytes=group["stream_bytes"][...],
        gains=group["stream_gains"][...],
        offsets=group["stream_offsets"][...],
        nt=nt,
        bits=bits,
        waveform_attrs=waveform_attrs,
        encoding=_encoding_back(group.attrs["waveform_encoding"]),
        exact_stations=exact_stations,
        exact_waveform=exact_waveform,
    )


def _same_besides_samples(
    original_rest: xr.Dataset,
    waveform: xr.DataArray,
    real: np.ndarray,
    rest: xr.Dataset,
    streams: _Streams,
) -> bool:
    """Say whether a compressed file holds all of its original but the waveform's samples.

    This is `verify_file`'s comparison of everything but the samples, which the check in
    `compress_file` makes too. Everything besides the waveform, every other variable,
    coordinate and attribute, has to be identical. The waveform's attributes have to
    come back as a reader of the file gets them, each in the type it is written with,
    and so do its ``_FillValue`` and ``missing_value``. The file has to hold exact copies
    of the real stations and of no others. The waveform's samples aren't read.

    Parameters
    ----------
    original_rest : xr.Dataset
        The original, in the standard layout, without its waveform.
    waveform : xr.DataArray
        The original's waveform, for its attributes and encoding.
    real : np.ndarray
        The original's real stations, in increasing order.
    rest : xr.Dataset
        The compressed file as xarray opens it: all of it but the waveform.
    streams : _Streams
        The compressed file's streams, with the waveform's attributes and encoding as a
        reader gets them.

    Returns
    -------
    bool
        Whether the file agrees with the original in all of these.

    Raises
    ------
    NotStandardLayout
        If the original's ``_FillValue`` or ``missing_value`` is of a type JSON can't
        hold, so it can't be compared.
    """
    # `identical` compares attribute values and not their types, so the types and the
    # encoding are compared on their own.
    same_attrs = xr.Dataset(attrs=dict(waveform.attrs)).identical(
        xr.Dataset(attrs=streams.waveform_attrs)
    ) and _written_dtypes(waveform.attrs) == _written_dtypes(streams.waveform_attrs)
    same_encoding = _comparable_fills(waveform.encoding) == _comparable_fills(
        streams.encoding
    )
    return (
        original_rest.identical(rest)
        and same_attrs
        and same_encoding
        and np.array_equal(streams.exact_stations, real)
    )


def _compare_block(
    streams: _Streams,
    block: np.ndarray,
    decoded: np.ndarray,
    real: np.ndarray,
    s0: int,
    s1: int,
) -> tuple[float, bool]:
    """Judge a decoded block of stations by verify's rules, against the original's.

    Every trace's FLAC stream counts, a real station's too, since a reader that ignores
    the exact copies decodes it. Each trace's largest error has to be within the bound
    the file's setting promises, ``1 / full_scale(bits)`` of its peak with a relative
    slack of 1e-9, whatever its gain says; and every sample has to be within one step of
    its own gain (see `waveform_compression_tools.codec.within_one_step`), so a trace of
    zeros has to come back as exact zeros. Then, with the file's exact copies put back,
    every real station has to come back bit for bit.

    Parameters
    ----------
    streams : _Streams
        The compressed file's streams.
    block : np.ndarray
        The original's traces of the block's stations, of shape
        (component, s1 - s0, nt).
    decoded : np.ndarray
        The same stations as their streams decode, of the same shape. The exact copies
        are put back into it.
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
    """
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
    _, here = _in_block(real, s0, s1)
    passed = passed and np.array_equal(
        block[:, here].view(np.uint32), decoded[:, here].view(np.uint32)
    )
    return float(error.max()), passed


def _stream_crcs(
    data: np.ndarray, starts: np.ndarray, nbytes: np.ndarray
) -> np.ndarray:
    """Compute the CRC-32 of each of a block's streams.

    Parameters
    ----------
    data : np.ndarray
        The bytes the block's streams span, as uint8.
    starts : np.ndarray
        Where each stream starts in `data`, of shape (component, station).
    nbytes : np.ndarray
        The length of each in bytes, of the same shape.

    Returns
    -------
    np.ndarray
        Each stream's CRC-32, as uint32 of the same shape.
    """
    crcs = [
        zlib.crc32(data[start : start + size])
        for start, size in zip(starts.flat, nbytes.flat)
    ]
    return np.array(crcs, np.uint32).reshape(starts.shape)


def _trace_crcs(traces: np.ndarray) -> np.ndarray:
    """Compute the CRC-32 of each trace's bytes.

    Parameters
    ----------
    traces : np.ndarray
        Traces of shape (component, station, nt), in C order.

    Returns
    -------
    np.ndarray
        Each trace's CRC-32, as uint32 of shape (component, station).
    """
    crcs = [zlib.crc32(traces[c, s]) for c, s in np.ndindex(*traces.shape[:2])]
    return np.array(crcs, np.uint32).reshape(traces.shape[:2])


def _check_block(
    streams: _Streams, block: np.ndarray, s0: int, s1: int, path: Path
) -> tuple[float, np.ndarray, np.ndarray]:
    """Check a block of stations as soon as it is written, against the original's.

    The bytes of the block's streams are read back from the file once, and decoded with
    the starts, sizes and gains recorded for them, and the block's exact copies are read
    back. The block has to pass verify's rules against the original's traces (see
    `_compare_block`). An error while decoding counts as a failed check. The CRC-32s of
    what passed are returned, so that the closed file can be shown to hold it.

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
        Where the file is going, which an error's message names.

    Returns
    -------
    error : float
        The block's largest error as the file decompresses, as a fraction of a
        trace's peak.
    stream_crcs : np.ndarray
        The CRC-32 of each stream, of the very bytes decoded: uint32 of shape
        (component, s1 - s0).
    exact_crcs : np.ndarray
        The CRC-32 of each of the block's exact copies, of the bytes read back and
        compared: uint32 of shape (component, the block's real stations).

    Raises
    ------
    CheckFailed
        If the block fails, or a stream doesn't decode.
    """
    read = streams.read(s0, s1)
    try:
        decoded = streams.decode(s0, s1, read)
    except ValueError as e:
        raise CheckFailed(
            f"{path}: stations {s0} to {s1 - 1} don't decode ({e})", path
        ) from e
    real = streams.exact_stations
    error, passed = _compare_block(streams, block, decoded, real, s0, s1)
    if not passed:
        raise CheckFailed(
            f"{path}: stations {s0} to {s1 - 1} don't come back as they should at "
            f"{streams.bits} bits",
            path,
        )
    data, starts = read
    _, copies = _in_block(real, s0, s1)
    stream_crcs = _stream_crcs(data, starts, streams.nbytes[:, s0:s1])
    return error, stream_crcs, _trace_crcs(decoded[:, copies])


@dataclass(frozen=True)
class _Record:
    """What a pass of `compress_file` wrote to the group, and the CRC-32s of what it judged.

    Attributes
    ----------
    datasets : dict[str, np.ndarray]
        The datasets written once the pass is done, by name: ``stream_starts``,
        ``stream_bytes``, ``stream_gains``, ``stream_offsets`` and ``exact_stations``.
    attrs : dict[str, object]
        The group's attributes, by name.
    nt : int
        The number of samples in every trace, which ``stream_size`` records.
    stream_crcs : np.ndarray
        The CRC-32 of each trace's stream, of the bytes decoded during the pass:
        uint32 of shape (component, station).
    exact_crcs : np.ndarray
        The CRC-32 of each exact copy, of the bytes read back and compared during the
        pass: uint32 of shape (component, real station).
    blocks : list[tuple[int, int]]
        The pass's blocks, each its first station and the station after its last.
    """

    datasets: dict[str, np.ndarray]
    attrs: dict[str, object]
    nt: int
    stream_crcs: np.ndarray
    exact_crcs: np.ndarray
    blocks: list[tuple[int, int]]


def _check_reopened(
    partial: Path,
    original_rest: xr.Dataset,
    waveform: xr.DataArray,
    record: _Record,
    path: Path,
) -> None:
    """Reopen a closed compressed file, and check it against its pass and its original.

    The file is reopened read-only, and compared with what the pass recorded:

    - the group holds just the datasets and the attributes written, each attribute with
      the value written, the setting's ``bits`` included;
    - ``stream_starts``, ``stream_bytes``, ``stream_gains``, ``stream_offsets`` and
      ``exact_stations`` hold what was written, in its type, and ``stream_size`` is the
      number of samples in a trace;
    - the streams and the exact copies are of their type and shape;
    - read back a block at a time, every stream and every exact copy has the CRC-32 of
      the bytes judged during the pass, so the file holds what was judged.

    Then it is compared with the original in everything but the waveform's samples, as
    `verify_file` compares them (see `_same_besides_samples`). The original isn't read
    again.

    Parameters
    ----------
    partial : Path
        The compressed file, closed since it was written, under its partial name.
    original_rest : xr.Dataset
        The original without its waveform, as it was read.
    waveform : xr.DataArray
        The original's waveform, for its attributes and encoding.
    record : _Record
        What the pass wrote, and the CRC-32s of what it judged.
    path : Path
        Where the file is going, which an error's message names.

    Raises
    ------
    CheckFailed
        If the file differs from the record or from the original in any of these.
    """
    datasets, nt = record.datasets, record.nt
    size = int(datasets["stream_bytes"].sum())
    with h5py.File(partial, "r") as f:
        group = _compressed_group(f)
        if group is None or set(group) != {*datasets, "compressed", "exact_waveform"}:
            raise CheckFailed(
                f"{path} doesn't hold the datasets that were written", path
            )
        if set(group.attrs) != set(record.attrs) or not all(
            np.array_equal(group.attrs[name], value)
            for name, value in record.attrs.items()
        ):
            raise CheckFailed(
                f"{path}: the attributes of {GROUP} differ from those written", path
            )
        for name, data in datasets.items():
            if group[name].dtype != data.dtype or not np.array_equal(
                group[name][...], data
            ):
                raise CheckFailed(f"{path}: {name} differs from what was written", path)
        compressed = group["compressed"]
        if (compressed.dtype, compressed.shape) != (np.uint8, (size,)):
            raise CheckFailed(
                f"{path}: compressed is {compressed.dtype} of shape "
                f"{compressed.shape}, not uint8 of shape {(size,)}",
                path,
            )
        if dict(group["stream_starts"].attrs) != {"stream_size": nt}:
            raise CheckFailed(
                f"{path}: stream_size isn't {nt}, the samples in a trace", path
            )
        # As written, so it opens as a reader opens it. That checks exact_waveform's
        # type and shape.
        try:
            streams = _open_streams(f, path)
        except ValueError as e:
            raise CheckFailed(str(e), path) from e
        for s0, s1 in record.blocks:
            data, starts = streams.read(s0, s1)
            crcs = _stream_crcs(data, starts, streams.nbytes[:, s0:s1])
            if not np.array_equal(crcs, record.stream_crcs[:, s0:s1]):
                raise CheckFailed(
                    f"{path}: a stream of stations {s0} to {s1 - 1} differs from the "
                    "one checked",
                    path,
                )
            here, _ = _in_block(streams.exact_stations, s0, s1)
            crcs = _trace_crcs(streams.exact_waveform[:, here])
            if not np.array_equal(crcs, record.exact_crcs[:, here]):
                raise CheckFailed(
                    f"{path}: an exact copy of stations {s0} to {s1 - 1} differs from "
                    "the one checked",
                    path,
                )
        real = datasets["exact_stations"]
        with xr.open_dataset(partial, engine="h5netcdf") as rest:
            if not _same_besides_samples(original_rest, waveform, real, rest, streams):
                raise CheckFailed(
                    f"{path} differs from the original besides the waveform's samples",
                    path,
                )


def decompress_file(path: str | os.PathLike[str]) -> xr.Dataset:
    """Decompress a file written by `compress_file`.

    Parameters
    ----------
    path : str or os.PathLike
        The compressed file.

    Returns
    -------
    xr.Dataset
        The original dataset, loaded into memory: every variable, coordinate and
        attribute of the original, and its ``waveform`` as float32 with the
        dimensions (component, station, time) and its original attributes. Each
        attribute is in its own type again, and the waveform's encoding has its
        ``_FillValue`` and ``missing_value`` as the original had them, so the dataset
        is written to a file as the original was. The real stations' traces are the
        exact copies the file holds; every other sample is within one quantisation
        step of the original. The whole waveform is decoded, so it has to fit in
        memory.

    Raises
    ------
    NotCompressed
        If `path` is a file that isn't HDF5, or holds no compressed waveform.
    UnknownFormat
        If it is in a format version other than 1, or has a setting outside 8 to 32.
    ValueError
        If its arrays don't agree with each other, its ``waveform_attr_types`` names an
        attribute that its ``waveform_attrs`` lacks, or a stream is damaged or decodes
        to the wrong number of samples.
    """
    path = Path(path)
    _refuse_unless_hdf5(path, NotCompressed)
    with h5py.File(path, "r") as f:
        streams = _open_streams(f, path)
        waveform = np.empty(streams.shape, np.float32)
        for s0, s1 in _station_blocks(streams.shape, None):
            # Straight into place, so no block is held besides the waveform.
            streams.decode(s0, s1, out=waveform[:, s0:s1])
            streams.put_back_exact(waveform[:, s0:s1], s0, s1)
    ds = xr.load_dataset(path, engine="h5netcdf")
    ds["waveform"] = (STANDARD_DIMS, waveform, streams.waveform_attrs)
    ds["waveform"].encoding.update(streams.encoding)
    return ds


def decompress_to(
    src: str | os.PathLike[str], dst: str | os.PathLike[str], overwrite: bool = False
) -> None:
    """Decompress a file written by `compress_file` to a netCDF file.

    The destination and its folder are checked first (see
    `waveform_compression_tools.destinations.check_destination`). The dataset that
    `decompress_file` gives is then written by xarray with the h5netcdf engine to `dst`
    with ``.partial`` added, and renamed to `dst` when it is complete (see
    `waveform_compression_tools.destinations.write_netcdf`). If anything goes wrong,
    the partial file is deleted and the exception raised again; a file already at
    `dst` is then as it was. As in `decompress_file`, the whole waveform is decoded
    into memory.

    Parameters
    ----------
    src : str or os.PathLike
        The compressed file.
    dst : str or os.PathLike
        Where to write the netCDF file.
    overwrite : bool, default False
        Whether to replace a file already at `dst`. `src` itself is never replaced.

    Raises
    ------
    SameFile
        If `dst`, or `dst` with ``.partial`` added, is `src`, whatever `overwrite`
        says. Nothing is written.
    OutputExists
        If a file is at `dst` already and `overwrite` is false. Nothing is written.
    FileNotFoundError
        If there is no folder to write `dst` in, before `src` is read.
    NotCompressed, UnknownFormat
        As `decompress_file` raises them. Nothing is written.
    """
    check_destination(dst, src, overwrite=overwrite)
    write_netcdf(decompress_file(src), dst)


def verify_file(
    original: str | os.PathLike[str],
    compressed: str | os.PathLike[str],
    block_stations: int | None = None,
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
    attribute), have to be identical to the original's. So do the types the waveform's
    attributes come back in, and its ``_FillValue`` and ``missing_value``.

    Parameters
    ----------
    original : str or os.PathLike
        The netCDF file that was compressed, in the standard layout.
    compressed : str or os.PathLike
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
        If `original` is a file that isn't HDF5, is not in the standard layout, has no
        ``station`` coordinate, or has a station whose name is bytes that are not
        ASCII.
    NotCompressed
        If `compressed` is a file that isn't HDF5, or holds no compressed waveform.
    UnknownFormat
        If `compressed` is in a format version other than 1, or has a setting outside
        8 to 32.
    ValueError
        If the arrays of `compressed` don't agree with each other, if its
        ``waveform_attr_types`` names an attribute that its ``waveform_attrs`` lacks,
        if it decodes to a waveform of another shape than the original's, if a stream
        is damaged or decodes to the wrong number of samples, or if `block_stations` is
        less than 1.
    """
    start = time.perf_counter()
    original, compressed = Path(original), Path(compressed)
    _refuse_unless_hdf5(original, NotStandardLayout)
    _refuse_unless_hdf5(compressed, NotCompressed)
    worst = np.float64(0.0)
    with (
        xr.open_dataset(original, engine="h5netcdf") as ds,
        xr.open_dataset(compressed, engine="h5netcdf") as rest,
        h5py.File(compressed, "r") as f,
    ):
        # A refusal of the original's layout, stations or fill values names it.
        with about(original):
            check_standard(ds)
            real = _real_stations(ds)
        waveform = ds["waveform"]
        streams = _open_streams(f, compressed)
        if streams.shape != waveform.shape:
            raise ValueError(
                f"{compressed} decodes to a waveform of shape {streams.shape}, but the "
                f"original {original} holds one of shape {waveform.shape}"
            )
        with about(original):
            passed = _same_besides_samples(
                ds.drop_vars("waveform"), waveform, real, rest, streams
            )
        chunks = waveform.encoding.get("chunksizes")
        for s0, s1 in _station_blocks(waveform.shape, chunks, block_stations):
            block = waveform[:, s0:s1, :].values
            decoded = streams.decode(s0, s1)
            error, block_passed = _compare_block(streams, block, decoded, real, s0, s1)
            passed = passed and block_passed
            worst = np.maximum(worst, error)  # A NaN stays.
            # So that neither is kept while the next block is read and decoded.
            del block, decoded
            # Each stream's decoder is kept in a reference cycle, as pyflac's encoders
            # are, which would otherwise wait for the next collection and add up. They
            # are all young, so a young collection frees them, without walking the
            # whole heap.
            gc.collect(1)
    return VerifyResult(
        max_relative_error=float(worst),
        passed=passed,
        n_exact=int(real.size),
        verify_s=time.perf_counter() - start,
    )


def is_compressed(path: str | os.PathLike[str]) -> bool:
    """Say whether a file was written by `compress_file`.

    Parameters
    ----------
    path : str or os.PathLike
        The file.

    Returns
    -------
    bool
        True if `path` is an HDF5 file with the group ``_flac_compressed_waveform``,
        and False otherwise, as when it does not exist.
    """
    path = Path(path)
    if not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as f:
        return _compressed_group(f) is not None
