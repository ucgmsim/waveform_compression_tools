"""Open a compressed file as an xarray dataset, decoding only the traces that are read.

`open_dataset` gives the dataset that `waveform_compression_tools.files.decompress_file`
gives, but its waveform stays in the file until it is read. Selecting with ``isel`` or
``sel`` reads nothing; reading a selection's values decodes only its traces, each from
its FLAC stream, of which only its own bytes are read. A real station's trace is then
its exact copy, as in `decompress_file`. A selection in time decodes whole traces, and
then cuts them. A selection of points decodes every chosen component of every chosen
station, and then picks.

It is built on xarray's backend machinery, as xarray's own netCDF backends are: the
dataset is opened by ``xr.open_dataset`` with this package's engine (see
`waveform_compression_tools.engine`), the waveform is a `xarray.backends.BackendArray`,
which xarray indexes lazily, and the file is opened through xarray's
`xarray.backends.CachingFileManager`. Nothing needs dask.
"""

import os
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from xarray.backends import BackendArray, CachingFileManager
from xarray.core import indexing

from .destinations import naming
from .engine import FlacBackendEntrypoint
from .errors import NotCompressed, about
from .files import _open_streams, _original, _refuse_unless_hdf5, _Streams


class _OpenFile:
    """A compressed file, open, with where its streams are read from it.

    It is what a `_CompressedFile` opens through xarray's file manager, so where the
    streams are is read each time the file is opened, and checked as
    `waveform_compression_tools.files.decompress_file` checks it. Nothing read from the
    file is kept once it is closed.

    Parameters
    ----------
    path : Path
        The file, as an absolute path.
    named : Path
        The file as it was given, which an error's message names.
    mode : str
        How h5py opens the file, as xarray's file manager gives it: ``"r"``.

    Attributes
    ----------
    streams : _Streams
        The file's streams, with its datasets.

    Raises
    ------
    NotCompressed
        If the file holds no compressed waveform.
    UnknownFormat
        If it is in a format version other than 1, or has a setting outside 8 to 32.
    Damaged
        If it lacks a part of the format, or its parts don't agree with each other.
    """

    def __init__(
        self, path: Path, named: Path, mode: str
    ) -> None:  # numpydoc ignore=GL08
        self._file = h5py.File(path, mode)
        try:
            self.streams = _open_streams(self._file, named)
        except BaseException:
            self._file.close()
            raise

    def close(self) -> None:
        """Close the file."""
        self._file.close()


class _CompressedFile:
    """A compressed file, open for a dataset that decodes its waveform as it is read.

    The file is opened through a `xarray.backends.CachingFileManager`, as xarray's
    h5netcdf backend opens its files: xarray keeps only so many files open, and opens a
    file again when it is read after it was closed to make room for others. It is
    opened by its absolute path, so it is the same file whatever the working folder is
    by then. A file closed by `close` stays closed.

    A deep copy of the dataset shares the file, and is closed with it: it reads through
    this very object, which a deep copy doesn't copy. A pickled copy is a copy of its
    own, as xarray's pickles are: it opens the file itself, in whatever process it is
    read, and either one can still be read once the other is closed.

    Parameters
    ----------
    path : Path
        The compressed file.

    Attributes
    ----------
    path : Path
        The compressed file, as it was given.
    """

    def __init__(self, path: Path) -> None:  # numpydoc ignore=GL08
        self.path = path
        # A mode is given, as xarray's engines give one. xarray marks a manager given
        # none with an object of its own, and a pickled or deep copy of the manager holds
        # a copy of that object, which xarray takes for a mode to pass to `_OpenFile`.
        self._manager = CachingFileManager(_OpenFile, path.absolute(), path, mode="r")
        self._closed = False

    def __deepcopy__(self, memo: dict[int, object]) -> "_CompressedFile":
        """Give a deep copy this very object, so that it shares the file.

        xarray's deep copy shares an engine's array, but not once a transpose has
        dropped the wrapper that makes it do so.

        Parameters
        ----------
        memo : dict
            What the deep copy has copied so far.

        Returns
        -------
        _CompressedFile
            This object, not a copy.
        """
        return self

    @contextmanager
    def reading(self) -> Generator[_Streams]:
        """Have the file open while its streams are read, and name it in what goes wrong.

        A read may come long after the file was opened, with other files open, so a
        damaged stream names the file it is in, and so does an error of h5py's, which
        names none.

        Yields
        ------
        _Streams
            The streams, with the file's datasets.

        Raises
        ------
        ValueError
            If the file has been closed.
        Damaged
            If the file fails as `_OpenFile` does as it is opened again, or a stream read
            is damaged. It names the file.
        OSError
            If the file can't be read, as when it is cut short, naming it.
        """
        if self._closed:
            raise ValueError(f"{self.path} is closed, so its waveform can't be read")
        with (
            naming(self.path),
            about(self.path),
            self._manager.acquire_context() as opened,
        ):
            yield opened.streams

    def close(self) -> None:
        """Close the file, for good."""
        self._closed = True
        self._manager.close()


def _chosen(key: int | slice | np.ndarray, n: int) -> np.ndarray:
    """Give the indices that an outer key chooses along a dimension.

    Parameters
    ----------
    key : int, slice or np.ndarray
        The key: an index, a slice or an array of indices.
    n : int
        The dimension's length.

    Returns
    -------
    np.ndarray
        The indices chosen, as an array of one dimension. It is no longer than the key
        chooses.
    """
    return np.arange(*key.indices(n)) if isinstance(key, slice) else np.atleast_1d(key)


def _read_traces(
    streams: _Streams,
    components: np.ndarray,
    stations: np.ndarray,
    times: slice | np.ndarray,
) -> np.ndarray:
    """Read some traces as `decompress_file` gives them, reading only what they need.

    The traces are every chosen component of every chosen station, each cut to the
    chosen samples. A station's chosen traces are decompressed together (see
    `_Streams.decompressed`), and cut as soon as they are, so memory holds the cut
    traces and one station's whole ones. When no sample is chosen, nothing is decoded.

    Parameters
    ----------
    streams : _Streams
        The file's streams, with its datasets open.
    components : np.ndarray
        The components, as indices.
    stations : np.ndarray
        The stations, as indices.
    times : slice or np.ndarray
        The samples of each trace, as a slice, which cuts a trace without copying it,
        or as indices.

    Returns
    -------
    np.ndarray
        The traces, as float32 of shape (components, stations, times).

    Raises
    ------
    Damaged
        If a stream is damaged, or decodes to the wrong number of samples (see
        `waveform_compression_tools.codec.decode_block`).
    """
    n_time = len(range(streams.nt)[times]) if isinstance(times, slice) else times.size
    traces = np.empty((components.size, stations.size, n_time), np.float32)
    if traces.size == 0:
        return traces
    for j, s in enumerate(stations.tolist()):
        traces[:, j] = streams.decompressed(s, s + 1, components)[:, 0, times]
    return traces


class _LazyWaveform(BackendArray):
    """A compressed file's waveform, decoded only where it is read.

    xarray indexes it lazily, and reads it with outer indexing: an integer, a slice or
    an array of indices along each dimension, so a read is all the chosen components of
    all the chosen stations. A selection of points, pairs of a component and a station,
    reads every component of the pairs at every station of the pairs.

    Parameters
    ----------
    file : _CompressedFile
        The file the waveform is in.
    shape : tuple[int, int, int]
        The waveform's shape: (component, station, time).

    Attributes
    ----------
    shape : tuple[int, int, int]
        The waveform's shape.
    dtype : np.dtype
        float32.
    """

    def __init__(
        self, file: _CompressedFile, shape: tuple[int, int, int]
    ) -> None:  # numpydoc ignore=GL08
        self.file = file
        self.shape = shape
        self.dtype = np.dtype(np.float32)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        """Read the part of the waveform that xarray's key selects.

        Parameters
        ----------
        key : indexing.ExplicitIndexer
            The selection. xarray turns one it can't hand to `_read` into one it can,
            and picks the values asked for from what that reads.

        Returns
        -------
        np.ndarray
            The selection's values.
        """
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._read
        )

    def _read(self, key: tuple[int | slice | np.ndarray, ...]) -> np.ndarray:
        """Read an outer selection of the waveform.

        Parameters
        ----------
        key : tuple
            An integer, a slice with a positive step, or an array of sorted indices,
            which may repeat, for each of the component, station and time dimensions.

        Returns
        -------
        np.ndarray
            The selection, as float32, without the dimensions an integer selects.

        Raises
        ------
        ValueError
            If the file has been closed.
        Damaged
            If a stream doesn't decode, naming the file.
        OSError
            If the file can't be read, naming it.
        """
        components, stations, times = key
        n_component, n_station, _ = self.shape
        with self.file.reading() as streams:
            traces = _read_traces(
                streams,
                _chosen(components, n_component),
                _chosen(stations, n_station),
                times if isinstance(times, slice) else np.atleast_1d(times),
            )
        # An integer drops its dimension, as it does in numpy.
        return traces[tuple(0 if isinstance(k, int) else slice(None) for k in key)]


def _open(
    path: str | os.PathLike[str], drop_variables: str | Iterable[str] | None
) -> xr.Dataset:
    """Build the dataset that xarray's engine, and so `open_dataset`, gives.

    Parameters
    ----------
    path : str or os.PathLike
        The compressed file.
    drop_variables : str or iterable of str, optional
        Variables to leave out, which aren't read. With ``waveform`` left out, the file
        is closed once it is checked.

    Returns
    -------
    xr.Dataset
        The dataset, which closes the file as it is closed.

    Raises
    ------
    NotCompressed, UnknownFormat, Damaged, OSError
        As `open_dataset` raises them.
    """
    path = Path(path)
    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    drop = list(drop_variables or ())
    with naming(path):
        _refuse_unless_hdf5(path, NotCompressed)
    file = _CompressedFile(path)
    try:
        # Opening the file checks it, as decompress_file does, before the rest is read.
        with file.reading() as streams:
            lazy = indexing.LazilyIndexedArray(_LazyWaveform(file, streams.shape))
            ds = _original(path, streams, lambda: lazy, drop)
    except BaseException:
        file.close()
        raise
    if "waveform" in drop:
        file.close()
    else:
        ds.set_close(file.close)
    return ds


def open_dataset(path: str | os.PathLike[str]) -> xr.Dataset:
    """Open a compressed file as an xarray dataset, decoding only the traces read.

    The dataset is the one `waveform_compression_tools.files.decompress_file` gives,
    except that its waveform is read from the file only as it is needed. Opening,
    printing and selecting read none of it. Reading a selection's values, with
    ``.values``, ``.load()`` or a computation, decodes its traces then, and only those,
    each from its own stream; a real station's trace is then its exact copy. A
    selection in time decodes whole traces and then cuts them. A selection of points
    decodes every chosen component of every chosen station and then picks. Every value
    read is bit for bit the one `decompress_file` gives, and a damaged stream fails as
    it does in `decompress_file`, with `waveform_compression_tools.errors.Damaged`
    naming the file.

    It is ``xr.open_dataset(path, engine="waveform_compression_tools")``, so the dataset
    behaves as the datasets of xarray's other engines do: the values read are kept with
    the selection that read them, and loading the whole waveform holds all of it. The
    file stays open until the dataset is closed, with ``.close()`` or at the end of a
    ``with`` block, and its waveform can't be read after that. Selections made from the
    dataset read through it, so close it once they are read.

    Parameters
    ----------
    path : str or os.PathLike
        The compressed file.

    Returns
    -------
    xr.Dataset
        The original dataset, with its waveform decoded as it is read. Everything else
        in it is read into memory as it is opened.

    Raises
    ------
    NotCompressed
        If `path` is a file that isn't HDF5, or holds no compressed waveform.
    UnknownFormat
        If it is in a format version other than 1, or has a setting outside 8 to 32.
    Damaged
        If it lacks a part of the format, or its parts don't agree with each other, as
        when its streams don't fit its dimensions, as for `decompress_file`.
    OSError
        If it can't be read, as when it is cut short, naming it.
    """
    return xr.open_dataset(path, engine=FlacBackendEntrypoint)
