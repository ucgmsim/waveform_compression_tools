"""Open a compressed file as an xarray dataset, decoding only the traces that are read.

`open_dataset` gives the dataset that `waveform_compression_tools.files.decompress_file`
gives, but its waveform stays in the file until it is read. Selecting with ``isel`` or
``sel`` reads nothing; reading a selection's values decodes only its traces, each from
its FLAC stream, of which only its own bytes are read. A real station's trace is then
its exact copy, as in `decompress_file`. A selection in time decodes whole traces, and
then cuts them.

It is built on xarray's backend machinery, as xarray's own netCDF backends are: the
waveform is a `xarray.backends.BackendArray`, which xarray indexes lazily, and the file
is opened through xarray's `xarray.backends.CachingFileManager`. Nothing needs dask.
"""

import dataclasses
import itertools
import os
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint, CachingFileManager
from xarray.core import indexing

from .errors import NotCompressed
from .files import GROUP, _open_streams, _refuse_unless_hdf5, _Streams


class _CompressedFile:
    """A compressed file, open for a dataset that decodes its waveform as it is read.

    The file is opened through a `xarray.backends.CachingFileManager`, as xarray's
    h5netcdf backend opens its files: xarray keeps only so many files open, and opens a
    file again when it is read after it was closed to make room for others. A file
    closed by `close` stays closed.

    Parameters
    ----------
    path : Path
        The compressed file.

    Attributes
    ----------
    path : Path
        The compressed file.
    streams : _Streams
        Where its streams are, and how to decode them, as it was opened. Its datasets
        are those of the file as it was first opened: `reading` gives them as the file
        is open at the time.

    Raises
    ------
    NotCompressed
        If the file holds no compressed waveform.
    UnknownFormat
        If it is in a format version other than 1, or has a setting outside 8 to 32.
    ValueError
        If its arrays don't agree with each other (see
        `waveform_compression_tools.files.decompress_file`).
    """

    def __init__(self, path: Path) -> None:  # numpydoc ignore=GL08
        self.path = path
        # Without HDF5's cache of chunks, which would read each chunk a stream is in
        # whole, a MiB, a stream's own bytes are all that is read.
        self._manager = CachingFileManager(
            h5py.File, path, mode="r", kwargs={"rdcc_nbytes": 0}
        )
        self._closed = False
        # A file refused here is closed again by the manager.
        with self._manager.acquire_context() as f:
            self.streams = _open_streams(f, path)

    @contextmanager
    def reading(self) -> Generator[_Streams]:
        """Have the file open while its streams are read.

        Yields
        ------
        _Streams
            The streams, with the file's datasets as it is open now.

        Raises
        ------
        ValueError
            If the file has been closed.
        """
        if self._closed:
            raise ValueError(f"{self.path} is closed, so its waveform can't be read")
        with self._manager.acquire_context() as f:
            group = f[GROUP]
            yield dataclasses.replace(
                self.streams,
                compressed=group["compressed"],
                exact_waveform=group["exact_waveform"],
            )

    def close(self) -> None:
        """Close the file, for good."""
        self._closed = True
        self._manager.close()


def _read_traces(
    streams: _Streams, components: np.ndarray, stations: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Read some traces as `decompress_file` gives them, reading only what they need.

    The traces are every chosen component of every chosen station, each cut to the
    chosen samples. Each is decoded on its own (see `_Streams.trace`), and cut as soon
    as it is decoded, so memory holds the cut traces and one whole one.

    Parameters
    ----------
    streams : _Streams
        The file's streams, with its datasets open.
    components : np.ndarray
        The components, as indices.
    stations : np.ndarray
        The stations, as indices.
    times : np.ndarray
        The samples of each trace, as indices.

    Returns
    -------
    np.ndarray
        The traces, as float32 of shape (components, stations, times).

    Raises
    ------
    ValueError
        If a stream is damaged, or decodes to the wrong number of samples (see
        `waveform_compression_tools.codec.decode_trace`).
    """
    traces = np.empty((components.size, stations.size, times.size), np.float32)
    for (i, c), (j, s) in itertools.product(enumerate(components), enumerate(stations)):
        traces[i, j] = streams.trace(int(c), int(s))[times]
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

    Attributes
    ----------
    shape : tuple[int, int, int]
        The waveform's shape: (component, station, time).
    dtype : np.dtype
        float32.
    """

    def __init__(self, file: _CompressedFile) -> None:  # numpydoc ignore=GL08
        self.file = file
        self.shape = file.streams.shape
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
            An integer, a slice with a positive step, or an array of indices in
            increasing order, for each of the component, station and time dimensions.

        Returns
        -------
        np.ndarray
            The selection, as float32, without the dimensions an integer selects.

        Raises
        ------
        ValueError
            If the file has been closed, or a stream doesn't decode.
        """
        chosen = [np.arange(n)[k] for n, k in zip(self.shape, key, strict=True)]
        with self.file.reading() as streams:
            traces = _read_traces(streams, *map(np.atleast_1d, chosen))
        return traces.reshape([n for indices in chosen for n in np.shape(indices)])


def open_dataset(path: str | os.PathLike[str]) -> xr.Dataset:
    """Open a compressed file as an xarray dataset, decoding only the traces read.

    The dataset is the one `waveform_compression_tools.files.decompress_file` gives,
    except that its waveform is read from the file only as it is needed. Opening,
    printing and selecting read none of it. Reading a selection's values, with
    ``.values``, ``.load()`` or a computation, decodes its traces then, and only those,
    each from its own stream; a real station's trace is then its exact copy. A
    selection in time decodes whole traces and then cuts them. Each read
    decodes again, so ``.load()`` a selection to keep it. Every value read is bit for bit
    the one `decompress_file` gives.

    The file stays open until the dataset is closed, with ``.close()`` or at the end of a
    ``with`` block, and its waveform can't be read after that. Selections made from the
    dataset read through it, so close it once they are read.

    Parameters
    ----------
    path : str or os.PathLike
        The compressed file.

    Returns
    -------
    xr.Dataset
        The original dataset, with its waveform decoded as it is read. Everything else in
        it is read into memory as it is opened.

    Raises
    ------
    NotCompressed
        If `path` is a file that isn't HDF5, or holds no compressed waveform.
    UnknownFormat
        If it is in a format version other than 1, or has a setting outside 8 to 32.
    ValueError
        If its arrays don't agree with each other, or its ``waveform_attr_types`` names
        an attribute that its ``waveform_attrs`` lacks, as for `decompress_file`.
    """
    path = Path(path)
    _refuse_unless_hdf5(path, NotCompressed)
    file = _CompressedFile(path)
    try:
        ds = xr.load_dataset(path, engine="h5netcdf")
        waveform = indexing.LazilyIndexedArray(_LazyWaveform(file))
        ds["waveform"] = file.streams.variable(waveform)
    except BaseException:
        file.close()
        raise
    ds.set_close(file.close)
    return ds


class FlacBackendEntrypoint(BackendEntrypoint):
    """xarray's engine for compressed files, ``"waveform_compression_tools"``.

    ``xr.open_dataset(path, engine="waveform_compression_tools")`` opens a compressed
    file as `open_dataset` does. It takes none of xarray's options for decoding, since
    the dataset is the original as it was compressed. xarray never picks this engine by
    itself: without it, xarray opens a compressed file as an ordinary netCDF file,
    without its waveform.
    """

    description = "Open a compressed file, decoding only the traces read"
    url = "https://github.com/ucgmsim/waveform_compression_tools"

    def open_dataset(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.Dataset:
        """Open a compressed file as `open_dataset` does.

        Parameters
        ----------
        filename_or_obj : str or os.PathLike
            The compressed file. xarray can give an engine a file object or bytes too,
            but a compressed file is opened only by its path.
        drop_variables : str or iterable of str, optional
            Variables to leave out. A name the file doesn't hold is ignored.

        Returns
        -------
        xr.Dataset
            The dataset, as `open_dataset` gives it, without `drop_variables`.

        Raises
        ------
        TypeError
            If `filename_or_obj` isn't a path.
        """
        if not isinstance(filename_or_obj, str | os.PathLike):
            raise TypeError(
                "a compressed file is opened by its path, not from a "
                f"{type(filename_or_obj).__name__}"
            )
        ds = open_dataset(filename_or_obj)
        kept = ds.drop_vars(drop_variables or [], errors="ignore")
        kept.set_close(ds.close)  # Dropping makes a dataset that wouldn't close it.
        return kept
