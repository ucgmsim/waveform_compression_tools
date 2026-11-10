"""Write outputs safely: never over the input, over an earlier file only if asked, and whole.

An output is written to its path with ``.partial`` added, and renamed to its path when
it is complete, so it appears there complete or not at all (see `writing`). Whatever is
at that name first, a partial file left by a run that was killed or a link, is removed,
and never written through. HDF5 writes an output through an `HDF5Output`, so that a
write that fails, as on a full disk, raises its own error.
"""

import errno
import io
import os
import signal
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType, TracebackType
from typing import TYPE_CHECKING

import xarray as xr

from .errors import OutputExists, SameFile

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer, WriteableBuffer

# From a failed write on, a write to an HDF5Output of this many bytes or more is held as
# zeros, not copied. Only a variable's data comes that big, as xarray writes it in one
# go, or as decompress writes a block of the waveform; HDF5 reads none of it back, and
# its own records are far smaller.
_BIG_WRITE = 16 * 2**20

# The signals whose handlers in Python wait while an HDF5Output is open: Ctrl-C's, and
# SIGTERM's, as a job may clean up with.
_WAITING_SIGNALS = (signal.SIGINT, signal.SIGTERM)

_Handler = Callable[[int, FrameType | None], object]


def _partial_name(dst: str | os.PathLike[str]) -> Path:
    """Return the name a file for `dst` is written under: `dst` with ``.partial`` added.

    Parameters
    ----------
    dst : str or os.PathLike
        Where the file is to go.

    Returns
    -------
    Path
        `dst` with ``.partial`` added.
    """
    return Path(f"{Path(dst)}.partial")


@contextmanager
def writing(dst: str | os.PathLike[str]) -> Generator[Path]:
    """Have a file written under a partial name, and put at `dst` only when it is whole.

    The name, `dst` with ``.partial`` added, is cleared first: whatever is there is
    removed, a partial file left by a run that was killed or a link, whose target stays
    as it was. If the block ends normally, the file is renamed to `dst`, replacing a
    file already there. If anything goes wrong, the partial file is deleted and the
    exception raised again; a file already at `dst` is then as it was. An `OSError` in
    any of this, the block's included, that names no file or the partial file names
    `dst` (see `naming`).

    Parameters
    ----------
    dst : str or os.PathLike
        Where the file is to go. A caller that may not replace a file there checks
        first (see `check_destination`).

    Yields
    ------
    Path
        The name to write the file under: `dst` with ``.partial`` added.
    """
    dst = Path(dst)
    partial = _partial_name(dst)
    with naming(dst):
        if os.path.lexists(partial):
            partial.unlink()  # A link itself, never its target.
        try:
            yield partial
            os.replace(partial, dst)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise


def named_file(error: Exception) -> str | None:
    """Return the file an error names, as an `OSError` or a refusal does, if it names one.

    Parameters
    ----------
    error : Exception
        The error.

    Returns
    -------
    str or None
        Its ``filename``, as text, or None if it has none, or one that isn't a path.
    """
    named = getattr(error, "filename", None)
    return os.fsdecode(named) if isinstance(named, str | bytes | os.PathLike) else None


@contextmanager
def naming(path: str | os.PathLike[str]) -> Generator[None]:
    """Have an `OSError` that names no file, or `path`'s partial file, name `path`.

    h5py's errors name no file, and nor does one that a write through an `HDF5Output`
    meets, so an error from reading or writing one doesn't say which it was. And the
    partial file that `path` is written as is `path` to anyone using it: `writing` names
    `path` this way.
    Such an error that has an error number and its words is raised again as an `OSError`
    with the same number and words, and `path` as its file, caused by the first. Any
    other is raised as it is, since a file name would take its words.

    Parameters
    ----------
    path : str or os.PathLike
        The file that the block's errors are about.

    Yields
    ------
    None
        Nothing; the block runs.

    Raises
    ------
    OSError
        Of the type its number gives, as `PermissionError` for EACCES, naming `path`,
        for an error of the block that names no file, or `path`'s partial file, and has
        a number and its words.
    """
    partial = os.fspath(_partial_name(path))
    try:
        yield
    except OSError as error:
        elsewhere = error.filename is not None and named_file(error) != partial
        if elsewhere or error.errno is None or error.strerror is None:
            raise
        raise OSError(error.errno, error.strerror, os.fspath(path)) from error


class HDF5Output(io.RawIOBase):
    """A new file for HDF5 to write through, which never tells HDF5 that a write failed.

    This is a workaround for a bug in HDF5 that gives a misleading error when one of its
    writes fails, as one does on a full disk or past a quota. Closing the file then
    fails too, and the error from the close, which says nothing of the disk, replaces
    the one that does. HDF5 2.0.0 to 2.2.0 are also left pointing at memory they have
    freed, so that a later call can crash Python. h5py and xarray write through a file
    object in place of a path, and HDF5 is never told here that a write, a truncation
    or a read failed.

    The first `OSError` that one meets is kept as `failure`, and a read that fails gives
    zeros. From then on the file on disk is left as it is, and what HDF5 writes is held
    in memory, where its reads find it, so that it closes the file cleanly; a write of
    16 MiB or more, which only a variable's data comes to, and which HDF5 doesn't read
    back, is held as zeros, not copied. The file isn't whole, so the writer raises the
    failure as soon as it can (see `raise_pending`), to stop what is held from adding
    up. Leaving the block raises it too, in place of any other exception but an
    interruption, since the run can't succeed once its file can't be written.

    A signal whose handler is in Python raises in whatever Python code is running, and
    as HDF5 writes, that is this file's: HDF5 would take a `KeyboardInterrupt` there for
    a failed write. So while the file is open, Ctrl-C and SIGTERM wait, to be acted on
    by their handlers in `raise_pending`, or as the file is closed. Sent again while it
    waits, either ends Python at once, as it does by default, leaving the partial file.
    Nothing waits off the main thread, whose code no signal stops, nor for a signal
    that is ignored or ends Python at once. An exception that another signal's handler
    raises, or any other raised in HDF5's call, as a `MemoryError`, still reaches HDF5.

    Open one at a time, or close them in the reverse order, as `with` blocks do, on the
    thread that opened them: each gives the signals back the handlers it found.

    Parameters
    ----------
    path : str or os.PathLike
        Where to write the file. Nothing may be there already, so no link is written
        through (see `writing`).

    Attributes
    ----------
    name : str
        The path of the file.
    failure : OSError or None
        The error that the first failed write, truncation or read met, if one has.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:  # numpydoc ignore=PR01
        """Make the file, and have Ctrl-C and SIGTERM wait while it is open."""
        super().__init__()
        self.name = os.fspath(path)
        self.failure: OSError | None = None
        self._position = self._size = self._on_disk = 0  # The size is as HDF5 sees it.
        # Writes after a failure, in order: where, how long, and what, or None if zeros.
        self._kept: list[tuple[int, int, bytes | None]] = []
        self._handlers: dict[int, _Handler] = {}  # Each waiting signal's own handler.
        self._waited: list[int] = []  # The signals that came while waiting, in order.
        try:
            self._file = io.FileIO(path, "x+")
        except BaseException:
            super().close()  # So that nothing is left for close() to do.
            raise
        for number in _WAITING_SIGNALS:
            handler = signal.getsignal(number)
            if callable(handler):
                try:
                    signal.signal(number, self._wait)
                except ValueError:  # Not the main thread.
                    break
                self._handlers[number] = handler

    def __repr__(self) -> str:
        """Name the file, for the messages of h5py and xarray that show this."""
        return f"{type(self).__name__}({self.name!r})"

    def _wait(self, signum: int, frame: FrameType | None) -> None:
        """Note a signal, to be acted on later, or end Python on a second one at once.

        Parameters
        ----------
        signum : int
            The signal.
        frame : FrameType or None
            Where the code was.
        """
        if signum in self._waited:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        self._waited.append(signum)

    def _act_on_signals(self, handlers: dict[int, _Handler]) -> None:
        """Have each signal that waited acted on by its own handler, in the order sent.

        Parameters
        ----------
        handlers : dict
            Each waiting signal's own handler, by its number.
        """
        while self._waited:
            number = self._waited.pop(0)
            if number in handlers:
                handlers[number](number, None)

    def raise_pending(self) -> None:
        """Have a signal that waited acted on, as by its handler, or raise the failure.

        Call it outside any HDF5 call, as often as is useful: from a failure on, what
        HDF5 writes is held in memory.

        Raises
        ------
        KeyboardInterrupt
            For a Ctrl-C that waited, as Python's own handler raises it, or whatever a
            waiting signal's handler raises.
        OSError
            The error that a write, a truncation or a read met, if one did.
        """
        self._act_on_signals(self._handlers)
        if self.failure is not None:
            raise self.failure

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Move to a place in the file, as HDF5 does before each read and write.

        Parameters
        ----------
        offset : int
            How far to move, in bytes.
        whence : int, default os.SEEK_SET
            What `offset` counts from: the start, the place now, or the end.

        Returns
        -------
        int
            The place moved to, from the start of the file.
        """
        start = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self._size}
        self._position = start[whence] + offset
        return self._position

    def tell(self) -> int:
        """Return the place in the file.

        Returns
        -------
        int
            The place, from the start of the file.
        """
        return self._position

    def readinto(self, buffer: "WriteableBuffer") -> int:
        """Read into a buffer, from the disk and from what was written after a failure.

        Parameters
        ----------
        buffer : buffer
            Where to put the bytes, as many as it holds.

        Returns
        -------
        int
            How many bytes were read: fewer than `buffer` holds only at the end of the
            file.
        """
        view = memoryview(buffer).cast("B")
        start = self._position
        n = max(0, min(len(view), self._size - start))
        on_disk = max(0, min(n, self._on_disk - start))
        done = 0
        try:
            while done < on_disk:
                self._file.seek(start + done)
                got = self._file.readinto(view[done:on_disk])
                if not got:
                    break
                done += got
        except OSError as error:  # What couldn't be read reads as zeros.
            self._fail(error)
        view[done:n] = bytes(n - done)
        for at, length, data in self._kept:
            low, high = max(at, start), min(at + length, start + n)
            if low < high:
                view[low - start : high - start] = (
                    bytes(high - low) if data is None else data[low - at : high - at]
                )
        self._position = start + n
        return n

    def write(self, buffer: "ReadableBuffer") -> int:
        """Write a buffer, and report it all written, even if it couldn't all be.

        Parameters
        ----------
        buffer : buffer
            The bytes to write.

        Returns
        -------
        int
            How many bytes were written: all of them.
        """
        data = memoryview(buffer).cast("B")
        start, n = self._position, len(data)
        if not n:
            return 0  # As in any file, nothing written makes it no longer.
        if self.failure is None:
            try:
                done = 0
                while done < n:
                    self._file.seek(start + done)
                    written = self._file.write(data[done:])
                    if not written:  # A file that takes none of it would never end.
                        raise OSError(errno.EIO, "the file took none of a write")
                    done += written
            except OSError as error:
                self._fail(error)
        if self.failure is None:
            self._on_disk = max(self._on_disk, start + n)
        else:
            self._kept.append((start, n, bytes(data) if n < _BIG_WRITE else None))
        self._position = start + n
        self._size = max(self._size, start + n)
        return n

    def truncate(self, size: int | None = None) -> int:
        """Cut or extend the file to a size, and report it done, even if it couldn't be.

        Parameters
        ----------
        size : int, optional
            The size, in bytes. By default, the place in the file.

        Returns
        -------
        int
            The size.
        """
        size = self._position if size is None else size
        if self.failure is None:
            try:
                self._file.truncate(size)
            except OSError as error:
                self._fail(error)
        if self.failure is None:
            self._on_disk = size
        else:
            self._on_disk = min(self._on_disk, size)
            self._kept = [
                (at, min(n, size - at), None if data is None else data[: size - at])
                for at, n, data in self._kept
                if at < size
            ]
        self._size = size
        return size

    def _fail(self, error: OSError) -> None:
        """Keep the first failure, without the frames it was raised in.

        Those are HDF5's caller's, as xarray's, and would hold its data, and the file,
        for as long as the error is kept.

        Parameters
        ----------
        error : OSError
            What a write, a truncation or a read met.
        """
        if self.failure is None:
            self.failure = error.with_traceback(None)

    def flush(self) -> None:
        """Do nothing: each write goes straight to the file."""

    def close(self) -> None:
        """Close the file, and give each waiting signal its handler back.

        A signal that waited is acted on then, by its handler.

        Raises
        ------
        KeyboardInterrupt
            For a Ctrl-C that waited, as Python's own handler raises it, or whatever a
            waiting signal's handler raises.
        OSError
            If the file can't be closed, as when a disk reports a failed write then.
        """
        handlers, self._handlers = self._handlers, {}
        self._kept = []  # Once HDF5 is done with the file, nothing reads it.
        try:
            if not self.closed:
                self._file.close()
        finally:
            super().close()
            for number, handler in handlers.items():
                signal.signal(number, handler)
            self._act_on_signals(handlers)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the file, then raise the failure, in place of any but an interruption.

        Parameters
        ----------
        exc_type : type or None
            The type of the exception being raised, if one is.
        exc_val : BaseException or None
            The exception being raised, if one is.
        exc_tb : TracebackType or None
            Its traceback.
        """
        self.close()
        if exc_val is None or (
            isinstance(exc_val, Exception) and exc_val is not self.failure
        ):
            self.raise_pending()


class _NoFolderError(FileNotFoundError):
    """No folder to write a file in: it names the file, and its words are the reason."""

    def __str__(self) -> str:
        """Give the reason alone, without the error number or the file's name.

        Returns
        -------
        str
            The reason.
        """
        return str(self.strerror)


def _require_folder(dst: str | os.PathLike[str]) -> None:
    """Fail unless the folder that a file is to be written in exists.

    Parameters
    ----------
    dst : str or os.PathLike
        The file.

    Raises
    ------
    FileNotFoundError
        If there is no folder at ``dst``'s parent. Its ``filename`` is `dst`, and its
        message says which folder is missing.
    """
    dst = Path(dst)
    if not dst.parent.is_dir():
        raise _NoFolderError(
            errno.ENOENT,
            f"there is no folder {dst.parent} to write {dst.name} in",
            os.fspath(dst),
        )


def check_destination(
    dst: str | os.PathLike[str],
    *sources: str | os.PathLike[str],
    overwrite: bool = False,
) -> None:
    """Check a destination before any work: refuse it, or fail if it has no folder.

    A file is written to `dst` with ``.partial`` added, once whatever is at that name
    is removed, and then renamed to `dst`. Either would destroy a source that is `dst`,
    or `dst` with ``.partial`` added. The checks are made in that order, for each
    source in turn, and then that of a file already at `dst`. Last, the folder `dst`
    is to be written in has to exist, so that a mistyped folder is found before any
    work is done.

    Parameters
    ----------
    dst : str or os.PathLike
        The file that is written.
    *sources : str or os.PathLike
        The files that are read.
    overwrite : bool, default False
        Whether a file already at `dst` may be replaced.

    Raises
    ------
    SameFile
        If `dst`, or `dst` with ``.partial`` added, is a source, by its name, through a
        link, or as a hard link to it, whatever `overwrite` says.
    OutputExists
        If something is at `dst` already, a broken link included, and `overwrite` is
        false.
    FileNotFoundError
        If there is no folder at ``dst``'s parent. Its ``filename`` is `dst`, and its
        message says which folder is missing.
    """
    dst = Path(dst)
    partial = _partial_name(dst)
    for src in map(Path, sources):
        same = src.resolve() in (dst.resolve(), partial.resolve())
        if not same and src.exists():
            # A hard link is another name for src, which only the files themselves show.
            same = any(
                path.exists() and os.path.samefile(src, path) for path in (dst, partial)
            )
        if same:
            raise SameFile(f"writing {dst} would overwrite {src}", dst)
    if not overwrite and os.path.lexists(dst):
        raise OutputExists(
            f"{dst} exists already, and is replaced only if asked (--overwrite, or "
            "overwrite=True)",
            dst,
        )
    _require_folder(dst)


def write_netcdf(ds: xr.Dataset, dst: str | os.PathLike[str]) -> None:
    """Write a dataset to a netCDF file that appears at `dst` only when it is complete.

    It is written by xarray with the h5netcdf engine, under a partial name (see
    `writing`), through an `HDF5Output`. If anything goes wrong, the partial file is
    deleted and the exception raised again; a file already at `dst` is then as it was.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to write.
    dst : str or os.PathLike
        Where to write the file. A file already there is replaced (see
        `check_destination`, which a caller makes first).

    Raises
    ------
    OSError
        If the file can't be written, as on a full disk, naming `dst`.
    """
    with writing(dst) as partial, HDF5Output(partial) as out:
        ds.to_netcdf(out, engine="h5netcdf")
