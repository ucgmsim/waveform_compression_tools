"""Write outputs safely: never over the input, over an earlier file only if asked, and whole.

An output is written to its path with ``.partial`` added, and renamed to its path when
it is complete, so it appears there complete or not at all (see `writing`). Whatever is
at that name first, a partial file left by a run that was killed or a link, is removed,
and never written through.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import xarray as xr

from .errors import OutputExists, SameFile


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


def fresh_partial(dst: str | os.PathLike[str]) -> Path:
    """Clear the name a file for `dst` is written under, and return it.

    That name is `dst` with ``.partial`` added. Whatever is there is removed: a partial
    file left by a run that was killed, or a link, whose target stays as it was.

    Parameters
    ----------
    dst : str or os.PathLike
        Where the file is to go.

    Returns
    -------
    Path
        `dst` with ``.partial`` added, with nothing there now.
    """
    partial = _partial_name(dst)
    if os.path.lexists(partial):
        partial.unlink()  # A link itself, never its target.
    return partial


@contextmanager
def writing(dst: str | os.PathLike[str]) -> Generator[Path]:
    """Have a file written under a partial name, and put at `dst` only when it is whole.

    The name is cleared first (see `fresh_partial`). If the block ends normally, the file
    is renamed to `dst`, replacing a file already there. If anything goes wrong, the
    partial file is deleted and the exception raised again; a file already at `dst` is
    then as it was.

    Parameters
    ----------
    dst : str or os.PathLike
        Where the file is to go. A caller that may not replace a file there checks
        first (see `refuse_overwriting`).

    Yields
    ------
    Path
        The name to write the file under: `dst` with ``.partial`` added.
    """
    dst = Path(dst)
    partial = fresh_partial(dst)
    try:
        yield partial
        os.replace(partial, dst)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


@contextmanager
def naming(path: str | os.PathLike[str]) -> Generator[None]:
    """Have an `OSError` from a block that names no file name `path`.

    h5py's errors name no file, so an error from reading or writing one doesn't say
    which it was. One that has an error number and its words is raised again as an
    `OSError` with the same number and words, and `path` as its file, caused by the
    first. Any other is raised as it is, since a file name would take its words.

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
        for an error of the block that names no file and has a number and its words.
    """
    try:
        yield
    except OSError as error:
        if error.filename is not None or error.errno is None or error.strerror is None:
            raise
        raise OSError(error.errno, error.strerror, os.fspath(path)) from error


def refuse_overwriting(
    src: str | os.PathLike[str], dst: str | os.PathLike[str], overwrite: bool = False
) -> None:
    """Refuse to write `dst` where that would destroy `src`, or replace a file unasked.

    A file is written to `dst` with ``.partial`` added, once whatever is at that name
    is removed, and then renamed to `dst`. Either would destroy `src` if it is `dst`,
    or `dst` with ``.partial`` added. The checks are made in that order.

    Parameters
    ----------
    src : str or os.PathLike
        The file that is read.
    dst : str or os.PathLike
        The file that is written.
    overwrite : bool, default False
        Whether a file already at `dst` may be replaced.

    Raises
    ------
    SameFile
        If `dst`, or `dst` with ``.partial`` added, is `src`, by its name, through a
        link, or as a hard link to it, whatever `overwrite` says.
    OutputExists
        If something is at `dst` already, a broken link included, and `overwrite` is
        false.
    """
    src, dst = Path(src), Path(dst)
    partial = _partial_name(dst)
    same = src.resolve() in (dst.resolve(), partial.resolve())
    if not same and src.exists():
        # A hard link is another name for src, which only the files themselves show.
        same = any(
            path.exists() and os.path.samefile(src, path) for path in (dst, partial)
        )
    if same:
        raise SameFile(f"writing {dst} would overwrite {src}")
    if not overwrite and os.path.lexists(dst):
        raise OutputExists(
            f"{dst} exists already, and is replaced only if asked (--overwrite, or "
            "overwrite=True)"
        )


def write_netcdf(ds: xr.Dataset, dst: str | os.PathLike[str]) -> None:
    """Write a dataset to a netCDF file that appears at `dst` only when it is complete.

    It is written by xarray with the h5netcdf engine, under a partial name (see
    `writing`). If anything goes wrong, the partial file is deleted and the exception
    raised again; a file already at `dst` is then as it was.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to write.
    dst : str or os.PathLike
        Where to write the file. A file already there is replaced (see
        `refuse_overwriting`, which a caller makes first).
    """
    with writing(dst) as partial:
        ds.to_netcdf(partial, engine="h5netcdf")
