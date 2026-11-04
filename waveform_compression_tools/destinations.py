"""Write outputs safely: never over the input, over an earlier file only if asked, and whole.

An output is written to its path with ``.partial`` added, and renamed to its path when
it is complete, so it appears there complete or not at all. Whatever is at that name
first, a partial file left by a run that was killed or a link, is removed, and never
written through.
"""

import os
from pathlib import Path

import xarray as xr

from .errors import OutputExists, SameFile


def fresh_partial(dst: Path) -> Path:
    """Clear the name a file for `dst` is written under, and return it.

    That name is `dst` with ``.partial`` added. Whatever is there is removed: a partial
    file left by a run that was killed, or a link, whose target stays as it was.

    Parameters
    ----------
    dst : Path
        Where the file is to go.

    Returns
    -------
    Path
        `dst` with ``.partial`` added, with nothing there now.
    """
    partial = Path(f"{dst}.partial")
    if os.path.lexists(partial):
        partial.unlink()  # A link itself, never its target.
    return partial


def refuse_overwriting(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Refuse to write `dst` where that would destroy `src`, or replace a file unasked.

    A file is written to `dst` with ``.partial`` added, once whatever is at that name
    is removed, and then renamed to `dst`. Either would destroy `src` if it is `dst`,
    or `dst` with ``.partial`` added. The checks are made in that order.

    Parameters
    ----------
    src : Path
        The file that is read.
    dst : Path
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
    partial = Path(f"{dst}.partial")
    same = src.resolve() in (dst.resolve(), partial.resolve())
    if not same and src.exists():
        # A hard link is another name for src, which only the files themselves show.
        same = any(
            path.exists() and os.path.samefile(src, path) for path in (dst, partial)
        )
    if same:
        raise SameFile(f"writing {dst} would overwrite {src}")
    if not overwrite and os.path.lexists(dst):
        raise OutputExists(f"{dst} exists already, and is replaced only if asked")


def write_netcdf(ds: xr.Dataset, dst: Path) -> None:
    """Write a dataset to a netCDF file that appears at `dst` only when it is complete.

    It is written by xarray with the h5netcdf engine to `dst` with ``.partial`` added,
    once whatever is at that name is removed (see `fresh_partial`), and renamed to
    `dst` when it is complete. If anything goes wrong, the partial file is deleted and
    the exception raised again; a file already at `dst` is then as it was.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to write.
    dst : Path
        Where to write the file. A file already there is replaced (see
        `refuse_overwriting`, which a caller makes first).
    """
    partial = fresh_partial(dst)
    try:
        ds.to_netcdf(partial, engine="h5netcdf")
        os.replace(partial, dst)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
