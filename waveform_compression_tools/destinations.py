"""Write outputs safely: never over the input, over an earlier file only if asked, and whole.

An output is written to its path with ``.partial`` added, and renamed to its path when
it is complete, so it appears there complete or not at all. A partial file left by a
run that was killed is replaced by the next run.
"""

import os
from pathlib import Path

import xarray as xr

from .errors import OutputExists, SameFile


def refuse_overwriting(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Refuse to write `dst` where that would destroy `src`, or replace a file unasked.

    A file is written to `dst` with ``.partial`` added, which is deleted if the write
    fails, and then renamed to `dst`. Either would destroy `src` if it is `dst`, or
    `dst` with ``.partial`` added. The checks are made in that order.

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
        If `dst`, or `dst` with ``.partial`` added, resolves to `src`, whatever
        `overwrite` says.
    OutputExists
        If something is at `dst` already, a broken link included, and `overwrite` is
        false.
    """
    partial = Path(f"{dst}.partial")
    if src.resolve() in (dst.resolve(), partial.resolve()):
        raise SameFile(f"writing {dst} would overwrite {src}")
    if not overwrite and os.path.lexists(dst):
        raise OutputExists(f"{dst} exists already, and is replaced only if asked")


def write_netcdf(ds: xr.Dataset, dst: Path) -> None:
    """Write a dataset to a netCDF file that appears at `dst` only when it is complete.

    It is written by xarray with the h5netcdf engine to `dst` with ``.partial`` added,
    which replaces any partial file already there, and renamed to `dst` when it is
    complete. If anything goes wrong, the partial file is deleted and the exception
    raised again; a file already at `dst` is then as it was.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to write.
    dst : Path
        Where to write the file. A file already there is replaced (see
        `refuse_overwriting`, which a caller makes first).
    """
    partial = Path(f"{dst}.partial")
    try:
        ds.to_netcdf(partial, engine="h5netcdf")
        os.replace(partial, dst)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
