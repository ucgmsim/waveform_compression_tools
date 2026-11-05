"""Convert a legacy BB.bin file to the standard netCDF layout.

A BB.bin file holds the broadband acceleration that a simulation recorded at its
stations. This is the legacy BB.bin layout (qcore's former BBSeis reader). Every
number in the file is in one byte order, the writer's, which `read_bbbin` tells from
the file's size:

- a header of 0x500 bytes: the int32 number of stations ``nstat`` and of samples
  ``nt``; the float32 ``duration``, ``dt`` and ``start_sec``, in seconds; and three
  256-byte strings, ``lf_dir``, ``lf_vm`` and ``hf_file``, padded with NULs;
- from byte 0x500, one record of 0x2C bytes for each station: the float32 ``lon`` and
  ``lat``; the name, 8 bytes padded with NULs; the int32 ``x``, ``y`` and ``z``; and
  the float32 ``e_dist``, ``hf_vs_ref``, ``lf_vs_ref`` and ``vsite``;
- then the acceleration in g, float32 of shape (nstat, nt, 3), with the components in
  the order 090, 000, ver.

`convert_bbbin` writes a netCDF file in the standard layout (see
`waveform_compression_tools.codec.check_standard`) with all of that in it, which
`waveform_compression_tools.files.compress_file` can then compress.
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr

from .codec import STANDARD_COMPONENTS, STANDARD_DIMS
from .destinations import refuse_overwriting, require_folder, write_netcdf
from .errors import NotBBBin

# The size of a file's header, and of each station's record that follows it.
_HEAD_BYTES = 0x500
_STATION_BYTES = 0x2C
# The components of a sample, in the order a file holds them, and the size of a sample
# of all of them in float32.
_FILE_COMPONENTS = ("090", "000", "ver")
_N_COMPONENT = len(_FILE_COMPONENTS)
_SAMPLE_BYTES = 4 * _N_COMPONENT
# The station fields that become variables of the dataset, with the type each has there.
_STATION_VARIABLES = {
    "x": np.int32,
    "y": np.int32,
    "z": np.int32,
    "e_dist": np.float32,
    "hf_vs_ref": np.float32,
    "lf_vs_ref": np.float32,
    "vsite": np.float32,
}


def _station_dtype(endian: str) -> np.dtype:
    """Give the layout of a station's record.

    Parameters
    ----------
    endian : str
        The file's byte order: ``"<"`` for little-endian or ``">"`` for big-endian.

    Returns
    -------
    np.dtype
        The record, 0x2C bytes long, with no padding between its fields.
    """
    return np.dtype(
        [
            ("lon", f"{endian}f4"),
            ("lat", f"{endian}f4"),
            ("name", "S8"),
            ("x", f"{endian}i4"),
            ("y", f"{endian}i4"),
            ("z", f"{endian}i4"),
            ("e_dist", f"{endian}f4"),
            ("hf_vs_ref", f"{endian}f4"),
            ("lf_vs_ref", f"{endian}f4"),
            ("vsite", f"{endian}f4"),
        ]
    )


def _layout(head: bytes, size: int) -> tuple[str, int, int] | None:
    """Find a file's byte order, and how many stations and samples it holds.

    Parameters
    ----------
    head : bytes
        The start of the file: its header, or all of it if it is shorter.
    size : int
        The size of the whole file in bytes.

    Returns
    -------
    tuple[str, int, int] | None
        The byte order (``"<"`` for little-endian or ``">"`` for big-endian), the
        number of stations, and the number of samples of each trace. They are those
        that the header gives in the first byte order in which a file of that many
        stations and samples is exactly `size` bytes long. A BB.bin holds at least one
        station and at least one sample. None if no byte order gives that.
    """
    if len(head) < _HEAD_BYTES:
        return None
    for endian in ("<", ">"):
        # Python integers, so that a count that is wrong cannot overflow.
        nstat, nt = (int(n) for n in np.frombuffer(head, f"{endian}i4", count=2))
        expected = _HEAD_BYTES + nstat * (_STATION_BYTES + nt * _SAMPLE_BYTES)
        if nstat > 0 and nt > 0 and size == expected:
            return endian, nstat, nt
    return None


def read_bbbin(path: str | os.PathLike[str]) -> xr.Dataset:
    """Read a BB.bin file into a dataset in the standard layout.

    The whole file is read into memory. A BB.bin file is typically about 0.5 GiB, and
    reading takes about twice its size at the peak, since the waveform is copied into
    the standard order while the file's bytes are still held.

    The file's byte order is the first of little-endian and big-endian in which the
    header's numbers of stations and samples make the file exactly as long as it is.
    That is checked before the rest of the file is read. Every number in the dataset
    is in the machine's byte order.

    Parameters
    ----------
    path : str or os.PathLike
        The BB.bin file.

    Returns
    -------
    xr.Dataset
        - ``waveform``: float32 of dimensions (component, station, time), in g, with
          the attribute ``units = "g"``. The component coordinate holds ``000``,
          ``090`` and ``ver``, in that order, so the first two are swapped from the
          file's order.
        - the coordinates ``station`` (the names, as strings without their NULs),
          ``latitude`` and ``longitude`` (float32) along station, and ``time``
          (float64): ``start_sec + dt * arange(nt)``.
        - the variables ``x``, ``y`` and ``z`` (int32) and ``e_dist``, ``hf_vs_ref``,
          ``lf_vs_ref`` and ``vsite`` (float32), along station.
        - the attributes ``dt``, ``start_sec`` and ``duration`` (numbers), ``lf_dir``,
          ``lf_vm`` and ``hf_file`` (strings without their NULs), and
          ``source_format = "BB.bin"``.

    Raises
    ------
    NotBBBin
        If `path` is not a BB.bin file: if it is shorter than a header, if neither
        byte order makes the header's numbers of stations and samples, both at least
        1, fit its size exactly, or if a station's name, or a string of the header, is
        not ASCII.
    ValueError
        If the file shrinks while it is read.
    """
    path = Path(path)
    with path.open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        head = f.read(_HEAD_BYTES)
        layout = _layout(head, size)
        if layout is None:
            raise NotBBBin(f"not a BB.bin file: {path} ({size} bytes)", path)
        endian, nstat, nt = layout
        records = f.read(nstat * _STATION_BYTES)
        samples = f.read(nstat * nt * _SAMPLE_BYTES)
    stations = np.frombuffer(records, _station_dtype(endian), count=nstat)
    # Text that isn't ASCII isn't changed into something that might differ from what
    # the file holds.
    try:
        names = stations["name"].astype(str)
        lf_dir, lf_vm, hf_file = (
            np.frombuffer(head, "S256", count=3, offset=20).astype(str).tolist()
        )
    except UnicodeDecodeError as e:
        raise NotBBBin(
            f"not a BB.bin file: {path} holds text that isn't ASCII", path
        ) from e
    raw = np.frombuffer(samples, f"{endian}f4", count=nstat * nt * _N_COMPONENT)
    raw = raw.reshape(nstat, nt, _N_COMPONENT)
    waveform = np.empty((_N_COMPONENT, nstat, nt), dtype=np.float32)
    for i, component in enumerate(STANDARD_COMPONENTS):
        waveform[i] = raw[:, :, _FILE_COMPONENTS.index(component)]
    duration, dt, start_sec = (
        float(v) for v in np.frombuffer(head, f"{endian}f4", count=3, offset=8)
    )
    return xr.Dataset(
        {
            "waveform": (STANDARD_DIMS, waveform, {"units": "g"}),
            **{
                name: ("station", stations[name].astype(dtype))
                for name, dtype in _STATION_VARIABLES.items()
            },
        },
        coords={
            "component": list(STANDARD_COMPONENTS),
            "station": names,
            "latitude": ("station", stations["lat"].astype(np.float32)),
            "longitude": ("station", stations["lon"].astype(np.float32)),
            "time": start_sec + dt * np.arange(nt),
        },
        attrs={
            "dt": dt,
            "start_sec": start_sec,
            "duration": duration,
            "lf_dir": lf_dir,
            "lf_vm": lf_vm,
            "hf_file": hf_file,
            "source_format": "BB.bin",
        },
    )


def convert_bbbin(
    src: str | os.PathLike[str], dst: str | os.PathLike[str], overwrite: bool = False
) -> None:
    """Convert a BB.bin file to a netCDF file in the standard layout.

    The destination is checked first (see
    `waveform_compression_tools.destinations.refuse_overwriting`), and its folder. The
    dataset that `read_bbbin` gives is then written by xarray with the h5netcdf engine
    to `dst` with ``.partial`` added, and renamed to `dst` when it is complete. If
    anything goes wrong, the partial file is deleted and the exception raised again; a
    file already at `dst` is then as it was. As in `read_bbbin`, the whole file is read into memory.

    Parameters
    ----------
    src : str or os.PathLike
        The BB.bin file.
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
    NotBBBin
        If `src` is not a BB.bin file (see `read_bbbin`). Nothing is written.
    """
    refuse_overwriting(dst, src, overwrite=overwrite)
    require_folder(dst)
    write_netcdf(read_bbbin(src), dst)
