"""Convert a legacy HF.bin file to the standard netCDF layout.

An HF.bin file holds the high-frequency acceleration that a simulation computed at its
stations, in cm/s^2. This is the layout in which the simulation's HF.bin writer wrote
them, and in which qcore's former HFSeis reader read them, but for the names: that
reader cut them to 7 characters, where the writer wrote 8. Every number in the file is
in one byte order, the writer's, which `read_hfbin` tells from the file's size, as
for a BB.bin (see `waveform_compression_tools.legacy`):

- a header of 0x200 bytes: 16 int32, ``nstat`` (the number of stations), ``nt`` (of
  samples), ``seed``, ``siteamp``, ``pdur_model``, ``nrayset``, ``rayset1`` to
  ``rayset4``, ``nbu``, ``ift``, ``nlskip``, ``icflag``, ``same_seed`` and
  ``site_specific_vm``; 24 float32, ``duration``, ``dt``, ``start_sec``, ``sdrop``,
  ``kappa``, ``qfexp``, ``fmax``, ``flo``, ``fhi``, ``rvfac``, ``rvfac_shal``,
  ``rvfac_deep``, ``czero``, ``calpha``, ``mom``, ``rupv``, ``vs_moho``, ``vp_sig``,
  ``vsh_sig``, ``rho_sig``, ``qs_sig``, ``fa_sig1``, ``fa_sig2`` and ``rv_sig1``; two
  64-byte strings, ``stoch_file`` and ``velocity_model``, padded with NULs; then
  padding;
- from byte 0x200, one record of 0x18 bytes for each station: the float32 ``lon`` and
  ``lat``; the name, 8 bytes padded with NULs; and the float32 ``e_dist`` and ``vs``,
  which the writer fills in once the station is done, so that a ``vs`` of 0 means the
  run stopped first;
- then the acceleration in cm/s^2, float32 of shape (nstat, nt, 3), with the
  components in the order 090, 000, ver.

`convert_hfbin` writes a netCDF file in the standard layout (see
`waveform_compression_tools.codec.check_standard`) with all of that in it, which
`waveform_compression_tools.files.compress_file` can then compress.
"""

import io
import os
from pathlib import Path

import numpy as np
import xarray as xr

from . import legacy
from .codec import STANDARD_COMPONENTS, STANDARD_DIMS
from .destinations import check_destination, write_netcdf
from .errors import IncompleteHFBin, NotHFBin

# The size of a file's header, and of each station's record that follows it.
_HEAD_BYTES = 0x200
_STATION_BYTES = 0x18
# The header's fields, in the file's order: int32s from its start, float32s after them,
# and then strings of 64 bytes.
_INT_FIELDS = (
    "nstat",
    "nt",
    "seed",
    "siteamp",
    "pdur_model",
    "nrayset",
    "rayset1",
    "rayset2",
    "rayset3",
    "rayset4",
    "nbu",
    "ift",
    "nlskip",
    "icflag",
    "same_seed",
    "site_specific_vm",
)
_FLOAT_FIELDS = (
    "duration",
    "dt",
    "start_sec",
    "sdrop",
    "kappa",
    "qfexp",
    "fmax",
    "flo",
    "fhi",
    "rvfac",
    "rvfac_shal",
    "rvfac_deep",
    "czero",
    "calpha",
    "mom",
    "rupv",
    "vs_moho",
    "vp_sig",
    "vsh_sig",
    "rho_sig",
    "qs_sig",
    "fa_sig1",
    "fa_sig2",
    "rv_sig1",
)
_TEXT_FIELDS = ("stoch_file", "velocity_model")
_FLOATS_AT = 4 * len(_INT_FIELDS)
_TEXTS_AT = _FLOATS_AT + 4 * len(_FLOAT_FIELDS)


def _station_dtype(endian: str) -> np.dtype:
    """Give the layout of a station's record.

    Parameters
    ----------
    endian : str
        The file's byte order: ``"<"`` for little-endian or ``">"`` for big-endian.

    Returns
    -------
    np.dtype
        The record, 0x18 bytes long, with no padding between its fields.
    """
    return np.dtype(
        [
            ("lon", f"{endian}f4"),
            ("lat", f"{endian}f4"),
            ("name", "S8"),
            ("e_dist", f"{endian}f4"),
            ("vs", f"{endian}f4"),
        ]
    )


def _not_hfbin(path: Path, f: io.BufferedIOBase, head: bytes) -> NotHFBin:
    """Refuse a file whose size doesn't fit its header, as it is now.

    Parameters
    ----------
    path : Path
        The file.
    f : io.BufferedIOBase
        The file, open.
    head : bytes
        The start of the file: its header, or all of it if it is shorter.

    Returns
    -------
    NotHFBin
        The refusal, which names the file and gives its size now, which is less than
        it was found to be if it has shrunk since. It adds that the file may be one
        cut short if it holds the whole header and every record, but not every trace
        (see `waveform_compression_tools.legacy.cut_short`).
    """
    size = os.fstat(f.fileno()).st_size
    what = "not an HF.bin file"
    if legacy.cut_short(head, size, _HEAD_BYTES, _STATION_BYTES):
        what += ", or one cut short"
    return NotHFBin(f"{what}: {path} ({size} bytes)", path)


def read_hfbin(path: str | os.PathLike[str]) -> xr.Dataset:
    """Read an HF.bin file into a dataset in the standard layout.

    The whole waveform is read into memory. Its traces are read a block of stations at
    a time, about 64 MiB of them, each copied into the standard order as it is read
    (see `waveform_compression_tools.legacy.standard_waveform`), so reading takes about
    the file's size at the peak, and a block more.

    The file's byte order is the first of little-endian and big-endian in which the
    header's numbers of stations and samples make the file exactly as long as it is.
    That is checked before the rest of the file is read, and the text and every
    station's vs before the traces are. Every number in the dataset is in the
    machine's byte order.

    Parameters
    ----------
    path : str or os.PathLike
        The HF.bin file.

    Returns
    -------
    xr.Dataset
        - ``waveform``: float32 of dimensions (component, station, time), in cm/s^2 as
          the file holds it, with the attribute ``units = "cm/s^2"``. The component
          coordinate holds ``000``, ``090`` and ``ver``, in that order, so the first
          two are swapped from the file's order.
        - the coordinates ``station`` (the names, as strings without their NULs),
          ``latitude`` and ``longitude`` (float32) along station, and ``time``
          (float64): ``start_sec + dt * arange(nt)``.
        - the variables ``e_dist`` and ``vs`` (float32), along station.
        - an attribute for every field of the header, as the file holds it: the
          int32s as numpy int32, the float32s as numpy float32, and the strings as
          text without their NULs; and ``source_format = "HF.bin"``.

    Raises
    ------
    NotHFBin
        If `path` is not an HF.bin file: if it is shorter than a header, if neither
        byte order makes the header's numbers of stations and samples, both at least
        1, fit its size exactly, or if a station's name, or a string of the header, is
        not ASCII. If the file's size doesn't fit its header, but it holds the whole
        header and every record, the message adds that it may be one cut short. A
        file that shrinks while it is read, so that by the time its records or its
        traces are read it is shorter than its header says, is refused as a file of
        its new size would be.
    IncompleteHFBin
        If a station's vs is 0, so the run that wrote the file stopped before that
        station was done. The message says how many stations are unfinished.
    """
    path = Path(path)
    with path.open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        head = f.read(_HEAD_BYTES)
        layout = legacy.layout(head, size, _HEAD_BYTES, _STATION_BYTES)
        if layout is None:
            raise _not_hfbin(path, f, head)
        endian, nstat, nt = layout
        records = f.read(nstat * _STATION_BYTES)
        # A record or a trace that the file doesn't hold means that it has shrunk since
        # its size was checked.
        if len(records) < nstat * _STATION_BYTES:
            raise _not_hfbin(path, f, head)
        stations = np.frombuffer(records, _station_dtype(endian), count=nstat)
        # Text that isn't ASCII isn't changed into something that might differ from
        # what the file holds.
        try:
            names = stations["name"].astype(str)
            texts = np.frombuffer(
                head, "S64", count=len(_TEXT_FIELDS), offset=_TEXTS_AT
            ).astype(str)
        except UnicodeDecodeError as e:
            raise NotHFBin(
                f"not an HF.bin file: {path} holds text that isn't ASCII", path
            ) from e
        unfinished = int(np.count_nonzero(stations["vs"] == 0))
        if unfinished:
            raise IncompleteHFBin(
                f"{path} is incomplete, with {unfinished} of its {nstat} stations "
                "unfinished: their vs is 0",
                path,
            )
        # Only a file that can be converted has its traces read.
        waveform = legacy.standard_waveform(f, endian, nstat, nt)
        if waveform is None:
            raise _not_hfbin(path, f, head)
    ints = np.frombuffer(head, f"{endian}i4", count=len(_INT_FIELDS))
    floats = np.frombuffer(
        head, f"{endian}f4", count=len(_FLOAT_FIELDS), offset=_FLOATS_AT
    )
    header = (
        dict(zip(_INT_FIELDS, ints.astype(np.int32)))
        | dict(zip(_FLOAT_FIELDS, floats.astype(np.float32)))
        | dict(zip(_TEXT_FIELDS, texts.tolist()))
    )
    start_sec, dt = float(header["start_sec"]), float(header["dt"])
    return xr.Dataset(
        {
            "waveform": (STANDARD_DIMS, waveform, {"units": "cm/s^2"}),
            "e_dist": ("station", stations["e_dist"].astype(np.float32)),
            "vs": ("station", stations["vs"].astype(np.float32)),
        },
        coords={
            "component": list(STANDARD_COMPONENTS),
            "station": names,
            "latitude": ("station", stations["lat"].astype(np.float32)),
            "longitude": ("station", stations["lon"].astype(np.float32)),
            "time": start_sec + dt * np.arange(nt),
        },
        attrs={**header, "source_format": "HF.bin"},
    )


def convert_hfbin(
    src: str | os.PathLike[str], dst: str | os.PathLike[str], overwrite: bool = False
) -> None:
    """Convert an HF.bin file to a netCDF file in the standard layout.

    The destination and its folder are checked first (see
    `waveform_compression_tools.destinations.check_destination`). The dataset that
    `read_hfbin` gives is then written by xarray with the h5netcdf engine to `dst` with
    ``.partial`` added, and renamed to `dst` when it is complete. If anything goes
    wrong, the partial file is deleted and the exception raised again; a file already
    at `dst` is then as it was. As in `read_hfbin`, the whole waveform is read into
    memory, which takes about the file's size at the peak, and a block of 64 MiB more.

    Parameters
    ----------
    src : str or os.PathLike
        The HF.bin file.
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
    NotHFBin
        If `src` is not an HF.bin file (see `read_hfbin`). Nothing is written.
    IncompleteHFBin
        If a station of `src` is unfinished (see `read_hfbin`). Nothing is written.
    """
    check_destination(dst, src, overwrite=overwrite)
    write_netcdf(read_hfbin(src), dst)
