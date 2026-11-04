"""The command line: compress, decompress and verify waveform files, and convert BB.bin.

``waveform-compression-tools`` has four commands, each a thin layer over a function of this package:

- ``compress SRC DST``: `waveform_compression_tools.files.compress_file`;
- ``decompress SRC DST``: `waveform_compression_tools.files.decompress_file`, written to a netCDF
  file;
- ``verify ORIGINAL COMPRESSED``: `waveform_compression_tools.files.verify_file`;
- ``convert-bbbin SRC DST``: `waveform_compression_tools.bbbin.convert_bbbin`.

A command that succeeds prints a summary to standard output and exits with status 0.
One that cannot do its work, or whose check fails, prints a message to standard error
and exits with status 1, with nothing written at its destination. Arguments that
Typer refuses, as a file that does not exist or a ``--level`` out of range, exit with
status 2.
"""

import os
from pathlib import Path
from typing import Annotated, NoReturn

import pyflac
import typer
import xarray as xr

from . import bbbin, files

# Plain help and error text: the paragraphs of a command's help are reflowed, and the
# output is the same whether or not Rich is installed.
app = typer.Typer(
    help="Store float32 waveforms as FLAC, one stream per trace, and check them.",
    rich_markup_mode=None,
)

# What the functions behind the commands document raising for a file they can't use.
_BAD_FILE = (OSError, ValueError, TypeError, pyflac.DecoderProcessException)


def _fail(message: str) -> NoReturn:
    """Print an error message to standard error, and exit with status 1.

    Parameters
    ----------
    message : str
        What went wrong.

    Raises
    ------
    typer.Exit
        Always, with status 1.
    """
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _explain(error: Exception, path: Path) -> str:
    """Say what went wrong with a file, in words that name the file where they must.

    Parameters
    ----------
    error : Exception
        What the function behind a command raised for a file it can't use.
    path : Path
        The file the command read, which is the one a stream that doesn't decode is in.

    Returns
    -------
    str
        The error's message. For a stream that doesn't decode, whose own message names
        nothing, it says that a stream in `path` is damaged.
    """
    if isinstance(error, pyflac.DecoderProcessException):
        return f"a stream in {path} is damaged ({error})"
    return str(error)


def _count(n: int, noun: str) -> str:
    """Write a number of things, as ``1 station`` or ``2 stations``.

    Parameters
    ----------
    n : int
        The number.
    noun : str
        What there are `n` of, in the singular.

    Returns
    -------
    str
        `n` and `noun`, with an ``s`` added to the noun unless `n` is 1.
    """
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _size(nbytes: int) -> str:
    """Write a size in bytes in the largest binary unit that keeps it above 1.

    Parameters
    ----------
    nbytes : int
        The size in bytes.

    Returns
    -------
    str
        The size, as ``12 bytes``, or in KiB, MiB, GiB or TiB to one decimal place.
    """
    if nbytes < 1024:
        return f"{nbytes} bytes"
    value = float(nbytes)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if value < 1024:
            break
    return f"{value:.1f} {unit}"


def _factor(before: int, after: int) -> str:
    """Say how much smaller or larger a size has become, as a factor.

    Parameters
    ----------
    before : int
        The size before, in bytes.
    after : int
        The size after, in bytes, greater than 0.

    Returns
    -------
    str
        ``2.5x smaller`` for the factor `before` / `after` when `after` is no more than
        `before`, and ``1.2x larger`` for `after` / `before` when it is more.
    """
    if after <= before:
        return f"{before / after:.1f}x smaller"
    return f"{after / before:.1f}x larger"


def _refuse_overwriting(src: Path, dst: Path) -> None:
    """Exit unless writing a file to `dst` leaves `src` alone.

    A file is written to `dst` with ``.partial`` added, which is deleted if the write
    fails, and then renamed to `dst`. Either would destroy `src` if it is `dst`, or
    `dst` with ``.partial`` added.

    Parameters
    ----------
    src : Path
        The file the command reads.
    dst : Path
        The file it writes.

    Raises
    ------
    typer.Exit
        With status 1, if `src` is `dst` or `dst` with ``.partial`` added, by their
        resolved paths.
    """
    if src.resolve() in (dst.resolve(), Path(f"{dst}.partial").resolve()):
        _fail(f"writing {dst} would overwrite {src}")


def _write_netcdf(ds: xr.Dataset, dst: Path) -> None:
    """Write a dataset to a netCDF file that appears at `dst` only when it is complete.

    It is written by xarray with the h5netcdf engine to `dst` with ``.partial`` added,
    and renamed to `dst` when it is complete. If anything goes wrong, the partial file
    is deleted and the exception raised again; a file already at `dst` is then as it
    was.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to write.
    dst : Path
        Where to write the file. A file already there is replaced.
    """
    partial = Path(f"{dst}.partial")
    try:
        ds.to_netcdf(partial, engine="h5netcdf")
        os.replace(partial, dst)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


@app.command()
def compress(
    src: Annotated[
        Path,
        typer.Argument(
            metavar="SRC",
            exists=True,
            dir_okay=False,
            help="The netCDF file to compress, in the standard layout.",
        ),
    ],
    dst: Annotated[
        Path,
        typer.Argument(
            metavar="DST",
            dir_okay=False,
            help="Where to write the compressed file. A file already there is replaced.",
        ),
    ],
    bits: Annotated[
        int,
        typer.Option(help="Bits of quantisation, from 8 to 32. 24 is FLAC-24."),
    ] = 32,
    level: Annotated[
        int,
        typer.Option(
            min=0,
            max=8,
            help="FLAC compression level, from 0 (fastest) to 8 (smallest).",
        ),
    ] = 5,
) -> None:
    """Compress a netCDF file of waveforms to FLAC.

    Each trace is scaled so that its largest sample is full scale for BITS, rounded to
    an integer and written as one FLAC stream. Every decoded sample is then within one
    quantisation step of the original, which is 1 / (2**(BITS - 1) - 1) of its trace's
    peak: 4.7e-10 at 32 bits and 1.2e-7 at 24. The waveform is read, encoded and
    written a block of stations at a time, so memory holds one block, however big the
    file is.

    A station whose name is not exactly 7 characters long is real, and its traces are
    also kept exactly as they were. A file whose stations are all real is not
    compressed: the command says so, exits with status 1 and writes nothing, so the
    file stays as it is.

    The compressed file appears at DST only when it is complete.
    \f
    Parameters
    ----------
    src : Path
        The netCDF file to compress, in the standard layout.
    dst : Path
        Where to write the compressed file. A file already there is replaced.
    bits : int, default 32
        Bits of quantisation, from 8 to 32.
    level : int, default 5
        FLAC compression level, from 0 (fastest) to 8 (smallest).

    Raises
    ------
    typer.Exit
        With status 1, if the file can't be compressed.
    """
    try:
        result = files.compress_file(src, dst, bits, level)
    except files.AllStationsReal:
        _fail(
            f"every station of {src} is real, so it isn't compressed: it stays as it "
            f"is, and nothing is written to {dst}"
        )
    except _BAD_FILE as e:
        _fail(_explain(e, src))
    typer.echo(
        f"compressed {src} to {dst} at {result.bits} bits: "
        f"{_count(result.n_traces, 'trace')}, "
        f"{_count(result.n_exact, 'real station')} also kept exactly"
    )
    typer.echo(
        f"{_size(result.bytes_in)} -> {_size(result.bytes_out)}, "
        f"{_factor(result.bytes_in, result.bytes_out)} "
        f"(read {result.read_s:.1f} s, encode {result.encode_s:.1f} s, "
        f"write {result.write_s:.1f} s)"
    )


@app.command()
def decompress(
    src: Annotated[
        Path,
        typer.Argument(
            metavar="SRC",
            exists=True,
            dir_okay=False,
            help="The compressed file, written by compress.",
        ),
    ],
    dst: Annotated[
        Path,
        typer.Argument(
            metavar="DST",
            dir_okay=False,
            help="Where to write the netCDF file. A file already there is replaced.",
        ),
    ],
) -> None:
    """Decompress a file written by compress.

    DST is a netCDF file in the standard layout: every variable, coordinate and
    attribute of the original, and its waveform as float32 with the dimensions
    (component, station, time). The real stations' traces are exactly as they were;
    every other sample is within one quantisation step of the original.

    The whole waveform is decoded into memory, so this command suits files whose
    waveform fits there: it needs 4 bytes for each sample, plus a block of 128 MiB
    and some working space. The file appears at DST only when it is complete.
    \f
    Parameters
    ----------
    src : Path
        The compressed file, written by compress.
    dst : Path
        Where to write the netCDF file. A file already there is replaced.

    Raises
    ------
    typer.Exit
        With status 1, if SRC is not a compressed file, is damaged, or can't be
        decoded or written, or if writing DST would overwrite SRC.
    """
    _refuse_overwriting(src, dst)
    try:
        ds = files.decompress_file(src)
        _write_netcdf(ds, dst)
    except _BAD_FILE as e:
        _fail(_explain(e, src))
    n_component, n_station, nt = ds["waveform"].shape
    typer.echo(
        f"decompressed {src} to {dst}: "
        f"{_count(n_component * n_station, 'trace')} of {nt} samples, "
        f"{_size(dst.stat().st_size)}"
    )


@app.command()
def verify(
    original: Annotated[
        Path,
        typer.Argument(
            metavar="ORIGINAL",
            exists=True,
            dir_okay=False,
            help="The netCDF file that was compressed.",
        ),
    ],
    compressed: Annotated[
        Path,
        typer.Argument(
            metavar="COMPRESSED",
            exists=True,
            dir_okay=False,
            help="The file that compress made from it.",
        ),
    ],
) -> None:
    """Check a compressed file against its original.

    The compressed file is decoded a block of stations at a time and compared with the
    original. It passes if every trace decodes within the bound its setting promises
    and within one quantisation step of the original, the real stations come back bit
    for bit, and everything else in the file, the waveform's attributes included, is
    identical to the original's.

    Exits with status 0 if the file passes and 1 if it fails or can't be checked.
    \f
    Parameters
    ----------
    original : Path
        The netCDF file that was compressed.
    compressed : Path
        The file that compress made from it.

    Raises
    ------
    typer.Exit
        With status 1, if the file fails the check or can't be checked.
    """
    try:
        result = files.verify_file(original, compressed)
    except _BAD_FILE as e:
        _fail(_explain(e, compressed))
    error = f"largest error {result.max_relative_error:.2g} of a trace's peak"
    if not result.passed:
        typer.echo(
            f"FAILED: {compressed} does not agree with {original} ({error})", err=True
        )
        raise typer.Exit(1)
    typer.echo(
        f"verified: {compressed} agrees with {original} ({error}; "
        f"{_count(result.n_exact, 'real station')} exact; "
        f"checked in {result.verify_s:.1f} s)"
    )


@app.command()
def convert_bbbin(
    src: Annotated[
        Path,
        typer.Argument(
            metavar="SRC", exists=True, dir_okay=False, help="The BB.bin file."
        ),
    ],
    dst: Annotated[
        Path,
        typer.Argument(
            metavar="DST",
            dir_okay=False,
            help="Where to write the netCDF file. A file already there is replaced.",
        ),
    ],
) -> None:
    """Convert a BB.bin file to a netCDF file.

    BB.bin is the legacy layout of a simulation's broadband waveforms. DST is in the
    standard layout, which the compress command takes. The whole BB.bin file is read
    into memory, which takes about twice its size at the peak. The file appears at DST
    only when it is complete.
    \f
    Parameters
    ----------
    src : Path
        The BB.bin file.
    dst : Path
        Where to write the netCDF file. A file already there is replaced.

    Raises
    ------
    typer.Exit
        With status 1, if SRC is not a BB.bin file or can't be converted, or if writing
        DST would overwrite SRC.
    """
    _refuse_overwriting(src, dst)
    try:
        bbbin.convert_bbbin(src, dst)
    except _BAD_FILE as e:
        _fail(_explain(e, src))
    typer.echo(
        f"converted {src} to {dst}: "
        f"{_size(src.stat().st_size)} -> {_size(dst.stat().st_size)}"
    )
