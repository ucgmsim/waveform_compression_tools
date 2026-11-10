"""The command line: compress, decompress and verify waveform files, and convert legacy ones.

``waveform-compression-tools`` has five commands, each a thin layer over this package:

- ``compress SRC DST``: `waveform_compression_tools.files.compress_file`, which checks
  the file it writes, with a JSON record of the run if asked (see
  `waveform_compression_tools.record`);
- ``decompress SRC DST``: `waveform_compression_tools.files.decompress_file`, written
  to a netCDF file;
- ``verify ORIGINAL COMPRESSED``: `waveform_compression_tools.files.verify_file`;
- ``convert-bbbin SRC DST``: `waveform_compression_tools.bbbin.convert_bbbin`;
- ``convert-hfbin SRC DST``: `waveform_compression_tools.hfbin.convert_hfbin`.

Every command exits with one of four statuses, which are published: `EXIT_OK` when it
succeeds; `EXIT_FAILED` when the run goes wrong, as when a file can't be read or
written, a check fails, or something unexpected happens; `EXIT_USAGE` for arguments
that Typer refuses; and `EXIT_REFUSED` for an input that can't be used as it is (see
`waveform_compression_tools.errors.Refused`). A command that succeeds prints a summary
to standard output. One that fails or refuses prints exactly one line to standard
error, ``waveform-compression-tools: <path>: <reason>``, where the path names the file
the problem is about and the reason doesn't name it again at its start, and leaves
nothing at its destination; only a compressed file whose record can't be written
stays. A file already at a destination, compress's record included, is replaced only
with ``--overwrite``.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, NoReturn

import pyflac
import typer

from . import bbbin, destinations, files, hfbin
from .content_hash import content_hash
from .errors import CheckFailed, Refused, SameFile
from .record import make_record, write_record

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

# How every error line starts.
_PROGRAM = "waveform-compression-tools"
# What the functions behind the commands document raising when an input can't be used
# or a run goes wrong. Anything else is unexpected, and its line gives its type.
_EXPECTED = (OSError, ValueError, CheckFailed)
# libFLAC's errors as it encodes a trace.
_ENCODING = (pyflac.EncoderInitException, pyflac.EncoderProcessException)

# Plain help and error text: the paragraphs of a command's help are reflowed, and the
# output is the same whether or not Rich is installed.
app = typer.Typer(
    help="Store float32 waveforms as FLAC, one stream per trace, and check them.",
    rich_markup_mode=None,
)

# The destination of decompress and the converters, and their --overwrite.
_NetcdfDst = Annotated[
    Path,
    typer.Argument(
        metavar="DST", dir_okay=False, help="Where to write the netCDF file."
    ),
]
_Overwrite = Annotated[
    bool, typer.Option("--overwrite", help="Replace a file already at DST.")
]


@app.callback()
def _quiet_pyflac() -> None:
    """Send pyflac's log nowhere, before any command runs.

    pyflac logs each error that libFLAC reports while it decodes, and the exception that
    follows says it again. A command says what went wrong in its one line. This is done
    when the app runs, not when this module is imported, so that a program that only
    imports it keeps its logging as it was.
    """
    log = logging.getLogger("pyflac")
    # A handler of its own, so that Python's last-resort handler doesn't print a record
    # that no other handler sees.
    if not any(isinstance(handler, logging.NullHandler) for handler in log.handlers):
        log.addHandler(logging.NullHandler())
    log.propagate = False


def _reason(error: Exception) -> str:
    """Say what went wrong, in words for the error line and the record.

    Parameters
    ----------
    error : Exception
        What a command's step raised.

    Returns
    -------
    str
        The error's message, which is never empty. libFLAC's errors as it encodes say
        that a trace couldn't be encoded. An unexpected error, or one without a
        message, is named by its type.
    """
    message = str(error)
    if isinstance(error, _ENCODING):
        return f"a trace couldn't be encoded ({message})"
    if isinstance(error, _EXPECTED) and message:
        return message
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _say(path: Path, reason: str) -> None:
    """Print a command's one error line to standard error.

    The line names the file once: a reason that starts with ``"<path>: "`` or
    ``"<path> "``, as many of this package's messages do, loses that start.

    Parameters
    ----------
    path : Path
        The file the problem is about.
    reason : str
        What went wrong.
    """
    for start in (f"{path}: ", f"{path} "):
        if reason.startswith(start) and len(reason) > len(start):
            reason = reason[len(start) :]
            break
    line = f"{_PROGRAM}: {path}: {reason}"
    # A message of several lines, as some of xarray's are, is joined into one.
    typer.echo(" ".join(line.splitlines()), err=True)


class _Stop(Exception):  # noqa: N818
    """Why a command stopped short of its work.

    Parameters
    ----------
    path : Path
        The file the problem is about: the input, the output or the record.
    error : Exception
        What went wrong.

    Attributes
    ----------
    path : Path
        The file the problem is about.
    reason : str
        What went wrong, in words (see `_reason`).
    status : str
        ``"refused"`` for a refusal of the input, and ``"failed"`` otherwise.
    code : int
        The status to exit with: `EXIT_REFUSED` or `EXIT_FAILED`.
    """

    def __init__(self, path: Path, error: Exception) -> None:  # numpydoc ignore=GL08
        super().__init__(path, error)
        refused = isinstance(error, Refused)
        self.path = path
        self.reason = _reason(error)
        self.status = "refused" if refused else "failed"
        self.code = EXIT_REFUSED if refused else EXIT_FAILED

    def exit(self) -> NoReturn:
        """Print the error line, and exit.

        Raises
        ------
        typer.Exit
            Always, with `code`.
        """
        _say(self.path, self.reason)
        raise typer.Exit(self.code)


@contextmanager
def _about(path: Path) -> Generator[None]:
    """Turn any error in a block into a `_Stop` that names the file it is about.

    An error that says which file it is about, as an `OSError`, a refusal or a failed
    check does, names that file instead.

    Parameters
    ----------
    path : Path
        The file an error in the block is about, unless it says otherwise.

    Yields
    ------
    None
        Nothing; the block runs.

    Raises
    ------
    _Stop
        For any exception the block raises. `KeyboardInterrupt` and the like are left
        alone.
    """
    try:
        yield
    # Every error becomes one line, never a traceback.
    except Exception as error:
        named = destinations.named_file(error)
        raise _Stop(path if named is None else Path(named), error) from error


def _write_compress_record(
    path: Path,
    *,
    src: Path,
    dst: Path,
    bits: int,
    level: int,
    start: float,
    hash_s: float,
    input_hash: str | None,
    outcome: files.CompressResult | _Stop,
) -> None:
    """Write the record of a compress run, or exit if it can't be written.

    Parameters
    ----------
    path : Path
        Where the record goes.
    src : Path
        The input.
    dst : Path
        Where the compressed file goes.
    bits : int
        The setting asked for.
    level : int
        The FLAC compression level asked for.
    start : float
        When the command started, by `time.perf_counter`.
    hash_s : float
        The time spent hashing the input, in seconds.
    input_hash : str or None
        The input's content hash, if the run got as far as hashing it.
    outcome : CompressResult or _Stop
        What compressing made, or why the run stopped short.

    Raises
    ------
    typer.Exit
        With `EXIT_FAILED`, after the error line, if the record can't be written. A
        compressed file stays where it is.
    """
    total_s = time.perf_counter() - start
    if isinstance(outcome, _Stop):
        result, failure = None, (outcome.status, outcome.reason)
    else:
        result, failure = outcome, None
    try:
        record = make_record(
            input_path=src,
            input_hash=input_hash,
            bits=bits,
            level=level,
            result=result,
            output_path=dst,
            failure=failure,
            hash_s=hash_s,
            total_s=total_s,
        )
        write_record(record, path)
    # Every error becomes one line, never a traceback.
    except Exception as error:
        _say(path, f"the record can't be written: {_reason(error)}")
        raise typer.Exit(EXIT_FAILED) from error


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
            help="Where to write the compressed file.",
        ),
    ],
    bits: Annotated[
        int,
        typer.Option(
            min=8, max=32, help="Bits of quantisation, from 8 to 32. 24 is FLAC-24."
        ),
    ] = 32,
    level: Annotated[
        int,
        typer.Option(
            min=0,
            max=8,
            help="FLAC compression level, from 0 (fastest) to 8 (smallest).",
        ),
    ] = 5,
    no_check: Annotated[
        bool,
        typer.Option(
            "--no-check",
            help="Don't check the compressed file against the original as it is "
            "written.",
        ),
    ] = False,
    record: Annotated[
        Path | None,
        typer.Option(
            metavar="PATH",
            dir_okay=False,
            help="Write a JSON record of the run to PATH, whatever its outcome. A "
            "file already there is replaced only with --overwrite.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace a file already at DST, or at the record's PATH.",
        ),
    ] = False,
) -> None:
    """Compress a netCDF file of waveforms to FLAC, and check it.

    Each trace is scaled so that its largest sample is full scale for BITS, rounded to
    an integer and written as one FLAC stream. Every decoded sample is then within one
    quantisation step of the original, which is 1 / (2**(BITS - 1) - 1) of its trace's
    peak: 4.7e-10 at 32 bits and 1.2e-7 at 24. The waveform is read, encoded and
    written a block of stations at a time, so memory holds one block, however big the
    file is.

    A station whose name is not exactly 7 characters long is real, and its traces are
    also kept exactly as they were. A file whose stations are all real is not
    compressed: it is refused, and stays as it is.

    The compressed file is checked as it is written, against the samples read, unless
    --no-check is given. It appears at DST only when it is complete, and has passed
    the check if one is made. A file already at DST, or at the record's PATH, is
    replaced only with --overwrite.

    With --record, a JSON record of the run is written to PATH whether the file is
    compressed, refused or fails: the input's and the output's paths, sizes and content
    hashes, the setting, the check's largest error, and the time spent. A file already
    at PATH is refused first, before the input is read, unless --overwrite is given;
    that refusal writes no record, so the earlier one stays as it was. If the record
    can't be written, the run fails, though a compressed file stays at DST.

    The command exits with status 0 when the file is compressed, 1 when the run fails,
    2 for a usage error, and 3 when the file is refused. A refusal or a failure is one
    line on standard error.
    \f
    Parameters
    ----------
    src : Path
        The netCDF file to compress, in the standard layout.
    dst : Path
        Where to write the compressed file.
    bits : int, default 32
        Bits of quantisation, from 8 to 32.
    level : int, default 5
        FLAC compression level, from 0 (fastest) to 8 (smallest).
    no_check : bool, default False
        Whether to leave out the check.
    record : Path, optional
        Where to write the record of the run.
    overwrite : bool, default False
        Whether to replace a file already at `dst`, or at `record`.

    Raises
    ------
    typer.Exit
        With `EXIT_REFUSED` if the file, or a file already at `record`, is refused,
        and `EXIT_FAILED` if the run fails or the record can't be written.
    """
    start = time.perf_counter()
    if record is not None:
        # The record is a destination too, checked before the input is read. One that
        # would be written over the input or the output is a usage error. Refusing it,
        # or finding no folder for it, writes no record, so an earlier one, which may
        # say that DST is complete, stays as it was.
        try:
            with _about(record):
                destinations.check_destination(record, src, dst, overwrite=overwrite)
        except _Stop as stop:
            if isinstance(stop.__cause__, SameFile):
                raise typer.BadParameter(
                    stop.reason, param_hint="'--record'"
                ) from stop.__cause__
            stop.exit()
    hash_s = 0.0
    input_hash: str | None = None
    outcome: files.CompressResult | _Stop
    try:
        with _about(src):
            files.check_compressible(src, dst, overwrite)
            # Only the record uses the hash. It comes after the checks, so that a
            # refused input or a mistyped folder costs no hash.
            if record is not None:
                hashing = time.perf_counter()
                try:
                    input_hash = content_hash(src)
                finally:
                    hash_s = time.perf_counter() - hashing
            outcome = files.compress_file(
                src, dst, bits, level, check=not no_check, overwrite=overwrite
            )
    except _Stop as stop:
        outcome = stop
    if record is not None:
        _write_compress_record(
            record,
            src=src,
            dst=dst,
            bits=bits,
            level=level,
            start=start,
            hash_s=hash_s,
            input_hash=input_hash,
            outcome=outcome,
        )
    if isinstance(outcome, _Stop):
        outcome.exit()
    typer.echo(
        f"compressed {src} to {dst} at {outcome.bits} bits: "
        f"{_count(outcome.n_traces, 'trace')}, "
        f"{_count(outcome.n_exact, 'real station')} also kept exactly"
    )
    typer.echo(
        f"{_size(outcome.bytes_in)} -> {_size(outcome.bytes_out)}, "
        f"{_factor(outcome.bytes_in, outcome.bytes_out)} "
        f"(read {outcome.read_s:.1f} s, encode {outcome.encode_s:.1f} s, "
        f"write {outcome.write_s:.1f} s, check {outcome.check_s:.1f} s)"
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
    dst: _NetcdfDst,
    overwrite: _Overwrite = False,
) -> None:
    """Decompress a file written by compress.

    DST is a netCDF file in the standard layout: every variable, coordinate and
    attribute of the original, and its waveform as float32 with the dimensions
    (component, station, time). The real stations' traces are exactly as they were;
    every other sample is within one quantisation step of the original.

    The whole waveform is decoded into memory, so this command suits files whose
    waveform fits there: it needs 4 bytes for each sample, and some working space.
    The file appears at DST only when it is complete. A file already at DST is
    replaced only with --overwrite.

    The command exits with status 0 when the file is decompressed, 1 when the run
    fails, as for a damaged file, 2 for a usage error, and 3 when SRC is refused, as a
    file that isn't compressed is. A refusal or a failure is one line on standard
    error.
    \f
    Parameters
    ----------
    src : Path
        The compressed file, written by compress.
    dst : Path
        Where to write the netCDF file.
    overwrite : bool, default False
        Whether to replace a file already at `dst`.

    Raises
    ------
    typer.Exit
        With `EXIT_REFUSED` if SRC isn't a compressed file that can be read, or if DST
        is SRC or is there already, and with `EXIT_FAILED` if the run fails.
    """
    try:
        with _about(dst):
            destinations.check_destination(dst, src, overwrite=overwrite)
        with _about(src):
            ds = files.decompress_file(src)
        with _about(dst):
            destinations.write_netcdf(ds, dst)
    except _Stop as stop:
        stop.exit()
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

    The command exits with status 0 if the file passes, 1 if it fails or can't be
    checked, 2 for a usage error, and 3 if either file is refused, as an original that
    isn't in the standard layout is, or a compressed file that isn't one. A refusal or
    a failure is one line on standard error.
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
        With `EXIT_REFUSED` if either file is refused, and `EXIT_FAILED` if the file
        fails the check or can't be checked.
    """
    try:
        with _about(compressed):
            result = files.verify_file(original, compressed)
    except _Stop as stop:
        stop.exit()
    error = f"largest error {result.max_relative_error:.2g} of a trace's peak"
    if not result.passed:
        _say(compressed, f"doesn't agree with {original} ({error})")
        raise typer.Exit(EXIT_FAILED)
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
    dst: _NetcdfDst,
    overwrite: _Overwrite = False,
) -> None:
    """Convert a BB.bin file to a netCDF file.

    BB.bin is the legacy layout of a simulation's broadband waveforms. DST is in the
    standard layout, which the compress command takes. The whole waveform is read into
    memory, a block of stations at a time, which takes about the BB.bin's size at the
    peak, and 64 MiB more. The file appears at DST only when it is complete. A file
    already at DST is replaced only with --overwrite.

    The command exits with status 0 when the file is converted, 1 when the run fails,
    2 for a usage error, and 3 when SRC is refused, as a file that can't be a BB.bin
    is. A refusal or a failure is one line on standard error.
    \f
    Parameters
    ----------
    src : Path
        The BB.bin file.
    dst : Path
        Where to write the netCDF file.
    overwrite : bool, default False
        Whether to replace a file already at `dst`.

    Raises
    ------
    typer.Exit
        With `EXIT_REFUSED` if SRC is not a BB.bin file, or if DST is SRC or is there
        already, and with `EXIT_FAILED` if the run fails.
    """
    try:
        with _about(dst):
            bbbin.convert_bbbin(src, dst, overwrite)
    except _Stop as stop:
        stop.exit()
    typer.echo(
        f"converted {src} to {dst}: "
        f"{_size(src.stat().st_size)} -> {_size(dst.stat().st_size)}"
    )


@app.command()
def convert_hfbin(
    src: Annotated[
        Path,
        typer.Argument(
            metavar="SRC", exists=True, dir_okay=False, help="The HF.bin file."
        ),
    ],
    dst: _NetcdfDst,
    overwrite: _Overwrite = False,
) -> None:
    """Convert an HF.bin file to a netCDF file.

    HF.bin is the legacy layout of a simulation's high-frequency waveforms. DST is in
    the standard layout, which the compress command takes, with the acceleration in
    cm/s^2, as the HF.bin holds it. The whole waveform is read into memory, a block of
    stations at a time, which takes about the HF.bin's size at the peak, and 64 MiB
    more. The file appears at DST only when it is complete. A file already at DST is
    replaced only with --overwrite.

    An HF.bin in which a station's vs is 0 is unfinished, since the run that wrote it
    stopped before that station was done. It is refused, and the error line says how
    many stations are unfinished.

    The command exits with status 0 when the file is converted, 1 when the run fails,
    2 for a usage error, and 3 when SRC is refused, as a file that can't be an HF.bin
    is, or one that is unfinished. A refusal or a failure is one line on standard
    error.
    \f
    Parameters
    ----------
    src : Path
        The HF.bin file.
    dst : Path
        Where to write the netCDF file.
    overwrite : bool, default False
        Whether to replace a file already at `dst`.

    Raises
    ------
    typer.Exit
        With `EXIT_REFUSED` if SRC is not an HF.bin file or is unfinished, or if DST is
        SRC or is there already, and with `EXIT_FAILED` if the run fails.
    """
    try:
        with _about(dst):
            hfbin.convert_hfbin(src, dst, overwrite)
    except _Stop as stop:
        stop.exit()
    typer.echo(
        f"converted {src} to {dst}: "
        f"{_size(src.stat().st_size)} -> {_size(dst.stat().st_size)}"
    )
