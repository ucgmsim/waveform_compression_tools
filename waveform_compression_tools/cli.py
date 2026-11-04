"""The command line: compress, decompress and verify waveform files, and convert BB.bin.

``waveform-compression-tools`` has four commands, each a thin layer over this package:

- ``compress SRC DST``: `waveform_compression_tools.files.compress_file`, which checks
  the file it writes, with a JSON record of the run if asked (see
  `waveform_compression_tools.record`);
- ``decompress SRC DST``: `waveform_compression_tools.files.decompress_file`, written
  to a netCDF file;
- ``verify ORIGINAL COMPRESSED``: `waveform_compression_tools.files.verify_file`;
- ``convert-bbbin SRC DST``: `waveform_compression_tools.bbbin.convert_bbbin`.

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
import os
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, NoReturn

import pyflac
import typer

from . import bbbin, destinations, files
from .content_hash import content_hash
from .errors import (
    CheckFailed,
    NotBBBin,
    NotStandardLayout,
    OutputExists,
    Refused,
    SameFile,
)
from .record import make_record, write_record

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

# How every error line starts.
_PROGRAM = "waveform-compression-tools"
# The errors that are about a command's output, whichever of its steps raises them.
_ABOUT_THE_OUTPUT = (SameFile, OutputExists, CheckFailed)
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
        The error's message, which is never empty. libFLAC's errors as it decodes say
        that a stream is damaged, and those as it encodes that a trace couldn't be
        encoded. An unexpected error, or one without a message, is named by its type.
    """
    message = str(error)
    if isinstance(error, pyflac.DecoderProcessException):
        return f"a FLAC stream is damaged ({message})"
    if isinstance(error, _ENCODING):
        return f"a trace couldn't be encoded ({message})"
    if isinstance(error, _EXPECTED) and message:
        return message
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _say(path: Path, reason: str) -> None:
    """Print a command's one error line to standard error.

    The line names the file once: a reason that starts with ``"<path>: "`` or
    ``"<path> "``, as many of this package's messages do, loses that start, and so
    does one that starts so with the file's partial name, as a failed check's does.

    Parameters
    ----------
    path : Path
        The file the problem is about.
    reason : str
        What went wrong.
    """
    for start in (f"{path}: ", f"{path} ", f"{path}.partial: ", f"{path}.partial "):
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
def _about(
    path: Path,
    others: Mapping[type[Exception] | tuple[type[Exception], ...], Path] | None = None,
) -> Generator[None]:
    """Turn any error in a block into a `_Stop` that names the file it is about.

    An `OSError` that says which file it is about names that file instead.

    Parameters
    ----------
    path : Path
        The file an error in the block is about.
    others : Mapping, optional
        Errors that are about another file instead: each kind of error, or a tuple of
        kinds, and the file.

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
        for kinds, other in (others or {}).items():
            if isinstance(error, kinds):
                path = other
                break
        if isinstance(error, OSError) and isinstance(
            error.filename, str | bytes | os.PathLike
        ):
            path = Path(os.fsdecode(error.filename))
        raise _Stop(path, error) from error


def _require_folder(path: Path) -> None:
    """Fail unless the folder that a file is to be written in exists.

    Parameters
    ----------
    path : Path
        The file.

    Raises
    ------
    FileNotFoundError
        If there is no folder at ``path.parent``.
    """
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"there is no folder {path.parent} to write {path.name} in"
        )


def _refuse_a_record_over(record: Path, src: Path, dst: Path) -> None:
    """Refuse a record that would be written over the input or the output.

    Parameters
    ----------
    record : Path
        Where the record is to go.
    src : Path
        The input.
    dst : Path
        The output.

    Raises
    ------
    typer.BadParameter
        A usage error, if `record`, or `record` with ``.partial`` added, is `src` or
        `dst` (see `waveform_compression_tools.destinations.refuse_overwriting`).
    """
    for path in (src, dst):
        try:
            destinations.refuse_overwriting(path, record, overwrite=True)
        except SameFile as error:
            raise typer.BadParameter(str(error), param_hint="'--record'") from error


def _hash(path: Path, seconds: dict[str, float]) -> str:
    """Hash a file's content, and count the time it takes.

    Parameters
    ----------
    path : Path
        The file.
    seconds : dict[str, float]
        The run's timings: the time is added to ``"hash"``.

    Returns
    -------
    str
        The file's content hash (see `waveform_compression_tools.content_hash`).
    """
    start = time.perf_counter()
    try:
        return content_hash(path)
    finally:
        seconds["hash"] += time.perf_counter() - start


def _hash_output(dst: Path, seconds: dict[str, float]) -> str:
    """Hash the compressed file, or delete it if hashing fails or is interrupted.

    A run whose output isn't hashed doesn't succeed, and a run that doesn't succeed
    leaves nothing at its destination.

    Parameters
    ----------
    dst : Path
        The compressed file.
    seconds : dict[str, float]
        The run's timings: the time is added to ``"hash"``.

    Returns
    -------
    str
        The file's content hash.
    """
    try:
        return _hash(dst, seconds)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise


def _write_compress_record(
    path: Path,
    *,
    src: Path,
    bits: int,
    start: float,
    seconds: dict[str, float],
    input_hash: str | None,
    result: files.CompressResult | None,
    output: tuple[Path, int, str] | None,
    stop: _Stop | None,
) -> None:
    """Write the record of a compress run, or exit if it can't be written.

    Parameters
    ----------
    path : Path
        Where the record goes.
    src : Path
        The input.
    bits : int
        The setting asked for.
    start : float
        When the command started, by `time.perf_counter`.
    seconds : dict[str, float]
        The run's timings so far: the time spent hashing, as ``"hash"``, and 0.0 for
        ``"compress"`` and ``"check"``, which come from `result`.
    input_hash : str or None
        The input's content hash, if the run got as far as hashing it.
    result : CompressResult or None
        What compressing made, if the run got that far.
    output : tuple[Path, int, str] or None
        The compressed file's path, size and content hash, if the run succeeded.
    stop : _Stop or None
        Why the run stopped short, if it did.

    Raises
    ------
    typer.Exit
        With `EXIT_FAILED`, after the error line, if the record can't be written. A
        compressed file stays where it is.
    """
    seconds = {**seconds, "total": time.perf_counter() - start}
    if result is not None:
        seconds["compress"] = result.read_s + result.encode_s + result.write_s
        seconds["check"] = result.check_s
    worst = None if result is None else result.max_relative_error
    output_path, output_bytes, output_hash = output or (None, None, None)
    try:
        # make_record takes plain Python values only.
        record = make_record(
            status="ok" if stop is None else stop.status,
            reason="" if stop is None else stop.reason,
            input_path=src,
            input_bytes=int(src.stat().st_size),
            input_hash=input_hash,
            output_path=output_path,
            output_bytes=None if output_bytes is None else int(output_bytes),
            output_hash=output_hash,
            bits=int(bits),
            checked=result is not None and bool(result.checked),
            max_relative_error=None if worst is None else float(worst),
            n_traces=None if result is None else int(result.n_traces),
            n_exact_stations=None if result is None else int(result.n_exact),
            seconds={name: float(value) for name, value in seconds.items()},
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
        _refuse_a_record_over(record, src, dst)
        # The record is a destination too, checked before the input is read. Refusing it
        # writes no record, so an earlier one, which may say that DST is complete, stays
        # as it was.
        try:
            with _about(record):
                destinations.refuse_overwriting(src, record, overwrite)
        except _Stop as stop:
            stop.exit()
    seconds = {"hash": 0.0, "compress": 0.0, "check": 0.0}
    input_hash: str | None = None
    result: files.CompressResult | None = None
    try:
        with _about(src, {_ABOUT_THE_OUTPUT: dst}):
            files.check_compressible(src, dst, overwrite)
        # Before the input is hashed, so that a mistyped folder costs no hash.
        for path in (dst,) if record is None else (dst, record):
            with _about(path):
                _require_folder(path)
        with _about(src):
            input_hash = _hash(src, seconds)
        with _about(src, {_ABOUT_THE_OUTPUT: dst}):
            result = files.compress_file(
                src, dst, bits, level, check=not no_check, overwrite=overwrite
            )
        with _about(dst):
            output_hash = _hash_output(dst, seconds)
    except _Stop as stop:
        # Unless it is the record's own folder that is missing.
        if record is not None and stop.path != record:
            _write_compress_record(
                record,
                src=src,
                bits=bits,
                start=start,
                seconds=seconds,
                input_hash=input_hash,
                result=result,
                output=None,
                stop=stop,
            )
        stop.exit()
    if record is not None:
        _write_compress_record(
            record,
            src=src,
            bits=bits,
            start=start,
            seconds=seconds,
            input_hash=input_hash,
            result=result,
            output=(dst, result.bytes_out, output_hash),
            stop=None,
        )
    typer.echo(
        f"compressed {src} to {dst} at {result.bits} bits: "
        f"{_count(result.n_traces, 'trace')}, "
        f"{_count(result.n_exact, 'real station')} also kept exactly"
    )
    typer.echo(
        f"{_size(result.bytes_in)} -> {_size(result.bytes_out)}, "
        f"{_factor(result.bytes_in, result.bytes_out)} "
        f"(read {result.read_s:.1f} s, encode {result.encode_s:.1f} s, "
        f"write {result.write_s:.1f} s, check {result.check_s:.1f} s)"
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
            metavar="DST", dir_okay=False, help="Where to write the netCDF file."
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace a file already at DST."),
    ] = False,
) -> None:
    """Decompress a file written by compress.

    DST is a netCDF file in the standard layout: every variable, coordinate and
    attribute of the original, and its waveform as float32 with the dimensions
    (component, station, time). The real stations' traces are exactly as they were;
    every other sample is within one quantisation step of the original.

    The whole waveform is decoded into memory, so this command suits files whose
    waveform fits there: it needs 4 bytes for each sample, plus a block of 128 MiB
    and some working space. The file appears at DST only when it is complete. A file
    already at DST is replaced only with --overwrite.

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
            destinations.refuse_overwriting(src, dst, overwrite)
            _require_folder(dst)
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
        # verify_file checks only the original's layout.
        with _about(compressed, {NotStandardLayout: original}):
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
    dst: Annotated[
        Path,
        typer.Argument(
            metavar="DST", dir_okay=False, help="Where to write the netCDF file."
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace a file already at DST."),
    ] = False,
) -> None:
    """Convert a BB.bin file to a netCDF file.

    BB.bin is the legacy layout of a simulation's broadband waveforms. DST is in the
    standard layout, which the compress command takes. The whole BB.bin file is read
    into memory, which takes about twice its size at the peak. The file appears at DST
    only when it is complete. A file already at DST is replaced only with --overwrite.

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
            _require_folder(dst)
        # Only a file that can't be a BB.bin is about SRC: the rest is the writing's.
        with _about(dst, {NotBBBin: src}):
            bbbin.convert_bbbin(src, dst, overwrite)
    except _Stop as stop:
        stop.exit()
    typer.echo(
        f"converted {src} to {dst}: "
        f"{_size(src.stat().st_size)} -> {_size(dst.stat().st_size)}"
    )
