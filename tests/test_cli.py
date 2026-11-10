import errno
import json
import logging
import logging.handlers
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyflac
import pytest
import xarray as xr
from typer.testing import CliRunner, Result

from waveform_compression_tools import bbbin, cli, files, legacy
from waveform_compression_tools.bbbin import read_bbbin
from waveform_compression_tools.cli import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
    app,
)
from waveform_compression_tools.codec import STEP_SLACK, EncodedBlock, encode_block
from waveform_compression_tools.content_hash import content_hash
from waveform_compression_tools.destinations import HDF5Output
from waveform_compression_tools.files import (
    GROUP,
    CompressResult,
    compress_file,
    decompress_file,
    is_compressed,
    verify_file,
)
from waveform_compression_tools.hfbin import read_hfbin
from waveform_compression_tools.quantise import full_scale, quantise
from waveform_compression_tools.stream import encode_stream

from helpers import (
    RealTraces,
    StandardDataset,
    contents,
    limit_file_sizes,
    limits_file_sizes,
    names_in,
    nothing_left,
    partial_of,
    read,
    refused_input,
    run_python,
    standard_dataset_of,
    with_a_flipped_byte,
    write,
    write_bbbin,
    write_hfbin,
)

runner = CliRunner()

# How every error line starts.
PROGRAM = "waveform-compression-tools"
# The least time a hash is made to take, where the record's timings are checked.
HASH_S = 0.02
# The commands that write a file, each from an input of its own kind (see an_input).
WRITING = ["compress", "decompress", "convert-bbbin", "convert-hfbin"]


def run(*args: str | int | Path) -> Result:
    # In a real run, where nothing sets up logging, a record of WARNING or above that
    # reaches the root logger is printed to stderr by Python's last-resort handler.
    # Under pytest it goes to pytest's own handlers instead, unseen, so none may reach
    # the root.
    logged = logging.handlers.BufferingHandler(capacity=1000)
    logged.setLevel(logging.WARNING)
    logging.getLogger().addHandler(logged)
    try:
        result = runner.invoke(app, [str(arg) for arg in args])
    finally:
        logging.getLogger().removeHandler(logged)
    assert [record.getMessage() for record in logged.buffer] == []
    return result


def assert_exit(result: Result, status: int) -> None:
    # The command chose its exit status. An exception that escaped it would give status
    # 1 as well, so a failure has to be a deliberate exit, which Typer raises as a
    # SystemExit.
    assert result.exit_code == status, result.output
    if status == EXIT_OK:
        assert result.exception is None
    else:
        assert isinstance(result.exception, SystemExit), repr(result.exception)


def assert_stopped(result: Result, status: int, path: Path) -> str:
    # A refusal or a failure: its status, and exactly one line on stderr in the published
    # form, naming the file the problem is about, with nothing on stdout. The reason
    # doesn't name that file again at its start, as some of the library's messages do.
    # Returns the reason the line gives.
    assert_exit(result, status)
    lines = result.stderr.splitlines()
    assert len(lines) == 1 and result.stderr == f"{lines[0]}\n", result.stderr
    prefix = f"{PROGRAM}: {path}: "
    assert lines[0].startswith(prefix), result.stderr
    assert result.stdout == ""
    reason = lines[0].removeprefix(prefix)
    assert reason.strip() != ""
    assert not reason.startswith((f"{path}: ", f"{path} ")), result.stderr
    return reason


def read_record(path: Path) -> dict:
    return json.loads(path.read_text())


def compressed_file(ds: xr.Dataset, tmp_path: Path) -> tuple[Path, Path]:
    # Writes a dataset and compresses it with the command, returning both files.
    src = write(ds, tmp_path / "in.nc")
    dst = tmp_path / "packed.nc"
    assert_exit(run("compress", src, dst), EXIT_OK)
    return src, dst


def an_input(command: str, folder: Path, standard_dataset: StandardDataset) -> Path:
    # A file that the command takes: a netCDF file with a real station for compress, the
    # file compressed from it for decompress, a BB.bin for convert-bbbin, and an HF.bin
    # for convert-hfbin.
    if command == "convert-bbbin":
        write_bbbin(folder / "BB.bin", 3, 200, names=["S000000", "S000001", "KIKS"])
        return folder / "BB.bin"
    if command == "convert-hfbin":
        write_hfbin(folder / "HF.bin", 3, 200, names=["S000000", "S000001", "KIKS"])
        return folder / "HF.bin"
    src = write(standard_dataset(3, nt=200, real=(2,)), folder / "in.nc")
    if command == "compress":
        return src
    compress_file(src, folder / "in.flac.nc")
    return folder / "in.flac.nc"


def command_args(
    command: str, src: Path, dst: Path, rec: Path, *options: str
) -> list[str | Path]:
    # The command's arguments, with compress's record at rec.
    args: list[str | Path] = [command, src, dst, *options]
    if command == "compress":
        args += ["--record", rec]
    return args


def assert_complete(
    command: str, src: Path, dst: Path, rec: Path | None = None
) -> None:
    # The output that the command wrote from src is whole: compress's passes verify, and
    # its record at rec, if given, says so; decompress's and the converters' hold what
    # the library reads from src.
    if command == "compress":
        assert verify_file(src, dst).passed
        assert rec is None or read_record(rec)["status"] == "ok"
        return
    reader = {
        "decompress": decompress_file,
        "convert-bbbin": read_bbbin,
        "convert-hfbin": read_hfbin,
    }[command]
    xr.testing.assert_identical(read(dst), reader(src))


@pytest.mark.parametrize(
    ("convert", "stem", "write_legacy"),
    [("convert-bbbin", "BB", write_bbbin), ("convert-hfbin", "HF", write_hfbin)],
)
def test_compress_then_verify(
    tmp_path: Path, convert: str, stem: str, write_legacy: Callable[..., np.ndarray]
) -> None:
    # Each command in turn, on what the one before it wrote, from a BB.bin or an HF.bin.
    # Each exits with status 0 and says nothing on stderr; its output is at its
    # destination; and the files it read are as they were.
    file_names = [f"{stem}.bin", f"{stem}.nc", f"{stem}.flac.nc", "back.nc"]
    legacy, nc, packed, back = (tmp_path / name for name in file_names)
    write_legacy(legacy, 3, 200, names=["S000000", "S000001", "KIKS"])
    steps = [
        ((convert, legacy, nc), [legacy], nc),
        (("compress", nc, packed), [nc], packed),
        (("verify", nc, packed), [nc, packed], None),
        (("decompress", packed, back), [packed], back),
    ]
    for args, inputs, output in steps:
        before = [path.read_bytes() for path in inputs]
        result = run(*args)
        assert_exit(result, EXIT_OK)
        assert result.stderr == "", args
        assert output is None or output.is_file(), args
        assert [path.read_bytes() for path in inputs] == before, args
    # Nothing else, no partial file included, was left behind.
    assert names_in(tmp_path) == sorted(file_names)


@pytest.mark.parametrize(
    ("options", "bits", "level", "checked"),
    [
        pytest.param([], 32, 5, True, id="the default"),
        pytest.param(["--bits", "8", "--level", "0"], 8, 0, True, id="8 bits"),
        pytest.param(
            ["--bits", "24", "--level", "8", "--no-check"],
            24,
            8,
            False,
            id="24 bits unchecked",
        ),
    ],
)
def test_compress_ok_writes_a_record(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    bits: int,
    level: int,
    checked: bool,
) -> None:
    src = write(standard_dataset(4, nt=300, real=(1,)), tmp_path / "in.nc")
    dst, rec = tmp_path / "out.nc", tmp_path / "out.json"
    # What compress_file gives the command, kept to be compared with the record; and
    # each hash made to take at least HASH_S, so that both are seen to count.
    results: list[CompressResult] = []

    def compress_and_keep(*args: Any, **kwargs: Any) -> CompressResult:
        results.append(compress_file(*args, **kwargs))
        return results[-1]

    def hash_slowly(path: Path) -> str:
        time.sleep(HASH_S)
        return content_hash(path)

    monkeypatch.setattr(files, "compress_file", compress_and_keep)
    for module in (cli, files):  # The input's hash, and the output's.
        monkeypatch.setattr(module, "content_hash", hash_slowly)
    result = run("compress", src, dst, "--record", rec, *options)
    assert_exit(result, EXIT_OK)
    assert result.stderr == ""
    assert names_in(tmp_path) == ["in.nc", "out.json", "out.nc"]
    [compressed] = results
    record = read_record(rec)
    assert (record["status"], record["reason"]) == ("ok", "")
    # The sizes are the files', and the hashes their content hashes.
    sizes = (src.stat().st_size, dst.stat().st_size)
    assert (compressed.bytes_in, compressed.bytes_out) == sizes
    assert record["input"] == {
        "path": str(src),
        "bytes": sizes[0],
        "content_hash": content_hash(src),
    }
    assert record["output"] == {
        "path": str(dst),
        "bytes": sizes[1],
        "content_hash": content_hash(dst),
    }
    # The setting given, or 32, reaches the file and the record, and the level given,
    # or 5, the record.
    with h5py.File(dst, "r") as f:
        assert int(f[GROUP].attrs["bits"]) == bits
    assert record["bits"] == compressed.bits == bits
    assert record["level"] == level
    # Four stations of three traces, one of them real.
    assert (record["n_traces"], record["n_exact_stations"]) == (12, 1)
    # The check's largest error is the file's, as verify finds it, within the bound the
    # setting promises.
    verified = verify_file(src, dst)
    assert verified.passed and verified.verify_s > 0
    assert 0 < verified.max_relative_error <= (1 + STEP_SLACK) / full_scale(bits)
    assert record["checked"] is checked
    assert record["max_relative_error"] == (
        verified.max_relative_error if checked else None
    )
    # The time hashing both files; reading, encoding and writing; checking; and the
    # command's own, which holds them all.
    assert compressed.read_s > 0 and compressed.encode_s > 0 and compressed.write_s > 0
    assert (compressed.check_s > 0) is checked
    seconds = record["seconds"]
    assert seconds["hash"] >= 2 * HASH_S
    assert seconds["compress"] == (
        compressed.read_s + compressed.encode_s + compressed.write_s
    )
    assert seconds["check"] == compressed.check_s
    assert seconds["hash"] + seconds["compress"] + seconds["check"] <= seconds["total"]


def test_a_higher_level_makes_a_smaller_file_that_decompresses_the_same(
    tmp_path: Path, real_traces: RealTraces
) -> None:
    # These traces' streams at 24 bits are about 1.3 times smaller at level 8 than at 0.
    src = write(standard_dataset_of(real_traces("hf")), tmp_path / "in.nc")
    sizes, back = {}, {}
    for level in (0, 8):
        dst, out = tmp_path / f"{level}.nc", tmp_path / f"{level}.back.nc"
        assert_exit(run("compress", src, dst, "--bits", 24, "--level", level), EXIT_OK)
        assert_exit(run("decompress", dst, out), EXIT_OK)
        sizes[level], back[level] = dst.stat().st_size, read(out)
    assert sizes[8] < sizes[0]
    xr.testing.assert_identical(back[0], back[8])


# Each refusal of compress, as helpers.refused_input writes it, one for each Refused
# class it raises and a few more: the file its line names, whether it comes after the
# input is hashed (only the refusals that need the samples do), and a word of its
# reason.
REFUSALS = {
    "the same path": ("dst", False, "overwrite"),  # SameFile
    "the output exists": ("dst", False, "exists"),  # OutputExists
    "a text file": ("src", False, "isn't an HDF5 file"),  # NotStandardLayout
    "already compressed": ("src", False, "compressed already"),  # AlreadyCompressed
    "an HDF5 group": ("src", False, "groups"),  # HasGroups
    "a float64 waveform": ("src", False, "float32"),  # NotStandardLayout
    "an attribute JSON can't hold": ("src", False, "JSON"),  # NotStandardLayout
    "a _FillValue JSON can't hold": ("src", False, "JSON"),  # NotStandardLayout
    "a _FillValue of text": ("src", False, "not a number"),  # NotStandardLayout
    "a boolean attribute": ("src", False, "boolean"),  # NotStandardLayout
    "every station real": ("src", False, "real"),  # AllStationsReal
    "a NaN": ("src", True, "non-finite"),  # NonFiniteSamples
    "a fill value inside the data": ("src", True, "fill value"),  # NotStandardLayout
}


@pytest.mark.parametrize("case", REFUSALS)
def test_each_refusal_exits_3_with_one_line_and_a_record(
    tmp_path: Path, standard_dataset: StandardDataset, case: str
) -> None:
    named, after_the_hash, word = REFUSALS[case]
    src, dst, overwrite = refused_input(case, tmp_path, standard_dataset)
    rec = tmp_path / "rec.json"
    before = contents(tmp_path)
    options = ["--overwrite"] if overwrite else []
    result = run("compress", src, dst, "--record", rec, *options)
    reason = assert_stopped(result, EXIT_REFUSED, {"src": src, "dst": dst}[named])
    assert word in reason
    # The line names its file once. Only a refusal of src being dst names it again, as
    # the file it would overwrite.
    assert reason.count(str({"src": src, "dst": dst}[named])) == (src == dst)
    # Nothing is written but the record: the input, and a file already at the
    # destination, are as they were, and no partial file is left.
    assert contents(tmp_path) == {**before, "rec.json": rec.read_bytes()}
    # The record keeps the library's whole message, which names its file.
    record = read_record(rec)
    assert record["status"] == "refused" and record["reason"].endswith(reason)
    if case == "already compressed":
        # That message, "SRC is compressed already", leads with the file, which the
        # line names only once.
        assert result.stderr.count(str(src)) == 1
        assert record["reason"] == f"{src} {reason}"
    assert record["input"] == {
        "path": str(src),
        "bytes": src.stat().st_size,
        "content_hash": content_hash(src) if after_the_hash else None,
    }
    assert record["output"] is None
    assert (
        record["bits"],
        record["level"],
        record["checked"],
        record["max_relative_error"],
    ) == (32, 5, False, None)
    assert (record["n_traces"], record["n_exact_stations"]) == (None, None)
    assert (record["seconds"]["hash"] > 0) is after_the_hash


# Failures once compression has begun: which file the line names, and a word of its
# reason. The output's hash is part of compressing, so none of them finishes it.
FAILURES = {
    "a damaged stream": ("dst", "don't decode"),
    "a stream libFLAC can't finish": ("src", "couldn't be encoded"),
    "an output it can't hash": ("dst", "Input/output error"),
}


@pytest.mark.parametrize("failure", FAILURES)
def test_a_failed_check_exits_1_and_leaves_nothing(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    # The encoder flips a byte of one trace's stream, which the check finds as it
    # decodes what was written: libFLAC reports it, and the command says it once. Or
    # libFLAC fails to finish a stream, as it has. Or the output passes its check but
    # can't be read back to be hashed, so nothing vouches for it. An earlier file at
    # DST, asked to be replaced, stays as it was in each case.
    named, word = FAILURES[failure]
    src = write(standard_dataset(3, nt=5_000), tmp_path / "in.nc")
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"
    dst.write_bytes(b"an earlier file")

    def encode_then_flip_a_byte(
        waveform: np.ndarray, bits: int, level: int
    ) -> EncodedBlock:
        return with_a_flipped_byte(encode_block(waveform, bits, level))

    def hash_all_but_the_output(path: Path) -> str:
        if path == partial_of(dst):
            raise OSError(errno.EIO, "Input/output error")
        return content_hash(path)

    if failure == "a damaged stream":
        monkeypatch.setattr(files, "encode_block", encode_then_flip_a_byte)
    elif failure == "a stream libFLAC can't finish":
        monkeypatch.setattr(pyflac.StreamEncoder, "finish", lambda self: False)
    else:
        monkeypatch.setattr(files, "content_hash", hash_all_but_the_output)
    before = contents(tmp_path)
    result = run("compress", src, dst, "--record", rec, "--overwrite")
    path = {"src": src, "dst": dst}[named]
    reason = assert_stopped(result, EXIT_FAILED, path)
    assert word in reason
    # The line names the file once, though the check's own message starts with it. An
    # OSError's words end with the file, as those of a write to DST that fails do.
    if failure != "an output it can't hash":
        assert result.stderr.count(str(path)) == 1
    # It names the file the run was making, not the partial file it was checking.
    assert ".partial" not in result.stderr
    # No new output and no partial file: nothing but the record.
    assert contents(tmp_path) == {**before, "rec.json": rec.read_bytes()}
    # The record keeps the library's whole message.
    record = read_record(rec)
    assert record["status"] == "failed" and record["reason"].endswith(reason)
    # The input was hashed before it was compressed, and there is no output.
    assert record["input"]["content_hash"] == content_hash(src)
    assert record["output"] is None
    # Compressing didn't finish, so nothing it would have made is recorded.
    assert (record["checked"], record["n_traces"]) == (False, None)


@pytest.mark.parametrize("outcome", ["compressed", "refused"])
def test_a_record_that_cannot_be_written_fails_naming_it(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    # Whatever became of the file, the run fails if its record can't be written, since
    # nothing then says what became of it. A compressed file that passed its check
    # stays where it is.
    real = (0, 1, 2) if outcome == "refused" else ()
    src = write(standard_dataset(3, nt=100, real=real), tmp_path / "in.nc")
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"

    def fail_to_write(record: dict, path: Path) -> None:
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(cli, "write_record", fail_to_write)
    result = run("compress", src, dst, "--record", rec)
    assert "Disk quota exceeded" in assert_stopped(result, EXIT_FAILED, rec)
    assert not rec.exists()
    if outcome == "compressed":
        assert verify_file(src, dst).passed
    else:
        assert not dst.exists()


@pytest.mark.parametrize("command", WRITING)
def test_a_stale_partial_is_replaced_and_the_rerun_succeeds(
    tmp_path: Path, standard_dataset: StandardDataset, command: str
) -> None:
    # A job killed partway through a file, by the OOM killer, a time limit or SIGKILL,
    # has no chance to delete its partial files. It leaves the first half of the file it
    # was writing, and of compress's record.
    src = an_input(command, tmp_path, standard_dataset)
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"
    assert_exit(run(command, src, dst), EXIT_OK)
    partial_of(dst).write_bytes(dst.read_bytes()[: dst.stat().st_size // 2])
    dst.unlink()
    if command == "compress":
        partial_of(rec).write_text('{"record_version": 1, "tool": ')
    # The same command again replaces them, and finishes.
    assert_exit(run(*command_args(command, src, dst, rec)), EXIT_OK)
    assert not partial_of(dst).exists() and not partial_of(rec).exists()
    assert_complete(command, src, dst, rec)


@pytest.mark.parametrize("command", WRITING)
def test_a_link_at_a_partial_name_is_removed_not_written_through(
    tmp_path: Path, standard_dataset: StandardDataset, command: str
) -> None:
    # A symbolic link at a destination's .partial name, and at compress's record's,
    # each to a file that has nothing to do with the run. Writing through a link would
    # destroy its target, and the rename would then put the link at the destination.
    src = an_input(command, tmp_path, standard_dataset)
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"
    links = {dst: tmp_path / "unrelated.nc"}
    if command == "compress":
        links[rec] = tmp_path / "unrelated.json"
    for path, target in links.items():
        target.write_bytes(b"an unrelated file")
        partial_of(path).symlink_to(target)
    assert_exit(run(*command_args(command, src, dst, rec)), EXIT_OK)
    for path, target in links.items():
        assert target.read_bytes() == b"an unrelated file"
        assert not path.is_symlink() and not os.path.lexists(partial_of(path))
    assert_complete(command, src, dst, rec)


@pytest.mark.parametrize(
    ("command", "missing"),
    [
        ("compress", "DST"),
        ("compress", "--record"),
        ("decompress", "DST"),
        ("convert-bbbin", "DST"),
        ("convert-hfbin", "DST"),
    ],
)
def test_a_missing_destination_folder_fails_naming_the_file(
    tmp_path: Path, standard_dataset: StandardDataset, command: str, missing: str
) -> None:
    src = an_input(command, tmp_path, standard_dataset)
    nowhere = tmp_path / "nowhere"
    dst = nowhere / "out.nc" if missing == "DST" else tmp_path / "out.nc"
    rec = nowhere / "rec.json" if missing == "--record" else tmp_path / "rec.json"
    before = contents(tmp_path)
    result = run(*command_args(command, src, dst, rec))
    reason = assert_stopped(result, EXIT_FAILED, dst if missing == "DST" else rec)
    # Said before any work is done, rather than found when the write fails.
    assert f"there is no folder {nowhere}" in reason
    # The input is as it was, the folder isn't made, and nothing is written but
    # compress's record, if its own folder is there.
    after = contents(tmp_path)
    if command == "compress" and missing == "DST":
        del after["rec.json"]
        record = read_record(rec)
        assert (record["status"], record["reason"]) == ("failed", reason)
        # The folders are checked before the input is hashed.
        assert record["input"]["content_hash"] is None
        assert record["seconds"]["hash"] == 0.0
    assert after == before


@pytest.mark.parametrize("missing", ["DST", "--record"])
def test_a_missing_folder_is_found_before_the_input_is_refused(
    tmp_path: Path, standard_dataset: StandardDataset, missing: str
) -> None:
    # Both destinations are checked, their folders included, before the input is: a
    # mistyped folder fails the run, though the input would be refused too.
    src = write(standard_dataset(3, nt=100, real=(0, 1, 2)), tmp_path / "in.nc")
    nowhere = tmp_path / "nowhere"
    dst = nowhere / "out.nc" if missing == "DST" else tmp_path / "out.nc"
    rec = nowhere / "rec.json" if missing == "--record" else tmp_path / "rec.json"
    result = run("compress", src, dst, "--record", rec)
    reason = assert_stopped(result, EXIT_FAILED, dst if missing == "DST" else rec)
    assert reason.startswith(f"there is no folder {nowhere}")
    if missing == "DST":
        record = read_record(rec)
        assert (record["status"], record["reason"]) == ("failed", reason)
    else:
        # No record can be written, and none is tried.
        assert names_in(tmp_path) == ["in.nc"]


@pytest.mark.parametrize("recorded", [False, True])
def test_the_input_is_hashed_only_for_the_record(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    recorded: bool,
) -> None:
    # The input's hash goes only in the record, so a run without one doesn't read the
    # input to hash it. The output is hashed by compress_file either way.
    src = write(standard_dataset(3, nt=100, real=(1,)), tmp_path / "in.nc")
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"
    hashed: list[Path] = []

    def hash_and_say(path: Path) -> str:
        hashed.append(Path(path))
        return content_hash(path)

    monkeypatch.setattr(cli, "content_hash", hash_and_say)
    record = ["--record", rec] if recorded else []
    assert_exit(run("compress", src, dst, *record), EXIT_OK)
    assert hashed == ([src] if recorded else [])
    if recorded:
        assert read_record(rec)["input"]["content_hash"] == content_hash(src)


@pytest.mark.parametrize(
    ("option", "value"),
    [("--bits", "7"), ("--bits", "33"), ("--level", "-1"), ("--level", "9")],
)
def test_bits_out_of_range_is_a_usage_error(
    tmp_path: Path, standard_dataset: StandardDataset, option: str, value: str
) -> None:
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"
    result = run("compress", src, dst, "--record", rec, option, value)
    assert_exit(result, EXIT_USAGE)  # As Typer reports one.
    assert option in result.stderr and result.stdout == ""
    # A usage error writes nothing, not even a record.
    assert names_in(tmp_path) == ["in.nc"]


# What decompress, verify and the converters each refuse, as compress refuses its own:
# which of their two arguments the line names, and a word of its reason. Their
# failures are the damage table's and the failed write's.
SAME_CODES = [
    ("decompress", "a file that isn't compressed", 0, "not compressed"),
    ("decompress", "a dataset with the group's name", 0, "not compressed"),
    ("decompress", "a text file", 0, "isn't an HDF5 file, so it is not compressed"),
    ("decompress", "an unknown format version", 0, "format version 2"),
    ("verify", "a file that isn't compressed", 1, "not compressed"),
    ("verify", "a dataset with the group's name", 1, "not compressed"),
    ("verify", "a text file", 1, "isn't an HDF5 file, so it is not compressed"),
    ("verify", "an unknown setting", 1, "bits must be from 8 to 32"),
    ("verify", "an original that isn't in the standard layout", 0, "float32"),
    ("verify", "an original that is a text file", 0, "isn't an HDF5 file"),
    ("convert-bbbin", "a file that isn't a BB.bin", 0, "BB.bin file (2000 bytes)"),
    ("convert-bbbin", "text that isn't ASCII", 0, "it holds text that isn't ASCII"),
    ("convert-hfbin", "a file that isn't an HF.bin", 0, "HF.bin file (2000 bytes)"),
    ("convert-hfbin", "text that isn't ASCII", 0, "it holds text that isn't ASCII"),
    ("convert-hfbin", "an unfinished HF.bin", 0, "with 2 of its 3 stations unfinished"),
]


@pytest.mark.parametrize(("command", "case", "named", "word"), SAME_CODES)
def test_decompress_verify_and_convert_use_the_same_codes(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    command: str,
    case: str,
    named: int,
    word: str,
) -> None:
    original = write(standard_dataset(3, nt=100), tmp_path / "in.nc")
    packed = tmp_path / "in.flac.nc"
    compress_file(original, packed)
    with h5py.File(packed, "r+") as f:
        if case == "an unknown format version":
            f[GROUP].attrs["format_version"] = 2
        elif case == "an unknown setting":
            f[GROUP].attrs["bits"] = 33
    text = tmp_path / "notes.nc"
    text.write_text("station,pga\nS000000,0.12\n")
    if case == "a file that isn't compressed":
        packed = original
    elif case == "a dataset with the group's name":
        packed = write(standard_dataset(3, nt=100), tmp_path / "dataset.nc")
        with h5py.File(packed, "r+") as f:
            f.create_dataset(GROUP, data=np.arange(3))
    elif case == "a text file":
        packed = text
    elif case == "an original that is a text file":
        original = text
    elif case == "an original that isn't in the standard layout":
        ds = standard_dataset(3, nt=100)
        ds["waveform"] = ds["waveform"].astype(np.float64)
        original = write(ds, tmp_path / "float64.nc")
    bb, hf = tmp_path / "BB.bin", tmp_path / "HF.bin"
    if case == "a file that isn't a BB.bin":
        bb.write_bytes(bytes(2000))
    else:
        write_bbbin(bb, 2, 20, names=["Ōtaki", "S000001"])
    if case == "a file that isn't an HF.bin":
        hf.write_bytes(bytes(2000))
    elif case == "an unfinished HF.bin":
        write_hfbin(hf, 3, 20, unfinished=(0, 2))
    else:
        write_hfbin(hf, 2, 20, names=["Ōtaki", "S000001"])
    # Decompress and the converters take the file and where to write; verify takes the
    # original and the compressed file.
    args = {
        "decompress": (packed, tmp_path / "out.nc"),
        "verify": (original, packed),
        "convert-bbbin": (bb, tmp_path / "BB.nc"),
        "convert-hfbin": (hf, tmp_path / "HF.nc"),
    }[command]
    before = contents(tmp_path)
    reason = assert_stopped(run(command, *args), EXIT_REFUSED, args[named])
    assert word in reason
    assert contents(tmp_path) == before  # Nothing is written.


@pytest.mark.parametrize("cut", ["in its records", "in its traces"])
@pytest.mark.parametrize(
    ("command", "head_bytes"), [("convert-bbbin", 0x500), ("convert-hfbin", 0x200)]
)
def test_a_legacy_file_that_shrinks_as_it_is_read_is_refused(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    head_bytes: int,
    cut: str,
) -> None:
    # A file as long as its header says when that is checked, but shorter by the time
    # its records or its traces are read, has been cut short meanwhile. Its line names
    # SRC, and is the one that converting it again gives. The file is cut first, and
    # the size check is given the size it had, since a cut after the check might not
    # be seen: the reader may hold the whole of a small file once it has the header.
    src, dst = an_input(command, tmp_path, standard_dataset), tmp_path / "out.nc"
    whole = src.stat().st_size
    os.truncate(src, head_bytes + 1 if cut == "in its records" else whole - 1)
    check = legacy.layout

    def check_as_whole(
        head: bytes, size: int, head_bytes: int, station_bytes: int
    ) -> tuple[str, int, int] | None:
        return check(head, whole, head_bytes, station_bytes)

    monkeypatch.setattr(legacy, "layout", check_as_whole)
    reason = assert_stopped(run(command, src, dst), EXIT_REFUSED, src)
    monkeypatch.undo()
    assert assert_stopped(run(command, src, dst), EXIT_REFUSED, src) == reason
    assert names_in(tmp_path) == [src.name]


@pytest.mark.parametrize(
    ("command", "earlier"),
    [
        ("compress", "DST"),
        ("decompress", "DST"),
        ("convert-bbbin", "DST"),
        ("convert-hfbin", "DST"),
        ("compress", "a record"),
        ("compress", "a finished run"),
    ],
)
def test_overwrite_replaces_an_existing_destination(
    tmp_path: Path, standard_dataset: StandardDataset, command: str, earlier: str
) -> None:
    # A file already at the destination, or at compress's record, is replaced only if
    # asked. The record is checked first, before the input is read, so a rerun of a
    # finished file that isn't asked to replace them leaves its "ok" record as it was,
    # rather than writing a "refused" one over it.
    src = an_input(command, tmp_path, standard_dataset)
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"
    args: list[str | Path] = [command, src, dst]
    if earlier != "DST":
        args += ["--record", rec]
    if earlier == "DST":
        dst.write_bytes(b"an earlier file")
    elif earlier == "a record":
        rec.write_bytes(b"an earlier record")
    else:
        assert_exit(run(*args), EXIT_OK)
    before = contents(tmp_path)
    named = dst if earlier == "DST" else rec
    assert "exists" in assert_stopped(run(*args), EXIT_REFUSED, named)
    assert contents(tmp_path) == before  # Nothing is written, not even a record.
    assert_exit(run(*args, "--overwrite"), EXIT_OK)
    assert not partial_of(dst).exists() and not partial_of(rec).exists()
    assert_complete(command, src, dst, None if earlier == "DST" else rec)


def test_decompress_gives_back_the_standard_layout(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # The file holds what decompress_file gives back, which test_files.py checks.
    _, packed = compressed_file(standard_dataset(5, nt=300, real=(1,)), tmp_path)
    back = tmp_path / "back.nc"
    result = run("decompress", packed, back)
    assert_exit(result, EXIT_OK)
    assert "15 traces of 300 samples" in result.stdout
    assert not partial_of(back).exists() and not is_compressed(back)
    assert_complete("decompress", packed, back)


# Damage that decompressing and verifying both meet, to one trace's stream, to the
# waveform's attributes or their types, to which streams the file holds, or to the
# file itself, and a word of the reason each gives. The errors of the last three name
# no file of their own, and the line names the compressed file all the same.
DAMAGE = {
    "a zeroed fLaC marker": "component 1, station 2: a FLAC stream is damaged (",
    "a flipped byte": "component 1, station 2: a FLAC stream is damaged (",
    "the last 10 bytes cut off": "component 1, station 2: a FLAC stream is damaged (",
    "a cut between frames": "component 1, station 2: the stream decodes to 8192",
    "a cut to nothing": "component 1, station 2: the stream decodes to 0 samples",
    "a type for an attribute that isn't there": "waveform_attr_types",
    "the streams of 2 stations of 3": "decodes to a waveform of shape (3, 2, 10000)",
    "attributes that aren't JSON": "can't be read (Expecting property name",
    "attributes of JSON that isn't an object": "can't be read (JSON that isn't an object",
    "an attribute outside its type's range": "out of bounds for int8",
    "a type that numpy doesn't know": "data type 'int33' not understood",
    "a file cut short, as by a download that stopped": "truncated file",
}
# Damage that only a check against the original can find.
CHECK_DAMAGE = {
    "a doubled gain": "doesn't agree",
    "an original of another shape": "shape",
    # The samples are as close as ever, so the line says what else differs.
    "an attribute of the file changed": "its variables, coordinates or attributes differ",
}


@pytest.mark.parametrize(
    ("command", "damage"),
    [("decompress", damage) for damage in DAMAGE]
    + [("verify", damage) for damage in DAMAGE | CHECK_DAMAGE],
)
def test_a_damaged_file_fails_the_check(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    damage: str,
) -> None:
    # 10,000 samples are two whole frames of 4,096 and a short third.
    ds = standard_dataset(3, nt=10_000)
    src, packed = compressed_file(ds, tmp_path)
    c, s = 1, 2
    with h5py.File(packed, "r+") as f:
        g = f[GROUP]
        start, size = int(g["stream_starts"][c, s]), int(g["stream_bytes"][c, s])
        if damage == "a zeroed fLaC marker":
            g["compressed"][start : start + 4] = 0
        elif damage == "a flipped byte":
            g["compressed"][start + size // 2] ^= 0xFF
        elif damage == "the last 10 bytes cut off":
            g["stream_bytes"][c, s] = size - 10
        elif damage == "a cut between frames":
            # The stream's first two frames are the first 8,192 samples encoded alone.
            # libFLAC decodes them without an error: only a count of the samples finds
            # the cut.
            gain = g["stream_gains"][c, s]
            frames = encode_stream(quantise(ds["waveform"].values[c, s, :8192], gain))
            assert g["compressed"][start : start + size].tobytes().startswith(frames)
            g["stream_bytes"][c, s] = len(frames)
        elif damage == "a cut to nothing":
            g["stream_bytes"][c, s] = 0
        elif damage == "a type for an attribute that isn't there":
            g.attrs["waveform_attr_types"] = json.dumps({"gain": "float32"})
        elif damage == "attributes that aren't JSON":
            g.attrs["waveform_attrs"] = "{not json"
        elif damage == "attributes of JSON that isn't an object":
            g.attrs["waveform_attrs"] = "[1, 2]"
        elif damage == "an attribute outside its type's range":
            g.attrs["waveform_attrs"] = json.dumps({"units": 300})
            g.attrs["waveform_attr_types"] = json.dumps({"units": "int8"})
        elif damage == "a type that numpy doesn't know":
            g.attrs["waveform_attr_types"] = json.dumps({"units": "int33"})
        elif damage == "the streams of 2 stations of 3":
            # The streams of a file of the first 2 stations, whole in themselves, in
            # place of these: the file's stations and its streams disagree.
            two, other = tmp_path / "two.nc", tmp_path / "two.flac.nc"
            compress_file(write(ds.isel(station=slice(0, 2)), two), other)
            del f[GROUP]
            with h5py.File(other, "r") as streams:
                streams.copy(GROUP, f)
        elif damage == "a doubled gain":
            g["stream_gains"][c, s] *= 2
        elif damage == "an attribute of the file changed":
            f.attrs["dt"] = 0.01
    if damage == "a file cut short, as by a download that stopped":
        os.truncate(packed, packed.stat().st_size // 2)
    original = src
    if damage == "an original of another shape":
        original = write(ds.isel(station=slice(0, 2)), tmp_path / "other.nc")
    back = tmp_path / "back.nc"
    if command == "decompress":
        # decompress writes each block as it decodes it. In blocks of 2 stations, a
        # damaged stream of station 2 is found in the second, as the file is written,
        # here over an earlier one. The line names SRC all the same.
        monkeypatch.setattr(files, "BLOCK_BYTES", 2 * 3 * 10_000 * 4)
        back.write_bytes(b"an earlier file")
    before = contents(tmp_path)
    # Decompress takes the compressed file and where to write; verify takes the
    # original and the compressed file.
    args = {"decompress": (packed, back, "--overwrite"), "verify": (original, packed)}
    reason = assert_stopped(run(command, *args[command]), EXIT_FAILED, packed)
    assert (DAMAGE | CHECK_DAMAGE)[damage] in reason
    assert contents(tmp_path) == before  # Nothing new anywhere, and DST as it was.


# What each command's line gives when its write fails, or None where the failure is an
# interruption, which isn't caught. The unexpected error's message is of two lines, as
# some of xarray's are, which the line joins into one.
@pytest.mark.parametrize("command", WRITING)
@pytest.mark.parametrize(
    ("error", "shown"),
    [
        pytest.param(OSError(errno.ENOSPC, "No space left on device"), "No space left"),
        pytest.param(KeyboardInterrupt(), None),
        pytest.param(
            RuntimeError("something\nunexpected"), "RuntimeError: something unexpected"
        ),
    ],
    ids=["disk", "interruption", "unexpected"],
)
def test_a_failed_write_leaves_no_partial_file_and_keeps_the_old_one(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    error: BaseException,
    shown: str | None,
) -> None:
    src = an_input(command, tmp_path, standard_dataset)
    dst, rec = tmp_path / "out.nc", tmp_path / "rec.json"
    dst.write_bytes(b"an earlier file")
    written = []

    def write_part_then_fail(
        self: xr.Dataset, out: HDF5Output, *args: object, **kwargs: object
    ) -> None:
        written.append(Path(out.name))
        out.write(b"half a file")
        raise error

    monkeypatch.setattr(xr.Dataset, "to_netcdf", write_part_then_fail)
    result = run(*command_args(command, src, dst, rec, "--overwrite"))
    assert written == [partial_of(dst)]  # The write began.
    if shown is None:
        # Typer ends an interruption its own way: it isn't a failure of the file, so no
        # line or record says so, and the status is none of the command's own.
        statuses = (EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED)
        assert result.exit_code not in statuses
        assert PROGRAM not in result.stderr and not rec.exists()
    else:
        # A failure, an unexpected one with its type. An OSError as the output is
        # written names the output, whichever command writes it. Compress names its
        # input for anything else that goes wrong while it compresses, but for a refusal
        # to write or a failed check; decompress and the converters name the file they
        # were writing.
        named = src if command == "compress" and not isinstance(error, OSError) else dst
        assert shown in assert_stopped(result, EXIT_FAILED, named)
        assert command != "compress" or read_record(rec)["status"] == "failed"
    assert dst.read_bytes() == b"an earlier file" and not partial_of(dst).exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write in any folder")
@pytest.mark.parametrize("command", WRITING)
def test_a_folder_that_cant_be_written_in_fails_naming_the_output(
    tmp_path: Path, standard_dataset: StandardDataset, command: str
) -> None:
    # The error names the partial file that can't be made, yet the line names the
    # output, not its partial file or the input.
    src = an_input(command, tmp_path, standard_dataset)
    folder = tmp_path / "read-only"
    folder.mkdir()
    dst = folder / "out.nc"
    folder.chmod(0o500)
    try:
        result = run(command, src, dst)
    finally:
        folder.chmod(0o700)
    reason = assert_stopped(result, EXIT_FAILED, dst)
    assert "Permission denied" in reason and reason.endswith(f"'{dst}'")
    assert list(folder.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write in any folder")
def test_a_record_that_cant_be_written_fails_naming_the_record(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # As for an output: the error names the record's partial file, which can't be
    # made, yet the line names the record. The compressed file stays.
    src = write(standard_dataset(3, nt=100, real=(1,)), tmp_path / "in.nc")
    dst, folder = tmp_path / "out.nc", tmp_path / "read-only"
    folder.mkdir()
    rec = folder / "rec.json"
    folder.chmod(0o500)
    try:
        result = run("compress", src, dst, "--record", rec)
    finally:
        folder.chmod(0o700)
    reason = assert_stopped(result, EXIT_FAILED, rec)
    assert reason.startswith(
        "the record can't be written: [Errno 13] Permission denied"
    )
    assert reason.endswith(f"'{rec}'")
    assert list(folder.iterdir()) == []
    assert verify_file(src, dst).passed


@limits_file_sizes
@pytest.mark.parametrize(
    ("command", "options", "where"),
    [
        ("compress", [], "in the coordinates"),
        ("compress", [], "at the last byte"),
        ("compress", ["--no-check"], "at the last byte"),
        ("decompress", [], "halfway"),
        ("convert-bbbin", [], "halfway"),
        ("convert-hfbin", [], "halfway"),
    ],
)
def test_a_disk_that_fills_partway_fails_the_run_naming_the_output(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    command: str,
    options: list[str],
    where: str,
) -> None:
    # A write that fails partway, as on a full disk or past a quota, fails the run: one
    # line that names the output with the system's reason, status 1, and nothing left
    # at DST. Each file the command writes is held to a limit on its size, which fails
    # writes the same way. Python mustn't crash either, as it did when HDF5 2.0.0 was
    # told of a failed write and then failed to close the file. compress writes its
    # coordinates first, 8 KiB here, where the record takes less than 1 KiB, and its
    # last byte as the file is closed, after every block; without the check, which
    # reads the file back, only the error that the write met can stop that run.
    src = an_input(command, tmp_path, standard_dataset)
    full = tmp_path / "full.nc"
    assert_exit(run(command, src, full, *options), EXIT_OK)
    size = full.stat().st_size
    limit = {
        "in the coordinates": 2048,
        "halfway": size // 2,
        "at the last byte": size - 1,
    }[where]
    args = [command, src.name, "out.nc", *options]
    if command == "compress":
        args += ["--record", "rec.json"]
    cli = "from waveform_compression_tools.cli import app; app()"
    ran = run_python(limit_file_sizes(limit) + cli, *args, cwd=tmp_path)
    assert ran.returncode == EXIT_FAILED, ran.stderr
    assert ran.stdout == "" and ran.stderr.count("\n") == 1, ran.stderr
    assert ran.stderr.startswith(f"{PROGRAM}: out.nc: "), ran.stderr
    assert "File too large" in ran.stderr
    assert nothing_left(tmp_path / "out.nc")
    assert command != "compress" or read_record(tmp_path / "rec.json")["status"] == (
        "failed"
    )


def test_an_os_error_names_the_file_it_is_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An OSError that says which file it is about names that file, whatever step raised
    # it: here, reading the BB.bin fails inside convert-bbbin, whose other failures are
    # its output's.
    src, dst = tmp_path / "BB.bin", tmp_path / "BB.nc"
    write_bbbin(src, 2, 20, names=["S000000", "S000001"])

    def fail_to_read(path: Path) -> xr.Dataset:
        raise OSError(errno.EIO, "Input/output error", str(path))

    monkeypatch.setattr(bbbin, "read_bbbin", fail_to_read)
    reason = assert_stopped(run("convert-bbbin", src, dst), EXIT_FAILED, src)
    assert "Input/output error" in reason
    assert names_in(tmp_path) == ["BB.bin"]


def test_importing_the_command_line_leaves_logging_alone(tmp_path: Path) -> None:
    # pyflac's logger is quietened when a command runs, not when cli.py is imported, so
    # a program that only imports it keeps its logging as it was. This needs a fresh
    # interpreter.
    code = (
        "import logging, waveform_compression_tools.cli; "
        "log = logging.getLogger('pyflac'); "
        "assert log.propagate and not log.handlers, (log.propagate, log.handlers)"
    )
    imported = run_python(code, cwd=tmp_path)
    assert imported.returncode == 0, imported.stderr


@pytest.mark.parametrize("command", WRITING)
@pytest.mark.parametrize("src_name", ["data.nc", "data.nc.partial"])
def test_a_command_will_not_write_over_its_source(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    src_name: str,
) -> None:
    # A file is written to its destination with .partial added, which is deleted if the
    # write fails, and then renamed to the destination: either would destroy a source
    # that is the destination, or the destination with .partial added. The source is
    # named relative to the folder, and the destination by its full path, since
    # whatever the spelling it is the same file. Asking for overwriting doesn't change
    # that.
    monkeypatch.chdir(tmp_path)
    src, dst = Path(src_name), tmp_path / "data.nc"
    if command == "compress":
        write(standard_dataset(2, nt=50), src)
    elif command == "decompress":
        original = write(standard_dataset(2, nt=50), tmp_path / "original.nc")
        assert_exit(run("compress", original, src), EXIT_OK)
    elif command == "convert-bbbin":
        write_bbbin(src, 2, 20, names=["S000000", "S000001"])
    else:
        write_hfbin(src, 2, 20, names=["S000000", "S000001"])
    before = src.read_bytes()
    result = run(command, src, dst, "--overwrite")
    assert "overwrite" in assert_stopped(result, EXIT_REFUSED, dst)
    assert src.read_bytes() == before


@pytest.mark.parametrize(
    "over", ["the input", "the input's partial name", "the output"]
)
def test_a_record_is_never_written_over_the_input_or_the_output(
    tmp_path: Path, standard_dataset: StandardDataset, over: str
) -> None:
    # The record, too, is written with .partial added and then renamed, so it would
    # destroy a file at either name. That is a usage error, found before anything is
    # read or written.
    src = tmp_path / (
        "data.nc.partial" if over == "the input's partial name" else "data.nc"
    )
    write(standard_dataset(2, nt=50), src)
    dst = tmp_path / "out.nc"
    rec = dst if over == "the output" else tmp_path / "data.nc"
    before = contents(tmp_path)
    result = run("compress", src, dst, "--record", rec)
    assert_exit(result, EXIT_USAGE)
    assert "--record" in result.stderr and "overwrite" in result.stderr
    assert contents(tmp_path) == before


# The arguments of the commands, each as its command and its position: the files that a
# command reads, and the one it writes.
SOURCES = [
    ("compress", 0),
    ("decompress", 0),
    ("verify", 0),
    ("verify", 1),
    ("convert-bbbin", 0),
    ("convert-hfbin", 0),
]
DESTINATIONS = [(command, 1) for command in WRITING]


# A source that doesn't exist, and a folder where any file is named.
@pytest.mark.parametrize(
    ("command", "position", "path"),
    [(command, position, "missing.nc") for command, position in SOURCES]
    + [(command, position, ".") for command, position in SOURCES + DESTINATIONS],
)
def test_a_missing_source_is_a_usage_error(
    tmp_path: Path, command: str, position: int, path: str
) -> None:
    exists = tmp_path / "exists.nc"
    exists.write_bytes(b"")
    args = [exists, exists]
    args[position] = tmp_path / path  # "." names the folder itself.
    result = run(command, *args)
    assert_exit(result, EXIT_USAGE)
    message = "is a directory" if path == "." else "does not exist"
    assert message in result.stderr
