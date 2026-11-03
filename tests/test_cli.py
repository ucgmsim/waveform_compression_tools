import errno
import struct
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner, Result

from waveform_compression_tools.cli import app
from waveform_compression_tools.codec import (
    STANDARD_DIMS,
    max_relative_error,
)
from waveform_compression_tools.files import GROUP, is_compressed
from waveform_compression_tools.quantise import full_scale

StandardDataset = Callable[..., xr.Dataset]

runner = CliRunner()


def run(*args: str | int | Path) -> Result:
    return runner.invoke(app, [str(arg) for arg in args])


def assert_exit(result: Result, status: int) -> None:
    # The command chose its exit status. An exception that escaped it would give status
    # 1 as well, so a failure has to be a deliberate exit, which Typer raises as a
    # SystemExit.
    assert result.exit_code == status, result.output
    if status == 0:
        assert result.exception is None
    else:
        assert isinstance(result.exception, SystemExit), repr(result.exception)


def write(ds: xr.Dataset, path: Path) -> Path:
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def partial_of(path: Path) -> Path:
    return Path(f"{path}.partial")


def write_bbbin(path: Path, names: list[str], nt: int) -> None:
    # A little-endian BB.bin file, packed with struct: a header of 0x500 bytes (the
    # numbers of stations and samples, then duration, dt and start_sec, then three
    # empty strings), a record of 0x2C bytes for each station, then float32 samples of
    # shape (station, time, 3).
    rng = np.random.default_rng(0)
    header = struct.pack("<2i3f", len(names), nt, nt * 0.005, 0.005, 0.0)
    records = b"".join(
        struct.pack(
            "<ff8s3i4f", 170.5, -40.5, name.encode(), 1, 2, 3, 4.0, 5.0, 6.0, 7.0
        )
        for name in names
    )
    samples = 0.01 * np.cumsum(rng.normal(size=(len(names), nt, 3)), axis=1)
    path.write_bytes(
        header.ljust(0x500, b"\0") + records + samples.astype("<f4").tobytes()
    )


def compressed_file(ds: xr.Dataset, tmp_path: Path) -> tuple[Path, Path]:
    # Writes a dataset and compresses it with the command, returning both files.
    src = write(ds, tmp_path / "in.nc")
    dst = tmp_path / "packed.nc"
    assert_exit(run("compress", src, dst), 0)
    return src, dst


# A small file grows, as the compressed bytes are stored in chunks of 4 MiB; a larger
# one shrinks. The size is reported as a factor of the bigger over the smaller.


@pytest.mark.parametrize("bits", ["7", "33"])
def test_compress_refuses_a_setting_outside_8_to_32(
    tmp_path: Path, standard_dataset: StandardDataset, bits: str
) -> None:
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    result = run("compress", src, dst, "--bits", bits)
    assert_exit(result, 1)
    assert "bits" in result.stderr and result.stdout == ""
    assert not dst.exists() and not partial_of(dst).exists()


def test_the_setting_reaches_the_file(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(3, nt=200), tmp_path / "in.nc")
    for bits in (8, 24):
        dst = tmp_path / f"bits{bits}.nc"
        result = run("compress", src, dst, "--bits", bits)
        assert_exit(result, 0)
        assert f"at {bits} bits" in result.stdout
        with h5py.File(dst, "r") as f:
            assert int(f[GROUP].attrs["bits"]) == bits
        assert_exit(run("verify", src, dst), 0)


@pytest.mark.parametrize("level", ["-1", "9"])
def test_compress_refuses_a_level_outside_0_to_8(
    tmp_path: Path, standard_dataset: StandardDataset, level: str
) -> None:
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    result = run("compress", src, dst, "--level", level)
    assert_exit(result, 2)  # A usage error, as Typer reports one.
    assert "--level" in result.stderr
    assert not dst.exists() and not partial_of(dst).exists()


def test_compress_leaves_a_file_of_real_stations_as_it_is(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(3, nt=100, real=(0, 1, 2)), tmp_path / "in.nc")
    before = src.read_bytes()
    dst = tmp_path / "out.nc"
    result = run("compress", src, dst)
    assert_exit(result, 1)
    assert result.stderr.startswith("Error: every station of ")
    assert "is real" in result.stderr
    assert "stays as it is" in result.stderr and result.stdout == ""
    assert not dst.exists() and not partial_of(dst).exists()
    assert src.read_bytes() == before


def test_compress_refuses_a_file_that_is_not_in_the_standard_layout(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"] = ds["waveform"].astype(np.float64)
    src = write(ds, tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    result = run("compress", src, dst)
    assert_exit(result, 1)
    assert "float32" in result.stderr and result.stdout == ""
    assert not dst.exists() and not partial_of(dst).exists()


def test_compress_refuses_an_attribute_that_cannot_be_stored(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # The waveform's attributes are stored as JSON, which has no complex numbers.
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    with h5py.File(src, "r+") as f:
        f["waveform"].attrs["gain"] = np.complex128(1 + 2j)
    dst = tmp_path / "out.nc"
    result = run("compress", src, dst)
    assert_exit(result, 1)
    assert "JSON" in result.stderr and result.stdout == ""
    assert not dst.exists() and not partial_of(dst).exists()


def test_decompress_gives_back_the_standard_layout(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src, packed = compressed_file(standard_dataset(5, nt=300, real=(1,)), tmp_path)
    back = tmp_path / "back.nc"
    result = run("decompress", packed, back)
    assert_exit(result, 0)
    assert "15 traces of 300 samples" in result.stdout
    assert not partial_of(back).exists() and not is_compressed(back)
    with (
        xr.open_dataset(src, engine="h5netcdf") as orig,
        xr.open_dataset(back, engine="h5netcdf") as got,
    ):
        assert got["waveform"].dims == STANDARD_DIMS
        assert got["waveform"].dtype == np.float32
        assert got["waveform"].attrs == {"units": "g"}
        xr.testing.assert_identical(
            got.drop_vars("waveform"), orig.drop_vars("waveform")
        )
        original, decoded = orig["waveform"].values, got["waveform"].values
    assert max_relative_error(original, decoded).max() <= 1 / full_scale(32)
    # Station 1 is real, so it comes back as it was, to the bit.
    assert np.array_equal(original[:, 1].view(np.uint32), decoded[:, 1].view(np.uint32))


def test_decompress_refuses_a_file_that_is_not_compressed(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    result = run("decompress", src, dst)
    assert_exit(result, 1)
    assert "not compressed" in result.stderr and result.stdout == ""
    assert not dst.exists() and not partial_of(dst).exists()


# What the command shows when the write fails, or None where the failure is not one it
# expects and the exception escapes it. Typer ends an interruption its own way, so
# nothing in particular is looked for in that one.
@pytest.mark.parametrize(
    ("error", "shown"),
    [
        pytest.param(OSError(errno.ENOSPC, "No space left on device"), "No space left"),
        pytest.param(KeyboardInterrupt(), ""),
        pytest.param(RuntimeError("something unexpected"), None),
    ],
    ids=["disk", "interruption", "unexpected"],
)
def test_a_failed_write_leaves_no_partial_file_and_keeps_the_old_one(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    shown: str | None,
) -> None:
    _, packed = compressed_file(standard_dataset(3, nt=100), tmp_path)
    back = tmp_path / "back.nc"
    back.write_bytes(b"an earlier file")

    def write_part_then_fail(
        self: xr.Dataset, path: Path, *args: object, **kwargs: object
    ) -> None:
        Path(path).write_bytes(b"half a file")
        raise error

    monkeypatch.setattr(xr.Dataset, "to_netcdf", write_part_then_fail)
    result = run("decompress", packed, back)
    if shown is None:
        # Not hidden behind a message: it's a bug, not a bad file.
        assert result.exit_code == 1 and result.exception is error
    else:
        assert result.exit_code != 0 and shown in result.output
    assert back.read_bytes() == b"an earlier file" and not partial_of(back).exists()


def test_convert_bbbin_refuses_a_file_that_is_not_a_bbbin(tmp_path: Path) -> None:
    src, dst = tmp_path / "BB.bin", tmp_path / "BB.nc"
    src.write_bytes(bytes(2000))
    result = run("convert-bbbin", src, dst)
    assert_exit(result, 1)
    assert "not a BB.bin file" in result.stderr and result.stdout == ""
    assert not dst.exists() and not partial_of(dst).exists()


@pytest.mark.parametrize("command", ["decompress", "convert-bbbin"])
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
    # whatever the spelling it is the same file.
    monkeypatch.chdir(tmp_path)
    src, dst = Path(src_name), tmp_path / "data.nc"
    if command == "decompress":
        original = write(standard_dataset(2, nt=50), tmp_path / "original.nc")
        assert_exit(run("compress", original, src), 0)
    else:
        write_bbbin(src, ["S000000", "S000001"], nt=20)
    before = src.read_bytes()
    result = run(command, src, dst)
    assert_exit(result, 1)
    assert "overwrite" in result.stderr
    assert src.read_bytes() == before


# The arguments of the commands, each as its command and its position: the files that a
# command reads, and the one it writes.
SOURCES = [
    ("compress", 0),
    ("decompress", 0),
    ("verify", 0),
    ("verify", 1),
    ("convert-bbbin", 0),
]
DESTINATIONS = [("compress", 1), ("decompress", 1), ("convert-bbbin", 1)]


@pytest.mark.parametrize(("command", "position"), SOURCES)
def test_a_missing_source_is_a_usage_error(
    tmp_path: Path, command: str, position: int
) -> None:
    exists = tmp_path / "exists.nc"
    exists.write_bytes(b"")
    args = [exists, exists]
    args[position] = tmp_path / "missing.nc"
    result = run(command, *args)
    assert_exit(result, 2)
    assert "does not exist" in result.stderr
