import re
import struct
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from waveform_compression_tools.bbbin import convert_bbbin, read_bbbin
from waveform_compression_tools.codec import (
    check_standard,
    max_relative_error,
)
from waveform_compression_tools.errors import NotBBBin
from waveform_compression_tools.files import (
    compress_file,
    decompress_file,
    verify_file,
)

from helpers import MakeBbbin, names_in, read


# Shapes with one station, one sample, or both, where a dimension could be lost.
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(("nstat", "nt"), [(1, 1), (1, 9), (4, 1), (2, 3), (5, 40)])
def test_every_sample_lands_in_its_place(
    tmp_path: Path, endian: str, nstat: int, nt: int, make_bbbin: MakeBbbin
) -> None:
    data = make_bbbin(tmp_path / "BB.bin", nstat=nstat, nt=nt, endian=endian)
    ds = read_bbbin(tmp_path / "BB.bin")
    # The file has (station, time, component) with 090 first; the dataset has
    # (component, station, time) with 000 first.
    expected = np.moveaxis(data[:, :, [1, 0, 2]], 2, 0)
    assert ds["waveform"].shape == (3, nstat, nt)
    np.testing.assert_array_equal(ds["waveform"].values, expected)
    assert list(ds["component"].values) == ["000", "090", "ver"]
    assert ds["waveform"].attrs["units"] == "g"
    assert ds["time"].shape == (nt,)
    assert ds["station"].shape == (nstat,)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_the_station_records(
    tmp_path: Path, endian: str, make_bbbin: MakeBbbin
) -> None:
    make_bbbin(tmp_path / "BB.bin", nstat=3, nt=7, endian=endian)
    ds = read_bbbin(tmp_path / "BB.bin")
    expected = {
        "longitude": ([170.5, 171.5, 172.5], np.float32),
        "latitude": ([-40.5, -39.5, -38.5], np.float32),
        "x": ([11, 12, 13], np.int32),
        "y": ([22, 23, 24], np.int32),
        "z": ([33, 34, 35], np.int32),
        "e_dist": ([44.5, 45.5, 46.5], np.float32),
        "hf_vs_ref": ([555.5, 556.5, 557.5], np.float32),
        "lf_vs_ref": ([666.5, 667.5, 668.5], np.float32),
        "vsite": ([777.5, 778.5, 779.5], np.float32),
    }
    for name, (values, dtype) in expected.items():
        assert ds[name].dims == ("station",), name
        assert ds[name].dtype == dtype, name
        np.testing.assert_array_equal(ds[name].values, values, err_msg=name)
    # Longitude and latitude are coordinates; the rest are variables.
    assert {"longitude", "latitude"} <= set(ds.coords)
    assert set(expected) - {"longitude", "latitude"} <= set(ds.data_vars)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_the_header(tmp_path: Path, endian: str, make_bbbin: MakeBbbin) -> None:
    make_bbbin(tmp_path / "BB.bin", nstat=3, nt=7, endian=endian)
    ds = read_bbbin(tmp_path / "BB.bin")
    # The header holds float32; dt is the float32 nearest 0.005, exactly.
    dt = float(np.float32(0.005))
    assert ds.attrs == {
        "dt": dt,
        "start_sec": -1.0,
        "duration": 123.5,
        "lf_dir": "/runs/fault1/rel01/LF",
        "lf_vm": "/models/fault1/vm",
        "hf_file": "/runs/fault1/rel01/HF/Acc/HF.bin",
        "source_format": "BB.bin",
    }
    assert ds["time"].dims == ("time",)
    assert ds["time"].dtype == np.float64
    np.testing.assert_array_equal(ds["time"].values, -1.0 + dt * np.arange(7))


def test_station_names_lose_only_their_padding(
    tmp_path: Path, make_bbbin: MakeBbbin
) -> None:
    # The record's name is 8 bytes, padded with NULs: a name of 8 characters has
    # none, and keeps all of them.
    names = ["A", "KIKS", "S000000", "ABCDEFGH"]
    make_bbbin(tmp_path / "BB.bin", nstat=4, nt=3, endian="<", names=names)
    ds = read_bbbin(tmp_path / "BB.bin")
    assert list(ds["station"].values) == names
    assert all(isinstance(name, str) for name in ds["station"].values)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        pytest.param("Ōtaki", "text that isn't ASCII", id="not ASCII"),
        pytest.param("AB\0CD", "text with a NUL inside it", id="a NUL inside"),
    ],
)
@pytest.mark.parametrize("where", ["a station's name", "a string of the header"])
def test_text_that_is_not_ascii_or_holds_a_nul_is_refused(
    tmp_path: Path, where: str, text: str, reason: str, make_bbbin: MakeBbbin
) -> None:
    # It isn't changed into something that might differ from what the file holds: it
    # loses only the NULs that pad it.
    path = tmp_path / "BB.bin"
    if where == "a station's name":
        make_bbbin(path, nstat=1, nt=3, endian="<", names=[text])
    else:
        make_bbbin(path, nstat=1, nt=3, endian="<")
        content, raw = bytearray(path.read_bytes()), text.encode()
        content[20 : 20 + len(raw)] = raw  # At the start of lf_dir.
        path.write_bytes(content)
    message = re.escape(f"{path}: not a BB.bin file: it holds {reason}")
    with pytest.raises(NotBBBin, match=f"^{message}$"):
        read_bbbin(path)
    with pytest.raises(NotBBBin, match=f"^{message}$"):
        convert_bbbin(path, tmp_path / "BB.nc")
    assert names_in(tmp_path) == ["BB.bin"]


def header_saying(nstat: int, nt: int, size: int) -> bytes:
    # The start of a file whose header counts nstat stations of nt samples, zero-filled
    # to `size` bytes.
    return struct.pack("<2i3f", nstat, nt, 0.0, 0.005, 0.0).ljust(size, b"\0")


@pytest.mark.parametrize(
    ("content", "cut_short"),
    [
        pytest.param(b"", False, id="empty"),
        pytest.param(b"\0" * 7, False, id="shorter than the counts"),
        pytest.param(header_saying(3, 7, 0x4FF), False, id="shorter than the header"),
        # These fit the size a BB.bin of their counts would have, so only the counts
        # can refuse them: a file of zeros the size of a header is not an empty BB.bin.
        pytest.param(b"\0" * 0x500, False, id="zeros the size of a header"),
        pytest.param(header_saying(0, 5, 0x500), False, id="no stations"),
        pytest.param(
            header_saying(3, 0, 0x500 + 3 * 0x2C), False, id="stations with no samples"
        ),
        # In 32-bit arithmetic, 2**30 samples of 12 bytes would take no room at all.
        # The file holds the header and the one record, so it may be one cut short.
        pytest.param(
            header_saying(1, 2**30, 0x500 + 0x2C), True, id="a size that wraps around"
        ),
        # Counts below zero would put any size between the end of the records and
        # the end of the traces.
        pytest.param(
            header_saying(-(2**31), -(2**31), 0x500 + 0x100),
            False,
            id="counts below zero",
        ),
    ],
)
def test_a_file_that_cannot_be_a_bbbin_is_refused(
    tmp_path: Path, content: bytes, cut_short: bool
) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(content)
    what = "not a BB.bin file, or one cut short" if cut_short else "not a BB.bin file"
    # The message names the file once, at its start, which the error line drops.
    message = re.escape(f"{path}: {what} ({len(content)} bytes)")
    with pytest.raises(ValueError, match=f"^{message}$"):
        read_bbbin(path)


# The size of the header and of the three records of the file the size test writes.
RECORDS_END = 0x500 + 3 * 0x2C


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(
    ("change", "cut_short"),
    [
        pytest.param(lambda content: content[:-1], True, id="a byte short"),
        pytest.param(lambda content: content + b"\0", False, id="a byte too many"),
        pytest.param(lambda content: content[:-12], True, id="a sample short"),
        pytest.param(
            lambda content: content[:RECORDS_END], True, id="every record, no trace"
        ),
        pytest.param(
            lambda content: content[: RECORDS_END - 1],
            False,
            id="a byte short of its records",
        ),
    ],
)
def test_the_size_has_to_fit_the_header(
    tmp_path: Path,
    endian: str,
    change: Callable[[bytes], bytes],
    cut_short: bool,
    make_bbbin: MakeBbbin,
) -> None:
    # A file that holds the whole header and every record, but not every trace, may be
    # one cut short, and its refusal says so.
    path = tmp_path / "BB.bin"
    make_bbbin(path, nstat=3, nt=7, endian=endian)
    path.write_bytes(change(path.read_bytes()))
    what = "not a BB.bin file, or one cut short" if cut_short else "not a BB.bin file"
    message = re.escape(f"{path}: {what} ({path.stat().st_size} bytes)")
    with pytest.raises(ValueError, match=f"^{message}$"):
        read_bbbin(path)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_convert_bbbin_writes_a_netcdf_file(
    tmp_path: Path, endian: str, make_bbbin: MakeBbbin
) -> None:
    # A big-endian file's numbers have to be turned into the machine's, in every
    # variable: a file written from them would keep their byte order, and a waveform
    # read back from it in that order isn't the standard layout's float32.
    make_bbbin(tmp_path / "BB.bin", nstat=3, nt=7, endian=endian)
    ds = read_bbbin(tmp_path / "BB.bin")
    nc = tmp_path / "BB.nc"
    convert_bbbin(tmp_path / "BB.bin", nc)
    # The BB.bin and the converted file, and no partial file.
    assert names_in(tmp_path) == ["BB.bin", "BB.nc"]
    assert h5py.is_hdf5(nc)
    written = read(nc)
    for dataset in (ds, written):
        check_standard(dataset)
        assert all(variable.dtype.isnative for variable in dataset.variables.values())
    xr.testing.assert_identical(written, ds)


def test_converted_file_compresses(tmp_path: Path, make_bbbin: MakeBbbin) -> None:
    # Seven characters name a point of a synthetic grid; KIKS, like any other name of
    # another length, is a real station, whose waveform is kept exactly.
    names = ["S000000", "S000001", "KIKS", "S000003", "S000004"]
    data = make_bbbin(tmp_path / "BB.bin", nstat=5, nt=400, endian="<", names=names)
    nc = tmp_path / "BB.nc"
    convert_bbbin(tmp_path / "BB.bin", nc)
    result = compress_file(nc, tmp_path / "BB.flac.nc")
    assert (result.n_traces, result.n_exact) == (15, 1)
    assert verify_file(nc, tmp_path / "BB.flac.nc").passed
    # What comes out is the BB.bin's waveform: the real station exactly, and the rest to
    # within the setting's bound.
    expected = np.moveaxis(data[:, :, [1, 0, 2]], 2, 0)
    got = decompress_file(tmp_path / "BB.flac.nc")["waveform"].values
    assert max_relative_error(expected, got).max() <= 4.7e-10
    np.testing.assert_array_equal(got[:, 2], expected[:, 2])
