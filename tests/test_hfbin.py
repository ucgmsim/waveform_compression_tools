import struct
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from waveform_compression_tools import hfbin
from waveform_compression_tools.bbbin import read_bbbin
from waveform_compression_tools.codec import check_standard, max_relative_error
from waveform_compression_tools.errors import IncompleteHFBin, NotBBBin, NotHFBin
from waveform_compression_tools.files import (
    compress_file,
    decompress_file,
    verify_file,
)
from waveform_compression_tools.hfbin import convert_hfbin, read_hfbin

from helpers import (
    HFBIN_FLOATS,
    HFBIN_INTS,
    HFBIN_TEXTS,
    MakeBbbin,
    MakeHfbin,
    names_in,
    read,
)


# Shapes with one station, one sample, or both, where a dimension could be lost.
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(("nstat", "nt"), [(1, 1), (1, 9), (4, 1), (2, 3), (5, 40)])
def test_every_sample_lands_in_its_place(
    tmp_path: Path, endian: str, nstat: int, nt: int, make_hfbin: MakeHfbin
) -> None:
    data = make_hfbin(tmp_path / "HF.bin", nstat=nstat, nt=nt, endian=endian)
    ds = read_hfbin(tmp_path / "HF.bin")
    # The file has (station, time, component) with 090 first; the dataset has
    # (component, station, time) with 000 first, and the values as stored, in cm/s^2.
    expected = np.moveaxis(data[:, :, [1, 0, 2]], 2, 0)
    assert ds["waveform"].shape == (3, nstat, nt)
    np.testing.assert_array_equal(ds["waveform"].values, expected)
    assert list(ds["component"].values) == ["000", "090", "ver"]
    assert ds["waveform"].attrs == {"units": "cm/s^2"}
    assert ds["time"].shape == (nt,)
    assert ds["station"].shape == (nstat,)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_the_station_records(
    tmp_path: Path, endian: str, make_hfbin: MakeHfbin
) -> None:
    # The record's name is 8 bytes, padded with NULs: a name of 8 characters has none,
    # and keeps all of them.
    names = ["A", "KIKS", "S000000", "ABCDEFGH"]
    make_hfbin(tmp_path / "HF.bin", nstat=4, nt=7, endian=endian, names=names)
    ds = read_hfbin(tmp_path / "HF.bin")
    assert ds["station"].values.tolist() == names
    expected = {
        "longitude": [170.5, 171.5, 172.5, 173.5],
        "latitude": [-40.5, -39.5, -38.5, -37.5],
        "e_dist": [44.5, 45.5, 46.5, 47.5],
        "vs": [555.5, 556.5, 557.5, 558.5],
    }
    for name, values in expected.items():
        assert ds[name].dims == ("station",), name
        assert ds[name].dtype == np.float32, name
        np.testing.assert_array_equal(ds[name].values, values, err_msg=name)
    # The names, longitudes and latitudes are coordinates; e_dist and vs are variables.
    assert set(ds.coords) == {"component", "station", "longitude", "latitude", "time"}
    assert set(ds.data_vars) == {"waveform", "e_dist", "vs"}


@pytest.mark.parametrize("endian", ["<", ">"])
def test_the_header(tmp_path: Path, endian: str, make_hfbin: MakeHfbin) -> None:
    # Every field becomes an attribute as the file holds it: the int32s as int32, the
    # float32s as float32, so dt is the float32 nearest 0.005, and the strings as text
    # without their NULs.
    make_hfbin(tmp_path / "HF.bin", nstat=3, nt=7, endian=endian)
    ds = read_hfbin(tmp_path / "HF.bin")
    expected = {
        "nstat": np.int32(3),
        "nt": np.int32(7),
        **{name: np.int32(value) for name, value in HFBIN_INTS.items()},
        **{name: np.float32(value) for name, value in HFBIN_FLOATS.items()},
        **HFBIN_TEXTS,
        "source_format": "HF.bin",
    }
    assert ds.attrs == expected
    assert {name: type(value) for name, value in ds.attrs.items()} == {
        name: type(value) for name, value in expected.items()
    }
    # Each sample's time, in float64, from the header's start and step.
    dt = float(np.float32(0.005))
    assert ds["time"].dims == ("time",)
    assert ds["time"].dtype == np.float64
    np.testing.assert_array_equal(ds["time"].values, -1.0 + dt * np.arange(7))


@pytest.mark.parametrize("where", ["a station's name", "a string of the header"])
def test_text_that_is_not_ascii_is_refused(
    tmp_path: Path, where: str, make_hfbin: MakeHfbin
) -> None:
    # It isn't changed into something that might differ from what the file holds.
    path = tmp_path / "HF.bin"
    if where == "a station's name":
        make_hfbin(path, nstat=1, nt=3, endian="<", names=["Ōtaki"])
    else:
        make_hfbin(path, nstat=1, nt=3, endian="<")
        content, text = bytearray(path.read_bytes()), "Ōtaki".encode()
        at = 4 * (16 + 24) + 64  # The start of velocity_model, the second string.
        content[at : at + len(text)] = text
        path.write_bytes(content)
    with pytest.raises(NotHFBin, match="ASCII"):
        read_hfbin(path)
    with pytest.raises(NotHFBin, match="ASCII"):
        convert_hfbin(path, tmp_path / "HF.nc")
    assert names_in(tmp_path) == ["HF.bin"]


def header_saying(nstat: int, nt: int, size: int) -> bytes:
    # The start of a file whose header counts nstat stations of nt samples, zero-filled
    # to `size` bytes.
    return struct.pack("<2i", nstat, nt).ljust(size, b"\0")


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\0" * 7, id="shorter than the counts"),
        pytest.param(header_saying(3, 7, 0x1FF), id="shorter than the header"),
        # These fit the size an HF.bin of their counts would have, so only the counts
        # can refuse them: a header of zeros, say, is not an empty HF.bin.
        pytest.param(header_saying(0, 5, 0x200), id="no stations"),
        pytest.param(
            header_saying(3, 0, 0x200 + 3 * 0x18), id="stations with no samples"
        ),
        # In 32-bit arithmetic, 2**30 samples of 12 bytes would take no room at all.
        pytest.param(
            header_saying(1, 2**30, 0x200 + 0x18), id="a size that wraps around"
        ),
    ],
)
def test_a_file_that_cannot_be_an_hfbin_is_refused(
    tmp_path: Path, content: bytes
) -> None:
    (tmp_path / "x.bin").write_bytes(content)
    with pytest.raises(NotHFBin, match="not an HF.bin file"):
        read_hfbin(tmp_path / "x.bin")


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda content: content[:-1], id="a byte short"),
        pytest.param(lambda content: content + b"\0", id="a byte too many"),
        pytest.param(lambda content: content[:-12], id="a sample short"),
    ],
)
def test_the_size_has_to_fit_the_header(
    tmp_path: Path,
    endian: str,
    change: Callable[[bytes], bytes],
    make_hfbin: MakeHfbin,
) -> None:
    path = tmp_path / "HF.bin"
    make_hfbin(path, nstat=3, nt=7, endian=endian)
    path.write_bytes(change(path.read_bytes()))
    with pytest.raises(NotHFBin, match="not an HF.bin file"):
        read_hfbin(path)


def test_neither_format_is_taken_for_the_other(
    tmp_path: Path, make_bbbin: MakeBbbin, make_hfbin: MakeHfbin
) -> None:
    # Their headers and records differ in size, so a file of one never fits the other's
    # header, whatever its numbers of stations and samples.
    make_bbbin(tmp_path / "BB.bin", nstat=3, nt=7)
    make_hfbin(tmp_path / "HF.bin", nstat=3, nt=7)
    with pytest.raises(NotHFBin, match="not an HF.bin file"):
        read_hfbin(tmp_path / "BB.bin")
    with pytest.raises(NotBBBin, match="not a BB.bin file"):
        read_bbbin(tmp_path / "HF.bin")


def test_an_unfinished_file_is_refused(tmp_path: Path, make_hfbin: MakeHfbin) -> None:
    # The writer fills in a station's vs once the station is done, so a vs of 0 means
    # that the run stopped first. The refusal says how many stations it left unfinished.
    path = tmp_path / "HF.bin"
    make_hfbin(path, nstat=5, nt=7, unfinished=(1, 3))
    with pytest.raises(IncompleteHFBin, match="with 2 of its 5 stations unfinished"):
        read_hfbin(path)
    with pytest.raises(IncompleteHFBin):
        convert_hfbin(path, tmp_path / "HF.nc")
    assert names_in(tmp_path) == ["HF.bin"]


@pytest.mark.parametrize("endian", ["<", ">"])
def test_convert_hfbin_writes_a_netcdf_file(
    tmp_path: Path, endian: str, make_hfbin: MakeHfbin
) -> None:
    # A big-endian file's numbers have to be turned into the machine's, in every
    # variable: a file written from them would keep their byte order, and a waveform
    # read back from it in that order isn't the standard layout's float32.
    make_hfbin(tmp_path / "HF.bin", nstat=3, nt=7, endian=endian)
    ds = read_hfbin(tmp_path / "HF.bin")
    nc = tmp_path / "HF.nc"
    convert_hfbin(tmp_path / "HF.bin", nc)
    # The HF.bin and the converted file, and no partial file.
    assert names_in(tmp_path) == ["HF.bin", "HF.nc"]
    assert h5py.is_hdf5(nc)
    written = read(nc)
    for dataset in (ds, written):
        check_standard(dataset)
        assert all(variable.dtype.isnative for variable in dataset.variables.values())
    xr.testing.assert_identical(written, ds)
    # The attributes keep their types in the file, as the header has them.
    assert {name: type(value) for name, value in written.attrs.items()} == {
        name: type(value) for name, value in ds.attrs.items()
    }


def test_converted_file_compresses(tmp_path: Path, make_hfbin: MakeHfbin) -> None:
    # Seven characters name a point of a synthetic grid; KIKS, like any other name of
    # another length, is a real station, whose waveform is kept exactly. The paths are
    # strings, as any path can be.
    names = ["S000000", "S000001", "KIKS", "S000003", "S000004"]
    data = make_hfbin(tmp_path / "HF.bin", nstat=5, nt=400, endian="<", names=names)
    nc = tmp_path / "HF.nc"
    convert_hfbin(str(tmp_path / "HF.bin"), str(nc))
    result = compress_file(nc, tmp_path / "HF.flac.nc")
    assert (result.n_traces, result.n_exact) == (15, 1)
    assert verify_file(nc, tmp_path / "HF.flac.nc").passed
    # What comes out is the HF.bin's waveform: the real station exactly, and the rest to
    # within the setting's bound.
    expected = np.moveaxis(data[:, :, [1, 0, 2]], 2, 0)
    got = decompress_file(tmp_path / "HF.flac.nc")["waveform"].values
    assert max_relative_error(expected, got).max() <= 4.7e-10
    np.testing.assert_array_equal(got[:, 2], expected[:, 2])


def test_the_destination_folder_is_checked_before_the_file_is_read(
    tmp_path: Path, make_hfbin: MakeHfbin, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As for convert_bbbin: a mistyped folder is found before any work is done.
    make_hfbin(tmp_path / "HF.bin", nstat=3, nt=7)

    def too_late(path: object) -> None:
        raise AssertionError("the input was read")

    monkeypatch.setattr(hfbin, "read_hfbin", too_late)
    nowhere = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError, match="there is no folder"):
        convert_hfbin(tmp_path / "HF.bin", nowhere / "out.nc")
    assert not nowhere.exists()
