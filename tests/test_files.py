import json
import shutil
import string
import struct
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pyflac
import pytest
import xarray as xr

from waveform_compression_tools import files
from waveform_compression_tools.bbbin import convert_bbbin, read_bbbin
from waveform_compression_tools.codec import (
    STANDARD_DIMS,
    EncodedBlock,
    encode_block,
    max_relative_error,
    within_one_step,
)
from waveform_compression_tools.content_hash import content_hash
from waveform_compression_tools.destinations import refuse_overwriting, write_netcdf
from waveform_compression_tools.errors import NotStandardLayout
from waveform_compression_tools.files import (
    GROUP,
    check_compressible,
    compress_file,
    decompress_file,
    is_compressed,
    verify_file,
)
from waveform_compression_tools.quantise import full_scale, gain_for, quantise
from waveform_compression_tools.record import write_record
from waveform_compression_tools.stream import encode_stream

from helpers import StandardDataset, partial_of, within_one_step_of_the_setting, write


def bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
    # Compares the bits of float32 arrays, so -0.0 differs from 0.0 and a NaN can match.
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


def rewrite(group: h5py.Group, name: str, data: np.ndarray) -> None:
    # Replaces a dataset in a group, whatever its shape or type.
    del group[name]
    group.create_dataset(name, data=data)


def streaminfo(stream: bytes) -> dict[str, int]:
    # A FLAC stream is the marker "fLaC" and then the STREAMINFO block: a 4-byte block
    # header and 34 bytes of data. Counting from the start of the stream, bytes 8 to
    # 12 of it are the minimum and maximum block sizes, and bytes 18 to 26 pack the
    # sample rate, channel count and bits per sample (20, 3 and 5 bits) ahead of the
    # total sample count.
    assert stream[:4] == b"fLaC"
    min_blocksize, max_blocksize = struct.unpack(">HH", stream[8:12])
    packed = int.from_bytes(stream[18:26], "big")
    return {
        "min_blocksize": min_blocksize,
        "max_blocksize": max_blocksize,
        "sample_rate": packed >> 44,
        "channels": ((packed >> 41) & 0b111) + 1,
        "bits_per_sample": ((packed >> 36) & 0b11111) + 1,
    }


def decode_with_pyflac(stream: bytes, folder: Path) -> np.ndarray:
    # pyflac's public StreamDecoder, which this package's decoder doesn't use.
    frames: list[np.ndarray] = []
    decoder = pyflac.StreamDecoder(
        write_callback=lambda samples, rate, channels, n: frames.append(samples.copy())
    )
    decoder.process(stream)
    decoder.finish()
    return np.concatenate(frames).ravel()


def decode_with_ffmpeg(stream: bytes, folder: Path) -> np.ndarray:
    # As the README says: ffmpeg -i trace.flac -f s32le trace.raw.
    flac, raw = folder / "trace.flac", folder / "trace.raw"
    flac.write_bytes(stream)
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(flac)]
    subprocess.run([*command, "-f", "s32le", str(raw)], check=True)
    return np.fromfile(raw, "<i4")


# FLAC decoders independent of this package's own. Each takes a stream's bytes, and a
# folder for any files it needs, and gives back its samples.
DECODERS = {"pyflac's StreamDecoder": decode_with_pyflac, "ffmpeg": decode_with_ffmpeg}


# Station counts on both sides of a multiple of 344, the stations in a block, and an
# odd number of samples: every station has to be compressed exactly once.
@pytest.mark.parametrize("bits", [8, 17, 24, 32])
@pytest.mark.parametrize("n_station", [1, 343, 344, 345, 700])
def test_round_trip_every_station(
    tmp_path: Path, standard_dataset: StandardDataset, n_station: int, bits: int
) -> None:
    src, dst = tmp_path / "realisation.bb", tmp_path / "out.bb"
    standard_dataset(n_station, nt=1001).to_netcdf(src, engine="h5netcdf")
    res = compress_file(src, dst, bits=bits, block_stations=344)
    assert res.n_traces == 3 * n_station
    assert dst.exists() and not partial_of(dst).exists()
    back = decompress_file(dst)
    with xr.open_dataset(src, engine="h5netcdf") as orig:
        assert back["waveform"].dims == STANDARD_DIMS
        assert back["waveform"].dtype == np.float32
        assert back["waveform"].attrs == {"units": "g"}
        xr.testing.assert_identical(
            back.drop_vars("waveform"), orig.drop_vars("waveform")
        )
        original, decoded = orig["waveform"].values, back["waveform"].values
    assert within_one_step_of_the_setting(original, decoded, bits).all()
    check = verify_file(src, dst, block_stations=344)
    assert check.passed
    assert check.max_relative_error == max_relative_error(original, decoded).max()


def test_every_path_can_be_a_string(
    tmp_path: Path, make_bbbin: Callable[..., np.ndarray]
) -> None:
    # Wherever a function takes a path, a string does as well as a Path: here each of
    # them in turn, from a BB.bin to the record of its compression.
    bb, nc, packed, back, rec = (
        str(tmp_path / name)
        for name in ("BB.bin", "bb.nc", "bb.flac.nc", "back.nc", "rec.json")
    )
    names = ["S000000", "KIKS", "S000002"]
    make_bbbin(Path(bb), nstat=3, nt=200, endian="<", names=names)
    assert read_bbbin(bb)["waveform"].shape == (3, 3, 200)
    assert convert_bbbin(bb, nc) == Path(nc)
    refuse_overwriting(nc, packed)
    check_compressible(nc, packed)
    assert compress_file(nc, packed, bits=24).n_exact == 1
    assert is_compressed(packed)
    assert verify_file(nc, packed).passed
    write_netcdf(decompress_file(packed), back)
    with xr.open_dataset(back, engine="h5netcdf") as got:
        assert got["waveform"].shape == (3, 3, 200)
    assert content_hash(packed) == content_hash(Path(packed))
    write_record({"status": "ok"}, rec)
    assert json.loads(Path(rec).read_text()) == {"status": "ok"}


def test_a_file_whose_components_are_x_y_z_comes_back_with_its_labels(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # Some files label their components x, y and z, not 000, 090 and ver.
    src, dst = tmp_path / "realisation.hf", tmp_path / "out.hf"
    original = standard_dataset(5, nt=1001).assign_coords(component=["x", "y", "z"])
    original.to_netcdf(src, engine="h5netcdf")
    compress_file(src, dst, bits=24)
    back = decompress_file(dst)
    assert list(back["component"].values) == ["x", "y", "z"]
    with xr.open_dataset(src, engine="h5netcdf") as orig:
        xr.testing.assert_identical(
            back.drop_vars("waveform"), orig.drop_vars("waveform")
        )
    assert verify_file(src, dst).passed


@pytest.mark.parametrize("decoder", DECODERS)
def test_a_file_decodes_by_the_readme_alone(
    tmp_path: Path, standard_dataset: StandardDataset, decoder: str
) -> None:
    # The README describes the file well enough to decode it with h5py and any FLAC
    # decoder, without this package. Traces of 5,000 samples take two frames each.
    if decoder == "ffmpeg" and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on the PATH")
    src = write(standard_dataset(4, nt=5_000, real=(1,)), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst, bits=24)
    with h5py.File(dst, "r") as f:
        assert "waveform" not in f
        # By the name the README gives it, as every file written so far holds it.
        g = f["_flac_compressed_waveform"]
        # The README's tables: the group's datasets, and its attributes.
        assert {name: (g[name].dtype, g[name].shape) for name in g} == {
            "compressed": (np.uint8, (g["compressed"].size,)),
            "stream_starts": (np.int64, (3, 4)),
            "stream_bytes": (np.int64, (3, 4)),
            "stream_gains": (np.float64, (3, 4)),
            "stream_offsets": (np.float64, (3, 4)),
            "exact_stations": (np.int64, (1,)),
            "exact_waveform": (np.float32, (3, 1, 5_000)),
        }
        assert g["stream_starts"].attrs["stream_size"] == 5_000
        assert g.attrs["format_version"] == 1
        assert list(g.attrs["dims"]) == ["component", "station", "time"]
        assert g.attrs["dtype"] == "float32"
        assert g.attrs["bits"] == 24
        assert json.loads(g.attrs["waveform_attrs"]) == {"units": "g"}
        assert json.loads(g.attrs["waveform_encoding"]) == {"_FillValue": "NaN"}
        assert json.loads(g.attrs["waveform_attr_types"]) == {}
        compressed = g["compressed"][...]
        starts, nbytes = g["stream_starts"][...], g["stream_bytes"][...]
        gains, offsets = g["stream_gains"][...], g["stream_offsets"][...]
        exact_stations = g["exact_stations"][...]
        exact_waveform = g["exact_waveform"][...]
    # The README's recipe, step by step.
    waveform = np.empty((3, 4, 5_000), np.float32)
    for c, s in np.ndindex(3, 4):
        stream = compressed[starts[c, s] : starts[c, s] + nbytes[c, s]].tobytes()
        info = streaminfo(stream)
        assert (info["channels"], info["bits_per_sample"]) == (1, 32)
        assert (info["min_blocksize"], info["max_blocksize"]) == (4_096, 4_096)
        assert info["sample_rate"] == 44_100
        q = DECODERS[decoder](stream, tmp_path)
        trace = q.astype(np.float64) / gains[c, s] + offsets[c, s]
        waveform[c, s] = trace.astype(np.float32)
    for k, s in enumerate(exact_stations):
        waveform[:, s] = exact_waveform[:, k]
    assert bit_identical(waveform, decompress_file(dst)["waveform"].values)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError, id="an error"),
        pytest.param(KeyboardInterrupt, id="an interruption"),
    ],
)
def test_failure_mid_write_leaves_nothing(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    error: type[BaseException],
) -> None:
    src = write(standard_dataset(5, nt=100), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    encoded: list[tuple[int, ...]] = []

    def encode_then_fail(waveform: np.ndarray, bits: int, level: int) -> EncodedBlock:
        if encoded:
            assert partial_of(dst).exists()  # The first block is in it.
            raise error("encoding failed")
        encoded.append(waveform.shape)
        return encode_block(waveform, bits, level)

    monkeypatch.setattr(files, "encode_block", encode_then_fail)
    with pytest.raises(error, match="encoding failed"):
        compress_file(src, dst, block_stations=2)
    assert encoded == [(3, 2, 100)]
    assert not dst.exists() and not partial_of(dst).exists()


def test_a_stream_that_decodes_short_is_named_by_its_station_in_the_file(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # In blocks of 2 stations, station 3 is the second of the second block: the message
    # counts it from the file's first station, not its block's.
    src = write(standard_dataset(5, nt=100), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r+") as f:
        f[GROUP]["stream_bytes"][1, 3] = 0  # Station 3's second stream, cut to nothing.
    message = "component 1, station 3: the stream decodes to 0 samples"
    with pytest.raises(ValueError, match=message):
        verify_file(src, dst, block_stations=2)


def test_a_small_file_has_chunks_no_larger_than_its_waveform(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # Chunks of 4 MiB would make every compressed file take at least that much.
    src = write(standard_dataset(3, nt=100), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r") as f:
        (chunk,) = f[GROUP]["compressed"].chunks
    assert chunk <= 3 * 3 * 100 * 4  # The waveform's bytes, as float32.
    assert verify_file(src, dst).passed


def test_verify_fails_a_trace_that_decodes_a_few_steps_off(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(3, nt=400), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst, bits=24)
    assert verify_file(src, dst).passed
    # A gain 4 parts in full scale too large brings the trace's peak in by 4 steps.
    step = 1 / full_scale(24)
    with h5py.File(dst, "r+") as f:
        gains = f[GROUP]["stream_gains"]
        gains[2, 1] = gains[2, 1] * (1 + 4 * step)
    check = verify_file(src, dst)
    assert not check.passed and 3 * step < check.max_relative_error < 6 * step


def test_verify_holds_a_file_to_the_bound_its_setting_promises(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # Quantised at 24 bits but labelled 32: every sample is within one step of its own
    # gain, but not within the 1 / full_scale(32) of its peak that 32 bits promise.
    src = write(standard_dataset(3, nt=400), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst, bits=24)
    assert verify_file(src, dst).passed
    with h5py.File(dst, "r+") as f:
        f[GROUP].attrs["bits"] = 32
    check = verify_file(src, dst)
    assert not check.passed
    assert 1 / full_scale(32) < check.max_relative_error <= 1 / full_scale(24)


def test_verify_fails_a_station_that_never_moved_but_decodes_to_something(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # Its peak is 0, so the setting's bound sees no error in it, whatever it decodes
    # to: only the one-step check, which wants exact zeros, catches it.
    ds = standard_dataset(3, nt=200)
    ds["waveform"][:, 1] = 0.0
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    assert verify_file(src, dst).passed
    with h5py.File(dst, "r+") as f:
        for name in ("stream_starts", "stream_bytes"):
            index = f[GROUP][name]
            index[0, 1] = index[0, 0]  # Station 1's first trace gets station 0's.
    check = verify_file(src, dst)
    assert not check.passed and check.max_relative_error <= 1 / full_scale(32)


def test_a_nan_in_the_original_fails_verification_and_is_reported(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # As when the original has changed since it was compressed. The NaN is in the
    # last block, after the largest finite error has been found.
    ds = standard_dataset(3, nt=200)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    ds["waveform"][1, 2, 100] = np.nan
    changed = write(ds, tmp_path / "changed.nc")
    check = verify_file(changed, dst, block_stations=1)
    assert not check.passed and np.isnan(check.max_relative_error)


@pytest.mark.parametrize(
    ("compressed", "original"),
    [
        pytest.param({}, {"station": slice(0, 4)}, id="a station fewer"),
        pytest.param({"station": slice(0, 4)}, {}, id="a station more"),
        pytest.param({}, {"time": slice(0, 399)}, id="a sample fewer"),
    ],
)
def test_verify_refuses_a_file_compressed_from_another_shape(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    compressed: dict[str, slice],
    original: dict[str, slice],
) -> None:
    # Both files are cut from one dataset, so the stations they share hold the same
    # traces: comparing the original's stations alone would pass the first case.
    ds = standard_dataset(5, nt=400)
    src = write(ds.isel(compressed), tmp_path / "in.nc")
    other = write(ds.isel(original), tmp_path / "other.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst)
    with pytest.raises(ValueError, match="decodes to a waveform of shape"):
        verify_file(other, dst)


@pytest.mark.parametrize("fill", [-999.0, None])
def test_the_fill_value_survives_compression(
    tmp_path: Path, standard_dataset: StandardDataset, fill: float | None
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].encoding["_FillValue"] = fill
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    back = decompress_file(dst)
    back.to_netcdf(tmp_path / "back.nc", engine="h5netcdf")
    with h5py.File(tmp_path / "back.nc", "r") as f:
        if fill is None:
            assert "_FillValue" not in f["waveform"].attrs
        else:
            assert f["waveform"].attrs["_FillValue"] == np.float32(fill)
            assert f["waveform"].attrs["_FillValue"].dtype == np.float32
    assert verify_file(src, dst).passed


def test_waveform_attributes_come_back_with_their_types(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].attrs.update(
        scale=np.float32(0.1),
        count=np.int32(7),
        wide=np.float64(2.5),
        long=np.int64(9),
        flags=np.array([1, 2, 3], np.int32),
        weights=np.array([0.5, 1.5], np.float32),
        names=["a", "bb"],
        unset=np.nan,
    )
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    back = decompress_file(dst)["waveform"].attrs
    with xr.open_dataset(src, engine="h5netcdf") as orig:
        expected = dict(orig["waveform"].attrs)
    assert back.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(back[name], value)
        assert type(back[name]) is type(value), name
        if isinstance(value, np.ndarray | np.generic):
            assert back[name].dtype == value.dtype, name
    assert back["scale"].dtype == np.float32
    assert back["count"].dtype == np.int32
    assert verify_file(src, dst).passed


@pytest.mark.parametrize("attribute", ["waveform_encoding", "waveform_attr_types"])
def test_verify_fails_when_the_fill_value_or_a_type_differs(
    tmp_path: Path, standard_dataset: StandardDataset, attribute: str
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].encoding["_FillValue"] = -999.0
    ds["waveform"].attrs["scale"] = np.float32(0.1)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    assert verify_file(src, dst).passed
    with h5py.File(dst, "r+") as f:
        g = f[GROUP]
        assert json.loads(g.attrs["waveform_encoding"]) == {"_FillValue": -999.0}
        assert json.loads(g.attrs["waveform_attr_types"]) == {"scale": "float32"}
        g.attrs[attribute] = (
            '{"_FillValue": "NaN"}'
            if attribute == "waveform_encoding"
            else '{"scale": "float64"}'
        )
    assert not verify_file(src, dst).passed


def test_a_nan_fill_value_is_written_as_strict_json(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].encoding["_FillValue"] = np.float32(np.nan)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r") as f:
        text = f[GROUP].attrs["waveform_encoding"]
    assert json.loads(text, parse_constant=pytest.fail) == {"_FillValue": "NaN"}
    assert verify_file(src, dst).passed
    assert np.isnan(decompress_file(dst)["waveform"].encoding["_FillValue"])


def test_a_non_finite_attribute_is_written_as_strict_json(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].attrs.update(
        unset=np.nan,
        high=np.float32(np.inf),
        low=-np.inf,
        weights=np.array([0.5, np.nan], np.float32),
    )
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r") as f:
        text = f[GROUP].attrs["waveform_attrs"]
    assert json.loads(text, parse_constant=pytest.fail) == {
        "units": "g",
        "unset": "NaN",
        "high": "Infinity",
        "low": "-Infinity",
        "weights": [0.5, "NaN"],
    }
    # Each comes back as a number again, in its own type, a NaN as a NaN.
    back = decompress_file(dst)["waveform"].attrs
    with xr.open_dataset(src, engine="h5netcdf") as orig:
        expected = dict(orig["waveform"].attrs)
    assert back.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(back[name], value)
        assert type(back[name]) is type(value), name
        if isinstance(value, np.ndarray | np.generic):
            assert back[name].dtype == value.dtype, name
    assert verify_file(src, dst).passed
    # A file written before 1.0.0 may hold bare tokens, and no types: it still decodes.
    with h5py.File(dst, "r+") as f:
        f[GROUP].attrs["waveform_attrs"] = (
            '{"units": "g", "unset": NaN, "weights": [0.5, NaN], "high": Infinity}'
        )
        del f[GROUP].attrs["waveform_attr_types"]
    older = decompress_file(dst)["waveform"].attrs
    assert older["units"] == "g" and np.isnan(older["unset"])
    assert np.isnan(older["weights"][1]) and older["high"] == np.inf


def test_a_non_finite_attribute_without_a_type_is_refused() -> None:
    # Attributes read from a file are numpy values, whose types bring a NaN back from
    # its name. A plain float has no type, so its name would come back as a string.
    for value in (float("nan"), [1.0, float("-inf")]):
        with pytest.raises(NotStandardLayout, match="'unset'"):
            files._attrs_json({"units": "g", "unset": value})
    typed = files._attrs_json({"unset": np.float64(np.nan)})
    assert json.loads(typed, parse_constant=pytest.fail) == {"unset": "NaN"}


def test_every_kind_of_encoding_value_is_strict_json() -> None:
    text = files._encoding_json(
        {"_FillValue": np.float32(np.inf), "missing_value": np.float32(-np.inf)}
    )
    assert json.loads(text, parse_constant=pytest.fail) == {
        "_FillValue": "Infinity",
        "missing_value": "-Infinity",
    }
    back = files._encoding_back(text)
    assert back == {"_FillValue": np.inf, "missing_value": -np.inf}
    assert all(type(v) is np.float32 for v in back.values())
    assert files._encoding_json({"_FillValue": None}) == '{"_FillValue": null}'


def test_a_file_without_the_new_attributes_still_decodes(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].attrs["count"] = np.int32(7)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r+") as f:
        del f[GROUP].attrs["waveform_encoding"]
        del f[GROUP].attrs["waveform_attr_types"]
    back = decompress_file(dst)
    assert back["waveform"].attrs["count"] == 7
    assert back["waveform"].shape == (3, 2, 50)


def test_an_unlimited_time_dimension_is_no_trouble(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # Without a time coordinate, nothing in the compressed file's root group has a
    # time dimension, so it can't be declared unlimited there.
    src, dst = tmp_path / "in.nc", tmp_path / "out.nc"
    standard_dataset(2, nt=50).to_netcdf(
        src, engine="h5netcdf", unlimited_dims=["time"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compress_file(src, dst)
    assert verify_file(src, dst).passed


def test_a_file_that_is_not_compressed_is_not_said_to_be(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # A compressed file is test_check.py's "already compressed" refusal.
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    (tmp_path / "notes.txt").write_text("not HDF5")
    assert not is_compressed(src)
    assert not is_compressed(tmp_path / "notes.txt")
    assert not is_compressed(tmp_path / "missing.nc")


@pytest.mark.parametrize("fault", ["missing", "cut short"])
@pytest.mark.parametrize(
    "call", ["check_compressible", "decompress_file", "verify_file"]
)
def test_a_missing_or_cut_short_input_fails_rather_than_being_refused(
    tmp_path: Path, standard_dataset: StandardDataset, call: str, fault: str
) -> None:
    # Only a file that is there and isn't HDF5 is refused, as the wrong kind of file.
    # One that isn't there, or an HDF5 file cut short, which keeps its signature, may be
    # the right kind: opening it fails, and that error is raised.
    original = write(standard_dataset(3, nt=100), tmp_path / "in.nc")
    packed = tmp_path / "in.flac.nc"
    compress_file(original, packed)
    bad = tmp_path / "bad.nc"
    if fault == "cut short":
        whole = (original if call == "check_compressible" else packed).read_bytes()
        bad.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(OSError):
        if call == "check_compressible":
            files.check_compressible(bad, tmp_path / "out.nc")
        elif call == "decompress_file":
            decompress_file(bad)
        else:
            verify_file(original, bad)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        pytest.param("format_version", 2, "format version 2", id="format version 2"),
        pytest.param("bits", 7, "bits must be from 8 to 32", id="7 bits"),
        pytest.param("bits", 33, "bits must be from 8 to 32", id="33 bits"),
    ],
)
def test_a_format_or_setting_that_cannot_be_read_is_refused(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    name: str,
    value: int,
    message: str,
) -> None:
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r+") as f:
        f[GROUP].attrs[name] = value
    with pytest.raises(ValueError, match=message):
        decompress_file(dst)
    with pytest.raises(ValueError, match=message):
        verify_file(src, dst)


def test_real_stations_come_back_exactly(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # In blocks of 3 stations, real station 2 ends a block, 3 starts the next, and 7 is
    # in the middle of another. At 17 bits no trace comes back exactly from its FLAC
    # stream, so only an exact copy can bring one back bit for bit.
    real, synthetic = [2, 3, 7], [0, 1, 4, 5, 6, 8, 9]
    ds = standard_dataset(10, nt=500, real=real)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    res = compress_file(src, dst, bits=17, block_stations=3)
    original = ds["waveform"].values
    with h5py.File(dst, "r") as f:
        exact_stations = f[GROUP]["exact_stations"][...]
        exact_waveform = f[GROUP]["exact_waveform"][...]
        gains = f[GROUP]["stream_gains"][...]
    assert exact_stations.dtype == np.int64 and exact_stations.tolist() == real
    assert exact_waveform.dtype == np.float32 and exact_waveform.shape == (3, 3, 500)
    assert bit_identical(exact_waveform, original[:, real])
    back = decompress_file(dst)["waveform"].values
    assert bit_identical(back[:, real], original[:, real])
    assert within_one_step(
        original[:, synthetic], back[:, synthetic], gains[:, synthetic]
    )
    assert not any(
        np.array_equal(back[c, s], original[c, s]) for c in range(3) for s in synthetic
    )
    check = verify_file(src, dst, block_stations=3)
    assert check.passed and res.n_exact == 3 and check.n_exact == 3
    assert verify_file(src, dst).passed  # All 10 stations in one block.


# A netCDF file holds station names as text, which can be any Unicode, padded with
# whitespace (HDF5 can't store a NUL in text), or as bytes of a fixed length, padded with
# NULs, which are read as ASCII. Bytes that aren't ASCII are refused, rather than read
# some other way: that is test_check.py's refusal of them.
@pytest.mark.parametrize(
    "names",
    [
        pytest.param(
            ["Ōtaki", " KIKS ", "S000002 ", "\tOPZ", "6c5SYSL", "22008d3\n"],
            id="text",
        ),
        pytest.param(
            [b"OPZ", b" KIKS ", b"6c5SYSL ", b"\0S000003", b"ABCDEFGH", b"22008d3"],
            id="fixed-length ASCII bytes",
        ),
    ],
)
def test_station_names_come_back_exactly(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    names: list[str] | list[bytes],
) -> None:
    ds = standard_dataset(len(names), nt=50).assign_coords(station=np.array(names))
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    back = decompress_file(dst)["station"]
    with xr.open_dataset(src, engine="h5netcdf") as orig:
        assert back.dtype == orig["station"].dtype
    assert back.values.tolist() == names  # Each name, and whether it is text or bytes.
    # A name is a grid's if it is 7 characters long without its padding.
    texts = [
        name.decode("ascii") if isinstance(name, bytes) else name for name in names
    ]
    unpadded = [text.strip(string.whitespace + "\0") for text in texts]
    with h5py.File(dst, "r") as f:
        exact_stations = f[GROUP]["exact_stations"][...].tolist()
    assert exact_stations == [i for i, name in enumerate(unpadded) if len(name) != 7]
    assert verify_file(src, dst).passed


@pytest.mark.parametrize("change", ["one ulp up", "a zero's sign"])
def test_verify_fails_an_exact_value_moved_within_one_step(
    tmp_path: Path, standard_dataset: StandardDataset, change: str
) -> None:
    # At 17 bits a step is about 1.5e-5 of the trace's peak, and an ulp of float32 is at
    # most 1.2e-7 of the value, so the change is far within one step. A zero's sign
    # doesn't change its value at all, only its bits.
    ds = standard_dataset(4, nt=300, real=(1,))
    if change == "a zero's sign":
        ds["waveform"][2, 1, 150] = -0.0
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst, bits=17)
    assert verify_file(src, dst).passed
    with h5py.File(dst, "r+") as f:
        exact = f[GROUP]["exact_waveform"]
        if change == "one ulp up":
            exact[2, 0, 150] = np.nextafter(exact[2, 0, 150], np.float32(np.inf))
        else:
            exact[2, 0, 150] = np.float32(0.0)
        gains = f[GROUP]["stream_gains"][...]
    original = ds["waveform"].values
    back = decompress_file(dst)["waveform"].values
    assert not bit_identical(back[:, 1], original[:, 1])
    assert within_one_step(original, back, gains)
    assert not verify_file(src, dst).passed


@pytest.mark.parametrize(
    "kept",
    [
        pytest.param([2, 7], id="a real station without one"),
        pytest.param([2, 3, 5, 7], id="a synthetic station with one"),
    ],
)
def test_verify_wants_exact_copies_of_the_real_stations_and_no_others(
    tmp_path: Path, standard_dataset: StandardDataset, kept: list[int]
) -> None:
    # Real station 3 never moved, so its FLAC stream alone brings it back exactly, and
    # synthetic station 5's copy is its own trace: every trace comes back as it should,
    # and only the list of exact stations shows that the file breaks the rule.
    ds = standard_dataset(10, nt=200, real=(2, 3, 7))
    ds["waveform"][:, 3] = 0.0
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    original = ds["waveform"].values
    with h5py.File(dst, "r+") as f:
        rewrite(f[GROUP], "exact_stations", np.array(kept, np.int64))
        rewrite(f[GROUP], "exact_waveform", original[:, kept])
    back = decompress_file(dst)["waveform"].values
    assert bit_identical(back[:, [2, 3, 7]], original[:, [2, 3, 7]])
    assert not verify_file(src, dst).passed


@pytest.mark.parametrize(
    "fault",
    [
        "a station past the last",
        "a negative station",
        "stations out of order",
        "stations as floats",
        "stations in two dimensions",
        "one component for three",
        "copies in float64",
    ],
)
def test_exact_copies_that_cannot_be_put_back_are_refused(
    tmp_path: Path, standard_dataset: StandardDataset, fault: str
) -> None:
    # Each would put a copy back at the wrong place, or not at all, or not exactly.
    ds = standard_dataset(4, nt=100, real=(1, 2))
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    stations, copies = np.array([1, 2]), ds["waveform"].values[:, [1, 2]]
    name = "exact_stations"
    if fault == "a station past the last":
        stations = np.array([1, 4])
    elif fault == "a negative station":
        stations = np.array([-3, 2])
    elif fault == "stations out of order":
        stations, copies = np.array([2, 1]), copies[:, ::-1]
    elif fault == "stations as floats":
        stations = np.array([1.0, 2.0])
    elif fault == "stations in two dimensions":
        stations = np.array([[1], [2]])
    elif fault == "one component for three":
        name, copies = "exact_waveform", copies[:1]
    else:
        name, copies = "exact_waveform", copies.astype(np.float64)
    with h5py.File(dst, "r+") as f:
        rewrite(f[GROUP], "exact_stations", stations)
        rewrite(f[GROUP], "exact_waveform", copies)
    with pytest.raises(ValueError, match=name):
        decompress_file(dst)
    with pytest.raises(ValueError, match=name):
        verify_file(src, dst)


# The waveform's attributes, which the file holds as JSON, changed from the original's
# {"units": "g"}.
WAVEFORM_ATTRS = {
    "a waveform attribute changed": {"units": "m/s"},
    "a waveform attribute added": {"units": "g", "scale": 2.0},
    "a waveform attribute gone": {},
}


@pytest.mark.parametrize(
    "change",
    [
        "an attribute of the file",
        "an attribute of a variable",
        "a variable's value",
        "a coordinate's value",
        *WAVEFORM_ATTRS,
    ],
)
def test_verify_fails_a_file_whose_other_content_changed(
    tmp_path: Path, standard_dataset: StandardDataset, change: str
) -> None:
    src = write(standard_dataset(3, nt=100), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst)
    assert verify_file(src, dst).passed
    with h5py.File(dst, "r+") as f:
        if change == "an attribute of the file":
            f.attrs["dt"] = 0.01
        elif change == "an attribute of a variable":
            f["vs30"].attrs["units"] = "m/s"
        elif change == "a variable's value":
            f["vs30"][1] += 1.0
        elif change == "a coordinate's value":
            f["latitude"][0] += 1.0
        else:
            f[GROUP].attrs["waveform_attrs"] = json.dumps(WAVEFORM_ATTRS[change])
    assert not verify_file(src, dst).passed


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param("a doubled gain", id="a doubled gain"),
        pytest.param("a stream at 24 bits", id="a stream at 24 bits"),
        pytest.param("another station's stream", id="another station's stream"),
    ],
)
def test_verify_checks_the_flac_streams_of_real_stations_too(
    tmp_path: Path, standard_dataset: StandardDataset, fault: str
) -> None:
    # A reader that ignores the exact copies decodes a real station's FLAC streams, so
    # they have to pass as any other station's do, though decompressing puts the exact
    # traces in their place. A doubled gain fails both checks. A stream quantised at
    # 24 bits in a 32-bit file is within one step of its own gain, but not within the
    # setting's bound. A station that never moved, given another station's stream,
    # shows no error over its peak of 0, but doesn't come back as exact zeros.
    ds = standard_dataset(4, nt=300, real=(1,))
    if fault == "another station's stream":
        ds["waveform"][:, 1] = 0.0
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    assert verify_file(src, dst).passed
    original = ds["waveform"].values
    with h5py.File(dst, "r+") as f:
        g = f[GROUP]
        if fault == "a doubled gain":
            g["stream_gains"][0, 1] = 2 * g["stream_gains"][0, 1]
        elif fault == "a stream at 24 bits":
            gain = gain_for(np.abs(original[0, 1].astype(np.float64)).max(), 24)
            stream = np.frombuffer(
                encode_stream(quantise(original[0, 1], gain)), np.uint8
            )
            end = g["compressed"].shape[0]
            g["compressed"].resize((end + stream.size,))
            g["compressed"][end:] = stream
            g["stream_starts"][0, 1], g["stream_bytes"][0, 1] = end, stream.size
            g["stream_gains"][0, 1] = gain
        else:
            for name in ("stream_starts", "stream_bytes"):
                g[name][0, 1] = g[name][0, 0]
    back = decompress_file(dst)["waveform"].values
    assert bit_identical(back[:, 1], original[:, 1])
    assert not verify_file(src, dst).passed


@pytest.mark.parametrize("fill", [np.nan, -999.0])
def test_verify_asks_whether_an_older_file_decompresses_to_the_original(
    tmp_path: Path, standard_dataset: StandardDataset, fill: float
) -> None:
    # Without the two new attributes, a file decodes with a NaN fill value, so it
    # verifies exactly when the original's was NaN.
    ds = standard_dataset(2, nt=50)
    ds["waveform"].encoding["_FillValue"] = np.float32(fill)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r+") as f:
        del f[GROUP].attrs["waveform_encoding"]
        del f[GROUP].attrs["waveform_attr_types"]
    assert verify_file(src, dst).passed is bool(np.isnan(fill))


@pytest.mark.parametrize(
    ("name", "value", "passes"),
    [
        ("wide", np.float64(2.5), True),
        ("long", np.int64(9), True),
        ("array", np.array([0.5, 1.5]), True),
        ("scale", np.float32(0.1), False),
        ("count", np.int32(7), False),
    ],
)
def test_verify_of_an_older_file_expects_jsons_types(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    name: str,
    value: np.generic,
    passes: bool,
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].attrs[name] = value
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    with h5py.File(dst, "r+") as f:
        del f[GROUP].attrs["waveform_encoding"]
        del f[GROUP].attrs["waveform_attr_types"]
    assert verify_file(src, dst).passed is passes


def test_a_finite_fill_value_far_from_the_data_round_trips(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].encoding["_FillValue"] = np.float32(-1e6)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst, bits=8)
    assert verify_file(src, dst).passed
    assert decompress_file(dst)["waveform"].encoding["_FillValue"] == np.float32(-1e6)


@pytest.mark.parametrize("fill", [-1e6, np.nan])
def test_an_equal_fill_value_and_missing_value_round_trip(
    tmp_path: Path, standard_dataset: StandardDataset, fill: float
) -> None:
    # xarray writes a _FillValue alone; another writer may give a missing_value too,
    # equal to it: far from the data, or NaN beside a NaN, which counts as equal.
    ds = standard_dataset(2, nt=50)
    ds["waveform"].encoding["_FillValue"] = np.float32(fill)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    with h5py.File(src, "r+") as f:
        f["waveform"].attrs["missing_value"] = np.float32(fill)
    compress_file(src, dst, bits=8)
    decompress_file(dst).to_netcdf(tmp_path / "back.nc", engine="h5netcdf")
    with h5py.File(tmp_path / "back.nc", "r") as f:
        for name in ("_FillValue", "missing_value"):
            np.testing.assert_array_equal(f["waveform"].attrs[name], np.float32(fill))
            assert f["waveform"].attrs[name].dtype == np.float32
    assert verify_file(src, dst).passed
