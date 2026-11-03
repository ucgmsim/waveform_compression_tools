import tracemalloc
import warnings
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from waveform_compression_tools import files
from waveform_compression_tools.codec import (
    EncodedBlock,
    NotStandardLayout,
    encode_block,
    within_one_step,
)
from waveform_compression_tools.files import (
    GROUP,
    AllStationsReal,
    compress_file,
    decompress_file,
    is_compressed,
    verify_file,
)
from waveform_compression_tools.quantise import full_scale, gain_for, quantise
from waveform_compression_tools.stream import encode_stream

StandardDataset = Callable[..., xr.Dataset]


def write(ds: xr.Dataset, path: Path) -> Path:
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def partial_of(path: Path) -> Path:
    return Path(f"{path}.partial")


def bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
    # Compares the bits of float32 arrays, so -0.0 differs from 0.0 and a NaN can match.
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


def rewrite(group: h5py.Group, name: str, data: np.ndarray) -> None:
    # Replaces a dataset in a group, whatever its shape or type.
    del group[name]
    group.create_dataset(name, data=data)


def assert_streams_tile_the_bytes(path: Path) -> None:
    # Laid end to end in order of their starts, the streams cover the compressed bytes
    # exactly: no stream is written twice, and nothing lies between them.
    with h5py.File(path, "r") as f:
        g = f[GROUP]
        starts, nbytes = g["stream_starts"][...].ravel(), g["stream_bytes"][...].ravel()
        size = g["compressed"].shape[0]
    order = np.argsort(starts)
    ends = np.cumsum(nbytes[order])
    assert np.array_equal(starts[order], ends - nbytes[order]) and ends[-1] == size


# Station counts on both sides of a multiple of 344, the stations in a block, and an
# odd number of samples: every station has to be compressed exactly once.


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


def test_refuses_non_standard_and_leaves_nothing(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    standard = write(ds, tmp_path / "standard.nc")
    ds["waveform"] = ds["waveform"].astype(np.float64)
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    with pytest.raises(NotStandardLayout):
        compress_file(src, dst)
    assert not dst.exists() and not partial_of(dst).exists()
    compress_file(standard, dst)
    with pytest.raises(NotStandardLayout):
        verify_file(src, dst)


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


def test_results_report_sizes_and_times(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(4, nt=500), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    res = compress_file(src, dst, bits=17)
    assert (res.bytes_in, res.bytes_out) == (src.stat().st_size, dst.stat().st_size)
    assert (res.n_traces, res.bits) == (12, 17)
    assert res.read_s > 0 and res.encode_s > 0 and res.write_s > 0
    check = verify_file(src, dst)
    assert check.passed and 0 < check.max_relative_error <= 1 / full_scale(17)
    assert check.verify_s > 0


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


def test_verify_fails_a_file_whose_gains_are_all_1(
    tmp_path: Path, standard_dataset: StandardDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As an encoder that wrote gains of 1 would: every trace rounded to whole units,
    # so every sample is within one step, 1, of the original, and far outside the
    # bound of the setting the file declares.
    def encode_with_gains_of_1(
        waveform: np.ndarray, bits: int, level: int
    ) -> EncodedBlock:
        gains = np.ones(waveform.shape[:2])
        streams = [
            encode_stream(quantise(waveform[c, s], 1.0), level)
            for c, s in np.ndindex(*gains.shape)
        ]
        nbytes = np.array([len(stream) for stream in streams], np.int64)
        return EncodedBlock(streams, nbytes.reshape(gains.shape), gains)

    src = write(standard_dataset(3, nt=400), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    monkeypatch.setattr(files, "encode_block", encode_with_gains_of_1)
    compress_file(src, dst)
    with xr.open_dataset(src, engine="h5netcdf") as orig:
        original = orig["waveform"].values
    decoded = decompress_file(dst)["waveform"].values
    assert within_one_step(original, decoded, np.ones((3, 3)))
    assert not verify_file(src, dst).passed


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


def test_waveform_attributes_with_numpy_values_come_back(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(2, nt=50)
    ds["waveform"].attrs.update(
        scale=np.float32(0.1),
        count=np.int32(7),
        flags=np.array([1, 2, 3], np.int32),
        names=["a", "bb"],
        unset=np.nan,
    )
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    compress_file(src, dst)
    back = decompress_file(dst)["waveform"].attrs
    with xr.open_dataset(src, engine="h5netcdf") as orig:
        expected = orig["waveform"].attrs
    assert back.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(back[name], value)
    # Their types came back from JSON as Python's, which isn't a difference.
    assert verify_file(src, dst).passed


@pytest.mark.parametrize("step", ["compress", "verify"])
def test_the_waveform_is_read_a_block_of_stations_at_a_time(
    tmp_path: Path, standard_dataset: StandardDataset, step: str
) -> None:
    # 300 stations of 4,000 samples hold 14.4 MB of waveform. Working through blocks
    # of 10 stations peaks at a few MB: a block, its copies and its streams. Reading
    # the whole waveform at once would peak above 14.4 MB.
    src = write(standard_dataset(300, nt=4_000), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst, block_stations=10)  # Also loads what loads on first use.
    tracemalloc.start()
    try:
        if step == "compress":
            compress_file(src, tmp_path / "again.nc", block_stations=10)
        else:
            assert verify_file(src, dst, block_stations=10).passed
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 14.4e6 / 2


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


def test_is_compressed(tmp_path: Path, standard_dataset: StandardDataset) -> None:
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    compress_file(src, dst)
    (tmp_path / "notes.txt").write_text("not HDF5")
    assert is_compressed(dst)
    assert not is_compressed(src)
    assert not is_compressed(tmp_path / "notes.txt")
    assert not is_compressed(tmp_path / "missing.nc")


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


def test_a_file_whose_stations_are_all_real_is_not_compressed(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(3, nt=100, real=(0, 1, 2)), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    with pytest.raises(AllStationsReal, match="real"):
        compress_file(src, dst)
    assert not dst.exists() and not partial_of(dst).exists()
    assert issubclass(AllStationsReal, ValueError)


@pytest.mark.parametrize("names", ["none", "along another dimension"])
def test_real_stations_are_found_by_a_station_coordinate(
    tmp_path: Path, standard_dataset: StandardDataset, names: str
) -> None:
    ds = standard_dataset(3, nt=100)
    standard, dst = write(ds, tmp_path / "standard.nc"), tmp_path / "out.nc"
    unnamed = ds.drop_vars("station")
    if names == "along another dimension":
        # Names that would make station 1 real, but they aren't the stations' names.
        unnamed = unnamed.assign_coords(station=("site", ["6c5SYSL", "OPZ", "22008d3"]))
    src = write(unnamed, tmp_path / "in.nc")
    with pytest.raises(NotStandardLayout, match="station"):
        compress_file(src, dst)
    assert not dst.exists() and not partial_of(dst).exists()
    compress_file(standard, dst)
    with pytest.raises(NotStandardLayout, match="station"):
        verify_file(src, dst)


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


def test_a_source_with_another_group_is_refused(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # xarray reads only the root group, so the compressed file would go without it.
    src = write(standard_dataset(2, nt=50), tmp_path / "in.nc")
    with h5py.File(src, "a") as f:
        f.create_group("provenance").create_dataset("notes", data=np.arange(3))
    dst = tmp_path / "out.nc"
    with pytest.raises(ValueError, match="provenance"):
        compress_file(src, dst)
    assert not dst.exists() and not partial_of(dst).exists()


@pytest.mark.parametrize(
    ("src_name", "dst_name"),
    [
        pytest.param("in.nc", "in.nc", id="the same path"),
        pytest.param("in.nc", "link.nc", id="a link to it"),
        # The partial file is written first, and deleted if anything fails.
        pytest.param("in.nc.partial", "in.nc", id="its partial"),
    ],
)
def test_a_file_is_never_compressed_over_itself(
    tmp_path: Path, standard_dataset: StandardDataset, src_name: str, dst_name: str
) -> None:
    src = write(standard_dataset(2, nt=50), tmp_path / src_name)
    (tmp_path / "link.nc").symlink_to(src)
    before = src.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        compress_file(src, tmp_path / dst_name)
    assert src.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        "an attribute of the file",
        "an attribute of a variable",
        "a variable's value",
        "a coordinate's value",
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
        else:
            f["latitude"][0] += 1.0
    assert not verify_file(src, dst).passed


def test_a_file_with_no_stations_is_refused_and_nothing_is_written(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # It would compress to empty arrays, and no real file has no stations. A file
    # already at the destination stays as it was.
    src = write(standard_dataset(0, nt=50), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    dst.write_bytes(b"an earlier file")
    with pytest.raises(NotStandardLayout, match="no stations"):
        compress_file(src, dst)
    assert dst.read_bytes() == b"an earlier file"
    assert not partial_of(dst).exists()


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
