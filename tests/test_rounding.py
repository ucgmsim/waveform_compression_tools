"""How a file says which way it was rounded, and what that holds it to."""

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest

from waveform_compression_tools import files, open_dataset
from waveform_compression_tools.cli import Rounding
from waveform_compression_tools.codec import EncodedBlock, encode_block
from waveform_compression_tools.errors import (
    CheckFailed,
    NonFiniteSamples,
    NotStandardLayout,
    UnknownFormat,
)
from waveform_compression_tools.files import (
    GROUP,
    CompressResult,
    VerifyResult,
    compress_file,
    decompress_file,
    decompress_to,
    verify_file,
)

from helpers import (
    RealTraces,
    StandardDataset,
    bit_identical,
    names_in,
    nothing_left,
    read,
    standard_dataset_of,
    within_carried_limits_of_the_setting,
    write,
    write_and_compress,
)


def label_of(path: Path) -> object:
    with h5py.File(path, "r") as f:
        return f[GROUP].attrs["rounding"]


def relabel(path: Path, rounding: object) -> None:
    with h5py.File(path, "r+") as f:
        f[GROUP].attrs["rounding"] = rounding


@pytest.mark.parametrize("bits", [24, 16, 8])
def test_a_carried_file_verifies_and_says_so(
    tmp_path: Path, real_traces: RealTraces, bits: int
) -> None:
    src = write(standard_dataset_of(real_traces("bb")), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    # Checked as it is written.
    assert compress_file(src, dst, bits, rounding="carried").rounding == "carried"
    assert label_of(dst) == "carried"
    check = verify_file(src, dst)
    assert check.passed and check.rounding == "carried"
    original = read(src)["waveform"].values
    back = decompress_file(dst)["waveform"].values
    assert within_carried_limits_of_the_setting(original, back, bits).all()


def test_a_file_is_carried_at_24_bits_unless_asked(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(3, nt=500), tmp_path / "in.nc")
    dst, asked = tmp_path / "out.nc", tmp_path / "asked.nc"
    result = compress_file(src, dst)
    assert result.bits == 24 and result.rounding == "carried"
    compress_file(src, asked, bits=24, rounding="carried")
    for path in (dst, asked):
        assert label_of(path) == "carried"
        check = verify_file(src, path)
        assert check.passed and check.rounding == "carried"
    with h5py.File(dst, "r") as f, h5py.File(asked, "r") as g:
        for name in f[GROUP]:
            np.testing.assert_array_equal(f[GROUP][name][...], g[GROUP][name][...])


@pytest.mark.parametrize("bits", [32, 25, 24, 8])
def test_a_file_is_rounded_to_nearest_if_asked(
    tmp_path: Path, standard_dataset: StandardDataset, bits: int
) -> None:
    src = write(standard_dataset(3, nt=500), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    result = compress_file(src, dst, bits, rounding="nearest")
    assert result.bits == bits and result.rounding == "nearest"
    assert label_of(dst) == "nearest"
    check = verify_file(src, dst)
    assert check.passed and check.rounding == "nearest"


@pytest.mark.parametrize(
    "asked",
    [np.str_("nearest"), np.str_("carried"), Rounding.NEAREST, Rounding.CARRIED],
)
def test_any_string_of_a_rounding_will_do(
    standard_dataset: StandardDataset, tmp_path: Path, asked: str
) -> None:
    src, dst = tmp_path / "in.nc", tmp_path / "out.nc"
    write(standard_dataset(3, nt=500), src)
    result = compress_file(src, dst, bits=24, rounding=asked)
    for said in (result.rounding, label_of(dst), verify_file(src, dst).rounding):
        assert type(said) is str and said == str(asked)


def test_the_results_say_the_rounding_beside_the_bits_and_the_verdict() -> None:
    compressed = [field.name for field in dataclasses.fields(CompressResult)]
    verified = [field.name for field in dataclasses.fields(VerifyResult)]
    assert compressed[compressed.index("bits") + 1] == "rounding"
    assert verified[verified.index("passed") + 1] == "rounding"


def test_a_carried_file_with_a_real_station_verifies(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # Its stream keeps the carried limits, as any other does, and its exact copy comes
    # back bit for bit.
    ds = standard_dataset(4, nt=900, real=(1,))
    src, dst = write_and_compress(ds, tmp_path, bits=24, rounding="carried")
    assert verify_file(src, dst).passed
    original = read(src)["waveform"].values
    back = decompress_file(dst)["waveform"].values
    assert bit_identical(back[:, 1], original[:, 1])
    assert not bit_identical(back[:, 2], original[:, 2])
    with h5py.File(dst, "r+") as f:
        # Its stream alone, spoilt: what a reader that ignores the exact copy decodes.
        f[GROUP]["stream_gains"][0, 1] *= 1.001
    assert not verify_file(src, dst).passed


def test_the_label_decides_the_rules(tmp_path: Path, real_traces: RealTraces) -> None:
    # At 24 bits, each way of rounding breaks the other's promise.
    src = write(standard_dataset_of(real_traces("bb")), tmp_path / "in.nc")
    for made, claimed in (("nearest", "carried"), ("carried", "nearest")):
        dst = tmp_path / f"{made}.nc"
        compress_file(src, dst, bits=24, rounding=made)
        assert verify_file(src, dst).passed
        relabel(dst, claimed)
        check = verify_file(src, dst)
        assert not check.passed and check.rounding == claimed
        assert check.same_besides_samples


@pytest.mark.parametrize("rounding", ["nearest", "carried"])
def test_a_file_without_the_label_is_judged_as_nearest(
    tmp_path: Path, real_traces: RealTraces, rounding: str
) -> None:
    # As every file written by 1.2.0 and earlier is.
    ds = standard_dataset_of(real_traces("bb"))
    src, dst = write_and_compress(ds, tmp_path, bits=24, rounding=rounding)
    with h5py.File(dst, "r+") as f:
        del f[GROUP].attrs["rounding"]
    check = verify_file(src, dst)
    assert check.rounding == "nearest" and check.passed == (rounding == "nearest")


def test_a_carried_file_whose_samples_drift_fails(
    tmp_path: Path, real_traces: RealTraces
) -> None:
    # Each sample of one trace moved by a hundredth of a step, the same way.
    ds = standard_dataset_of(real_traces("bb"))
    src, dst = write_and_compress(ds, tmp_path, bits=16, rounding="carried")
    with h5py.File(dst, "r+") as f:
        f[GROUP]["stream_offsets"][1, 2] = 0.01 / f[GROUP]["stream_gains"][1, 2]
    assert not verify_file(src, dst).passed


UNKNOWN_LABELS = [
    "another word",
    "no word",
    "a number",
    "bytes",
    "a list",
    "an empty value",
    "a value h5py can't read",
    "carried at 25 bits",
]


@pytest.mark.parametrize("label", UNKNOWN_LABELS)
def test_verify_refuses_a_label_it_doesnt_know_but_the_file_still_decodes(
    tmp_path: Path, standard_dataset: StandardDataset, label: str
) -> None:
    bits, rounding = (25, "nearest") if "25 bits" in label else (24, "carried")
    ds = standard_dataset(3, nt=400, real=(1,))
    src, dst = write_and_compress(ds, tmp_path, bits=bits, rounding=rounding)
    before = decompress_file(dst)["waveform"].values
    values = {
        "another word": "other",
        "no word": "",
        "a number": 7,
        "bytes": np.bytes_(b"carried"),
        "a list": ["carried"],
        "an empty value": h5py.Empty("S1"),
        "carried at 25 bits": "carried",
    }
    if label in values:
        relabel(dst, values[label])
    else:
        with h5py.File(dst, "r+") as f:
            # HDF5's type for a time, which h5py has nothing to read it as.
            del f[GROUP].attrs["rounding"]
            scalar = h5py.h5s.create(h5py.h5s.SCALAR)
            h5py.h5a.create(f[GROUP].id, b"rounding", h5py.h5t.UNIX_D32LE, scalar)
    with pytest.raises(UnknownFormat, match="rounding") as refused:
        verify_file(src, dst)
    assert refused.value.filename == dst and str(dst) in str(refused.value)
    assert bit_identical(decompress_file(dst)["waveform"].values, before)
    decompress_to(dst, tmp_path / "back.nc")
    assert bit_identical(read(tmp_path / "back.nc")["waveform"].values, before)
    with open_dataset(dst) as opened:
        assert bit_identical(opened["waveform"].values, before)


def rounded_to_nearest(
    waveform: np.ndarray, bits: int, level: int, rounding: str
) -> EncodedBlock:
    return encode_block(waveform, bits, level, "nearest")


def at_one_bit_fewer(
    waveform: np.ndarray, bits: int, level: int, rounding: str
) -> EncodedBlock:
    # Steps twice as long as the setting promises, which the gains agree with.
    return encode_block(waveform, bits - 1, level, rounding)


def test_the_check_fails_a_nearest_file_of_longer_steps(
    tmp_path: Path, real_traces: RealTraces, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = write(standard_dataset_of(real_traces("bb")), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    monkeypatch.setattr(files, "encode_block", at_one_bit_fewer)
    with pytest.raises(CheckFailed):
        compress_file(src, dst, bits=24, rounding="nearest")
    assert nothing_left(dst)


@pytest.mark.parametrize("encoder", [rounded_to_nearest, at_one_bit_fewer])
def test_the_check_fails_a_carried_file_that_isnt(
    tmp_path: Path,
    real_traces: RealTraces,
    monkeypatch: pytest.MonkeyPatch,
    encoder: object,
) -> None:
    src = write(standard_dataset_of(real_traces("bb")), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    monkeypatch.setattr(files, "encode_block", encoder)
    with pytest.raises(CheckFailed):
        compress_file(src, dst, bits=24, rounding="carried")
    assert nothing_left(dst)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("rounding", ["nearest", "carried"])
@pytest.mark.parametrize("steps", [0.75, 1.5, 2.25])
def test_a_fill_value_near_the_data_is_refused(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    sign: int,
    rounding: str,
    steps: float,
) -> None:
    # Traces from -1 to 0.25, where a step at 24 bits, and the spacing of float32 values
    # at the peak, are 2 ** -23 each. A sample can decode up to one step, or with the
    # error carried the step and the spacing, beyond its trace's range.
    ds = standard_dataset(2, nt=300)
    rng = np.random.default_rng(0)
    waveform = sign * rng.uniform(-1.0, 0.25, ds["waveform"].shape).astype(np.float32)
    waveform[..., 0], waveform[..., 1] = -sign, 0.25 * sign
    ds["waveform"].values = waveform
    fill = np.float32(sign * (0.25 + steps * 2.0**-23))
    ds["waveform"].encoding["_FillValue"] = fill
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    if steps < (2 if rounding == "carried" else 1):
        with pytest.raises(NotStandardLayout, match="fill value"):
            compress_file(src, dst, bits=24, rounding=rounding)
        assert nothing_left(dst)
    else:
        compress_file(src, dst, bits=24, rounding=rounding)
        assert verify_file(src, dst).passed


def test_blocks_dont_change_carried_streams(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(5, nt=700), tmp_path / "in.nc")
    streams = []
    for block_stations in (1, 2, 5):
        dst = tmp_path / f"{block_stations}.nc"
        compress_file(
            src, dst, bits=24, block_stations=block_stations, rounding="carried"
        )
        assert verify_file(src, dst, block_stations=3).passed
        with h5py.File(dst, "r") as f:
            g = f[GROUP]
            # Where each stream is differs, as a file's streams are laid out by block.
            compressed = g["compressed"][...].tobytes()
            starts, nbytes = g["stream_starts"][...], g["stream_bytes"][...]
            streams.append(
                [compressed[a : a + n] for a, n in zip(starts.flat, nbytes.flat)]
            )
    assert streams[0] == streams[1] == streams[2] and len(set(streams[0])) == 15


@pytest.mark.parametrize(
    ("bits", "rounding", "message"),
    [
        (25, "carried", "carried rounding takes 24 bits or fewer, got 25"),
        (32, "carried", "carried rounding takes 24 bits or fewer, got 32"),
        (24, "other", "rounding must be 'nearest' or 'carried', got 'other'"),
        (33, "carried", "bits must be from 8 to 32, got 33"),
    ],
)
def test_a_rounding_is_refused_before_the_input_is_opened(
    tmp_path: Path, bits: int, rounding: str, message: str
) -> None:
    # The input isn't there, which would be an error of its own.
    with pytest.raises(ValueError, match=message):
        compress_file(
            tmp_path / "in.nc", tmp_path / "out.nc", bits=bits, rounding=rounding
        )
    assert names_in(tmp_path) == []


@pytest.mark.parametrize("check", [True, False])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_carried_file_is_refused_for_samples_that_are_not_finite(
    tmp_path: Path, standard_dataset: StandardDataset, bad: float, check: bool
) -> None:
    ds = standard_dataset(3, nt=400)
    ds["waveform"][2, 2, 50] = bad
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    with pytest.raises(NonFiniteSamples):
        compress_file(src, dst, bits=24, check=check, rounding="carried")
    assert nothing_left(dst)


def test_stations_that_never_moved_come_back_as_zeros(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    ds = standard_dataset(3, nt=400)
    ds["waveform"][:, 1] = 0.0
    ds["waveform"][:, 2, :150] = 0.0
    src, dst = write_and_compress(ds, tmp_path, bits=8, rounding="carried")
    back = decompress_file(dst)["waveform"].values
    assert not back[:, 1].any() and not back[:, 2, :150].any()
    with h5py.File(dst, "r+") as f:
        for name in ("stream_starts", "stream_bytes"):
            index = f[GROUP][name]
            index[0, 1] = index[0, 0]  # Station 1's first trace gets station 0's.
    assert not verify_file(src, dst).passed
