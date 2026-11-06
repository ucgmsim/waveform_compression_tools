import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyflac
import pytest

from waveform_compression_tools import files
from waveform_compression_tools.codec import (
    STEP_SLACK,
    EncodedBlock,
    encode_block,
    within_one_step,
)
from waveform_compression_tools.content_hash import content_hash
from waveform_compression_tools.errors import (
    AllStationsReal,
    AlreadyCompressed,
    CheckFailed,
    HasGroups,
    NonFiniteSamples,
    NotStandardLayout,
    OutputExists,
    Refused,
    SameFile,
)
from waveform_compression_tools.files import (
    GROUP,
    check_compressible,
    compress_file,
    decompress_file,
    verify_file,
)
from waveform_compression_tools.quantise import full_scale, quantise
from waveform_compression_tools.stream import encode_stream

from helpers import (
    StandardDataset,
    contents,
    nothing_left,
    read,
    refused_input,
    stream_of,
    with_a_flipped_byte,
    with_stream,
    write,
)


def change_as_written(
    monkeypatch: pytest.MonkeyPatch, name: str, change: Callable[[np.ndarray], None]
) -> None:
    # Every write to the group's dataset `name` is changed by `change`, in a copy, as
    # storage that didn't keep what it was given would change it.
    setitem = h5py.Dataset.__setitem__

    def changed(dataset: h5py.Dataset, key: object, value: np.ndarray) -> None:
        if dataset.name == f"/{GROUP}/{name}":
            value = value.copy()
            change(value)
        setitem(dataset, key, value)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", changed)


def test_compress_checks_and_reports(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    src = write(standard_dataset(5, nt=500, real=(1,)), tmp_path / "in.nc")
    res = compress_file(src, tmp_path / "out.nc", bits=24)
    # The output's content hash, taken before it was renamed into place.
    assert res.output_hash == content_hash(tmp_path / "out.nc") and res.hash_s > 0
    error = res.max_relative_error
    assert error is not None and 0 < error <= (1 + STEP_SLACK) / full_scale(24)
    # The error of the file as it decompresses, which verify reports too: the real
    # station's exact copy comes back as it was.
    assert error == verify_file(src, tmp_path / "out.nc").max_relative_error
    assert res.check_s > 0 and res.check_cpu_s > 0
    unchecked = compress_file(src, tmp_path / "unchecked.nc", bits=24, check=False)
    assert unchecked.max_relative_error is None
    assert (unchecked.check_s, unchecked.check_cpu_s) == (0.0, 0.0)
    # Every station but the real one holds zeros, which come back exactly, so the error
    # is 0, though the real station's own streams are lossy: its exact copy is what
    # decompressing gives back.
    zeros = standard_dataset(3, nt=50, real=(1,))
    zeros["waveform"][:, [0, 2]] = 0.0
    src = write(zeros, tmp_path / "zeros.nc")
    assert compress_file(src, tmp_path / "zeros.out.nc").max_relative_error == 0.0


# Damage to one trace's stream, done after the encoder knows the gains, and what the
# decoder's ValueError, which the failed check is raised from, is raised from in turn:
# libFLAC's own error, or nothing. libFLAC finds a flipped byte. A cut between frames
# decodes without an error, so only a count of the samples finds it. A stream cut to
# nothing holds no frames at all. A trace of zeros given another trace's stream decodes,
# but not to zeros, which only the one-step rule sees: the setting's bound sees a peak
# of 0, and nothing fails to decode. A byte flipped as it is written leaves the
# encoder's streams as they should be, so the pass finds it, raised from libFLAC's
# error, only by decoding what the file holds.
STREAM_DAMAGE = {
    "a flipped byte": pyflac.DecoderProcessException,
    "a cut between frames": None,
    "a cut to nothing": None,
    "a trace of zeros given another's stream": None,
    "a byte flipped as it is written": pyflac.DecoderProcessException,
}


@pytest.mark.parametrize("damage", STREAM_DAMAGE)
def test_a_corrupted_stream_fails_the_check_and_leaves_nothing(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    def encode_then_damage(waveform: np.ndarray, bits: int, level: int) -> EncodedBlock:
        encoded = encode_block(waveform, bits, level)
        if damage == "a flipped byte":
            return with_a_flipped_byte(encoded)
        # Component 1 of station 2's stream is replaced.
        if damage == "a cut between frames":
            # The trace's first frame of 4,096 samples, as a stream of its own.
            q = quantise(waveform[1, 2, :4096], encoded.gains[1, 2])
            return with_stream(encoded, encode_stream(q, level))
        if damage == "a cut to nothing":
            return with_stream(encoded, b"")
        # A trace of zeros given component 1 of station 1's.
        return with_stream(encoded, stream_of(encoded, 1, 1))

    def flip_the_middle_byte(value: np.ndarray) -> None:
        value[value.size // 2] ^= 0xFF

    ds = standard_dataset(3, nt=5_000)
    if damage == "a trace of zeros given another's stream":
        ds["waveform"][1, 2] = 0.0
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"
    if damage == "a byte flipped as it is written":
        change_as_written(monkeypatch, "compressed", flip_the_middle_byte)
    else:
        monkeypatch.setattr(files, "encode_block", encode_then_damage)
    with pytest.raises(CheckFailed) as failure:
        compress_file(src, dst)
    # Raised from the decoder's error, which names the trace, if there was one.
    cause, flac = failure.value.__cause__, STREAM_DAMAGE[damage]
    if damage == "a trace of zeros given another's stream":
        assert cause is None
    else:
        assert isinstance(cause, ValueError)
        # The encoder's damage is to component 1 of station 2; the write's is to the
        # middle byte of all the streams.
        assert re.match(r"component \d, station \d: ", str(cause)), cause
        assert isinstance(cause.__cause__, flac) if flac else cause.__cause__ is None
    assert nothing_left(dst)


def encode_with_a_doubled_gain(
    waveform: np.ndarray, bits: int, level: int
) -> EncodedBlock:
    # One trace's stream as encode_block writes it, but with twice its gain.
    encoded = encode_block(waveform, bits, level)
    gains = encoded.gains.copy()
    gains[1, 2] *= 2
    return EncodedBlock(encoded.streams, encoded.nbytes, gains)


def encode_with_gains_of_1(waveform: np.ndarray, bits: int, level: int) -> EncodedBlock:
    # As an encoder that wrote gains of 1 would: every trace rounded to whole units, so
    # every sample is within one step, 1, of the original, and far outside the bound of
    # the setting the file declares.
    gains = np.ones(waveform.shape[:2])
    streams = [
        encode_stream(quantise(waveform[c, s], 1.0), level)
        for c, s in np.ndindex(*gains.shape)
    ]
    nbytes = np.array([len(stream) for stream in streams], np.int64)
    return EncodedBlock(streams, nbytes.reshape(gains.shape), gains)


WRONG_GAINS = {
    "a doubled gain": encode_with_a_doubled_gain,
    "gains of 1": encode_with_gains_of_1,
}


@pytest.mark.parametrize("encoder", WRONG_GAINS)
def test_a_wrong_gain_fails_the_check(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    encoder: str,
) -> None:
    src = write(standard_dataset(3, nt=400), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    monkeypatch.setattr(files, "encode_block", WRONG_GAINS[encoder])
    with pytest.raises(CheckFailed):
        compress_file(src, dst)
    assert nothing_left(dst)
    # Unchecked, the file is written, and verify fails it too.
    compress_file(src, dst, check=False)
    assert not verify_file(src, dst).passed
    if encoder == "gains of 1":
        # Every sample is within one step of its own gain, so only the setting's bound
        # finds it.
        original = read(src)["waveform"].values
        decoded = decompress_file(dst)["waveform"].values
        assert within_one_step(original, decoded, np.ones((3, 3)))


@pytest.mark.parametrize("change", ["one ulp up", "a zero's sign"])
def test_an_altered_exact_station_fails_the_check(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    # Real station 1's exact copy is written with one value changed, as storage that
    # didn't keep what it was given would. The streams are as they should be, so only a
    # comparison of the copy's bits finds it: -0.0 equals 0.0.
    ds = standard_dataset(4, nt=300, real=(1,))
    if change == "a zero's sign":
        ds["waveform"][2, 1, 150] = -0.0
    src, dst = write(ds, tmp_path / "in.nc"), tmp_path / "out.nc"

    def change_the_copy(value: np.ndarray) -> None:
        # The copies are of the one real station.
        if change == "one ulp up":
            value[2, 0, 150] = np.nextafter(value[2, 0, 150], np.float32(np.inf))
        else:
            value[2, 0, 150] = 0.0

    change_as_written(monkeypatch, "exact_waveform", change_the_copy)
    with pytest.raises(CheckFailed):
        compress_file(src, dst)
    assert nothing_left(dst)


# Damage to the closed partial file, after the pass has checked every block and before
# the file is reopened. An exact copy is compared with its CRC-32 from the pass, and a
# coordinate, outside the group, with the original. The file is written at 24 bits, so
# a claim of 32 would hold it to a bound it can't meet: only the group's attributes,
# compared with those written, show that.
DAMAGE_AFTER_THE_PASS = [
    "a flipped byte in a stream",
    "an exact copy one ulp up",
    "a gain one ulp up",
    "an offset that isn't zero",
    "another exact station",
    "a byte more of streams",
    "another trace length",
    "a finer setting claimed",
    "another waveform_encoding",
    "no waveform_attr_types",
    "a dataset more",
    "a coordinate's value",
]


@pytest.mark.parametrize("damage", DAMAGE_AFTER_THE_PASS)
def test_damage_after_the_pass_fails_the_structure_check(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    src = write(standard_dataset(4, nt=400, real=(1,)), tmp_path / "in.nc")
    dst = tmp_path / "out.nc"
    # An earlier file, asked to be replaced, but only by a file that passes.
    dst.write_bytes(b"an earlier file")
    before = contents(tmp_path)
    check_reopened = files._check_reopened

    def damage_then_check(path: Path, *recorded: Any) -> None:
        with h5py.File(path, "r+") as f:
            g = f[GROUP]
            if damage == "a flipped byte in a stream":
                g["compressed"][int(g["stream_starts"][1, 2]) + 20] ^= 0xFF
            elif damage == "an exact copy one ulp up":
                copy = g["exact_waveform"]
                copy[2, 0, 150] = np.nextafter(copy[2, 0, 150], np.float32(np.inf))
            elif damage == "a gain one ulp up":
                g["stream_gains"][0, 3] = np.nextafter(g["stream_gains"][0, 3], np.inf)
            elif damage == "an offset that isn't zero":
                g["stream_offsets"][2, 0] = 1e-300
            elif damage == "another exact station":
                g["exact_stations"][0] = 2
            elif damage == "a byte more of streams":
                g["compressed"].resize((g["compressed"].shape[0] + 1,))
            elif damage == "another trace length":
                g["stream_starts"].attrs["stream_size"] = 399
            elif damage == "a finer setting claimed":
                g.attrs["bits"] = 32
            elif damage == "another waveform_encoding":
                g.attrs["waveform_encoding"] = '{"_FillValue": -999.0}'
            elif damage == "no waveform_attr_types":
                del g.attrs["waveform_attr_types"]
            elif damage == "a dataset more":
                g.create_dataset("notes", data=np.arange(3))
            else:
                f["latitude"][0] += 1.0
        check_reopened(path, *recorded)

    monkeypatch.setattr(files, "_check_reopened", damage_then_check)
    with pytest.raises(CheckFailed):
        compress_file(src, dst, bits=24, overwrite=True)
    # The input and the earlier file are as they were, and no partial file is left.
    assert contents(tmp_path) == before


# Each refusal, as helpers.refused_input writes it: what it raises, whether
# check_compressible finds it before any sample is read, and what its message says. The
# others need the samples, so the pass finds them.
REFUSALS = {
    "the same path": (SameFile, True, "overwrite"),
    "a link to it": (SameFile, True, "overwrite"),
    "a hard link to it": (SameFile, True, "overwrite"),
    "its own partial": (SameFile, True, "overwrite"),
    "a hard link to it at its partial": (SameFile, True, "overwrite"),
    "the output exists": (OutputExists, True, "exists"),
    "a broken link at the output": (OutputExists, True, "exists"),
    "a text file": (NotStandardLayout, True, "isn't an HDF5 file"),
    "already compressed": (AlreadyCompressed, True, "compressed already"),
    "an HDF5 group": (HasGroups, True, "provenance"),
    "a float64 waveform": (NotStandardLayout, True, "float32"),
    "dimensions in another order": (NotStandardLayout, True, "dimensions"),
    "no waveform": (NotStandardLayout, True, "waveform"),
    "no station coordinate": (NotStandardLayout, True, "station"),
    "station names along another dimension": (NotStandardLayout, True, "station"),
    "station names of bytes that aren't ASCII": (NotStandardLayout, True, "ASCII"),
    "no stations": (NotStandardLayout, True, "no stations"),
    "an attribute JSON can't hold": (NotStandardLayout, True, "JSON"),
    "a _FillValue JSON can't hold": (NotStandardLayout, True, "JSON"),
    "a _FillValue of text": (NotStandardLayout, True, "not a number"),
    "an attribute whose type can't be kept": (NotStandardLayout, True, "pair"),
    "conflicting fill values": (NotStandardLayout, True, "differ"),
    "a vector missing_value": (NotStandardLayout, True, "missing_value"),
    "every station real": (AllStationsReal, True, "real"),
    "a NaN": (NonFiniteSamples, False, "non-finite"),
    "an infinity": (NonFiniteSamples, False, "non-finite"),
    "a negative infinity": (NonFiniteSamples, False, "non-finite"),
    "a fill value inside the data": (NotStandardLayout, False, "fill value"),
    "a missing_value inside the data": (NotStandardLayout, False, "fill value"),
}
# xarray warns as it opens a waveform with several fill values, which it reads as
# missing, before the refusal.
SEVERAL_FILL_VALUES = pytest.mark.filterwarnings(
    "ignore:variable 'waveform' has multiple fill values:xarray.SerializationWarning"
)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=SEVERAL_FILL_VALUES)
        if case == "a vector missing_value"
        else case
        for case in REFUSALS
    ],
)
def test_refusals_raise_refused_and_write_nothing(
    tmp_path: Path, standard_dataset: StandardDataset, case: str
) -> None:
    refusal, before_the_pass, message = REFUSALS[case]
    assert issubclass(refusal, Refused) and issubclass(refusal, ValueError)
    src, dst, overwrite = refused_input(case, tmp_path, standard_dataset)
    # A refusal names the file it is about: the destination's own, and the input's
    # every other, whichever step finds it.
    about = dst if refusal in (SameFile, OutputExists) else src
    before = contents(tmp_path)
    if before_the_pass:
        with pytest.raises(refusal, match=message) as refused:
            check_compressible(src, dst, overwrite)
        assert str(refused.value.filename) == str(about)
    else:
        check_compressible(src, dst, overwrite)
    with pytest.raises(refusal, match=message) as refused:
        compress_file(src, dst, block_stations=2, overwrite=overwrite)
    assert str(refused.value.filename) == str(about)
    # Nothing is written: the input and whatever was at the destination are as they
    # were, and no partial file is left.
    assert contents(tmp_path) == before


@pytest.mark.parametrize(
    "case",
    [
        "a float64 waveform",
        "dimensions in another order",
        "no waveform",
        "no station coordinate",
        "station names along another dimension",
        "station names of bytes that aren't ASCII",
    ],
)
def test_verify_refuses_an_original_that_compress_refuses(
    tmp_path: Path, standard_dataset: StandardDataset, case: str
) -> None:
    original, _, _ = refused_input(case, tmp_path, standard_dataset)
    standard = write(standard_dataset(3, nt=100), tmp_path / "standard.nc")
    compressed = tmp_path / "standard.flac.nc"
    compress_file(standard, compressed)
    with pytest.raises(NotStandardLayout, match=REFUSALS[case][2]) as refused:
        verify_file(original, compressed)
    assert refused.value.filename == original
