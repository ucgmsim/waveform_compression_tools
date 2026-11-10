import contextlib
import io
import pickle
import re
import shutil
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import xarray as xr

from waveform_compression_tools import codec, open_dataset
from waveform_compression_tools.errors import NotCompressed, UnknownFormat
from waveform_compression_tools.files import GROUP, compress_file, decompress_file

from helpers import bit_identical, build_standard_dataset, rewrite, run_python, write

N_STATION, NT, DT = 12, 400, 0.005
# Real stations 2 and 3 are in the first block of 5 stations, and 7 in the second.
REAL = (2, 3, 7)


@pytest.fixture(scope="module")
def compressed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # A file of three blocks, with real stations, a _FillValue and a time coordinate,
    # whose waveform has attributes of several types. At 17 bits no trace comes back
    # exactly from its stream, so only an exact copy brings a real station back. Its
    # vs30 is in chunks of 4 stations, compressed with gzip, as other variables can be.
    folder = tmp_path_factory.mktemp("dataset")
    ds = build_standard_dataset(N_STATION, NT, real=REAL)
    ds = ds.assign_coords(time=DT * np.arange(NT))
    ds["waveform"].attrs.update(
        scale=np.float32(0.5), count=np.int32(7), weights=np.array([0.5, 1.5], "f4")
    )
    ds["waveform"].encoding["_FillValue"] = np.float32(-999.0)
    ds["vs30"].encoding.update(zlib=True, chunksizes=(4,))
    dst = folder / "out.nc"
    compress_file(write(ds, folder / "in.nc"), dst, bits=17, block_stations=5)
    return dst


def held_open(path: Path) -> bool:
    # HDF5 won't open a file for writing while this process has it open to read.
    try:
        h5py.File(path, "r+").close()
    except OSError as error:
        if "already open" in str(error):
            return True
        raise
    return False


def assert_same(got: xr.Dataset, want: xr.Dataset) -> None:
    # Every variable, coordinate and attribute, the waveform's samples bit for bit, and
    # each attribute and encoding in its own type: repr shows each value with its type,
    # and a NaN as a NaN.
    xr.testing.assert_identical(got, want)
    assert bit_identical(got["waveform"].values, want["waveform"].values)
    assert repr((got.attrs, got.encoding)) == repr((want.attrs, want.encoding))
    for name, variable in want.variables.items():
        assert repr((got[name].attrs, got[name].encoding)) == repr(
            (variable.attrs, variable.encoding)
        ), name


def record_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    # Each read of an HDF5 dataset from now on: the dataset's name, and its key.
    reads: list[tuple[str, Any]] = []
    read = h5py.Dataset.__getitem__

    def spy(self: h5py.Dataset, args: object, new_dtype: object = None) -> object:
        reads.append((self.name, args))
        return read(self, args, new_dtype)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    return reads


SELECTIONS: dict[str, Callable[[xr.Dataset], xr.Dataset]] = {
    "everything": lambda ds: ds,
    "stations by name": lambda ds: ds.sel(station=["S000010", "R003", "S000005"]),
    "a station by name": lambda ds: ds.sel(station="R007"),
    "stations by index, out of order": lambda ds: ds.isel(station=[9, 2, 9, -1]),
    "stations by index, in order": lambda ds: ds.isel(station=[4, 4, 8]),
    "a slice across blocks": lambda ds: ds.isel(station=slice(3, 11)),
    "a slice backwards": lambda ds: ds.isel(station=slice(None, None, -4)),
    "a boolean mask": lambda ds: ds.isel(station=np.arange(N_STATION) % 3 == 1),
    "a latitude range": lambda ds: ds.isel(
        station=(ds["latitude"] > -40) & (ds["latitude"] < -36)
    ),
    "a latitude range, by where": lambda ds: ds.where(
        (ds["latitude"] > -40) & (ds["latitude"] < -36), drop=True
    ),
    "sorted by latitude": lambda ds: ds.sortby("latitude"),
    "no stations": lambda ds: ds.isel(station=[]),
    "a component by name": lambda ds: ds.sel(component="090"),
    "components by index": lambda ds: ds.isel(component=[2, 0]),
    "a time range": lambda ds: ds.sel(time=slice(0.2, 0.9)),
    "a sample": lambda ds: ds.isel(time=150),
    "all three": lambda ds: ds.isel(
        component=1, station=[3, 4, 8], time=slice(10, None, 7)
    ),
    "points": lambda ds: ds.isel(
        component=xr.DataArray([0, 2], dims="point"),
        station=xr.DataArray([7, 4], dims="point"),
    ),
    "a deep copy": lambda ds: ds.copy(deep=True),
    "pickled": lambda ds: pickle.loads(pickle.dumps(ds)),
}


@pytest.mark.parametrize("selection", SELECTIONS)
def test_a_selection_comes_back_as_decompress_file_gives_it(
    compressed: Path, selection: str
) -> None:
    select = SELECTIONS[selection]
    with open_dataset(compressed) as ds:
        got = select(ds).load()
    assert_same(got, select(decompress_file(compressed)))


def test_only_the_selected_traces_are_read_and_decoded(
    compressed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Components 000 and ver of real station 3 and of stations 5 and 10. Only their
    # streams are read, once, and decoded, and the real station's exact copies are read
    # too, as decompress_file decodes every stream before it puts the copies back.
    with h5py.File(compressed, "r") as f:
        group = f[GROUP]
        stream_bytes = group["compressed"][...]
        starts, nbytes = group["stream_starts"][...], group["stream_bytes"][...]
        exact_shape = group["exact_waveform"].shape
    selected = [(c, s) for c in (0, 2) for s in (3, 5, 10)]
    streams = sorted(
        stream_bytes[starts[c, s] : starts[c, s] + nbytes[c, s]].tobytes()
        for c, s in selected
    )
    decoded: list[bytes] = []
    decode = codec.decode_stream

    def spy_decode(blob: bytes) -> np.ndarray:
        decoded.append(bytes(blob))
        return decode(blob)

    monkeypatch.setattr(codec, "decode_stream", spy_decode)
    reads = record_reads(monkeypatch)
    names = {f"/{GROUP}/compressed", f"/{GROUP}/exact_waveform"}
    with open_dataset(compressed) as ds:
        repr(ds)
        picked = ds.sel(
            component=["000", "ver"], station=["R003", "S000005", "S000010"]
        )
        # Opening, printing and selecting read no sample.
        assert [name for name, _ in reads if name in names] == [] and decoded == []
        picked.load()
    assert sorted(decoded) == streams
    read_bytes, read_copies = np.zeros(stream_bytes.size), np.zeros(exact_shape)
    for name, key in reads:
        if name == f"/{GROUP}/compressed":
            read_bytes[key] += 1
        elif name == f"/{GROUP}/exact_waveform":
            read_copies[key] = 1
    in_selected = np.zeros(stream_bytes.size)
    for c, s in selected:
        in_selected[starts[c, s] : starts[c, s] + nbytes[c, s]] = 1
    assert np.array_equal(read_bytes, in_selected)
    copies = np.zeros(exact_shape)
    copies[[0, 2], REAL.index(3)] = 1
    assert np.array_equal(read_copies, copies)


def bytes_read() -> int:
    # What this process has asked the system to read, from the disk or its cache, with
    # the hundred bytes or so of this count's last reading.
    with open("/proc/self/io") as f:
        return int(next(line for line in f if line.startswith("rchar:")).split()[1])


linux_counts_reads = pytest.mark.skipif(
    not Path("/proc/self/io").exists(), reason="Linux counts a process's reads"
)


@linux_counts_reads
@pytest.mark.parametrize("already_open", [False, True])
def test_the_file_is_read_for_the_selected_streams_alone(
    compressed: Path, already_open: bool
) -> None:
    # HDF5 can keep chunks in a cache, reading each chunk a stream is in whole: here
    # the first block's streams, and a MiB in a file of many stations. A file has the
    # settings of its first opening in the process, here xarray's if it is open
    # already, so the cache is turned off for the streams alone. Station 0's streams
    # are in the first chunk, and the last station's are in another, so reading the
    # last station first loads what is read once, such as HDF5's index of chunks.
    with h5py.File(compressed, "r") as f:
        streams = int(f[GROUP]["stream_bytes"][:, 0].sum())
    first = (
        xr.open_dataset(compressed, engine="h5netcdf")
        if already_open
        else contextlib.nullcontext()
    )
    with first, open_dataset(compressed) as ds:
        ds["waveform"].isel(station=[-1]).load()
        before = bytes_read()
        ds["waveform"].isel(station=[0]).load()
        read = bytes_read() - before
    assert streams <= read < streams + 1024


@linux_counts_reads
def test_the_file_keeps_its_cache_for_everything_else(compressed: Path) -> None:
    # While the dataset is open, another handle of the file reads vs30's first station
    # 200 times from HDF5's cache. Without the cache, each read would read its chunk.
    with open_dataset(compressed), h5py.File(compressed, "r") as f:
        vs30 = f["vs30"]
        vs30[0]
        before = bytes_read()
        for _ in range(200):
            vs30[0]
        read = bytes_read() - before
    assert read < 1024


# Station 3 is real: its trace is its exact copy, but its stream is decoded too.
@pytest.mark.parametrize("station", [5, 3])
def test_a_damaged_stream_fails_its_own_trace_alone(
    compressed: Path, tmp_path: Path, station: int
) -> None:
    # Component 1 of the station is damaged, and fails as it does in decompress_file.
    # The error names the trace as the file counts its traces, not as the selection
    # does, and every other trace still comes back.
    path = Path(shutil.copy(compressed, tmp_path / "out.nc"))
    with h5py.File(path, "r+") as f:
        group = f[GROUP]
        start = group["stream_starts"][1, station]
        group["compressed"][start + group["stream_bytes"][1, station] // 2] ^= 0xFF
    with pytest.raises(ValueError, match=f"component 1, station {station}") as failed:
        decompress_file(path)
    want = decompress_file(compressed)["waveform"].isel(component=[0, 2])
    with open_dataset(path) as ds:
        with pytest.raises(ValueError) as read:
            ds["waveform"].isel(component=[1], station=[4, station]).load()
        got = ds["waveform"].isel(component=[0, 2]).values
    assert str(read.value) == str(failed.value)
    assert bit_identical(got, want.values)


# A file of a few stations, and one of four times as many. A station's waveform is 3
# traces of 1,000 float32 samples, 12 kB.
FEW, MANY, NT_MEMORY = 100, 400, 1_000


@pytest.fixture(scope="module")
def few_and_many(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    folder = tmp_path_factory.mktemp("sizes")
    paths = {}
    for n_station in (FEW, MANY):
        ds = build_standard_dataset(n_station, NT_MEMORY)
        src, paths[n_station] = folder / "in.nc", folder / f"out{n_station}.nc"
        compress_file(write(ds, src), paths[n_station], bits=8)
        src.unlink()
    return paths


def peak_reading_three_stations(path: Path) -> int:
    tracemalloc.start()
    try:
        with open_dataset(path) as ds:
            ds["waveform"].isel(station=[1, 50, 99]).load()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_reading_a_few_stations_takes_little_memory_for_each_station_in_the_file(
    few_and_many: dict[int, Path],
) -> None:
    # Opening keeps a few hundred bytes of each station: its name and coordinates, and
    # where its streams are, with their gains. Decoding the waveform would take 12 kB.
    peak_reading_three_stations(few_and_many[FEW])  # What loads on first use, untraced.
    few, many = (peak_reading_three_stations(few_and_many[n]) for n in (FEW, MANY))
    assert (many - few) / (MANY - FEW) < 1024


def test_closing_closes_the_file_and_reading_then_fails(
    compressed: Path, tmp_path: Path
) -> None:
    path = Path(shutil.copy(compressed, tmp_path / "out.nc"))
    ds = open_dataset(path)
    picked, copied = ds.isel(station=[1, 2]), ds.copy(deep=True)
    assert held_open(path)
    ds.close()
    assert not held_open(path)
    # A selection made before the file was closed fails too, and so does a deep copy,
    # which reads through the same file.
    for waveform in (ds["waveform"], picked["waveform"], copied["waveform"]):
        with pytest.raises(ValueError, match=f"{re.escape(str(path))} is closed"):
            waveform.load()
    ds.close()  # Again, which does nothing.


def test_several_datasets_can_be_open_at_once(compressed: Path, tmp_path: Path) -> None:
    # Two of one file and one of another, while xarray's cache keeps only one file
    # open: each closes the file of the one before, which reopens it when it is read.
    other = Path(shutil.copy(compressed, tmp_path / "other.nc"))
    want = decompress_file(compressed)["waveform"].isel(station=[0, 3, 8])
    with xr.set_options(file_cache_maxsize=1):
        first, second, third = (
            open_dataset(p) for p in (compressed, compressed, other)
        )
        second.close()
        for ds in (first, third, first):
            got = ds["waveform"].isel(station=[0, 3, 8]).values
            assert bit_identical(got, want.values)
        first.close()
        third.close()


def test_a_file_opened_by_a_relative_path_opens_again_wherever_the_folder_is_then(
    compressed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # xarray's cache closes the file to open another, and the working folder changes
    # before the file is read again.
    want = decompress_file(compressed)["waveform"].isel(station=[0, 3])
    other = Path(shutil.copy(compressed, tmp_path / "other.nc"))
    monkeypatch.chdir(compressed.parent)
    with xr.set_options(file_cache_maxsize=1), open_dataset(compressed.name) as ds:
        open_dataset(other).close()
        monkeypatch.chdir(tmp_path)
        got = ds["waveform"].isel(station=[0, 3]).values
    assert bit_identical(got, want.values)


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("not compressed", NotCompressed),
        ("not HDF5", NotCompressed),
        ("format version 2", UnknownFormat),
        # Its streams decode, but to a station fewer than its coordinates name, which
        # fails as the waveform is put in the dataset, once the file is open.
        ("a station short", ValueError),
    ],
)
def test_a_file_that_cannot_be_read_fails_as_it_is_opened(
    compressed: Path, tmp_path: Path, fault: str, error: type[ValueError]
) -> None:
    # With decompress_file's error, and the file isn't left open.
    path = tmp_path / "bad.nc"
    if fault == "not compressed":
        write(build_standard_dataset(3, 50), path)
    elif fault == "not HDF5":
        path.write_text("station,pga\nS000000,0.12\n")
    else:
        shutil.copy(compressed, path)
        with h5py.File(path, "r+") as f:
            group = f[GROUP]
            if fault == "format version 2":
                group.attrs["format_version"] = 2
            else:
                for name in ("starts", "bytes", "gains", "offsets"):
                    rewrite(group, f"stream_{name}", group[f"stream_{name}"][:, :-1])
                group["stream_starts"].attrs["stream_size"] = NT
    with pytest.raises(error) as failed:
        decompress_file(path)
    with pytest.raises(error) as opened:
        open_dataset(path)
    # The same type, message and attributes: a refusal names the file as its filename.
    assert type(opened.value) is type(failed.value)
    assert (str(opened.value), vars(opened.value)) == (
        str(failed.value),
        vars(failed.value),
    )
    if fault != "not HDF5":
        assert not held_open(path)


def test_xarray_opens_a_compressed_file_with_its_engine(
    compressed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As open_dataset does. A variable dropped isn't read, so neither the dataset nor
    # its close can keep it, and with the waveform dropped, no file is kept open.
    path = Path(shutil.copy(compressed, tmp_path / "out.nc"))
    engine = "waveform_compression_tools"
    reads = record_reads(monkeypatch)
    with xr.open_dataset(path, engine=engine, drop_variables=["vs30"]) as ds:
        got = ds.isel(station=[2, 5]).load()
        assert held_open(path)
    assert not held_open(path)
    names = {name for name, _ in reads}
    assert "/latitude" in names and "/vs30" not in names
    with xr.open_dataset(path, engine=engine, drop_variables="waveform") as ds:
        assert "waveform" not in ds and not held_open(path)
    monkeypatch.undo()
    assert_same(got, decompress_file(path).drop_vars("vs30").isel(station=[2, 5]))
    with pytest.raises(TypeError, match="by its path"):
        xr.open_dataset(io.BytesIO(path.read_bytes()), engine=engine)


def test_importing_the_package_and_its_engine_imports_neither_h5py_nor_pyflac(
    tmp_path: Path,
) -> None:
    # xarray imports the engine, and so the package, to list its engines, as it does
    # for any file it opens without being told one of its own engines.
    code = (
        "import sys, waveform_compression_tools, waveform_compression_tools.engine; "
        "print(sorted({'h5py', 'pyflac'} & sys.modules.keys()))"
    )
    ran = run_python(code, cwd=tmp_path)
    assert (ran.returncode, ran.stdout) == (0, "[]\n"), ran.stderr
