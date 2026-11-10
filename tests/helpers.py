"""What several test files share: plain functions, where conftest.py has the fixtures."""

import os
import struct
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import xarray as xr

from waveform_compression_tools.codec import EncodedBlock
from waveform_compression_tools.files import compress_file

# The types of conftest.py's fixtures standard_dataset, make_bbbin, make_hfbin and
# real_traces.
StandardDataset = Callable[..., xr.Dataset]
MakeBbbin = Callable[..., np.ndarray]
MakeHfbin = Callable[..., np.ndarray]
RealTraces = Callable[[str], np.ndarray]

# For a test that fails writes with a limit on the size of a file (see
# limit_file_sizes), which Windows doesn't have.
limits_file_sizes = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows can't limit the size of a file"
)


def limit_file_sizes(limit: int) -> str:
    """Return Python code that keeps each file the process writes to `limit` bytes.

    A write past the limit then fails as one does on a full disk or past a quota, with
    EFBIG ("File too large"), since Python ignores the signal the kernel sends first.
    """
    return (
        "import resource; resource.setrlimit(resource.RLIMIT_FSIZE, "
        f"({limit}, resource.getrlimit(resource.RLIMIT_FSIZE)[1])); "
    )


def run_python(
    code: str, *args: str | Path, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run Python code, with arguments, in a fresh interpreter, in the folder `cwd`.

    The interpreter is this one, with its -S flag if it has it, so that it imports the
    same package. It is a process of its own so that what it does to its files, its
    signals and its HDF5 library can't reach the tests, even if it crashes. Coverage
    isn't measured there, since its data is a file that a limit could stop.
    """
    flags = ["-S"] if sys.flags.no_site else []
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("COV_CORE_", "COVERAGE_"))
    }
    return subprocess.run(
        [sys.executable, *flags, "-c", code, *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,  # A test shows the process's own error instead.
    )


def build_standard_dataset(
    n_station: int, nt: int, real: Sequence[int] = ()
) -> xr.Dataset:
    # conftest.py's standard_dataset fixture gives this, and says what it builds.
    rng = np.random.default_rng(0)
    walks = np.cumsum(rng.normal(size=(3, n_station, nt)), axis=2)
    return xr.Dataset(
        {
            "waveform": (
                ("component", "station", "time"),
                walks.astype(np.float32),
                {"units": "g"},
            ),
            "vs30": ("station", rng.uniform(150.0, 1500.0, n_station)),
        },
        coords={
            "component": ["000", "090", "ver"],
            "station": [
                f"R{i:03d}" if i in real else f"S{i:06d}" for i in range(n_station)
            ],
            "latitude": ("station", rng.uniform(-47.0, -34.0, n_station)),
            "longitude": ("station", rng.uniform(166.0, 179.0, n_station)),
        },
        attrs={"dt": 0.005, "start_sec": -3.0},
    )


def standard_dataset_of(waveform: np.ndarray) -> xr.Dataset:
    # The dataset of build_standard_dataset, holding these traces instead of its own.
    ds = build_standard_dataset(waveform.shape[1], waveform.shape[2])
    ds["waveform"].values = waveform
    return ds


def write(ds: xr.Dataset, path: Path, **options: Any) -> Path:
    ds.to_netcdf(path, engine="h5netcdf", **options)
    return path


def read(path: str | Path) -> xr.Dataset:
    # The whole of a netCDF file, loaded, and the file closed.
    return xr.load_dataset(path, engine="h5netcdf")


def names_in(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


# What write_bbbin writes in the header. Every number but dt is exact in float32.
DURATION, DT, START_SEC = 123.5, 0.005, -1.0
LF_DIR = "/runs/fault1/rel01/LF"
LF_VM = "/models/fault1/vm"
HF_FILE = "/runs/fault1/rel01/HF/Acc/HF.bin"


def write_bbbin(
    path: Path,
    nstat: int,
    nt: int,
    endian: str = "<",
    names: Sequence[str] | None = None,
) -> np.ndarray:
    # conftest.py's make_bbbin fixture gives this, and says what it writes.
    names = [f"STA{i}" for i in range(nstat)] if names is None else names
    assert len(names) == nstat
    rng = np.random.default_rng(0)
    # Random walks of a few hundredths of a g, like an acceleration record.
    data = (0.01 * np.cumsum(rng.normal(size=(nstat, nt, 3)), axis=1)).astype(
        np.float32
    )
    header = struct.pack(f"{endian}2i3f", nstat, nt, DURATION, DT, START_SEC)
    header += b"".join(s.encode().ljust(256, b"\0") for s in (LF_DIR, LF_VM, HF_FILE))
    records = b"".join(
        struct.pack(
            f"{endian}ff8s3i4f",
            170.5 + i,
            -40.5 + i,
            name.encode(),
            11 + i,
            22 + i,
            33 + i,
            44.5 + i,
            555.5 + i,
            666.5 + i,
            777.5 + i,
        )
        for i, name in enumerate(names)
    )
    path.write_bytes(
        header.ljust(0x500, b"\0") + records + data.astype(f"{endian}f4").tobytes()
    )
    return data


# What write_hfbin writes in the header after the numbers of stations and samples, in
# the file's order: the int32s, the float32s, then the strings. Each field has a value
# of its own, so that one read from the wrong place shows.
HFBIN_INTS = {
    "seed": 5_234_567,
    "siteamp": 1,
    "pdur_model": 11,
    "nrayset": 2,
    "rayset1": 21,
    "rayset2": 22,
    "rayset3": 23,
    "rayset4": 24,
    "nbu": 4,
    "ift": 5,
    "nlskip": -99,
    "icflag": 6,
    "same_seed": 7,
    "site_specific_vm": 8,
}
HFBIN_FLOATS = {
    "duration": 123.5,
    "dt": 0.005,
    "start_sec": -1.0,
    "sdrop": 50.0,
    "kappa": 0.045,
    "qfexp": 0.6,
    "fmax": 10.0,
    "flo": 0.02,
    "fhi": 19.9,
    "rvfac": 0.8,
    "rvfac_shal": 0.7,
    "rvfac_deep": 0.75,
    "czero": 2.1,
    "calpha": -99.0,
    "mom": -2.5,
    "rupv": -3.5,
    "vs_moho": 999.9,
    "vp_sig": 0.11,
    "vsh_sig": 0.12,
    "rho_sig": 0.13,
    "qs_sig": 0.14,
    "fa_sig1": 0.21,
    "fa_sig2": 0.22,
    "rv_sig1": 0.1,
}
HFBIN_TEXTS = {"stoch_file": "rel01.stoch", "velocity_model": "model.1d"}


def write_hfbin(
    path: Path,
    nstat: int,
    nt: int,
    endian: str = "<",
    names: Sequence[str] | None = None,
    unfinished: Sequence[int] = (),
) -> np.ndarray:
    # conftest.py's make_hfbin fixture gives this, and says what it writes.
    names = [f"STA{i}" for i in range(nstat)] if names is None else names
    assert len(names) == nstat
    rng = np.random.default_rng(0)
    # Random walks of tens of cm/s^2, like a high-frequency acceleration record.
    data = (10 * np.cumsum(rng.normal(size=(nstat, nt, 3)), axis=1)).astype(np.float32)
    header = struct.pack(
        f"{endian}16i24f64s64s",
        nstat,
        nt,
        *HFBIN_INTS.values(),
        *HFBIN_FLOATS.values(),
        *(text.encode() for text in HFBIN_TEXTS.values()),
    )
    records = b"".join(
        struct.pack(
            f"{endian}ff8sff",
            170.5 + i,
            -40.5 + i,
            name.encode(),
            0.0 if i in unfinished else 44.5 + i,
            0.0 if i in unfinished else 555.5 + i,
        )
        for i, name in enumerate(names)
    )
    path.write_bytes(
        header.ljust(0x200, b"\0") + records + data.astype(f"{endian}f4").tobytes()
    )
    return data


def partial_of(path: Path) -> Path:
    return Path(f"{path}.partial")


def nothing_left(dst: Path) -> bool:
    # Neither the output nor its partial file is there.
    return not dst.exists() and not partial_of(dst).exists()


def contents(folder: Path) -> dict[str, bytes | Path]:
    # Each file in a folder, by name: a file's bytes, and where a link points.
    return {
        path.name: path.readlink() if path.is_symlink() else path.read_bytes()
        for path in folder.iterdir()
    }


def write_and_compress(
    ds: xr.Dataset, folder: Path, **options: Any
) -> tuple[Path, Path]:
    # Writes a dataset to in.nc and compresses it to out.nc, returning both.
    src, dst = write(ds, folder / "in.nc"), folder / "out.nc"
    compress_file(src, dst, **options)
    return src, dst


def rewrite(group: h5py.Group, name: str, data: np.ndarray) -> None:
    # Replaces a dataset in a group, whatever its shape or type.
    del group[name]
    group.create_dataset(name, data=data)


def stream_of(encoded: EncodedBlock, c: int, s: int) -> bytes:
    # Component c of station s's stream, in encode_block's (component, station) order.
    return encoded.streams[c * encoded.gains.shape[1] + s]


def with_stream(
    encoded: EncodedBlock, stream: bytes, c: int = 1, s: int = 2
) -> EncodedBlock:
    # A block's streams, with component c of station s's stream, and its size, replaced.
    streams = list(encoded.streams)
    streams[c * encoded.gains.shape[1] + s] = stream
    nbytes = encoded.nbytes.copy()
    nbytes[c, s] = len(stream)
    return EncodedBlock(streams, nbytes, encoded.gains)


def with_a_flipped_byte(encoded: EncodedBlock, c: int = 1, s: int = 2) -> EncodedBlock:
    # A block's streams, with the middle byte of component c of station s flipped.
    stream = bytearray(stream_of(encoded, c, s))
    stream[len(stream) // 2] ^= 0xFF
    return with_stream(encoded, bytes(stream), c, s)


# The non-finite values of refused_input's cases.
NON_FINITE = {"a NaN": np.nan, "an infinity": np.inf, "a negative infinity": -np.inf}


def refused_input(
    case: str, folder: Path, standard_dataset: StandardDataset
) -> tuple[Path, Path, bool]:
    # Writes the input for one of compress's refusals, named by `case`, and an earlier
    # file at the destination unless that is the input. Returns the input, the
    # destination, and whether to ask for overwriting. Every case asks, so that the
    # earlier file is seen to stay, but three: the two that don't ask, so something at
    # the destination is refused, and "the same path", which is the same file though a
    # file is there.
    src, dst = folder / "in.nc", folder / "out.nc"
    ds = standard_dataset(3, nt=100)
    if case == "its own partial":
        src = partial_of(dst)
    elif case == "every station real":
        ds = standard_dataset(3, nt=100, real=(0, 1, 2))
    elif case == "no stations":
        ds = standard_dataset(0, nt=100)
    elif case == "a float64 waveform":
        ds["waveform"] = ds["waveform"].astype(np.float64)
    elif case == "dimensions in another order":
        ds["waveform"] = ds["waveform"].transpose("station", "component", "time")
    elif case == "no waveform":
        ds = ds.rename_vars(waveform="other")
    elif case == "no station coordinate":
        ds = ds.drop_vars("station")
    elif case == "station names along another dimension":
        # Names that would make station 1 real, but they aren't the stations' names.
        names = ("site", ["6c5SYSL", "OPZ", "22008d3"])
        ds = ds.drop_vars("station").assign_coords(station=names)
    elif case == "station names of bytes that aren't ASCII":
        names = np.array(["Ōtaki".encode(), b"S000001", b"S000002"])
        ds = ds.assign_coords(station=names)
    elif case in NON_FINITE:
        # Station 2 is in the second of two blocks, after the first is in the partial
        # file.
        ds["waveform"][2, 2, 50] = NON_FINITE[case]
    elif case == "a fill value inside the data":
        ds["waveform"].encoding["_FillValue"] = np.float32(0.0)
    elif case == "a missing_value inside the data":
        # Without a _FillValue, so that only the missing_value is inside the data.
        ds["waveform"].encoding.update(_FillValue=None, missing_value=np.float32(0.0))
    write(ds, src)
    if case == "an HDF5 group":
        with h5py.File(src, "a") as f:
            f.create_group("provenance").create_dataset("notes", data=np.arange(3))
    elif case == "conflicting fill values":
        # Beside xarray's NaN _FillValue, as another writer might give it. xarray
        # couldn't write a decompressed file with both.
        with h5py.File(src, "r+") as f:
            f["waveform"].attrs["missing_value"] = np.float32(-1e6)
    elif case == "a vector missing_value":
        with h5py.File(src, "r+") as f:
            f["waveform"].attrs["missing_value"] = np.array([-1e6, -2e6], np.float32)
    elif case == "an attribute JSON can't hold":
        with h5py.File(src, "r+") as f:
            f["waveform"].attrs["gain"] = np.complex128(1 + 2j)
    elif case == "a _FillValue JSON can't hold":
        # One value, so that only its JSON can refuse it.
        with h5py.File(src, "r+") as f:
            f["waveform"].attrs["_FillValue"] = np.complex64(1 + 2j)
    elif case == "a _FillValue of text":
        # JSON holds text, so only the check of the fill value itself can refuse it.
        with h5py.File(src, "r+") as f:
            f["waveform"].attrs["_FillValue"] = "-999"
    elif case == "an attribute whose type can't be kept":
        with h5py.File(src, "r+") as f:
            pair = np.zeros((), np.dtype([("a", "<i4"), ("b", "<f4")]))
            f["waveform"].attrs.create("pair", pair)
    elif case == "already compressed":
        src = folder / "in.flac.nc"
        compress_file(folder / "in.nc", src)
    elif case == "a text file":
        src = folder / "notes.nc"
        src.write_text("station,pga\nS000000,0.12\n")
    if case == "the same path":
        return src, src, False
    if case == "a link to it":
        dst.symlink_to(src)
    elif case == "a hard link to it":
        # Another name for the input, which no comparison of names can see.
        dst.hardlink_to(src)
    elif case == "a hard link to it at its partial":
        dst.write_bytes(b"an earlier file")
        partial_of(dst).hardlink_to(src)
    elif case == "a broken link at the output":
        # Nothing to read there, but replacing the link would be overwriting.
        dst.symlink_to(folder / "missing.nc")
    else:
        dst.write_bytes(b"an earlier file")
    return src, dst, case not in ("the output exists", "a broken link at the output")


def within_one_step_of_the_setting(
    original: np.ndarray, decoded: np.ndarray, bits: int
) -> np.ndarray:
    """Say, for each trace, whether every sample is within the setting's one step.

    The step is the trace's peak over 2 ** (bits - 1) - 1, with a relative slack of
    1e-9 for rounding, as the README promises. A trace of zeros has a step of 0, so
    it has to come back as exact zeros.
    """
    original = original.astype(np.float64)
    step = np.abs(original).max(axis=2, keepdims=True) / (2 ** (bits - 1) - 1)
    return np.all(np.abs(decoded - original) <= step * (1 + 1e-9), axis=2)
