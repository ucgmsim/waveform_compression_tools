import os
import struct
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

# What make_bbbin writes in the header. Every number but dt is exact in float32.
DURATION, DT, START_SEC = 123.5, 0.005, -1.0
LF_DIR = "/runs/fault1/rel01/LF"
LF_VM = "/models/fault1/vm"
HF_FILE = "/runs/fault1/rel01/HF/Acc/HF.bin"


@pytest.fixture
def standard_dataset() -> Callable[..., xr.Dataset]:
    """Return a function that builds a dataset in the standard layout.

    ``standard_dataset(n_station, nt)`` holds a float32 ``waveform`` of random walks,
    with dimensions (component, station, time) and the attribute ``units = "g"``; a
    ``component`` coordinate of ``000``, ``090`` and ``ver``; stations named
    ``S000000``, ``S000001``, ... with ``latitude`` and ``longitude`` coordinates and a
    ``vs30`` variable; and the attributes ``dt = 0.005`` and ``start_sec = -3.0``.

    Names of 7 characters are a synthetic grid's. ``standard_dataset(n_station, nt,
    real=(2, 3))`` gives stations 2 and 3 4-character names instead,
    like a real station's.
    """

    def build(n_station: int, nt: int, real: Sequence[int] = ()) -> xr.Dataset:
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

    return build


@pytest.fixture
def make_bbbin() -> Callable[..., np.ndarray]:
    """Return a function that writes a legacy BB.bin file.

    ``make_bbbin(path, nstat, nt, endian, names=None)`` writes a BB.bin file in the
    byte order `endian` (``"<"`` or ``">"``, for everything in it) and returns its
    data: float32 of shape (nstat, nt, 3), with the components in the file's order,
    090, 000 and ver. The stations are named STA0, STA1, ... unless `names` says
    otherwise; those names are 4 characters long, like a real station's. The header
    holds `DURATION`, `DT`, `START_SEC`, `LF_DIR`, `LF_VM` and `HF_FILE`.

    The records are packed with struct, a description of the layout that is
    independent of the reader's. Each number in a record has a value of its own, and
    station i's are station 0's plus i, so a field that is read from the wrong place
    shows.
    """

    def write(
        path: Path,
        nstat: int,
        nt: int,
        endian: str,
        names: Sequence[str] | None = None,
    ) -> np.ndarray:
        names = [f"STA{i}" for i in range(nstat)] if names is None else names
        assert len(names) == nstat
        rng = np.random.default_rng(0)
        # Random walks of a few hundredths of a g, like an acceleration record.
        data = (0.01 * np.cumsum(rng.normal(size=(nstat, nt, 3)), axis=1)).astype(
            np.float32
        )
        header = struct.pack(f"{endian}2i3f", nstat, nt, DURATION, DT, START_SEC)
        header += b"".join(
            s.encode().ljust(256, b"\0") for s in (LF_DIR, LF_VM, HF_FILE)
        )
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

    return write


@pytest.fixture
def kit() -> Callable[[str], np.ndarray]:
    """Return a function that loads one set of real traces from the kit.

    The kit is a folder of real traces that isn't distributed: HDF5 files, each
    holding a float32 dataset ``waveform`` of shape (3, 10, nt). The folder is named
    by ``$WAVEFORM_COMPRESSION_KIT``, which has no default. A test that needs the kit
    is skipped when that is not set, or when the folder lacks the file it asks for.
    """
    env = os.environ.get("WAVEFORM_COMPRESSION_KIT")
    if not env:
        pytest.skip("WAVEFORM_COMPRESSION_KIT is not set")
    folder = Path(env).expanduser()

    def load(name: str) -> np.ndarray:
        path = folder / f"{name}.h5"
        if not path.is_file():
            pytest.skip(
                f"{path} not found (set WAVEFORM_COMPRESSION_KIT to the kit's folder)"
            )
        with h5py.File(path, "r") as f:
            return f["waveform"][...]

    return load
