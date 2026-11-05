import struct
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

from helpers import MakeBbbin, RealTraces, StandardDataset, build_standard_dataset

# What make_bbbin writes in the header. Every number but dt is exact in float32.
DURATION, DT, START_SEC = 123.5, 0.005, -1.0
LF_DIR = "/runs/fault1/rel01/LF"
LF_VM = "/models/fault1/vm"
HF_FILE = "/runs/fault1/rel01/HF/Acc/HF.bin"


@pytest.fixture
def standard_dataset() -> StandardDataset:
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

    return build_standard_dataset


@pytest.fixture
def make_bbbin() -> MakeBbbin:
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


REAL_TRACES = Path(__file__).parent / "real_traces"


@pytest.fixture
def real_traces() -> RealTraces:
    """Return a function that loads one set of real traces from ``tests/real_traces``.

    Each file there holds one float32 dataset, ``waveform``, of shape (3, 10, nt): 10
    stations' records from one physics-based simulation of an earthquake, as its
    low-frequency (``lf``), high-frequency (``hf``) and broadband (``bb``) traces. The
    numbers are copied bit for bit from the simulation's files, and nothing else is.
    """

    def load(name: str) -> np.ndarray:
        with h5py.File(REAL_TRACES / f"{name}.h5", "r") as f:
            return f["waveform"][...]

    return load
