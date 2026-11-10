from pathlib import Path

import h5py
import numpy as np
import pytest

from helpers import (
    MakeBbbin,
    MakeHfbin,
    RealTraces,
    StandardDataset,
    build_standard_dataset,
    write_bbbin,
    write_hfbin,
)


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

    ``make_bbbin(path, nstat, nt, endian="<", names=None)`` writes a BB.bin file in
    the byte order `endian` (``"<"`` or ``">"``, for everything in it) and returns its
    data: float32 of shape (nstat, nt, 3), with the components in the file's order,
    090, 000 and ver. The stations are named STA0, STA1, ... unless `names` says
    otherwise; those names are 4 characters long, like a real station's. The header
    holds helpers.py's `DURATION`, `DT`, `START_SEC`, `LF_DIR`, `LF_VM` and
    `HF_FILE`.

    The records are packed with struct, a description of the layout that is
    independent of the reader's. Each number in a record has a value of its own, and
    station i's are station 0's plus i, so a field that is read from the wrong place
    shows.
    """

    return write_bbbin


@pytest.fixture
def make_hfbin() -> MakeHfbin:
    """Return a function that writes a legacy HF.bin file.

    ``make_hfbin(path, nstat, nt, endian="<", names=None, unfinished=())`` writes an
    HF.bin file in the byte order `endian` (``"<"`` or ``">"``, for everything in it)
    and returns its data: float32 of shape (nstat, nt, 3), in cm/s^2, with the
    components in the file's order, 090, 000 and ver. The stations are named STA0,
    STA1, ... unless `names` says otherwise; those names are 4 characters long, like a
    real station's. The header holds nstat and nt, then helpers.py's `HFBIN_INTS`,
    `HFBIN_FLOATS` and `HFBIN_TEXTS`.

    The header and the records are packed with struct, a description of the layout
    that is independent of the reader's. Each number has a value of its own, and
    station i's are station 0's plus i, so a field that is read from the wrong place
    shows. The stations whose indices are in `unfinished` have an ``e_dist`` and a
    ``vs`` of 0, as the writer of an HF.bin leaves a station until it is done.
    """

    return write_hfbin


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
