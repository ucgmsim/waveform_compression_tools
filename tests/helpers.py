"""What several test files share: plain functions, where conftest.py has the fixtures."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# The type of conftest.py's standard_dataset fixture.
StandardDataset = Callable[..., xr.Dataset]

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


def write(ds: xr.Dataset, path: Path) -> Path:
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def partial_of(path: Path) -> Path:
    return Path(f"{path}.partial")


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
