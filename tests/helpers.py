"""What several test files share: plain functions, where conftest.py has the fixtures."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

# The type of conftest.py's standard_dataset fixture.
StandardDataset = Callable[..., xr.Dataset]


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
