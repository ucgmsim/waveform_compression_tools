"""Store float32 waveforms as FLAC, one stream per trace."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("waveform_compression_tools")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"

if TYPE_CHECKING:
    from .dataset import open_dataset

__all__ = ["__version__", "open_dataset"]


def __getattr__(name: str) -> object:
    """Import `open_dataset` when it is first used.

    It imports h5py, pyflac and the rest of the package, which importing the package
    needn't: xarray imports it, through its engine, to list its engines.
    """
    if name == "open_dataset":
        from .dataset import open_dataset

        return open_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
