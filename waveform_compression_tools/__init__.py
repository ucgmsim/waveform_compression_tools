"""Store float32 waveforms as FLAC, one stream per trace."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waveform_compression_tools")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"
