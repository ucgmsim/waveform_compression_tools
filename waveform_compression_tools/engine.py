"""xarray's engine for compressed files, ``engine="waveform_compression_tools"``.

It is a module of its own, which imports only xarray, since xarray imports every
installed engine to list them: it does that for any file it opens without being told
one of its own engines. The engine imports `waveform_compression_tools.dataset`, with
h5py and pyflac, only when it opens a file.
"""

import os
from collections.abc import Iterable

import xarray as xr
from xarray.backends import BackendEntrypoint


class FlacBackendEntrypoint(BackendEntrypoint):
    """xarray's engine for compressed files, ``"waveform_compression_tools"``.

    ``xr.open_dataset(path, engine="waveform_compression_tools")`` opens a compressed
    file as `waveform_compression_tools.open_dataset` does. It takes
    ``drop_variables``, but none of xarray's options for decoding, since the dataset is
    the original as it was compressed: ``decode_cf=False`` changes nothing, and
    another, such as ``decode_times=False``, raises a `TypeError`. xarray never picks
    this engine by itself: without it, xarray opens a compressed file as an ordinary
    netCDF file, without its waveform.
    """

    description = "Open a compressed file, decoding only the traces read"
    url = "https://github.com/ucgmsim/waveform_compression_tools"

    def open_dataset(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.Dataset:
        """Open a compressed file as `waveform_compression_tools.open_dataset` does.

        Parameters
        ----------
        filename_or_obj : str or os.PathLike
            The compressed file. xarray can give an engine a file object or bytes too,
            but a compressed file is opened only by its path.
        drop_variables : str or iterable of str, optional
            Variables to leave out, which aren't read. A name the file doesn't hold is
            ignored. With ``waveform`` left out, the file isn't kept open.

        Returns
        -------
        xr.Dataset
            The dataset, without `drop_variables`.

        Raises
        ------
        TypeError
            If `filename_or_obj` isn't a path.
        """
        if not isinstance(filename_or_obj, str | os.PathLike):
            raise TypeError(
                "a compressed file is opened by its path, not from a "
                f"{type(filename_or_obj).__name__}"
            )
        from .dataset import _open  # Here, for the reason the module's docstring gives.

        return _open(filename_or_obj, drop_variables)
