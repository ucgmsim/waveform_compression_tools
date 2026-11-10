"""Why an input is refused, and how a compressed file fails its check.

A refusal, `Refused` or one of its subclasses, says that an input can't be used as it
is: it can't be compressed, decompressed, verified or converted. Nothing is written: no
file appears at the destination, and a file already there stays as it was. A failed
check, `CheckFailed`, says that the compressed output didn't come back as it should. It
is deleted, so nothing appears at the destination either.

Each names the file it is about as its `filename`, where that is known, as an `OSError`
does; its message is the same either way. A function that knows which file a step is
about, but not the step itself, names it with `about`.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager


class Refused(ValueError):  # noqa: N818
    """An input can't be used as it is. Nothing is written.

    Parameters
    ----------
    message : str
        What is wrong.
    filename : str or os.PathLike, optional
        The file the refusal is about.

    Attributes
    ----------
    filename : str or os.PathLike or None
        The file the refusal is about, or None if that isn't known.
    """

    def __init__(  # numpydoc ignore=PR01
        self, message: str, filename: str | os.PathLike[str] | None = None
    ) -> None:
        """Keep the message, and the file it is about."""
        super().__init__(message)
        self.filename = filename


class NotStandardLayout(Refused):
    """A dataset does not hold its waveform in the layout this package handles."""


class AllStationsReal(Refused):
    """Every station of a file is real, so the whole file stays exact, uncompressed."""


class HasGroups(Refused):
    """A file holds an HDF5 group besides its root, which its compressed file would lack.

    xarray reads only the root group, so a group would be left behind.
    """


class NonFiniteSamples(Refused):
    """A trace holds a NaN or an infinity, which can't be quantised."""


class AlreadyCompressed(Refused):
    """A file holds a compressed waveform already."""


class OutputExists(Refused):
    """A file is at the destination already, and replacing it wasn't asked for."""


class SameFile(Refused):
    """The destination, or the destination with ``.partial`` added, is the input."""


class NotCompressed(Refused):
    """A file holds no compressed waveform, so there is nothing to decompress or verify."""


class UnknownFormat(Refused):
    """A compressed file is in a format version, or has a setting, that can't be read."""


class NotBBBin(Refused):
    """A file can't be a BB.bin, or is one cut short.

    Its size doesn't fit its header, or its text isn't ASCII or has a NUL inside it.
    """


class NotHFBin(Refused):
    """A file can't be an HF.bin, or is one cut short.

    Its size doesn't fit its header, or its text isn't ASCII or has a NUL inside it.
    """


class IncompleteHFBin(Refused):
    """An HF.bin is unfinished: a station's vs is 0, as it is until the station is done."""


class CheckFailed(RuntimeError):  # noqa: N818
    """The compressed output didn't pass the check. Nothing is left at the destination.

    An error while decoding during the check counts as a failed check.

    Parameters
    ----------
    message : str
        What is wrong.
    filename : str or os.PathLike, optional
        Where the compressed file was going.

    Attributes
    ----------
    filename : str or os.PathLike or None
        Where the compressed file was going, or None if that isn't known.
    """

    def __init__(  # numpydoc ignore=PR01
        self, message: str, filename: str | os.PathLike[str] | None = None
    ) -> None:
        """Keep the message, and the file it is about."""
        super().__init__(message)
        self.filename = filename


@contextmanager
def about(path: str | os.PathLike[str]) -> Generator[None]:
    """Have a refusal or a failed check in a block that names no file name `path`.

    One that names a file already keeps it.

    Parameters
    ----------
    path : str or os.PathLike
        The file the block's refusals and failed checks are about.

    Yields
    ------
    None
        Nothing; the block runs.
    """
    try:
        yield
    except (Refused, CheckFailed) as error:
        if error.filename is None:
            error.filename = path
        raise
