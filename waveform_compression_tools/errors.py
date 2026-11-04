"""Why an input is refused, and how a compressed file fails its check.

A refusal, `Refused` or one of its subclasses, says that an input can't be used as it
is: it can't be compressed, decompressed, verified or converted. Nothing is written: no
file appears at the destination, and a file already there stays as it was. A failed
check, `CheckFailed`, says that the compressed output didn't come back as it should. It
is deleted, so nothing appears at the destination either.
"""


class Refused(ValueError):  # noqa: N818
    """An input can't be used as it is. Nothing is written."""


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
    """A file can't be a BB.bin: its size doesn't fit its header, or its text isn't ASCII."""


class CheckFailed(RuntimeError):  # noqa: N818
    """The compressed output didn't pass the check. Nothing is left at the destination.

    An error while decoding during the check counts as a failed check.
    """
