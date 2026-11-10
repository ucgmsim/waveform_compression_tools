"""What the legacy BB.bin and HF.bin layouts share.

A BB.bin and an HF.bin file are laid out alike, though their headers and their
stations' records differ in size and in what they hold (see
`waveform_compression_tools.bbbin` and `waveform_compression_tools.hfbin`):

- the header starts with the int32 numbers of stations ``nstat`` and of samples
  ``nt``;
- a record of the same size for each station follows the header;
- then the traces: float32 of shape (nstat, nt, 3), with the components in the order
  090, 000, ver.

Every number in a file is in one byte order, the writer's, which neither format
records. `layout` tells it from the file's size: in the writer's byte order, the
counts make the file exactly as long as it is.
"""

import numpy as np

from .codec import STANDARD_COMPONENTS

# The components of a sample, in the order a file holds them, and the size of a sample
# of all of them in float32.
FILE_COMPONENTS = ("090", "000", "ver")
N_COMPONENT = len(FILE_COMPONENTS)
SAMPLE_BYTES = 4 * N_COMPONENT


def layout(
    head: bytes, size: int, head_bytes: int, station_bytes: int
) -> tuple[str, int, int] | None:
    """Find a file's byte order, and how many stations and samples it holds.

    Parameters
    ----------
    head : bytes
        The start of the file: its header, or all of it if it is shorter.
    size : int
        The size of the whole file in bytes.
    head_bytes : int
        The size of the header, which starts with the int32 numbers of stations and of
        samples: 0x500 bytes in a BB.bin, and 0x200 in an HF.bin.
    station_bytes : int
        The size of each station's record, which follows the header: 0x2C bytes in a
        BB.bin, and 0x18 in an HF.bin.

    Returns
    -------
    tuple[str, int, int] | None
        The byte order (``"<"`` for little-endian or ``">"`` for big-endian), the
        number of stations, and the number of samples of each trace. They are those
        that the header gives in the first byte order in which a file of that many
        stations and samples is exactly `size` bytes long. A file holds at least one
        station and at least one sample. None if no byte order gives that.
    """
    if len(head) < head_bytes:
        return None
    for endian in ("<", ">"):
        # Python integers, so that a count that is wrong cannot overflow.
        nstat, nt = (int(n) for n in np.frombuffer(head, f"{endian}i4", count=2))
        expected = head_bytes + nstat * (station_bytes + nt * SAMPLE_BYTES)
        if nstat > 0 and nt > 0 and size == expected:
            return endian, nstat, nt
    return None


def standard_waveform(samples: bytes, endian: str, nstat: int, nt: int) -> np.ndarray:
    """Put a file's traces in the standard order, in the machine's byte order.

    Parameters
    ----------
    samples : bytes
        The traces as the file holds them: float32 of shape (nstat, nt, 3), in the
        byte order `endian`, with the components in the order 090, 000, ver.
    endian : str
        The file's byte order: ``"<"`` for little-endian or ``">"`` for big-endian.
    nstat : int
        The number of stations.
    nt : int
        The number of samples of each trace.

    Returns
    -------
    np.ndarray
        The traces, float32 of shape (3, nstat, nt), with the components in the order
        000, 090, ver, so the first two are swapped from the file's order.

    Raises
    ------
    ValueError
        If `samples` is too short for that, as when the file shrinks while it is read.
    """
    raw = np.frombuffer(samples, f"{endian}f4", count=nstat * nt * N_COMPONENT)
    raw = raw.reshape(nstat, nt, N_COMPONENT)
    waveform = np.empty((N_COMPONENT, nstat, nt), dtype=np.float32)
    for i, component in enumerate(STANDARD_COMPONENTS):
        waveform[i] = raw[:, :, FILE_COMPONENTS.index(component)]
    return waveform
