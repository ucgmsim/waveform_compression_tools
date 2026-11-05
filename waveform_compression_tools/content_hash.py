"""Hash a file's content the way Dropbox does."""

import hashlib
import os
from pathlib import Path

BLOCK_BYTES = 4 * 1024 * 1024


def content_hash(path: str | os.PathLike[str]) -> str:
    """Hash a file by Dropbox's content hash algorithm.

    The file is cut into blocks of 4 MiB (the last may be shorter), each block is
    hashed with SHA-256, and the raw 32-byte digests, laid end to end, are hashed
    again. See Dropbox's developer reference, "Content hash". Only one block is
    held in memory at a time.

    Parameters
    ----------
    path : str or os.PathLike
        The file to hash.

    Returns
    -------
    str
        The hash, as 64 lower-case hexadecimal digits.
    """
    outer = hashlib.sha256()
    # A buffered read returns a whole block, or what is left at the end of the file.
    with Path(path).open("rb") as f:
        while block := f.read(BLOCK_BYTES):
            outer.update(hashlib.sha256(block).digest())
    return outer.hexdigest()
