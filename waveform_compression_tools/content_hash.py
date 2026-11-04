"""Hash a file's content the way Dropbox does."""

import hashlib
from pathlib import Path

BLOCK_BYTES = 4 * 1024 * 1024


def content_hash(path: Path) -> str:
    """Hash a file by Dropbox's content hash algorithm.

    The file is cut into blocks of 4 MiB (the last may be shorter), each block is
    hashed with SHA-256, and the raw 32-byte digests, laid end to end, are hashed
    again. See Dropbox's developer reference, "Content hash". Only one block is
    held in memory at a time.

    Parameters
    ----------
    path : Path
        The file to hash.

    Returns
    -------
    str
        The hash, as 64 lower-case hexadecimal digits.
    """
    outer = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(BLOCK_BYTES)
            # A short read does not mean the end: fill the block.
            while block and len(block) < BLOCK_BYTES:
                more = f.read(BLOCK_BYTES - len(block))
                if not more:
                    break
                block += more
            if not block:
                break
            outer.update(hashlib.sha256(block).digest())
    return outer.hexdigest()
