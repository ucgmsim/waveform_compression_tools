"""The per-file JSON record that says what became of one input file."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from waveform_compression_tools import __version__
from waveform_compression_tools.destinations import writing
from waveform_compression_tools.files import _FORMAT_VERSION, CompressResult

RECORD_VERSION = 1


def make_record(
    *,
    input_path: Path,
    input_hash: str | None,
    bits: int,
    result: CompressResult | None,
    output_path: Path,
    failure: tuple[str, str] | None,
    hash_s: float,
    total_s: float,
) -> dict:
    """Build the record of one input file.

    Parameters
    ----------
    input_path : Path
        The input, recorded as given. Its size is recorded as it is now.
    input_hash : str or None
        The input's content hash, if it was computed.
    bits : int
        The quantisation setting asked for.
    result : CompressResult or None
        What compressing made, if it finished: the record's check, its largest error,
        the numbers of traces and of real stations, the output's size and content hash,
        and the time spent compressing, checking and hashing the output come from it.
    output_path : Path
        Where the compressed file goes, recorded as given if compressing finished.
    failure : tuple[str, str] or None
        Why the run stopped short, if it did: ``"refused"`` or ``"failed"``, and the
        reason.
    hash_s : float
        The time spent hashing the input, in seconds. That of the output comes from
        `result`.
    total_s : float
        The time the whole run took, in seconds.

    Returns
    -------
    dict
        The record, ready for `write_record`.
    """
    status, reason = ("ok", "") if failure is None else failure
    compress_s = check_s = 0.0
    if result is not None:
        hash_s += result.hash_s
        compress_s = result.read_s + result.encode_s + result.write_s
        check_s = result.check_s
    return {
        "record_version": RECORD_VERSION,
        "tool": "waveform_compression_tools",
        "tool_version": __version__,
        "status": status,
        "reason": reason,
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "content_hash": input_hash,
        },
        "output": None
        if result is None
        else {
            "path": str(output_path),
            "bytes": result.bytes_out,
            "content_hash": result.output_hash,
        },
        "bits": bits,
        "format_version": _FORMAT_VERSION,
        "checked": result is not None and result.max_relative_error is not None,
        "max_relative_error": None if result is None else result.max_relative_error,
        "n_traces": None if result is None else result.n_traces,
        "n_exact_stations": None if result is None else result.n_exact,
        "seconds": {
            "hash": float(hash_s),
            "compress": float(compress_s),
            "check": float(check_s),
            "total": float(total_s),
        },
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_record(record: dict, path: str | os.PathLike[str]) -> None:
    """Write a record as strict JSON, appearing at `path` complete or not at all.

    Parameters
    ----------
    record : dict
        The record from `make_record`.
    path : str or os.PathLike
        Where to write it. An earlier file there is replaced only on success. It is
        written under a partial name first (see
        `waveform_compression_tools.destinations.writing`).

    Raises
    ------
    ValueError
        If the record holds a NaN or infinity, which JSON cannot carry.
    """
    with writing(path) as partial:
        partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
