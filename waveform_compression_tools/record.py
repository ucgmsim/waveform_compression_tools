"""The per-file JSON record that says what became of one input file."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from waveform_compression_tools import __version__
from waveform_compression_tools.files import _FORMAT_VERSION

RECORD_VERSION = 1

_STATUSES = ("ok", "refused", "failed")
_SECONDS_KEYS = {"hash", "compress", "check", "total"}


def make_record(
    *,
    status: str,
    reason: str,
    input_path: Path,
    input_bytes: int,
    input_hash: str | None,
    output_path: Path | None,
    output_bytes: int | None,
    output_hash: str | None,
    bits: int,
    checked: bool,
    max_relative_error: float | None,
    n_traces: int | None,
    n_exact_stations: int | None,
    seconds: dict[str, float],
) -> dict:
    """Build the record of one input file.

    Parameters
    ----------
    status : str
        "ok", "refused" or "failed".
    reason : str
        Why the file was refused or failed; empty exactly when `status` is "ok".
    input_path : Path
        The input, recorded as given.
    input_bytes : int
        The input's size in bytes.
    input_hash : str or None
        The input's content hash, if it was computed.
    output_path : Path or None
        The compressed file, or None if none was written.
    output_bytes : int or None
        The output's size; None exactly when `output_path` is None.
    output_hash : str or None
        The output's content hash; None exactly when `output_path` is None.
    bits : int
        The quantisation setting used.
    checked : bool
        Whether the output was checked against the input.
    max_relative_error : float or None
        The largest relative error found by the check, if any.
    n_traces : int or None
        The number of traces, if known.
    n_exact_stations : int or None
        The number of stations stored exactly, if known.
    seconds : dict of str to float
        Timings, with exactly the keys "hash", "compress", "check" and "total".

    Returns
    -------
    dict
        The record, ready for `write_record`.

    Raises
    ------
    ValueError
        If the status, reason, output fields or timing keys are inconsistent.
    """
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}, not {status!r}")
    if (status == "ok") != (reason == ""):
        raise ValueError("reason must be empty when status is ok, and given otherwise")
    trio = (output_path, output_bytes, output_hash)
    if any(v is None for v in trio) and any(v is not None for v in trio):
        raise ValueError("output_path, output_bytes and output_hash go together")
    if set(seconds) != _SECONDS_KEYS:
        raise ValueError(f"seconds must have exactly the keys {sorted(_SECONDS_KEYS)}")

    output = None
    if output_path is not None:
        output = {
            "path": str(output_path),
            "bytes": output_bytes,
            "content_hash": output_hash,
        }
    return {
        "record_version": RECORD_VERSION,
        "tool": "waveform_compression_tools",
        "tool_version": __version__,
        "status": status,
        "reason": reason,
        "input": {
            "path": str(input_path),
            "bytes": input_bytes,
            "content_hash": input_hash,
        },
        "output": output,
        "bits": bits,
        "format_version": _FORMAT_VERSION,
        "checked": checked,
        "max_relative_error": max_relative_error,
        "n_traces": n_traces,
        "n_exact_stations": n_exact_stations,
        "seconds": {k: float(v) for k, v in seconds.items()},
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_record(record: dict, path: Path) -> None:
    """Write a record as strict JSON, appearing at `path` complete or not at all.

    Parameters
    ----------
    record : dict
        The record from `make_record`.
    path : Path
        Where to write it. An earlier file there is replaced only on success.

    Raises
    ------
    ValueError
        If the record holds a NaN or infinity, which JSON cannot carry.
    """
    partial = path.with_name(path.name + ".partial")
    try:
        text = json.dumps(record, indent=2, allow_nan=False) + "\n"
        partial.write_text(text)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
