import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from waveform_compression_tools import __version__, record
from waveform_compression_tools.content_hash import content_hash
from waveform_compression_tools.files import CompressResult
from waveform_compression_tools.record import make_record, write_record

from helpers import partial_of

BLOCK = 4_194_304


def reference_hash(data: bytes) -> str:
    pieces = [data[i : i + BLOCK] for i in range(0, len(data), BLOCK)]
    digests = b"".join(hashlib.sha256(p).digest() for p in pieces)
    return hashlib.sha256(digests).hexdigest()


@pytest.mark.parametrize("size", [0, 1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 5])
def test_content_hash_matches_the_plain_algorithm(tmp_path: Path, size: int):
    data = os.urandom(size)
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    assert content_hash(path) == reference_hash(data)
    if size == 0:
        assert content_hash(path) == hashlib.sha256(b"").hexdigest()


def a_result(**changes: Any) -> CompressResult:
    fields: dict[str, Any] = {
        "bytes_in": 100,
        "bytes_out": 10,
        "n_traces": 30,
        "n_exact": 2,
        "bits": 16,
        "max_relative_error": 1e-5,
        "output_hash": "ab" * 32,
        "read_s": 0.5,
        "encode_s": 1.0,
        "write_s": 0.5,
        "check_s": 3.0,
        "hash_s": 0.25,
        "read_cpu_s": 0.5,
        "encode_cpu_s": 1.0,
        "write_cpu_s": 0.5,
        "check_cpu_s": 3.0,
    }
    fields.update(changes)
    return CompressResult(**fields)


def good(folder: Path, **changes: Any) -> dict:
    # The record of a run that compressed a file of 100 bytes and checked it.
    src = folder / "a.nc"
    src.write_bytes(bytes(100))
    args: dict[str, Any] = {
        "input_path": src,
        "input_hash": None,
        "bits": 16,
        "result": a_result(),
        "output": (Path("out/a.nc"), 10, "ab" * 32),
        "failure": None,
        "hash_s": 0.75,
        "total_s": 6.0,
    }
    args.update(changes)
    return make_record(**args)


def test_a_record_has_exactly_the_schema(tmp_path: Path) -> None:
    rec = good(tmp_path)
    assert list(rec) == [
        "record_version", "tool", "tool_version", "status", "reason", "input",
        "output", "bits", "format_version", "checked", "max_relative_error",
        "n_traces", "n_exact_stations", "seconds", "finished_at",
    ]  # fmt: skip
    assert rec["record_version"] == 1
    assert rec["tool"] == "waveform_compression_tools"
    assert rec["tool_version"] == __version__
    assert (rec["status"], rec["reason"]) == ("ok", "")
    assert rec["format_version"] == 1
    path = str(tmp_path / "a.nc")
    assert rec["input"] == {"path": path, "bytes": 100, "content_hash": None}
    assert rec["output"] == {
        "path": "out/a.nc",
        "bytes": 10,
        "content_hash": "ab" * 32,
    }
    assert (rec["checked"], rec["max_relative_error"]) == (True, 1e-5)
    assert (rec["n_traces"], rec["n_exact_stations"]) == (30, 2)
    # Compressing is reading, encoding and writing; hashing is the input's and the
    # output's.
    assert rec["seconds"] == {"hash": 1.0, "compress": 2.0, "check": 3.0, "total": 6.0}
    assert rec["finished_at"].endswith("+00:00")
    assert json.loads(json.dumps(rec, allow_nan=False)) == rec
    unchecked = good(tmp_path, result=a_result(max_relative_error=None, check_s=0.0))
    assert (unchecked["checked"], unchecked["max_relative_error"]) == (False, None)
    refused = good(tmp_path, result=None, output=None, failure=("refused", "no"))
    assert (refused["status"], refused["reason"]) == ("refused", "no")
    assert refused["output"] is None and refused["checked"] is False
    assert (refused["n_traces"], refused["n_exact_stations"]) == (None, None)
    assert refused["seconds"] == {
        "hash": 0.75,
        "compress": 0.0,
        "check": 0.0,
        "total": 6.0,
    }


def test_write_record_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "r.json"
    write_record(good(tmp_path), path)
    text = path.read_text()
    assert text.endswith("\n") and json.loads(text)["status"] == "ok"
    assert not partial_of(path).exists()

    # NaN is not strict JSON: the earlier file stays, no partial is left.
    with pytest.raises(ValueError):
        write_record(
            good(tmp_path, result=a_result(max_relative_error=float("nan"))), path
        )
    assert path.read_text() == text

    # A failure after the partial is written removes it and leaves the old file.
    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk")

    monkeypatch.setattr(record.os, "replace", boom)
    with pytest.raises(OSError):
        write_record(good(tmp_path, bits=8), path)
    assert path.read_text() == text
    assert sorted(tmp_path.iterdir()) == [tmp_path / "a.nc", path]
