import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from waveform_compression_tools import __version__, record
from waveform_compression_tools.content_hash import content_hash
from waveform_compression_tools.record import make_record, write_record

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


def test_content_hash_survives_short_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = os.urandom(BLOCK + 1000)
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    real_open = Path.open

    class Trickle:
        def __init__(self, f: Any) -> None:
            self.f = f

        def read(self, n: int) -> bytes:
            return self.f.read(min(n, 1_000_003))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

    monkeypatch.setattr(
        Path, "open", lambda self, *a, **k: Trickle(real_open(self, *a, **k))
    )
    assert content_hash(path) == reference_hash(data)


def good(**changes: Any) -> dict:
    args: dict[str, Any] = {
        "status": "ok",
        "reason": "",
        "input_path": Path("in/a.nc"),
        "input_bytes": 100,
        "input_hash": None,
        "output_path": Path("out/a.nc"),
        "output_bytes": 10,
        "output_hash": "ab" * 32,
        "bits": 16,
        "checked": True,
        "max_relative_error": 1e-5,
        "n_traces": 30,
        "n_exact_stations": None,
        "seconds": {"hash": 1.0, "compress": 2.0, "check": 3.0, "total": 6.0},
    }
    args.update(changes)
    return make_record(**args)


def test_a_record_has_exactly_the_schema():
    rec = good()
    assert list(rec) == [
        "record_version", "tool", "tool_version", "status", "reason", "input",
        "output", "bits", "format_version", "checked", "max_relative_error",
        "n_traces", "n_exact_stations", "seconds", "finished_at",
    ]  # fmt: skip
    assert rec["record_version"] == 1
    assert rec["tool"] == "waveform_compression_tools"
    assert rec["tool_version"] == __version__
    assert rec["format_version"] == 1
    assert rec["input"] == {"path": "in/a.nc", "bytes": 100, "content_hash": None}
    assert rec["output"] == {
        "path": "out/a.nc",
        "bytes": 10,
        "content_hash": "ab" * 32,
    }
    assert rec["finished_at"].endswith("+00:00")
    assert json.loads(json.dumps(rec, allow_nan=False)) == rec
    refused = good(
        status="refused", reason="no", output_path=None, output_bytes=None,
        output_hash=None, max_relative_error=None,
    )  # fmt: skip
    assert refused["output"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "maybe"},
        {"reason": "why"},
        {"status": "failed", "reason": ""},
        {"output_bytes": None},
        {"output_path": None},
        {"seconds": {"hash": 1.0, "compress": 2.0, "check": 3.0}},
        {
            "seconds": {
                "hash": 1.0,
                "compress": 2.0,
                "check": 3.0,
                "total": 6.0,
                "x": 1.0,
            }
        },
    ],
)
def test_inconsistent_records_are_refused(changes: dict) -> None:
    with pytest.raises(ValueError):
        good(**changes)


def test_write_record_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "r.json"
    write_record(good(), path)
    text = path.read_text()
    assert text.endswith("\n") and json.loads(text)["status"] == "ok"
    assert not (tmp_path / "r.json.partial").exists()

    # NaN is not strict JSON: the earlier file stays, no partial is left.
    with pytest.raises(ValueError):
        write_record(good(max_relative_error=float("nan")), path)
    assert path.read_text() == text

    # A failure after the partial is written removes it and leaves the old file.
    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk")

    monkeypatch.setattr(record.os, "replace", boom)
    with pytest.raises(OSError):
        write_record(good(bits=8), path)
    assert path.read_text() == text
    assert list(tmp_path.iterdir()) == [path]
