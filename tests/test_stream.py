import threading

import numpy as np
import pyflac
import pytest

from waveform_compression_tools.stream import decode_stream, encode_stream


@pytest.mark.parametrize("n", [1, 4095, 4096, 4097, 147_499])
def test_round_trip_exact(n: int) -> None:
    q = np.random.default_rng(n).integers(-(2**31) + 1, 2**31, n).astype(np.int32)
    q[0] = 2**31 - 1
    assert np.array_equal(decode_stream(encode_stream(q)), q)


def test_a_stream_that_cannot_be_finished_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # libFLAC writes its last frame as the stream is finished, and says whether that
    # worked. A stream that failed to finish ends a frame short: that has happened, and
    # only verification noticed it.
    monkeypatch.setattr(pyflac.StreamEncoder, "finish", lambda self: False)
    with pytest.raises(pyflac.EncoderProcessException, match="finish"):
        encode_stream(np.arange(10_000, dtype=np.int32))


def test_decoding_does_not_even_try_to_start_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pyflac's own StreamDecoder starts a thread for each stream and polls it, which
    # makes decoding a file of many traces slow. A count of threads can't see one that
    # has ended by the time decoding returns, so this refuses to let any thread start.
    def refuse(self: threading.Thread) -> None:
        raise AssertionError("decoding started a thread")

    blob = encode_stream(np.arange(10_000, dtype=np.int32))
    monkeypatch.setattr(threading.Thread, "start", refuse)
    decode_stream(blob)
