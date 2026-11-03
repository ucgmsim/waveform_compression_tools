import struct
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


# The tests below pin behaviour that the tests above leave open.


def streaminfo(blob: bytes) -> dict[str, int]:
    # A FLAC stream is the marker "fLaC" and then the STREAMINFO block: a 4-byte block
    # header and 34 bytes of data. Counting from the start of the stream, bytes 8 to
    # 12 of it are the minimum and maximum block sizes, and bytes 18 to 26 pack the
    # sample rate, channel count and bits per sample (20, 3 and 5 bits) ahead of the
    # total sample count.
    assert blob[:4] == b"fLaC"
    min_blocksize, max_blocksize = struct.unpack(">HH", blob[8:12])
    packed = int.from_bytes(blob[18:26], "big")
    return {
        "min_blocksize": min_blocksize,
        "max_blocksize": max_blocksize,
        "sample_rate": packed >> 44,
        "channels": ((packed >> 41) & 0b111) + 1,
        "bits_per_sample": ((packed >> 36) & 0b11111) + 1,
    }


def test_a_stream_that_cannot_be_finished_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # libFLAC writes its last frame as the stream is finished, and says whether that
    # worked. A stream that failed to finish would end a frame short.
    monkeypatch.setattr(pyflac.StreamEncoder, "finish", lambda self: False)
    with pytest.raises(pyflac.EncoderProcessException, match="finish"):
        encode_stream(np.arange(10_000, dtype=np.int32))


def test_decoding_does_not_even_try_to_start_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The thread count test above can't see a thread that has finished by the time
    # decoding returns, so this one refuses to let any thread start.
    def refuse(self: threading.Thread) -> None:
        raise AssertionError("decoding started a thread")

    blob = encode_stream(np.arange(10_000, dtype=np.int32))
    monkeypatch.setattr(threading.Thread, "start", refuse)
    decode_stream(blob)


def flip_a_byte(blob: bytes) -> bytes:
    damaged = bytearray(blob)
    damaged[5000] ^= 0xFF
    return bytes(damaged)


DAMAGE = {
    "a flipped byte": flip_a_byte,
    "the last 10 bytes cut off": lambda blob: blob[:-10],
    "not FLAC at all": lambda blob: bytes(len(blob)),
}
