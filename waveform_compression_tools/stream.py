"""One FLAC stream per trace: encode int32 samples, and decode them back.

A trace of quantised samples is stored as a single mono FLAC stream with 32 bits
per sample, whatever the quantisation setting. The stream's sample-rate field
says 44100. That means nothing here, but a stream has to carry a rate, and at
44100 FLAC's streamable subset allows blocks of at most 4608 samples. A stream is
written in one pass, so its header records neither the number of samples nor a
checksum of them: whoever decodes a stream has to know how many samples to
expect.

pyflac's own decoders do not suit a single stream held in memory.
``StreamDecoder`` starts a thread and polls it with 10 ms sleeps, and
``OneShotDecoder`` stops while libFLAC still holds read-ahead bytes, which drops
the last frames. `decode_stream` runs libFLAC to the end of the stream in the
calling thread instead, through pyflac's internals.
"""

import threading
from collections import deque

import numpy as np
import pyflac
from pyflac import decoder as pyflac_decoder


def encode_stream(q: np.ndarray, level: int = 5, blocksize: int = 4096) -> bytes:
    """Encode one trace as a single mono FLAC stream.

    Parameters
    ----------
    q : np.ndarray
        The samples: a non-empty 1-D int32 array. Any other kind of array is
        refused rather than converted, because pyflac would truncate floats to
        integers and flatten a 2-D array without a word.
    level : int, default 5
        The FLAC compression level, from 0 (fastest) to 8 (slowest).
    blocksize : int, default 4096
        The number of samples in each FLAC frame. The stream's sample-rate field
        is 44100, which allows at most 4608.

    Returns
    -------
    bytes
        The complete stream, from its ``fLaC`` marker to its last frame.

    Raises
    ------
    ValueError
        If `q` is not a non-empty 1-D int32 array.
    pyflac.EncoderInitException
        If libFLAC can't start the stream.
    pyflac.EncoderProcessException
        If libFLAC fails to encode the samples, or to finish the stream, which would
        then end a frame short.
    """
    if q.dtype != np.int32 or q.ndim != 1 or q.size == 0:
        raise ValueError(
            f"q must be a non-empty 1-D int32 array, got {q.dtype} with shape {q.shape}"
        )
    chunks: list[bytes] = []
    encoder = pyflac.StreamEncoder(
        sample_rate=44100,  # Meaningless for a trace, but a stream has to carry one.
        compression_level=level,
        blocksize=blocksize,
        write_callback=lambda buf, nbytes, nsamp, frame: chunks.append(bytes(buf)),
    )
    encoder.process(q.reshape(-1, 1))
    # Finishing writes the last frame. If that fails, as it does when the write
    # callback raises (pyflac swallows the exception), the stream would end short.
    if not encoder.finish():
        raise pyflac.EncoderProcessException(
            f"libFLAC could not finish the stream ({encoder.state})"
        )
    stream = b"".join(chunks)
    # The encoder is in a reference cycle, so it lives until the next collection, and
    # its write callback with it. The chunks needn't wait with them.
    chunks.clear()
    return stream


# The decoder below depends on pyflac 3.0.0's internals, not on its public API: the
# base class pyflac.decoder._Decoder, the cffi objects pyflac.decoder._lib and _ffi,
# and the attributes that pyflac's read callback expects a decoder to have (_buffer,
# _done, _event and _lock). Any new pyflac release can change them without notice,
# which is why the package pins pyflac==3.0.0. Run the tests again before moving the
# pin.


class _EndWhenEmpty:
    """Stand in for the `threading.Event` that pyflac's read callback waits on.

    Before every read, pyflac's read callback calls ``wait()`` on its decoder's
    event, to block until another thread supplies more bytes. Here every byte is
    supplied up front, so waiting never blocks. Once every byte has gone to
    libFLAC, ``wait()`` marks the decoder done, so the next read reports the end
    of the stream instead of waiting for data that will never come.

    Parameters
    ----------
    decoder : _WholeStreamDecoder
        The decoder whose buffer is watched and which is marked done.
    """

    def __init__(self, decoder: "_WholeStreamDecoder") -> None:  # numpydoc ignore=GL08
        self.decoder = decoder

    def wait(self, timeout: float | None = None) -> bool:
        """Mark the decoder done if no bytes are left, and return at once.

        Parameters
        ----------
        timeout : float | None, default None
            Ignored; it is here because `threading.Event.wait` takes one.

        Returns
        -------
        bool
            Always True, as `threading.Event.wait` does once its event is set.
        """
        if not any(len(b) for b in self.decoder._buffer):
            self.decoder._done = True
        return True

    def set(self) -> None:
        """Do nothing; pyflac's error callback calls this to wake a waiting thread."""

    def clear(self) -> None:
        """Do nothing; pyflac's read callback calls this when it runs out of bytes."""


class _WholeStreamDecoder(pyflac_decoder._Decoder):
    """Decode one complete FLAC stream held in memory, in the calling thread.

    Constructing the decoder does all the work: it hands the stream to libFLAC
    and runs ``process_until_end_of_stream``, reusing pyflac's own callbacks.
    The decoded frames are then in `frames`.

    Parameters
    ----------
    blob : bytes
        The complete stream.

    Attributes
    ----------
    frames : list[np.ndarray]
        The decoded frames in order, each an int32 array of shape
        (block size, 1).

    Raises
    ------
    pyflac.DecoderInitException
        If libFLAC refuses to start decoding.
    pyflac.DecoderProcessException
        If libFLAC reports an error, or stops before the end of the stream.
    """

    def __init__(self, blob: bytes) -> None:  # numpydoc ignore=GL08
        super().__init__()
        self._done = False
        # A view, since pyflac's read callback slices off what it reads each time: a
        # slice of bytes would copy the rest of the stream.
        self._buffer: deque[memoryview] = deque([memoryview(blob)])
        self._lock = threading.Lock()
        self._event = _EndWhenEmpty(self)
        self.frames: list[np.ndarray] = []
        lib, ffi = pyflac_decoder._lib, pyflac_decoder._ffi
        # libFLAC takes the read callback, then optional seek, tell, length and
        # end-of-file callbacks, the write callback, an optional metadata callback
        # and the error callback. The stream is read once, in order, so the optional
        # ones are left out (NULL).
        rc = lib.FLAC__stream_decoder_init_stream(
            self._decoder,
            lib._read_callback,
            ffi.NULL,
            ffi.NULL,
            ffi.NULL,
            ffi.NULL,
            lib._write_callback,
            ffi.NULL,
            lib._error_callback,
            self._decoder_handle,
        )
        if rc != lib.FLAC__STREAM_DECODER_INIT_STATUS_OK:
            raise pyflac_decoder.DecoderInitException(rc)
        lib.FLAC__stream_decoder_process_until_end_of_stream(self._decoder)
        state = self.state
        self.finish()
        if self._error or state != pyflac_decoder.DecoderState.END_OF_STREAM:
            raise pyflac_decoder.DecoderProcessException(f"{state} {self._error}")

    def write_callback(
        self, data: np.ndarray, sample_rate: int, channels: int, blocksize: int
    ) -> None:
        """Keep a copy of a decoded frame; pyflac's write callback calls this.

        Parameters
        ----------
        data : np.ndarray
            The frame's samples, of shape (block size, channels). The array may
            not be ours to keep, so a copy is stored.
        sample_rate : int
            The stream's sample-rate field. Ignored.
        channels : int
            The number of channels. Ignored.
        blocksize : int
            The number of samples in the frame. Ignored.
        """
        self.frames.append(np.array(data, copy=True))


def decode_stream(blob: bytes) -> np.ndarray:
    """Decode a FLAC stream written by `encode_stream` back to its samples.

    Parameters
    ----------
    blob : bytes
        One complete stream. It has to be bytes (a bytearray or a memoryview
        works too), not a numpy array: for a slice of a uint8 array, pass
        ``array.tobytes()``.

    Returns
    -------
    np.ndarray
        The samples, as a 1-D int32 array: empty if the blob holds no frames at all,
        as when it is empty.

    Raises
    ------
    pyflac.DecoderProcessException
        If libFLAC reports an error, as it does for a blob that is not a FLAC
        stream, a damaged one, and most that are cut short.

    Notes
    -----
    A stream carries no sample count, and a cut stream does not always raise: one
    that is cut between frames, or at some places inside a frame, decodes without
    an error to the whole frames before the cut, and one cut to nothing decodes to
    no samples. A caller that knows how many samples to expect must compare it with
    the length of the result.
    """
    frames = _WholeStreamDecoder(blob).frames
    if not frames:
        return np.empty(0, np.int32)
    samples = np.concatenate(frames).ravel()
    # The decoder is in reference cycles, so it lives until the next collection. Its
    # frames needn't wait with it.
    frames.clear()
    return samples
