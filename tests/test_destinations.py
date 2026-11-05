import errno
import io
import json
import signal
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from waveform_compression_tools import destinations
from waveform_compression_tools.destinations import HDF5Output, write_netcdf

from helpers import StandardDataset, limits_file_sizes, run_python, write

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer, WriteableBuffer


@limits_file_sizes
def test_a_failed_write_reports_success_and_is_raised_on_leaving_the_block(
    tmp_path: Path,
) -> None:
    # HDF5 is never told that a write failed: the write reports every byte written, and
    # reads find what it held back, as HDF5 expects of a file it goes on writing until
    # it is closed; the same for a truncation, as HDF5 makes to close the file. The
    # error is raised as the block is left, in place of any other error but an
    # interruption, since a run can't succeed once its file can't be written. A limit
    # on the size of each file fails a write or a truncation as a full disk does, with
    # EFBIG.
    code = textwrap.dedent(
        """
        import json, resource
        from waveform_compression_tools.destinations import HDF5Output

        seen = {}
        hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
        try:
            with HDF5Output("a.bin") as out:
                out.write(b"a" * 1000)
                resource.setrlimit(resource.RLIMIT_FSIZE, (1500, hard))
                seen["written"] = out.write(b"b" * 1000)  # Only 500 bytes fit.
                out.seek(0)
                seen["read"] = out.read().decode()
        except OSError as error:
            seen["raised"] = error.errno
        try:
            with HDF5Output("b.bin") as out:
                out.write(b"c" * 2000)  # Past the limit too.
                raise ValueError("raised in the block")
        except OSError as error:
            seen["instead"] = [error.errno, repr(error.__context__)]
        try:
            with HDF5Output("c.bin") as out:
                out.write(b"c" * 2000)
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            seen["interrupted"] = True
        try:
            with HDF5Output("d.bin") as out:
                out.write(b"d" * 100)
                seen["truncated"] = out.truncate(3000)  # Past the limit.
                seen["size"] = out.seek(0, 2)
                out.seek(0)
                seen["after"] = out.read().decode()
        except OSError as error:
            seen["raised too"] = error.errno
        print(json.dumps(seen))
        """
    )
    ran = run_python(code, cwd=tmp_path)
    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout) == {
        "written": 1000,
        "read": "a" * 1000 + "b" * 1000,
        "raised": errno.EFBIG,
        "instead": [errno.EFBIG, "ValueError('raised in the block')"],
        "interrupted": True,
        "truncated": 3000,
        "size": 3000,
        "after": "d" * 100 + "\0" * 2900,
        "raised too": errno.EFBIG,
    }


class _Full(io.FileIO):
    # A file whose writes fail past `room` bytes, after a short one, as past a limit on
    # a file's size; a truncation past it fails too, and `broken` fails every read.
    room = 0
    broken = False

    def write(self, b: "ReadableBuffer") -> int:
        data = memoryview(b).cast("B")
        fits = self.room - self.tell()
        if len(data) and fits <= 0:
            raise OSError(errno.EFBIG, "File too large")
        written = super().write(data[: max(0, fits)])
        assert written is not None
        return written

    def truncate(self, size: int | None = None) -> int:
        if size is not None and size > self.room:
            raise OSError(errno.EFBIG, "File too large")
        return super().truncate(size)

    def readinto(self, b: "WriteableBuffer") -> int | None:
        if self.broken:
            raise OSError(errno.EIO, "Input/output error")
        return super().readinto(b)


def on_full(path: Path, room: int) -> tuple[HDF5Output, _Full]:
    # An HDF5Output that writes through a _Full with room for `room` bytes, and that.
    out = HDF5Output(path)
    out._file.close()
    full = _Full(path, "r+")
    full.room = room
    out._file = full
    return out, full


_BIG = 300  # The size from which a write after a failure is held as zeros, here.
_STEPS = st.lists(
    st.one_of(
        st.tuples(st.just("write"), st.integers(0, 2000), st.integers(0, 700)),
        st.tuples(st.just("truncate"), st.integers(0, 2000), st.just(0)),
        st.tuples(st.just("read"), st.integers(0, 2000), st.integers(0, 700)),
    ),
    max_size=25,
)


@settings(max_examples=300, deadline=None)
@given(room=st.integers(0, 2500), steps=_STEPS)
def test_what_is_read_is_what_was_written_before_and_after_a_failure(
    room: int, steps: list[tuple[str, int, int]]
) -> None:
    # Whatever HDF5 writes, truncates and reads, and wherever the disk fills, a read
    # gives what was written, and the file's size is its own, as in a file that never
    # failed: but that a write of _BIG bytes or more after a failure reads as zeros.
    # The failure is raised as the block is left. The model is a bytearray.
    model, failed = bytearray(), False
    with (
        tempfile.TemporaryDirectory() as folder,
        mock.patch.object(destinations, "_BIG_WRITE", _BIG),
    ):
        out, _ = on_full(Path(folder) / "f", room)
        try:
            with out:
                for number, (step, at, n) in enumerate(steps):
                    out.seek(at)
                    if step == "write":
                        data = bytes((number + i) % 251 + 1 for i in range(n))
                        assert out.write(data) == n
                        failed = failed or (n > 0 and at + n > room)
                        if n:
                            held = bytes(n) if failed and n >= _BIG else data
                            model[len(model) : at] = bytes(max(0, at - len(model)))
                            model[at : at + n] = held
                    elif step == "truncate":
                        assert out.truncate(at) == at
                        failed = failed or at > room
                        model[:] = model[:at] + bytes(max(0, at - len(model)))
                    else:
                        assert out.read(n) == bytes(model[at : at + n])
                    assert out.seek(0, 2) == len(model)
        except OSError as error:
            assert failed and error.errno == errno.EFBIG
        else:
            assert not failed


def test_a_failed_read_reads_as_zeros_and_is_raised_on_leaving_the_block(
    tmp_path: Path,
) -> None:
    # A read that fails isn't HDF5's to see either: it gets zeros, and the error is
    # raised as the block is left.
    out, full = on_full(tmp_path / "f", room=100)
    read = []
    with pytest.raises(OSError) as raised, out:
        out.write(b"abc")
        full.broken = True
        out.seek(0)
        read.append(out.read(3))
    assert read == [b"\0\0\0"] and raised.value.errno == errno.EIO


@limits_file_sizes
def test_a_failed_write_leaves_nothing_held_in_memory(tmp_path: Path) -> None:
    # A program that writes many files, and goes on past one that fails, must get its
    # memory back: neither the error nor the file it kept may keep the dataset, or what
    # was written after the failure, once the program lets go of them. A limit on the
    # size of each file fails the write as a full disk does.
    code = textwrap.dedent(
        """
        import gc, resource, weakref
        import numpy as np, xarray as xr
        from waveform_compression_tools.destinations import write_netcdf

        wave = np.ones((3, 100, 1000), np.float32)
        ds = xr.Dataset({"waveform": (("component", "station", "time"), wave)})
        watched = weakref.ref(wave)
        del wave
        hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
        resource.setrlimit(resource.RLIMIT_FSIZE, (100_000, hard))
        try:
            write_netcdf(ds, "out.nc")
        except OSError as error:
            print(error.errno)
        del ds
        gc.collect()
        print(watched() is None)
        """
    )
    ran = run_python(code, cwd=tmp_path)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [str(errno.EFBIG), "True"]


@limits_file_sizes
def test_a_failed_write_of_a_big_variable_holds_no_copy_of_it(tmp_path: Path) -> None:
    # From a failed write on, what HDF5 writes is held in memory, for its reads to find.
    # xarray writes each variable of a netCDF file in one go, as decompress and
    # convert-bbbin have it do, and HDF5 reads none of that back, so a write that big
    # isn't copied: a run that fails on a full disk mustn't need twice the memory that
    # it would to succeed. A limit on the size of each file fails the write as a full
    # disk does; tracemalloc follows numpy's memory and Python's.
    code = textwrap.dedent(
        """
        import resource, tracemalloc
        import numpy as np, xarray as xr
        from waveform_compression_tools.destinations import write_netcdf

        wave = np.ones((3, 1000, 5600), np.float32)  # 64 MiB, written in one go.
        ds = xr.Dataset({"waveform": (("component", "station", "time"), wave)})
        hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, hard))
        tracemalloc.start()
        try:
            write_netcdf(ds, "out.nc")
        except OSError as error:
            print(error.errno)
        print(tracemalloc.get_traced_memory()[1] < wave.nbytes // 2)
        """
    )
    ran = run_python(code, cwd=tmp_path)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [str(errno.EFBIG), "True"]


@pytest.mark.parametrize(
    ("number", "writer", "stopped"),
    [
        ("SIGINT", "compress", "KeyboardInterrupt"),
        ("SIGTERM", "write_netcdf", "SystemExit"),
    ],
)
def test_a_signal_as_hdf5_closes_the_file_waits_then_stops_the_run(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    number: str,
    writer: str,
    stopped: str,
) -> None:
    # A signal whose handler is in Python, as Ctrl-C's is, and SIGTERM's when a job
    # cleans up, raises in whatever Python code is running; when HDF5 writes, that's the
    # code it writes through. An error there is a failed write to HDF5, and one as it
    # closes the file leaves HDF5 2.0.0 to crash Python. So the signal waits until HDF5
    # is done with its call, then stops the run as its handler does: nothing is left at
    # DST, HDF5 carries on, and the handler is the signal's own again. Here the signal
    # comes with the last write, as HDF5 closes the file, counted in a run before.
    write(standard_dataset(5, nt=100), tmp_path / "in.nc")
    code = textwrap.dedent(
        """
        import signal, sys
        import h5py, numpy as np, xarray as xr
        from waveform_compression_tools import files
        from waveform_compression_tools.destinations import HDF5Output, write_netcdf

        number = getattr(signal, sys.argv[1])
        if number == signal.SIGTERM:  # As a job's own cleanup might.
            signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
        handler = signal.getsignal(number)

        def run(dst):
            if sys.argv[2] == "compress":
                files.compress_file("in.nc", dst)
            else:
                with xr.open_dataset("in.nc", engine="h5netcdf") as ds:
                    write_netcdf(ds.load(), dst)

        def counted(self, data):
            writes.append(None)
            if len(writes) == last:
                signal.raise_signal(number)
            return write(self, data)

        write, HDF5Output.write = HDF5Output.write, counted
        writes, last = [], 0
        run("counted.nc")
        writes, last = [], len(writes)
        try:
            run("out.nc")
        except (KeyboardInterrupt, SystemExit) as stop:
            print(type(stop).__name__)
        HDF5Output.write = write
        with h5py.File("again.h5", "w") as f:
            f["x"] = np.arange(10)
        print("HDF5 carries on")
        print(signal.getsignal(number) is handler)
        """
    )
    ran = run_python(code, number, writer, cwd=tmp_path)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [stopped, "HDF5 carries on", "True"]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["again.h5", "counted.nc", "in.nc"]


@pytest.mark.skipif(sys.platform == "win32", reason="a signal ends Windows' processes")
def test_a_second_ctrl_c_ends_python_at_once(tmp_path: Path) -> None:
    # A Ctrl-C waits while a file is written, which for decompress is until the whole
    # file is. Pressed again, it ends Python at once, as it does by default, and leaves
    # the partial file, which the next run replaces.
    code = textwrap.dedent(
        """
        import signal
        from waveform_compression_tools.destinations import HDF5Output

        with HDF5Output("out.nc.partial") as out:
            out.write(b"part of a file")
            signal.raise_signal(signal.SIGINT)
            print("waits", flush=True)
            signal.raise_signal(signal.SIGINT)
            print("still running")
        """
    )
    ran = run_python(code, cwd=tmp_path)
    assert ran.returncode == -signal.SIGINT, ran.stderr
    assert ran.stdout == "waits\n"
    assert (tmp_path / "out.nc.partial").read_bytes() == b"part of a file"


def test_a_file_written_off_the_main_thread_leaves_ctrl_c_alone(
    tmp_path: Path, standard_dataset: StandardDataset
) -> None:
    # Only the main thread may say what Ctrl-C does, and no Ctrl-C stops another
    # thread's code, so off it nothing waits, and the file is written as ever.
    ds = standard_dataset(3, nt=50)
    handler = signal.getsignal(signal.SIGINT)
    with ThreadPoolExecutor(1) as pool:
        pool.submit(write_netcdf, ds, tmp_path / "out.nc").result()
    assert signal.getsignal(signal.SIGINT) is handler
    with xr.open_dataset(tmp_path / "out.nc", engine="h5netcdf") as written:
        xr.testing.assert_identical(written, ds)
