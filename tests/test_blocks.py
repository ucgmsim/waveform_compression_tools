import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from waveform_compression_tools import files, legacy
from waveform_compression_tools.bbbin import read_bbbin
from waveform_compression_tools.files import (
    block_stations_for,
    compress_file,
    decompress_file,
    decompress_to,
    verify_file,
)
from waveform_compression_tools.hfbin import read_hfbin

from helpers import (
    StandardDataset,
    build_standard_dataset,
    write,
    write_bbbin,
    write_hfbin,
)


def test_block_sizes() -> None:
    assert block_stations_for(3, 39_629, None) == 282
    assert block_stations_for(3, 39_629, 322) == 322  # a whole chunk; 322 <= 282 + 70
    assert block_stations_for(3, 39_629, 560) == 282  # 560 > 352, so the budget wins
    assert block_stations_for(3, 39_629, 4_096) == 282  # the chunk would break it
    assert block_stations_for(3, 400_000, None) == 27  # long traces get small blocks
    assert block_stations_for(3, 10**9, None) == 1


def test_blocks_start_and_end_on_chunk_boundaries(
    tmp_path: Path, standard_dataset: StandardDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 3 components x 64 samples x 4 bytes = 768 bytes a station: 60 stations a block.
    monkeypatch.setattr(files, "BLOCK_BYTES", 60 * 768)
    src = write(
        standard_dataset(260, nt=64),
        tmp_path / "in.nc",
        encoding={"waveform": {"chunksizes": (1, 50, 64)}},
    )
    seen: list[list[tuple[int, int]]] = []
    real_blocks = files._blocks

    def spy(n_station: int, block_stations: int) -> list[tuple[int, int]]:
        seen.append(real_blocks(n_station, block_stations))
        return seen[-1]

    monkeypatch.setattr(files, "_blocks", spy)
    compress_file(src, tmp_path / "out.nc")
    assert verify_file(src, tmp_path / "out.nc").passed
    assert len(seen) == 2
    for blocks in seen:
        assert len(blocks) > 1
        for s0, s1 in blocks:
            assert s0 % 50 == 0
            assert s1 % 50 == 0 or s1 == 260


def _peak(step: str, src: Path, dst: Path, again: Path) -> int:
    tracemalloc.start()
    try:
        if step == "compress":
            compress_file(src, again)
        elif step == "decompress":
            decompress_to(dst, again)
            again.unlink()  # A whole waveform, and the tests' folder is in memory.
        else:
            assert verify_file(src, dst).passed
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


# Some stations of 2,000 samples, four times as many stations, and four times as many
# samples. A budget of 512 KiB makes blocks of 21 stations of 2,000 samples.
SIZES = [(250, 2_000), (1_000, 2_000), (250, 8_000)]
SMALL_BLOCK_BYTES = 512 * 2**10


@pytest.fixture(scope="module")
def compressed_sizes(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[int, int], tuple[Path, Path]]:
    # Each size, written and compressed once, untraced, which also loads what loads on
    # first use.
    folder = tmp_path_factory.mktemp("sizes")
    written = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(files, "BLOCK_BYTES", SMALL_BLOCK_BYTES)
        for n_station, nt in SIZES:
            ds = build_standard_dataset(n_station, nt=nt)
            src = write(ds, folder / f"in{n_station}_{nt}.nc")
            dst = folder / f"out{n_station}_{nt}.nc"
            compress_file(src, dst)
            written[n_station, nt] = src, dst
    return written


@pytest.mark.parametrize("step", ["compress", "verify", "decompress"])
def test_peak_memory_stays_flat(
    tmp_path: Path,
    compressed_sizes: dict[tuple[int, int], tuple[Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
    step: str,
) -> None:
    # The peak must not follow the number of stations or the length of the traces.
    monkeypatch.setattr(files, "BLOCK_BYTES", SMALL_BLOCK_BYTES)
    peaks = {
        size: _peak(step, src, dst, tmp_path / f"again{size[0]}_{size[1]}.nc")
        for size, (src, dst) in compressed_sizes.items()
    }
    print(step, peaks)
    few, many, long = (peaks[size] for size in SIZES)
    assert many < 1.5 * few and few < 1.5 * many
    assert long < 1.5 * few and few < 1.5 * long


@pytest.mark.parametrize(
    ("read_legacy", "write_legacy"),
    [(read_bbbin, write_bbbin), (read_hfbin, write_hfbin)],
    ids=["BB.bin", "HF.bin"],
)
def test_a_legacy_file_is_read_a_block_of_stations_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_legacy: Callable[[Path], xr.Dataset],
    write_legacy: Callable[..., np.ndarray],
) -> None:
    # Each block of traces is copied into place as it is read, so reading the file
    # takes little more than its waveform at the peak, not the file's traces too. The
    # 3 MB of traces of 252 stations of 1,000 samples come in a budget of 64 KiB as 50
    # blocks of 5 stations and a last block of 2.
    monkeypatch.setattr(legacy, "BLOCK_BYTES", 64 * 2**10)
    path = tmp_path / "legacy.bin"
    data = write_legacy(path, 252, 1_000)
    read_legacy(path)  # Untraced, which loads what loads on first use.
    tracemalloc.start()
    try:
        waveform = read_legacy(path)["waveform"].values
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.25 * data.nbytes
    np.testing.assert_array_equal(waveform, np.moveaxis(data[:, :, [1, 0, 2]], 2, 0))


@pytest.mark.parametrize(
    "compressed", ["in one block", "from an original chunked along station"]
)
def test_a_block_reads_only_its_own_streams(
    tmp_path: Path,
    compressed_sizes: dict[tuple[int, int], tuple[Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
    compressed: str,
) -> None:
    # Decompress reads a file in blocks of its own, of 21 stations here, whatever blocks
    # it was compressed in: one block, or whole chunks of 24 stations of an original
    # chunked 8 stations at a time. compress lays out each of its blocks' streams one
    # component after another, and a block reads only its own, so neither what it reads
    # nor its peak follows the blocks the file was compressed in. The peak is the one
    # for the same file compressed in decompress's own blocks.
    monkeypatch.setattr(files, "BLOCK_BYTES", SMALL_BLOCK_BYTES)
    n_station, nt = SIZES[0]
    src, in_its_blocks = compressed_sizes[n_station, nt]
    packed = tmp_path / "out.nc"
    if compressed == "in one block":
        compress_file(src, packed, block_stations=n_station)
    else:
        chunks = {"waveform": {"chunksizes": (1, 8, nt)}}
        ds = build_standard_dataset(n_station, nt=nt)
        compress_file(write(ds, tmp_path / "in.nc", encoding=chunks), packed)
        (tmp_path / "in.nc").unlink()
    read = files._Streams.read
    unneeded: list[int] = []

    def counted(
        self: Any, s0: int, s1: int, components: None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        data, starts = read(self, s0, s1, components)
        unneeded.append(data.nbytes - int(self.nbytes[:, s0:s1].sum()))
        return data, starts

    monkeypatch.setattr(files._Streams, "read", counted)
    peak = _peak("decompress", src, packed, tmp_path / "back.nc")
    assert unneeded and not any(unneeded)
    assert peak < 1.25 * _peak("decompress", src, in_its_blocks, tmp_path / "back.nc")
    # And what it reads is each stream's own: the file decodes to what the same file
    # compressed in decompress's own blocks does, which a block reads at once.
    back = decompress_file(packed)["waveform"].values
    assert np.array_equal(back, decompress_file(in_its_blocks)["waveform"].values)
