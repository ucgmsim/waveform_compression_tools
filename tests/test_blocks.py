import tracemalloc
from pathlib import Path

import pytest

from waveform_compression_tools import files
from waveform_compression_tools.files import (
    block_stations_for,
    compress_file,
    verify_file,
)

from helpers import StandardDataset, build_standard_dataset, write


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


@pytest.mark.parametrize("step", ["compress", "verify"])
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
