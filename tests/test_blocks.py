import tracemalloc
from pathlib import Path

import pytest

from waveform_compression_tools import files
from waveform_compression_tools.files import (
    block_stations_for,
    compress_file,
    verify_file,
)

from helpers import StandardDataset


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
    src = tmp_path / "in.nc"
    standard_dataset(260, nt=64).to_netcdf(
        src, engine="h5netcdf", encoding={"waveform": {"chunksizes": (1, 50, 64)}}
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


@pytest.mark.parametrize("step", ["compress", "verify"])
def test_peak_memory_stays_flat(
    tmp_path: Path,
    standard_dataset: StandardDataset,
    monkeypatch: pytest.MonkeyPatch,
    step: str,
) -> None:
    # A budget of 2 MiB makes blocks of 87 stations of 2,000 samples. The peak must
    # not follow the number of stations or the length of the traces.
    monkeypatch.setattr(files, "BLOCK_BYTES", 2 * 2**20)
    peaks = {}
    for n_station, nt in [(1_000, 2_000), (4_000, 2_000), (1_000, 8_000)]:
        src, dst = tmp_path / f"in{n_station}_{nt}.nc", tmp_path / "out.nc"
        standard_dataset(n_station, nt=nt).to_netcdf(src, engine="h5netcdf")
        dst.unlink(missing_ok=True)
        compress_file(src, dst)  # Also loads what loads on first use.
        again = tmp_path / "again.nc"
        again.unlink(missing_ok=True)
        peaks[n_station, nt] = _peak(step, src, dst, again)
    print(step, peaks)
    assert peaks[4_000, 2_000] < 1.5 * peaks[1_000, 2_000]
    assert peaks[1_000, 2_000] < 1.5 * peaks[4_000, 2_000]
    assert peaks[1_000, 8_000] < 1.5 * peaks[1_000, 2_000]
    assert peaks[1_000, 2_000] < 1.5 * peaks[1_000, 8_000]
