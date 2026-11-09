# Changelog

## 1.1.0

The file format is unchanged, at version 1: 1.0.0 reads the files that 1.1.0 writes, and 1.1.0 reads 1.0.0's.

- **A small file is made smaller too.** 1.0.0 kept the streams in chunks of 4 MiB, or of the waveform's size if that was smaller, and HDF5 keeps every chunk whole, however little of it is filled. So a file whose waveform was under 4 MiB came out no smaller, and a bigger one held up to 4 MiB more than its streams needed. Now the streams of a file compressed in one block, which holds up to about 128 MiB of waveform, are kept in exactly their size: the 2.6 MiB of LF traces in `tests/real_traces` come out 9.3× smaller at 24 bits, where 1.0.0 left them a little larger. A file compressed in more blocks keeps its streams in chunks of 1 MiB, or of the first block's streams if those are smaller, and so holds less than 1 MiB more than they need.
- **The record names the FLAC level,** as `level`, after `bits`, so `record_version` is 2.

## 1.0.0

The first release: the FLAC codec for float32 waveforms in netCDF files, as a package of its own. The file format is unchanged: the group `_flac_compressed_waveform`, at format version 1.

- **A package of its own,** `waveform_compression_tools`, installed from a release tag, with the command `waveform-compression-tools`.
- **The waveform's encoding is kept:** its `_FillValue` and `missing_value`, in the new group attribute `waveform_encoding`, and its attributes' types, in `waveform_attr_types`.
- **The waveform's attributes are strict JSON:** in `waveform_attrs`, a float that isn't finite is the string `"NaN"`, `"Infinity"` or `"-Infinity"`, which its type in `waveform_attr_types` makes a number again, so any JSON parser reads them.
- **Blocks follow a memory budget,** `BLOCK_BYTES` (128 MiB), in whole station chunks of the file unless those would exceed the budget by more than a quarter.
- **compress checks its output in the same pass,** against the samples it read, and then the closed file against its original. `--no-check` or `check=False` leaves the check out.
- **Refusals and failures have their own types:** `Refused` and its subclasses for an input that can't be used as it is, and `CheckFailed` for an output that fails the check. Neither leaves a new file at the destination; a file already there stays as it was.
- **A content hash and a per-file record:** `compress --record PATH` writes a JSON record of the run, with the input's and the output's content hashes.
- **The command line's exit codes are published:** 0 when it did its work, 1 when the run failed, 2 for a usage error and 3 when an input is refused. A failure or a refusal is one line that names the file. A file already at a destination is replaced only with `--overwrite`.
- **A full disk or quota fails the run with its own reason,** naming the output, and `compress` stops at the end of the block it was writing. HDF5 writes each output through `destinations.HDF5Output`, a file object that never tells it a write failed: a workaround for an HDF5 bug, whose own error would hide the system's, and after which HDF5 2.0.0 to 2.2.0 can crash Python. While a file is written, Ctrl-C, and SIGTERM where Python handles it, wait until they are safe to act on: until the end of the block for `compress`, and until the file is written for `decompress` and `convert-bbbin`. Sent again, either ends the run at once.
- **Memory:** on a synthetic 36.4 GiB file of random walks, 82,000 stations of 39,629 samples, compress at 24 bits peaked at 0.74 GB of resident memory, and made the file 1.8× smaller in 6 minutes. verify peaked at 0.76 GB.
