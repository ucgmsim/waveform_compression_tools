# Changelog

## 1.0.0

The first release: the FLAC codec for float32 waveforms in netCDF files, as a package of its own. The file format is unchanged: the group `_flac_compressed_waveform`, at format version 1.

- **A package of its own,** `waveform_compression_tools`, installed from a release tag, with the command `waveform-compression-tools`.
- **The waveform's encoding is kept:** its `_FillValue` and `missing_value`, in the new group attribute `waveform_encoding`, and its attributes' types, in `waveform_attr_types`. Files written before 1.0.0 don't have them: they decompress with a NaN fill value, and their attributes in JSON's types.
- **Blocks follow a memory budget,** `BLOCK_BYTES` (128 MiB), in whole station chunks of the file unless those would exceed the budget by more than a quarter.
- **compress checks its output in the same pass,** against the samples it read, and then the closed file against its original. `--no-check` or `check=False` leaves the check out.
- **Refusals and failures have their own types:** `Refused` and its subclasses for an input that can't be used as it is, and `CheckFailed` for an output that fails the check. Neither leaves anything at the destination.
- **A content hash and a per-file record:** `compress --record PATH` writes a JSON record of the run, with the input's and the output's content hashes.
- **The command line's exit codes are published:** 0 when it did its work, 1 when the run failed, 2 for a usage error and 3 when an input is refused. A failure or a refusal is one line that names the file. A file already at a destination is replaced only with `--overwrite`.
- **Memory:** on a synthetic 36.4 GiB file of random walks, 82,000 stations of 39,629 samples, compress at 24 bits peaked at 0.74 GB of resident memory, and made the file 1.8× smaller in 6 minutes. verify peaked at 0.76 GB.
