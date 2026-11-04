# waveform_compression_tools

Compresses the float32 seismic waveforms in netCDF files to FLAC, one stream per trace (one component at one station), and checks what it writes. Every sample comes back within one quantisation step of the original, at a setting from 8 to 32 bits, and the traces of real stations come back exactly. Everything else in the file is kept as it was. A compressed file is still an ordinary netCDF-4 file, and its streams decode with any FLAC tool that handles 32-bit samples.

## What it is

A Python package, `waveform_compression_tools`, and a command, `waveform-compression-tools`, with four subcommands:

- `compress` writes a compressed copy of a netCDF file, and checks it as it is written;
- `decompress` turns a compressed file back into an ordinary netCDF file;
- `verify` checks a compressed file against its original;
- `convert-bbbin` turns a BB.bin file, a legacy binary format of broadband seismograms, into a netCDF file that `compress` takes.

An input has to be in the standard layout: a float32 data variable `waveform` with exactly the dimensions (component, station, time), a `station` coordinate that names the stations, and no HDF5 group besides the root. The components can be named anything, such as `x`, `y`, `z` or `000`, `090`, `ver`.

`compress` and `verify` work a block of stations at a time, so their memory doesn't grow with the file. On a synthetic file of 36.4 GiB, `compress` peaked at 0.74 GB of resident memory, and `verify` at 0.76 GB.

### Choosing a setting

The setting, `--bits` on the command line or `bits` in Python, is from 8 to 32, and 32 by default. A trace's gain is `full_scale / peak`, where `peak` is its largest absolute sample and `full_scale = 2**(bits - 1) - 1`, in float64; a trace of zeros has a gain of 1. Each sample is multiplied by the gain and rounded to the nearest integer. One quantisation step is `1 / gain`, which is `peak / full_scale`, so every decoded sample is within `1 / full_scale` of its trace's peak, with a relative slack of 1e-9 for rounding in float64. A trace of zeros comes back as exact zeros.

| `bits` | One step, as a fraction of the trace's peak | Random walks: N× smaller |
|---|---|---|
| 32, the default | 4.7e-10 | 1.25× |
| 24 | 1.2e-7 | 1.8× |
| 16 | 3.1e-5 | 3.4× |
| 8 | 7.9e-3 | 16× |

- At 24 bits, a step is about the spacing of float32 values at the trace's peak. At 32 bits it is 256 times finer.
- The bound is relative to the trace's peak, not to each sample, so a sample far below the peak has a larger relative error.
- A lower setting makes smaller files: the streams hold 32-bit integers at every setting, and a lower setting makes the integers smaller.
- How much smaller depends on the waveforms. The factors above are for synthetic random walks like the tests' (300 stations of 39,629 samples, each trace the cumulative sum of unit normal steps), in which every step is new noise; on them, each bit fewer took about one bit off each sample in the streams. To choose, compress a typical file at the settings you are weighing, and compare the factors that `compress` prints.

`--level`, from 0 (fastest) to 8 (smallest) and 5 by default, is FLAC's compression level. It changes only the size of the streams and the time taken, never the samples, and the file doesn't record it.

### Real stations stay exact

A station whose name isn't exactly 7 characters long is taken to be real: seismic networks' station codes, such as `ABC` or `WXYZ`, have 3 or 4 characters, and the points of a synthetic grid have codes of 7. Whitespace and NULs at either end of a name don't count. A compressed file also holds an exact float32 copy of each real station's traces, which decompressing puts back, so real stations come back bit for bit at any setting.

A file whose stations are all real isn't compressed, so that it stays exact as it is: it is refused, and nothing is written.

## Installing

There is no PyPI release. Install from a release tag, with pip:

```bash
pip install "waveform_compression_tools @ git+https://github.com/ucgmsim/waveform_compression_tools@v1.0.0"
```

or with uv:

```bash
uv pip install "waveform_compression_tools @ git+https://github.com/ucgmsim/waveform_compression_tools@v1.0.0"
uv add git+https://github.com/ucgmsim/waveform_compression_tools --tag v1.0.0  # as a dependency of a uv project
```

It needs Python 3.11 or newer, and git, to fetch the tag. pyflac 3.0.0, which the package pins, is published only as source, so installing it compiles two small extension modules: the machine needs a C compiler and Python's headers. pyflac brings its own libFLAC, so no FLAC library needs to be installed.

In a container, the image's build runs the same pip line, so the build needs git, a C compiler and Python's headers. Once installed, the tool downloads nothing when it runs.

## The command line

```text
waveform-compression-tools compress SRC DST [--bits 32] [--level 5] [--no-check] [--record PATH] [--overwrite]
waveform-compression-tools decompress SRC DST [--overwrite]
waveform-compression-tools verify ORIGINAL COMPRESSED
waveform-compression-tools convert-bbbin SRC DST [--overwrite]
```

Each command handles one file, and its `--help` says more.

### compress

```bash
waveform-compression-tools compress realisation.nc realisation.flac.nc --bits 24 --record realisation.json
```

| Option | Default | What it does |
|---|---|---|
| `--bits N` | 32 | the setting, from 8 to 32 |
| `--level N` | 5 | FLAC's compression level, from 0 (fastest) to 8 (smallest) |
| `--no-check` | | leaves out the check |
| `--record PATH` | | writes a JSON record of the run to PATH, whatever its outcome |
| `--overwrite` | | replaces a file already at DST, or at the record's PATH |

`compress` works in this order. A problem at any step ends the run there, and the record still says why, except in the cases listed under [The record](#the-record):

1. With `--record`, a file already at PATH is refused, unless `--overwrite` is given. This refusal writes no record, so the earlier one stays as it was.
2. The refusals that need no samples: DST is SRC, a file is at DST already, or SRC isn't an HDF5 file, is compressed already, holds an HDF5 group besides its root, isn't in the standard layout, or has only real stations. Only metadata and station names are read.
3. DST's folder, and the record's, have to exist.
4. SRC is hashed.
5. SRC is read, encoded, written and checked, a block of stations at a time. Two refusals need the samples, so they come here: a trace that holds a NaN or an infinity, and a finite fill value within one step of a trace's range.
6. DST is hashed.
7. The record is written.

On success it prints what it did, such as:

```text
compressed realisation.nc to realisation.flac.nc at 24 bits: 246000 traces, 41 real stations also kept exactly
36.4 GiB -> 19.7 GiB, 1.8x smaller (read 23.1 s, encode 150.8 s, write 15.0 s, check 139.3 s)
```

#### Blocks and memory

A block holds about `BLOCK_BYTES` (128 MiB) of float32 samples, so a block has fewer stations when the traces are longer. For a chunked waveform, a block is a whole number of the file's station chunks, so that no chunk is read twice, unless whole chunks would exceed the budget by more than a quarter; then the budget wins, and chunks are read more than once. Memory is a few blocks, however many stations the file has.

#### The check

`compress` checks its output by default, against the samples it has just read, so the original is read only once:

- As soon as a block is written, its streams are read back from the file and decoded with the gains recorded for them. Every sample has to be within one step of the original, and every trace within the setting's bound of its peak; a trace of zeros has to come back as exact zeros, and each exact copy bit for bit. A CRC-32 is taken of each stream, from the bytes decoded, and of each exact copy.
- When every block is written, the closed file is reopened. It has to hold just what was written, every stream and every exact copy with the CRC-32 it had when it was judged. It also has to agree with the original in everything but the waveform's samples, as `verify` compares them: every other variable, coordinate and attribute identical, the waveform's attributes and encoding as a reader gets them back, and exact copies of the real stations and of no others.

Only then is the file renamed to DST. A file that fails the check is deleted, and the run fails; a stream that can't be decoded fails the check too.

So `"checked": true` in the record means that DST passed the same tests as `verify`, against the samples as `compress` read them, and holds the very bytes that passed. A file that passed the check passes `verify`, unless it was changed afterwards in a way that its CRC-32s miss. The check doesn't read the original a second time; `verify` does. `--no-check` (`check=False` in Python) leaves the check out, and the record then says `"checked": false`.

#### The record

`--record PATH` writes a JSON file that says what became of the input, whether it was compressed, refused or failed. It is written last, after the output is in place and hashed. A run is complete only when its record says `"status": "ok"`.

To run a file again when its record is missing (because the run was killed, say) or doesn't say `"ok"`, give the same command with `--overwrite`. That replaces the record, and any output that a killed run left. Without `--overwrite`, a file already at PATH is refused before the input is read, and stays as it was.

No record is written for a usage error, for a refused record, when the record's own folder doesn't exist, or for an interruption. If the record can't be written, the run fails, with a line naming the record, and a compressed file stays at DST.

The record's keys, in order:

| Key | Type | Holds | Null |
|---|---|---|---|
| `record_version` | integer | 1 | never |
| `tool` | string | `"waveform_compression_tools"` | never |
| `tool_version` | string | the package's version | never |
| `status` | string | `"ok"`, `"refused"` or `"failed"` | never |
| `reason` | string | why the file was refused or the run failed: the library's whole message, which may start with the file's path, though the error line drops that start. `""` when `status` is `"ok"`. | never |
| `input.path` | string | SRC, as given | never |
| `input.bytes` | integer | SRC's size in bytes | never |
| `input.content_hash` | string | SRC's content hash (see below) | when the run stopped before hashing SRC: a refusal at step 2, or no folder for DST |
| `output` | object | `path`, `bytes` and `content_hash`, as for `input`, of DST | when `status` isn't `"ok"` |
| `bits` | integer | the setting asked for | never |
| `format_version` | integer | the format version written, 1 | never |
| `checked` | boolean | whether compression finished with its check, which passed | never; false with `--no-check`, or when compression didn't finish |
| `max_relative_error` | float | the largest error the check found, as a fraction of a trace's peak, with the exact copies in place, as `verify` measures it | with `--no-check`, or when compression didn't finish |
| `n_traces` | integer | the number of traces: components times stations | when compression didn't finish |
| `n_exact_stations` | integer | the number of real stations, kept exactly | when compression didn't finish |
| `seconds.hash` | float | wall time spent hashing SRC and DST | never |
| `seconds.compress` | float | wall time spent reading, encoding and writing: 0 if compression didn't finish | never |
| `seconds.check` | float | wall time spent checking: 0 if compression didn't finish, or with `--no-check` | never |
| `seconds.total` | float | the command's wall time, up to the record | never |
| `finished_at` | string | when the record was made, in UTC to the second, such as `"2026-11-04T06:26:37+00:00"` | never |

A content hash is computed as Dropbox's is: each 4 MiB block of the file is hashed with SHA-256, and the 32-byte digests, laid end to end, are hashed with SHA-256 again. It is written as 64 lower-case hexadecimal digits.

### decompress

```bash
waveform-compression-tools decompress realisation.flac.nc back.nc
```

Writes the netCDF file that SRC was made from: every variable, coordinate and attribute of the original, and its waveform as float32 with the dimensions (component, station, time), with its attributes in their types, and its `_FillValue` and `missing_value`. The real stations' traces are exactly as they were; every other sample is within one step of the original. The whole waveform is decoded into memory first (see [Limits in 1.0.0](#limits-in-100)).

### verify

```bash
waveform-compression-tools verify realisation.nc realisation.flac.nc
```

ORIGINAL comes first, then COMPRESSED. `verify` decodes COMPRESSED a block of stations at a time, and compares it with ORIGINAL. It exits with status 0 only if:

- every trace's stream, a real station's too, decodes within the bound its setting promises and within one step of every original sample, and a trace of zeros to exact zeros;
- with the exact copies in place, every real station comes back bit for bit, and the file holds exact copies of the real stations and of no others;
- the waveform's attributes and encoding, and everything else in the file, are identical to the original's.

A file that doesn't pass exits with status 1, with a line such as `waveform-compression-tools: realisation.flac.nc: doesn't agree with realisation.nc (largest error … of a trace's peak)`. Given in the wrong order, `verify` refuses the compressed file as an original that isn't in the standard layout.

### convert-bbbin

```bash
waveform-compression-tools convert-bbbin BB.bin bb.nc
```

A BB.bin file holds the broadband acceleration that a simulation recorded at its stations, in g: a header, a record for each station, then the traces. `convert-bbbin` writes it as a netCDF file in the standard layout, ready for `compress`: the waveform with the components `000`, `090`, `ver` in that order (the file holds 090, 000, ver), the station names, `latitude`, `longitude` and `time` coordinates, the stations' other fields as variables, and the header's values as attributes, with `source_format = "BB.bin"`. The whole BB.bin file is read into memory, which takes about twice its size at the peak.

### Exit codes

Every command exits with one of these statuses. They are published, and are the constants `EXIT_OK`, `EXIT_FAILED`, `EXIT_USAGE` and `EXIT_REFUSED` in `waveform_compression_tools.cli`.

| Status | Constant | Means |
|---|---|---|
| 0 | `EXIT_OK` | The command did its work. For `compress`, the output is in place, and checked unless `--no-check` was given; for `verify`, the file passed. |
| 1 | `EXIT_FAILED` | The run went wrong: a failed check (`compress`'s own, or a file that doesn't agree with its original in `verify`); a damaged compressed file; a compressed file whose waveform has another shape than its original's; an error from libFLAC; a file that can't be read or written, including a missing, unreadable or cut-short input and a missing destination or record folder; an output that can't be hashed; a record that can't be written; or anything unexpected. |
| 2 | `EXIT_USAGE` | Typer refused the arguments: a missing input, a folder where a file is named, `--bits` outside 8 to 32, `--level` outside 0 to 8, an unknown option, or a `--record` that is SRC or DST, or whose `.partial` is. Typer prints its own message, of a few lines. |
| 3 | `EXIT_REFUSED` | An input can't be used as it is, or a destination is taken (see below). |

The refusals, by command:

- `compress`: DST is SRC; a file is at DST, or at the record's PATH, and `--overwrite` isn't given; SRC isn't an HDF5 file, is compressed already, or holds an HDF5 group besides its root; SRC isn't in the standard layout, which includes having no `station` coordinate, a station name that is bytes but not ASCII, no stations, a waveform attribute that JSON can't hold or whose type can't be kept, a `_FillValue` or `missing_value` that isn't one value, a `_FillValue` and a `missing_value` that differ, and a finite fill value within one step of a trace's range; every station is real; a trace holds a NaN or an infinity.
- `decompress`: SRC isn't an HDF5 file, isn't compressed, or is in another format version or has a setting outside 8 to 32; DST is SRC; a file is at DST, and `--overwrite` isn't given.
- `verify`: ORIGINAL isn't an HDF5 file, or isn't in the standard layout; COMPRESSED isn't an HDF5 file, isn't compressed, or is in another format version or has a setting outside 8 to 32.
- `convert-bbbin`: SRC can't be a BB.bin file (it is shorter than a header, its size doesn't fit its header, or it holds text that isn't ASCII); DST is SRC; a file is at DST, and `--overwrite` isn't given.

A refusal writes nothing but `compress`'s record, and leaves a file already at the destination as it was. A file that isn't HDF5 at all is refused, while an HDF5 file that is damaged or cut short is a failure: it may be the right kind of file, and the run went wrong.

An interruption (Ctrl-C) is left to Typer, which exits with status 130 and prints nothing; `compress` writes no record then.

### The error line

A command that fails or refuses prints exactly one line to standard error, and nothing to standard output:

```text
waveform-compression-tools: <path>: <reason>
```

`<path>` is the file that the problem is about: the input, the output or the record, or, for an error that names its own file, that file. The reason doesn't name it again:

```text
waveform-compression-tools: x.nc: is compressed already
```

An unexpected error's reason starts with its type, such as `RuntimeError: …`. A command that succeeds prints a summary to standard output.

### Destinations, partial files and reruns

- Every command that writes a file writes it to the destination with `.partial` added, and renames it when it is complete, so a file appears at its destination complete or not at all. A run that fails deletes its partial file.
- A run that is killed, by a time limit, the out-of-memory killer or SIGKILL, may leave `DST.partial` behind. The next run replaces it, and it is safe to delete.
- A file already at DST, or at the record's PATH, is replaced only with `--overwrite` (`overwrite=True` in Python).
- With `--overwrite`, the earlier output stays until the new, checked file is renamed into place. If the run fails after that, while hashing the new file, the new file is deleted too, and nothing is left at DST. The input is untouched, so a rerun makes it again.
- A destination that is the input, by its name, through a symbolic link or as a hard link, or whose `.partial` is, is refused, with or without `--overwrite`, so a command doesn't write over its own input. Anything else at a destination's `.partial` name, a link included, is removed before the file is written, and never written through.

## Python use

```python
from pathlib import Path

from waveform_compression_tools.bbbin import convert_bbbin, read_bbbin
from waveform_compression_tools.files import compress_file, decompress_file, verify_file

src, dst = Path("realisation.nc"), Path("realisation.flac.nc")
result = compress_file(src, dst, bits=24)  # checks dst as it writes it
print(result.bytes_in / result.bytes_out, result.max_relative_error)

check = verify_file(src, dst)  # the original first
print(check.passed, check.max_relative_error)

ds = decompress_file(dst)  # an xarray Dataset, held in memory
ds.to_netcdf("back.nc", engine="h5netcdf")

bb = read_bbbin(Path("BB.bin"))  # a Dataset in the standard layout
convert_bbbin(Path("BB.bin"), Path("bb.nc"))  # the same, written to a netCDF file
```

A path can be given as a string or as a `pathlib.Path` object.

- `compress_file(src, dst, bits=32, level=5, block_stations=None, check=True, overwrite=False)` returns a `CompressResult`: `bytes_in` and `bytes_out`, `n_traces`, `n_exact` (the real stations), `bits`, `checked`, `max_relative_error` (None without the check), and the wall and CPU time spent reading, encoding, writing and checking, from `read_s` to `check_cpu_s`. It first calls `check_compressible(src, dst, overwrite)`, which makes the refusals that need no samples, and which can be called on its own.
- `verify_file(original, compressed, block_stations=None)` returns a `VerifyResult`: `passed`, `max_relative_error`, `n_exact` and `verify_s`. A file that doesn't agree gives `passed=False`; it doesn't raise.
- `decompress_file(path)` returns the original dataset, loaded into memory.
- `read_bbbin(path)` reads a BB.bin file into a dataset in the standard layout, and `convert_bbbin(src, dst, overwrite=False)` writes that dataset to a netCDF file, through a `.partial`.
- `is_compressed(path)` says whether a file holds a compressed waveform.

`block_stations` is the most stations in a block. By default it follows the memory budget and the file's chunks, as in [Blocks and memory](#blocks-and-memory).

### Errors

A refusal raises `Refused`, or one of its subclasses, from `waveform_compression_tools.errors`. It means that the input can't be used as it is, and that nothing was written: no file appears at the destination, and a file already there stays as it was. `Refused` is a `ValueError`, so catch it before `ValueError`.

| Class | Raised when |
|---|---|
| `SameFile` | the destination, or the destination with `.partial` added, is the input |
| `OutputExists` | a file is at the destination already, and `overwrite` is false |
| `NotStandardLayout` | a file isn't HDF5, or isn't in the standard layout, as listed under [Exit codes](#exit-codes) |
| `HasGroups` | a file holds an HDF5 group besides its root |
| `AlreadyCompressed` | a file is compressed already |
| `AllStationsReal` | every station of a file is real, so it stays exact as it is |
| `NonFiniteSamples` | a trace holds a NaN or an infinity |
| `NotCompressed` | a file to decompress or verify isn't HDF5, or holds no compressed waveform |
| `UnknownFormat` | a compressed file is in another format version, or has a setting outside 8 to 32 |
| `NotBBBin` | a file can't be a BB.bin file |

`CheckFailed`, a `RuntimeError`, means that the output of `compress_file` failed its check. The output is deleted, so nothing is left at the destination, and a file already there stays as it was.

Anything else is a failure of the run: `OSError` for a file that can't be read or written, `ValueError` for a bad argument (such as `bits` outside 8 to 32) or a damaged compressed file, and pyflac's exceptions when libFLAC can't encode a trace or decode a stream.

## The file format

This section describes a compressed file well enough to decode it with any FLAC tool, without this package.

A compressed file is the original netCDF-4 file without its data variable `waveform`, plus one HDF5 group, `_flac_compressed_waveform`, that holds the waveform. Every other variable, coordinate and attribute is as it was, written by xarray with the h5netcdf engine. xarray reads only the root group, so it doesn't see the waveform until `decompress_file` puts it back.

The waveform is float32 with the dimensions (component, station, time), its components in the original file's order, which its `component` coordinate names: `000`, `090`, `ver` in a converted BB.bin file, and `x`, `y`, `z` in some other files. Each (component, station) pair is a trace of `nt` samples.

| Dataset in the group | Type | Shape | Holds |
|---|---|---|---|
| `compressed` | uint8 | (bytes,) | every trace's FLAC stream, one after another. Their order isn't part of the format: use the next two datasets. |
| `stream_starts` | int64 | (component, station) | where each trace's stream starts in `compressed`. Its attribute `stream_size` is `nt`, the number of samples in every trace. |
| `stream_bytes` | int64 | (component, station) | the length of each stream in bytes |
| `stream_gains` | float64 | (component, station) | each trace's gain |
| `stream_offsets` | float64 | (component, station) | each trace's offset, always 0 |
| `exact_stations` | int64 | (real stations,) | the indices of the real stations along the station dimension, in increasing order |
| `exact_waveform` | float32 | (component, real stations, time) | their traces, exactly as they were |

| Attribute of the group | Holds |
|---|---|
| `format_version` | `1`. A reader should refuse any other. |
| `dims` | `["component", "station", "time"]` |
| `dtype` | `"float32"` |
| `waveform_attrs` | the original waveform's attributes, as a JSON string. A float that isn't finite, on its own or in a list, is the string `"NaN"`, `"Infinity"` or `"-Infinity"`, and its type in `waveform_attr_types` makes a number of it again. Files written before 1.0.0 may hold the bare tokens `NaN`, `Infinity` and `-Infinity` instead, which Python's `json` reads and stricter parsers refuse. |
| `waveform_encoding` | a JSON string of the waveform's `_FillValue` and `missing_value`, as the original had them; a value that isn't finite is the string `"NaN"`, `"Infinity"` or `"-Infinity"`. An original with no `_FillValue` has `{"_FillValue": null}`: write the waveform without one. Files written before 1.0.0 don't have it, and decompress with a NaN `_FillValue`. |
| `waveform_attr_types` | a JSON string mapping each numpy-typed attribute of `waveform_attrs` to its dtype, such as `"float32"` or `"int32"`. Files written before 1.0.0 don't have it, and their attributes come back in JSON's own types: float64 and int64. |
| `bits` | the setting, an integer from 8 to 32 (see [Choosing a setting](#choosing-a-setting)) |
| `encoder` | `"waveform_compression_tools"` and the version that wrote the file, such as `"waveform_compression_tools 1.0.0"`. Decoding doesn't use it. |
| `exact_rule` | which stations are real, in words (see [Real stations stay exact](#real-stations-stay-exact)) |

### Decoding with any FLAC tool

To rebuild the trace of component `c` at station `s`:

1. Read its stream: the bytes `compressed[start : start + size]`, with `start = stream_starts[c, s]` and `size = stream_bytes[c, s]`. That is a complete FLAC file of one channel, with 32 bits per sample at every setting, in frames of 4096 samples. Its sample-rate field says 44100, which means nothing here. It was written in one pass, so its header leaves the number of samples (`nt`) and the MD5 checksum unset.
2. Decode it to signed 32-bit integers, `q`, with a decoder that handles 32 bits per sample. libFLAC and the `flac` tool do from version 1.4 (2022) on, and older versions may not. ffmpeg does too: `ffmpeg -i trace.flac -f s32le trace.raw` writes little-endian integers. In Python, `decode_stream` in `waveform_compression_tools.stream` decodes a stream's bytes.
3. The trace is `float32(q / gain + offset)`, with `gain = stream_gains[c, s]` and `offset = stream_offsets[c, s]`, the division and the addition done in float64.
4. If `s` is in `exact_stations`, at position `k`, the trace is `exact_waveform[c, k]` instead, exactly as it was. Its stream decodes to within one step of the same trace, so a reader that ignores the exact copies still gets a valid waveform.

```python
import h5py
import numpy as np

with h5py.File("realisation.flac.nc", "r") as f:
    g = f["_flac_compressed_waveform"]
    c, s = 0, 5  # the first component (000, or x) of the station at index 5
    start, size = g["stream_starts"][c, s], g["stream_bytes"][c, s]
    stream = g["compressed"][start : start + size].tobytes()
    gain, offset = g["stream_gains"][c, s], g["stream_offsets"][c, s]
q = decode_flac(stream)  # any FLAC decoder, to int32 samples
trace = (q.astype(np.float64) / gain + offset).astype(np.float32)
```

## Limits in 1.0.0

- A file that holds an HDF5 group besides its root is refused: xarray reads only the root group, so its compressed file would go without the group.
- `decompress` and `decompress_file` load the whole waveform into memory: 4 bytes a sample, plus a block of 128 MiB and some working space. A file whose waveform doesn't fit in memory can be compressed and verified, but not yet decompressed. A later release is to decompress a block at a time.
- The streams are stored in chunks of 4 MiB, so a compressed file takes at least 4 MiB, and a file that is smaller than that comes out larger.

## Developing

```bash
uv sync --all-extras
```

These checks have to pass before a commit:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run deptry .
uvx npdlint check --extend-exclude "**/__init__.py"
```

`lefthook.yml` runs the shared CI checks as git hooks, once `uvx lefthook install` has set them up. pyflac is pinned to 3.0.0 because `stream.py` decodes through its internals: run the tests again before moving the pin. A few tests read real traces, which aren't distributed, from the folder named by `WAVEFORM_COMPRESSION_KIT`, and skip when it isn't set.
