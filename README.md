# waveform_compression_tools

Stores a float32 waveform as FLAC, one mono stream per trace (one component at one station), inside an ordinary netCDF file. Every sample comes back within one quantisation step of the original, and the traces of real stations come back exactly. This note describes the file well enough to decode it with any FLAC tool, even without this code. It then covers the setting, the commands and how to move the package.

## The file

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
| `waveform_attrs` | the original waveform's attributes, as a JSON string |
| `waveform_encoding` | optional. A JSON string of the waveform's `_FillValue` and `missing_value`, as the original had them; a value that isn't finite is the string `"NaN"`, `"Infinity"` or `"-Infinity"`. An original with no `_FillValue` has `{"_FillValue": null}`: write the waveform without one. A reader without it gets xarray's default, a NaN `_FillValue`. |
| `waveform_attr_types` | optional. A JSON string mapping each numpy-typed attribute of `waveform_attrs` to its dtype, such as `"float32"` or `"int32"`. A reader without it gets the JSON's own types: float64 and int64. |
| `bits` | the setting, an integer from 8 to 32 (see below) |
| `encoder` | `"waveform_compression_tools"` and the version that wrote the file |
| `exact_rule` | which stations are real, in words (see below) |

## Decoding with any FLAC tool

To rebuild the trace of component `c` at station `s`:

1. Read its stream: the bytes `compressed[start : start + size]`, with `start = stream_starts[c, s]` and `size = stream_bytes[c, s]`. That is a complete FLAC file of one channel, with 32 bits per sample at every setting, in frames of 4096 samples. Its sample-rate field says 44100, which means nothing here. It was written in one pass, so its header leaves the number of samples (`nt`) and the MD5 checksum unset.
2. Decode it to signed 32-bit integers, `q`, with a decoder that handles 32 bits per sample. libFLAC and the `flac` tool do from version 1.4 (2022) on, and older versions may not. ffmpeg does too: `ffmpeg -i trace.flac -f s32le trace.raw` writes little-endian integers. In Python, `decode_stream` in `waveform_compression_tools.stream` decodes a stream's bytes.
3. The trace is `float32(q / gain + offset)`, with `gain = stream_gains[c, s]` and `offset = stream_offsets[c, s]`, the division and the addition done in float64.
4. If `s` is in `exact_stations`, at position `k`, the trace is `exact_waveform[c, k]` instead, exactly as it was. Its stream decodes to within one step of the same trace, so a reader that ignores the exact copies still gets a valid waveform.

```python
import h5py
import numpy as np

with h5py.File("realisation.nc", "r") as f:
    g = f["_flac_compressed_waveform"]
    c, s = 0, 5  # the first component (000, or x) of the station at index 5
    start, size = g["stream_starts"][c, s], g["stream_bytes"][c, s]
    stream = g["compressed"][start : start + size].tobytes()
    gain, offset = g["stream_gains"][c, s], g["stream_offsets"][c, s]
q = decode_flac(stream)  # any FLAC decoder, to int32 samples
trace = (q.astype(np.float64) / gain + offset).astype(np.float32)
```

## The setting

`bits` is how finely each trace is quantised, from 8 to 32, and 32 by default. A trace's gain is `full_scale / peak`, where `peak` is its largest absolute sample and `full_scale = 2**(bits - 1) - 1`, in float64; a trace of zeros has a gain of 1. The samples are the trace times its gain, rounded to the nearest integer. The streams hold 32-bit integers at every setting: a lower setting only makes the integers smaller, and the streams with them.

Each decoded sample is within one quantisation step of the original, which is `1 / gain = peak / full_scale`. So the largest error of a trace is a fraction `1 / full_scale` of that trace's peak, with a relative slack of 1e-9 allowed for rounding in float64. A trace of zeros comes back as exact zeros, and a NaN or an infinity in a waveform is refused when compressing.

| `bits` | `full_scale` | largest error, as a fraction of the trace's peak |
|---|---|---|
| 8 | 127 | 7.9e-3 |
| 16 | 32,767 | 3.1e-5 |
| 24 (FLAC-24) | 8,388,607 | 1.2e-7 |
| 32 (FLAC-32, the default) | 2,147,483,647 | 4.7e-10 |

The error is relative to the trace's peak, not to each sample, so a sample far below the peak has a larger relative error. The FLAC compression level (0 to 8, 5 by default) changes only the size of the streams, and the file doesn't record it.

## Real stations stay exact

A station whose name is not exactly 7 characters long is real: seismic networks' station codes, such as `KIKS` or `OPZ`, are 3 or 4 characters, and the points of a synthetic grid have codes of 7. Whitespace and NULs at either end of a name don't count. For each real station the group holds an exact float32 copy of its traces, which decompressing puts back, so real stations come back bit for bit.

A file whose stations are all real isn't compressed, so that it stays exact as it is: `compress_file` raises `AllStationsReal` and the command exits with status 1, writing nothing. A file without a `station` coordinate can't be compressed either, since nothing names its stations, and nor can a file with no stations at all.

## The commands

The package installs the script `waveform-compression-tools`, which runs `cli.py`.

```bash
waveform-compression-tools compress SRC DST [--bits 32] [--level 5]   # netCDF file -> compressed file
waveform-compression-tools decompress SRC DST                         # compressed file -> netCDF file
waveform-compression-tools verify ORIGINAL COMPRESSED                 # exits 1 if the check fails
waveform-compression-tools convert-bbbin SRC DST                      # legacy BB.bin -> netCDF file
```

- `compress` reads, encodes and writes a block of 344 stations at a time, so its memory is a few times one block, however big the file is.
- `decompress` holds the whole waveform in memory, so it suits files that fit there: it needs 4 bytes a sample, plus a block of up to 344 stations and some working space. Python's `decompress_file` does the same.
- `verify` decodes the compressed file a block at a time and exits with status 0 only if every trace decodes within the bound its setting promises and within one step of the original, the real stations come back bit for bit, the file holds an exact copy of every real station and of no other, and everything else in the file, the waveform's attributes included, is identical to the original's.
- A command that can't do its work prints a message and exits with status 1, leaving nothing at its destination: a file appears there only when it is complete (it is written to the destination with `.partial` added, then renamed). Arguments that Typer refuses, such as a file that doesn't exist or a `--level` outside 0 to 8, exit with status 2.

The same functions are in Python, in `waveform_compression_tools.files` (`compress_file`, `decompress_file`, `verify_file` and `is_compressed`) and in `waveform_compression_tools.bbbin` (`read_bbbin` and `convert_bbbin`). A file must be in the standard layout: a float32 variable `waveform` with exactly the dimensions (component, station, time) and any component names (a `component` coordinate is optional). `compress_file` also refuses a file that holds an HDF5 group besides its root, which the compressed file would go without.

## Moving the package

To use the code elsewhere, copy `waveform_compression_tools/` and `tests/`, and add `pyflac==3.0.0`, `h5netcdf` and `h5py` to the dependencies. It also needs numpy and xarray, and typer for `cli.py`. Its own imports are relative, so it works wherever it sits; the tests import it as `waveform_compression_tools`, so change that prefix in them if it is no longer a top-level package. In the new `pyproject.toml`, point a `waveform-compression-tools` script at the `app` in `cli.py`.

pyflac is pinned because `stream.py` decodes through pyflac 3.0.0's internals: run the tests again before moving the pin. A few tests read real traces from the folder named by `$WAVEFORM_COMPRESSION_KIT` and skip when it isn't there.
