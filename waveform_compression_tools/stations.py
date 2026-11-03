"""Tell real stations from a synthetic grid's, by the length of their names.

A simulation records its waveforms at two kinds of station. Real stations are a
seismic network's, named by its station codes, such as GeoNet's ``KIKS`` or ``OPZ``,
3 or 4 characters long. The others are the points of a synthetic grid, named by codes
7 characters long, such as ``6c5SYSL`` or ``22008d3``. So a station is taken to be
real unless its name is exactly 7 characters long.
"""

import string

# How long every name on a synthetic grid is.
_GRID_NAME_LENGTH = 7

# What padding a name may carry at either end: whitespace, and the NULs that pad a
# fixed-length string.
_PADDING = string.whitespace + "\0"

# The rule `is_real_station` applies, in words, as a compressed file records it.
REAL_STATION_RULE = "stations whose names are not exactly 7 characters long"


def is_real_station(name: str | bytes) -> bool:
    """Say whether a station is real, rather than a point of a synthetic grid.

    Parameters
    ----------
    name : str | bytes
        The station's name, as a netCDF file holds it. Bytes are decoded as ASCII.
        Whitespace and NULs are stripped from both ends, in any mix.

    Returns
    -------
    bool
        True unless the name is exactly 7 characters long.

    Raises
    ------
    UnicodeDecodeError
        If `name` is bytes that are not ASCII.
    """
    text = name.decode("ascii") if isinstance(name, bytes) else name
    return len(text.strip(_PADDING)) != _GRID_NAME_LENGTH
