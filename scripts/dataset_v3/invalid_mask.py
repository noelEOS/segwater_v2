#!/usr/bin/env python3
"""Measure the invalid fraction of a chip footprint against the new valid mask.

The new lineage's valid mask is the Level 2 label: pixel value 255 is invalid,
0 is land and 1 is water. This module is the one place that turns a lon/lat box
into an invalid fraction, so a chip already in the corpus and a window being
considered for rescue are measured by identical code.

Windows are located by **geometry**, never by pixel indices. The label rasters
are cropped per pair ("CHIP-EXT"), and the offset between the original chipping
grid and the label grid differs from pair to pair -- PAIR_0 is (224, 224),
PAIR_1001 is (7840, 224), PAIR_143 is (3584, 896). Any code that computes a
window as ``row * 224`` is wrong by thousands of pixels for most pairs.
``from_bounds`` is correct for every pair, so it is the only route offered here.
"""

from __future__ import annotations

import re

import numpy as np
from rasterio.windows import from_bounds

# A chip is 224x224 on the S2 10 m lattice.
CHIP_PX = 224
CHIP_PIXELS = CHIP_PX * CHIP_PX

# Label encoding of the Level 2 rasters: 0=land, 1=water, 255=invalid.
INVALID_VALUE = 255

# Names the rule these numbers came from, so a table can always be traced back
# to the mask that produced it.
MASK_RULE = "l2_label_255"

# The old per-window invalid percentage is only recorded as free text in the
# GPKG `reason` column, e.g. "... threshold (17.25%)."
RE_OLD_PCT = re.compile(r"\(([\d.]+)%\)")

# A window is expected to land exactly on the pixel lattice. Verified zero for
# both populations; anything above this means the grid moved under us.
MAX_SUBPIXEL_RESIDUAL = 1e-6


def parse_old_percent(reason: str | None) -> float:
    """The invalid fraction the original gate measured, from the reason text.

    Returns NaN when the text carries no percentage, which the caller should
    treat as a parse failure rather than a zero.
    """
    if not reason:
        return float("nan")
    match = RE_OLD_PCT.search(reason)
    return float(match.group(1)) / 100.0 if match else float("nan")


def window_residual(window) -> float:
    """How far a window is from sitting exactly on the pixel lattice.

    Combines the offset fraction and the departure from a 224x224 size, so a
    single number can be asserted against.
    """
    return max(
        abs(window.col_off - round(window.col_off)),
        abs(window.row_off - round(window.row_off)),
        abs(window.width - CHIP_PX),
        abs(window.height - CHIP_PX),
    )


def invalid_fractions(src, boxes: np.ndarray) -> dict[str, np.ndarray]:
    """Invalid fraction for each (west, south, east, north) box.

    ``src`` must be an already-open rasterio dataset -- opening is the caller's
    job, because the caller knows to open each label once per pair rather than
    once per chip.

    Reads are ``boundless`` with ``fill_value=255``, so a footprint that
    overlaps the raster only partly counts the missing part as invalid. That is
    conservative in **both** directions, which is the point: for a rescue
    candidate it suppresses the rescue, and for a chip already in the corpus it
    argues for removal. Either way we never keep a chip whose label we could not
    fully read.

    A window that misses the raster **entirely** is a different thing, and is
    reported as such. 6.88% of the rescue-eligible windows fall outside their
    pair's label, because the label is cropped to the chip extent and never
    reached them. Those get ``frac = NaN`` and ``covered = False``, never a
    fraction of 1.0 -- scoring them as fully invalid would turn "we cannot
    measure this" into "we measured it and it failed".

    Returns arrays parallel to ``boxes``: ``frac``, ``npix`` (real pixels read),
    ``boundless`` (the read was partly filled), ``covered`` (the window overlaps
    the raster at all), plus the scalar ``max_residual``.
    """
    n = len(boxes)
    frac = np.full(n, np.nan, dtype=np.float32)
    npix = np.zeros(n, dtype=np.int32)
    boundless = np.zeros(n, dtype=bool)
    covered = np.ones(n, dtype=bool)
    max_residual = 0.0

    height, width = src.height, src.width
    for i, (west, south, east, north) in enumerate(boxes):
        window = from_bounds(west, south, east, north, transform=src.transform)
        max_residual = max(max_residual, window_residual(window))

        # Round to the lattice only after measuring the residual, so a genuinely
        # off-grid window is reported rather than quietly snapped into place.
        window = window.round_offsets().round_lengths()

        # How much of the window actually sits on the raster?
        left, top = max(0, window.col_off), max(0, window.row_off)
        right = min(width, window.col_off + window.width)
        bottom = min(height, window.row_off + window.height)
        inside = max(0, right - left) * max(0, bottom - top)

        if inside == 0:
            covered[i] = False
            continue

        boundless[i] = inside < window.width * window.height
        data = src.read(1, window=window, boundless=True, fill_value=INVALID_VALUE)
        if data.size == 0:
            covered[i] = False
            continue
        frac[i] = float((data == INVALID_VALUE).mean())
        npix[i] = int(data.size)

    return {
        "frac": frac,
        "npix": npix,
        "boundless": boundless,
        "covered": covered,
        "max_residual": max_residual,
    }


def sort_key_for_locality(src, boxes: np.ndarray) -> np.ndarray:
    """Order boxes so successive reads hit the same raster tiles.

    The labels are tiled 512x512 and a 224x224 window straddles up to four
    tiles, so reading in raster order lets GDAL's block cache serve most of the
    overlap instead of decompressing the same tile repeatedly.
    """
    inverse = ~src.transform
    cols = np.empty(len(boxes))
    rows = np.empty(len(boxes))
    for i, (west, _south, _east, north) in enumerate(boxes):
        cols[i], rows[i] = inverse * (west, north)
    return np.lexsort((cols, rows))
