#!/usr/bin/env python3
"""Read the SCL-off window GPKGs without geopandas.

Each ``PAIR_<N>_scl_lost.gpkg`` records the 224x224 windows the original
chipper visited and rejected on the SCL gate. A GeoPackage is a SQLite
database, and every feature stores its own bounding box in the GeoPackage
binary header, so the bounds can be read directly -- no geometry decode, and no
dependency on geopandas or fiona, neither of which is installed on the VM.

Two things about this data that shape how it may be used:

* ``chip_id`` is the literal string ``'<NA>'`` on all 217,894 features. These
  windows were never cut, so they have no chip identity and no memmap row. The
  column is read and deliberately dropped -- carrying it would invite a join
  against the chip corpus that can only produce nonsense.
* ``verdict`` separates the two clauses of the original gate: ``cloud15`` for
  the >15% cloud/shadow/cirrus rule, ``scl0`` for a window containing a pixel
  of genuinely absent data. A new *cloud* mask cannot redeem absent data, so
  ``scl0`` windows are excluded from rescue by the caller.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pandas as pd

LAYER = "scl_lost"

# GeoPackage binary header: magic "GP", version, flags, srs_id, then the
# envelope. Bit 0 of flags is byte order; bits 1-3 are the envelope indicator.
_HEADER_PREFIX = 8
_ENVELOPE_DOUBLES = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}

COLUMNS = ["pair_name", "row", "col", "verdict", "reason",
           "scl_passed", "has_data", "in_buffer"]


def envelope_bounds(blob: bytes) -> tuple[float, float, float, float]:
    """(west, south, east, north) from a GeoPackage geometry blob's header."""
    if len(blob) < _HEADER_PREFIX or blob[:2] != b"GP":
        raise ValueError("not a GeoPackage geometry blob")
    flags = blob[3]
    indicator = (flags >> 1) & 0x07
    count = _ENVELOPE_DOUBLES.get(indicator)
    if not count:
        raise ValueError(
            "feature stores no envelope (indicator %d); this reader needs one. "
            "Install geopandas and parse the geometry if the writer changed."
            % indicator
        )
    order = "<" if flags & 0x01 else ">"
    values = struct.unpack_from("%s%dd" % (order, count), blob, _HEADER_PREFIX)
    # Envelope order is minx, maxx, miny, maxy.
    return values[0], values[2], values[1], values[3]


def read_scl_lost(path: Path, verdicts: set[str] | None = None) -> pd.DataFrame:
    """One row per rejected window, with its footprint.

    ``verdicts`` filters at read time -- pass ``{"cloud15"}`` to drop the
    ``scl0`` windows before they are ever scored.
    """
    with sqlite3.connect("file:%s?mode=ro" % path, uri=True) as connection:
        rows = connection.execute(
            "SELECT geom, %s FROM %s" % (", ".join(COLUMNS), LAYER)
        ).fetchall()

    records = []
    for blob, *rest in rows:
        record = dict(zip(COLUMNS, rest))
        if verdicts is not None and record["verdict"] not in verdicts:
            continue
        west, south, east, north = envelope_bounds(blob)
        record.update(bbox_w=west, bbox_s=south, bbox_e=east, bbox_n=north)
        records.append(record)

    frame = pd.DataFrame.from_records(records, columns=COLUMNS + [
        "bbox_w", "bbox_s", "bbox_e", "bbox_n"])
    for column in ("scl_passed", "has_data", "in_buffer"):
        frame[column] = frame[column].astype("boolean")
    return frame


def pair_name_from_path(path: Path) -> str:
    """PAIR_1234 from .../PAIR_1234_scl_lost.gpkg."""
    return path.name.split("_scl_lost")[0]


if __name__ == "__main__":
    import sys

    import paths

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        paths.CHIPS_SCL_LOST / "PAIR_0_scl_lost.gpkg")
    table = read_scl_lost(target)
    print("%s: %d features" % (target.name, len(table)))
    print(table["verdict"].value_counts().to_string())
    print(table.head(3).to_string(index=False))
