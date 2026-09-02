#!/usr/bin/env python3
"""Run every snippet from docs/dataset_v3/FES2022B_RECIPE.md, verbatim.

The recipe is only useful if its code actually runs. Running this caught two
errors in its first draft: an example bbox that straddled lon 0 -- raising the
very error the seam section warns about -- and a window length quoted as 201
rather than 200.

    ~/miniforge3/envs/fes/bin/python fes_recipe_selftest.py

Expects the FES atlas at /mnt/local_ssd/fes/fes2022b and the ocean-edge points
beside it. Prints what the recipe says it should print.
"""
import os, pathlib, struct, sqlite3
import numpy as np, pandas as pd, pyfes
from scipy.signal import find_peaks

FES_ROOT = pathlib.Path("/mnt/local_ssd/fes/fes2022b")
GRID_EPOCH = pd.Timestamp("2014-01-01T00:00:00", tz="UTC")

# --- snippet 1: load handlers with a bbox ---
bbox = (-6.0, 51.0, -2.0, 55.0)   # Irish Sea; wholly negative, no seam
cwd = os.getcwd(); os.chdir(FES_ROOT)
try:
    handlers = pyfes.config.load(FES_ROOT / "fes2022.yaml", bbox).models
finally:
    os.chdir(cwd)
print("handlers keys:", sorted(handlers.keys()))

# --- snippet 2: generate wide, then filter ---
sat = pd.Timestamp("2020-06-15T10:23:41", tz="UTC")
step = pd.Timedelta(minutes=15)
lo = GRID_EPOCH + int(np.floor((sat - pd.Timedelta(hours=25) - GRID_EPOCH)/step))*step
hi = GRID_EPOCH + int(np.ceil((sat + pd.Timedelta(hours=25) - GRID_EPOCH)/step)+1)*step
grid = pd.date_range(lo, hi, freq="15min", inclusive="left", tz="UTC")
keep = (grid >= sat - pd.Timedelta(hours=25)) & (grid <= sat + pd.Timedelta(hours=25))
times = grid[keep]
print("window samples:", len(times), "(recipe says 201)")

# --- snippet 3: predict ---
dates = times.tz_localize(None).to_numpy(dtype="datetime64[us]")
lat, lon = 53.4, -3.5
tide, lp, _ = pyfes.evaluate_tide(handlers["tide"], dates,
                                  np.full(len(dates), lon), np.full(len(dates), lat))
pure = np.asarray(tide) + np.asarray(lp)
print("pure_tide cm: min %.2f max %.2f" % (pure.min(), pure.max()))

# --- snippet 4: labels for the .loc replica ---
labels = pd.Index(((times - GRID_EPOCH) // pd.Timedelta(minutes=15)).astype(int))
print("label range: %d .. %d" % (labels[0], labels[-1]))

# --- snippet 5: point GPKG parse ---
G = "/mnt/local_ssd/fes/ancillary/Global_Ocean_Edge_centroids.gpkg"
c = sqlite3.connect("file:%s?mode=ro" % G, uri=True)
layer = c.execute("SELECT table_name FROM gpkg_contents").fetchone()[0]
blob = c.execute('SELECT geom FROM "%s" LIMIT 1' % layer).fetchone()[0]
n = c.execute('SELECT COUNT(*) FROM "%s"' % layer).fetchone()[0]; c.close()
flags = blob[3]; envelope = (flags >> 1) & 0x07
offset = 8 + {0:0,1:32,2:48,3:48,4:64}[envelope]
order = "<" if blob[offset] == 1 else ">"
x, y = struct.unpack_from("%s2d" % order, blob, offset + 5)
print("points: %d (recipe says 489673) | first lon %.4f lat %.4f | envelope indicator %d" % (n, x, y, envelope))

# --- snippet 6: S1 id parse ---
import re
sid = "S1A_IW_GRDH_1SDV_20181218T122733_20181218T122802_025080_02C492_121D"
m = re.search(r"_(\d{8}T\d{6})_", sid)
print("S1 id ->", pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", utc=True))
print()
print("ALL RECIPE SNIPPETS RAN")
