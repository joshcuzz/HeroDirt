#!/usr/bin/env python3

"""
Hero Dirt Forecast
Latest SMAP surface soil-moisture ingest.

Product
-------
SPL3SMP_E Version 006
Enhanced L3 Radiometer Global Daily 9 km EASE-Grid Soil Moisture.

This script explicitly reads:

    Soil_Moisture_Retrieval_Data_AM

to avoid accidentally mixing the standard, PM, and polar grids.

SMAP is used as a regional observational constraint on
near-surface (~top few cm) soil moisture. It is NOT treated
as a 200 m observation.

Outputs
-------
data/smap/raw/
data/smap/processed/HeroDirt_SMAP_latest.npz
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import re

import numpy as np
import h5py
import earthaccess
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

GRID_FILE = (
    ROOT
    / "static"
    / "grid"
    / "HeroDirt_grid_200m.npz"
)

RAW_DIR = (
    ROOT
    / "data"
    / "smap"
    / "raw"
)

PROC_DIR = (
    ROOT
    / "data"
    / "smap"
    / "processed"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_FILE = (
    PROC_DIR
    / "HeroDirt_SMAP_latest.npz"
)


# ============================================================
# PRODUCT SETTINGS
# ============================================================

SHORT_NAME = "SPL3SMP_E"
VERSION = "006"

LOOKBACK_DAYS = 8

GROUP_NAME = (
    "Soil_Moisture_Retrieval_Data_AM"
)

BUFFER_DEG = 0.15


print()
print("============================================")
print(" HERO DIRT SMAP INGEST")
print("============================================")
print()


# ============================================================
# HERO DIRT DOMAIN -> WGS84
# ============================================================

G = np.load(
    GRID_FILE
)

west = float(G["west"])
east = float(G["east"])
south = float(G["south"])
north = float(G["north"])


transformer = Transformer.from_crs(
    "EPSG:26911",
    "EPSG:4326",
    always_xy=True,
)


lon1, lat1 = transformer.transform(
    west,
    south,
)

lon2, lat2 = transformer.transform(
    east,
    north,
)


MIN_LON = min(lon1, lon2) - BUFFER_DEG
MAX_LON = max(lon1, lon2) + BUFFER_DEG

MIN_LAT = min(lat1, lat2) - BUFFER_DEG
MAX_LAT = max(lat1, lat2) + BUFFER_DEG


print("SMAP search region:")
print(
    f"  lon: {MIN_LON:.4f} "
    f"to {MAX_LON:.4f}"
)
print(
    f"  lat: {MIN_LAT:.4f} "
    f"to {MAX_LAT:.4f}"
)
print()


# ============================================================
# EARTHDATA LOGIN
# ============================================================

print(
    "Authenticating with NASA Earthdata..."
)

earthaccess.login(
    strategy="netrc"
)

print(
    "Earthdata authentication complete."
)

print()


# ============================================================
# SEARCH RECENT GRANULES
# ============================================================

now = datetime.now(
    timezone.utc
)

start = (
    now
    -
    timedelta(
        days=LOOKBACK_DAYS
    )
)


print(
    f"Searching {SHORT_NAME} V{VERSION}"
)

print(
    f"Search window: "
    f"{start.date()} to {now.date()}"
)

print()


results = earthaccess.search_data(
    short_name=SHORT_NAME,
    version=VERSION,
    temporal=(
        start.strftime(
            "%Y-%m-%d"
        ),
        now.strftime(
            "%Y-%m-%d"
        ),
    ),
)


if not results:

    raise RuntimeError(
        "No recent SMAP granules found."
    )


print(
    f"Granules found: "
    f"{len(results)}"
)

print()


# ============================================================
# DOWNLOAD
# ============================================================

print(
    "Downloading / checking recent SMAP granules..."
)


downloaded = earthaccess.download(
    results,
    RAW_DIR,
)


downloaded = [
    Path(p)
    for p in downloaded
]


print(
    f"Available local files: "
    f"{len(downloaded)}"
)

print()


# ============================================================
# SORT NEWEST FIRST
#
# Filename contains YYYYMMDD.
# For duplicate revisions on same date, lexicographic reverse
# also places _002 ahead of _001.
# ============================================================

downloaded = sorted(
    downloaded,
    key=lambda p: p.name,
    reverse=True,
)


# ============================================================
# TIME PARSER
# ============================================================

def decode_smap_times(
    time_array,
    mask,
):

    raw = time_array[
        mask
    ]

    times = []


    for item in raw:

        if isinstance(
            item,
            bytes,
        ):

            text = item.decode(
                "utf-8",
                errors="ignore",
            )

        else:

            text = str(
                item
            )


        text = text.strip(
            "\x00 "
        )


        if (
            not text
            or
            text.startswith(
                "-9999"
            )
        ):

            continue


        try:

            # Examples generally resemble:
            # 2026-09-10T13:42:xx.xxxZ
            t = np.datetime64(
                text.replace(
                    "Z",
                    ""
                )
            )

            times.append(
                t.astype(
                    "datetime64[s]"
                )
            )

        except Exception:

            continue


    if not times:

        return (
            np.array(
                [],
                dtype="datetime64[s]",
            ),
            np.datetime64(
                "NaT",
                "s",
            ),
        )


    times = np.array(
        times,
        dtype="datetime64[s]",
    )


    # Median observation time across valid regional cells.
    seconds = times.astype(
        np.int64
    )

    median_seconds = int(
        np.median(
            seconds
        )
    )


    median_time = np.datetime64(
        median_seconds,
        "s",
    )


    return (
        times,
        median_time,
    )


# ============================================================
# READ ONE STANDARD-AM GRANULE
# ============================================================

def read_smap_file(
    path,
):

    with h5py.File(
        path,
        "r",
    ) as h5:


        if GROUP_NAME not in h5:

            raise RuntimeError(
                f"{GROUP_NAME} not found"
            )


        g = h5[
            GROUP_NAME
        ]


        required = [
            "soil_moisture",
            "latitude",
            "longitude",
            "retrieval_qual_flag",
            "tb_time_utc",
        ]


        for field in required:

            if field not in g:

                raise RuntimeError(
                    f"{field} not found "
                    f"in {GROUP_NAME}"
                )


        soil_ds = g[
            "soil_moisture"
        ]


        soil = soil_ds[
            :
        ].astype(
            np.float32
        )


        lat = g[
            "latitude"
        ][
            :
        ].astype(
            np.float32
        )


        lon = g[
            "longitude"
        ][
            :
        ].astype(
            np.float32
        )


        qual = g[
            "retrieval_qual_flag"
        ][
            :
        ]


        tb_time = g[
            "tb_time_utc"
        ][
            :
        ]


        # ----------------------------------------------------
        # Missing-data handling
        # ----------------------------------------------------

        fill_value = soil_ds.attrs.get(
            "_FillValue",
            -9999.0,
        )


        invalid_soil = (
            ~np.isfinite(
                soil
            )
            |
            (soil == fill_value)
            |
            (soil < 0.0)
            |
            (soil > 1.0)
        )


        soil[
            invalid_soil
        ] = np.nan


        # ----------------------------------------------------
        # Geographic selection
        # ----------------------------------------------------

        in_domain = (
            np.isfinite(lat)
            &
            np.isfinite(lon)
            &
            (lat >= MIN_LAT)
            &
            (lat <= MAX_LAT)
            &
            (lon >= MIN_LON)
            &
            (lon <= MAX_LON)
        )


        # ----------------------------------------------------
        # SMAP retrieval quality
        #
        # Keep quality flags 0 and 1, matching the previous
        # Hero Dirt prototype.
        # ----------------------------------------------------

        good_quality = (
            (qual == 0)
            |
            (qual == 1)
        )


        good = (
            in_domain
            &
            good_quality
            &
            np.isfinite(
                soil
            )
        )


        values = soil[
            good
        ]


        lats = lat[
            good
        ]


        lons = lon[
            good
        ]


        quals = qual[
            good
        ]


        observation_times, median_time = (
            decode_smap_times(
                tb_time,
                good,
            )
        )


        return {

            "values":
                values,

            "lat":
                lats,

            "lon":
                lons,

            "quality":
                quals,

            "times":
                observation_times,

            "median_time":
                median_time,

            "bbox_count":
                int(
                    np.count_nonzero(
                        in_domain
                    )
                ),
        }


# ============================================================
# FIND NEWEST VALID GRANULE
# ============================================================

chosen = None


print(
    "Searching newest standard-AM granule "
    "with valid San Gabriel retrievals..."
)

print()


for path in downloaded:

    if not path.exists():

        continue


    try:

        result = read_smap_file(
            path
        )

    except Exception as exc:

        print(
            f"  skipping {path.name}"
        )

        print(
            f"    {exc}"
        )

        continue


    print(
        f"  {path.name}"
    )

    print(
        f"    geometric cells: "
        f"{result['bbox_count']}"
    )

    print(
        f"    good retrievals: "
        f"{len(result['values'])}"
    )


    if len(
        result[
            "values"
        ]
    ) > 0:

        chosen = (
            path,
            result,
        )

        break


if chosen is None:

    raise RuntimeError(
        "No recent standard-AM SMAP granule "
        "contained valid San Gabriel retrievals."
    )


path, result = chosen


values = result[
    "values"
]

lats = result[
    "lat"
]

lons = result[
    "lon"
]

quals = result[
    "quality"
]

times = result[
    "times"
]

obs_time = result[
    "median_time"
]


# ============================================================
# DATE FROM FILENAME
# ============================================================

m = re.search(
    r"(\d{8})",
    path.name,
)


if m:

    obs_date = np.datetime64(
        datetime.strptime(
            m.group(1),
            "%Y%m%d",
        ).date()
    )

else:

    obs_date = np.datetime64(
        "NaT"
    )


# ============================================================
# STATISTICS
# ============================================================

mean_sm = float(
    np.mean(
        values
    )
)

median_sm = float(
    np.median(
        values
    )
)

std_sm = float(
    np.std(
        values
    )
)

min_sm = float(
    np.min(
        values
    )
)

max_sm = float(
    np.max(
        values
    )
)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(

    OUT_FILE,

    observation_date=obs_date,

    observation_time_utc=obs_time,

    cell_observation_time_utc=times,

    soil_moisture=values,

    latitude=lats,

    longitude=lons,

    retrieval_qual_flag=quals,

    mean_soil_moisture=np.float64(
        mean_sm
    ),

    median_soil_moisture=np.float64(
        median_sm
    ),

    std_soil_moisture=np.float64(
        std_sm
    ),

    min_soil_moisture=np.float64(
        min_sm
    ),

    max_soil_moisture=np.float64(
        max_sm
    ),

    source_file=np.array(
        path.name
    ),

    retrieval_group=np.array(
        GROUP_NAME
    ),
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("============================================")
print(" SMAP SUMMARY")
print("============================================")
print()


print("Source file:")
print(
    f"  {path.name}"
)

print()


print("Retrieval group:")
print(
    f"  {GROUP_NAME}"
)

print()


print("Observation date:")
print(
    f"  {obs_date}"
)

print()


print("Median observation time UTC:")
print(
    f"  {obs_time}"
)

print()


print("Valid San Gabriel retrievals:")
print(
    f"  {len(values)}"
)

print()


print(
    "Retrieval quality counts:"
)

for q in np.unique(
    quals
):

    print(
        f"  flag {int(q)}: "
        f"{np.count_nonzero(quals == q)}"
    )


print()


print(
    "Surface soil moisture [m3/m3]:"
)

print(
    f"  mean:   {mean_sm:.4f}"
)

print(
    f"  median: {median_sm:.4f}"
)

print(
    f"  std:    {std_sm:.4f}"
)

print(
    f"  range:  "
    f"{min_sm:.4f} "
    f"to "
    f"{max_sm:.4f}"
)

print()


print("Saved:")
print(
    f"  {OUT_FILE}"
)

print()
print("============================================")
print(" HERO DIRT SMAP INGEST COMPLETE")
print("============================================")
print()
