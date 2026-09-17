#!/usr/bin/env python3

"""
Hero Dirt Forecast
Download and process recent NOAA MRMS hourly precipitation.

Product
-------
MultiSensor_QPE_01H_Pass2

Workflow
--------
1. Read NOAA's live MRMS directory.
2. Identify timestamped hourly files.
3. Download files not already stored locally.
4. Decompress GRIB2 files.
5. Read precipitation with cfgrib/xarray.
6. Interpolate each hourly field to the Hero Dirt 200 m grid.
7. Save hourly precipitation history.
8. Sum hourly fields into 6-hour model forcing blocks.

Important
---------
The model may be run irregularly.

Downloaded MRMS files are therefore retained in a local archive:

    data/mrms/raw/

Over time, this archive lets Hero Dirt reconstruct forcing
between runs even after NOAA removes older files from the
live directory.

MRMS QPE values are interpreted as millimeters.
"""

from pathlib import Path
from datetime import datetime, timezone
import gzip
import re
import shutil

import numpy as np
import requests
from bs4 import BeautifulSoup
import xarray as xr
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling


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
    / "mrms"
    / "raw"
)

PROC_DIR = (
    ROOT
    / "data"
    / "mrms"
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

OUT_HOURLY = (
    PROC_DIR
    / "HeroDirt_MRMS_hourly.npz"
)

OUT_6H = (
    PROC_DIR
    / "HeroDirt_MRMS_6hourly.npz"
)


# ============================================================
# NOAA MRMS
# ============================================================

BASE_URL = (
    "https://mrms.ncep.noaa.gov/"
    "2D/MultiSensor_QPE_01H_Pass2/"
)


# ============================================================
# LOAD HERO DIRT GRID
# ============================================================

G = np.load(
    GRID_FILE
)

west = float(G["west"])
east = float(G["east"])
south = float(G["south"])
north = float(G["north"])

width = int(G["width"])
height = int(G["height"])

x = G["x"]
y = G["y"]

res = float(G["resolution"])

TARGET_CRS = CRS.from_epsg(
    26911
)

transform200 = from_origin(
    west,
    north,
    res,
    res,
)

shape200 = (
    height,
    width,
)


print()
print("============================================")
print(" HERO DIRT MRMS INGEST")
print("============================================")
print()

print("Hero Dirt grid:")
print(
    f"  {width} x {height}"
)
print(
    f"  resolution: {res:.0f} m"
)
print()


# ============================================================
# GET NOAA DIRECTORY LISTING
# ============================================================

print(
    "Reading NOAA MRMS directory..."
)

r = requests.get(
    BASE_URL,
    timeout=60,
)

r.raise_for_status()


soup = BeautifulSoup(
    r.text,
    "html.parser",
)


# ============================================================
# IDENTIFY TIMESTAMPED FILES
# ============================================================

pattern = re.compile(
    r"MRMS_MultiSensor_QPE_01H_Pass2_"
    r"00\.00_"
    r"(\d{8})-(\d{6})"
    r"\.grib2\.gz$"
)


records = []


for link in soup.find_all(
    "a"
):

    href = link.get(
        "href"
    )

    if not href:
        continue

    m = pattern.match(
        href
    )

    if not m:
        continue


    date_string = (
        m.group(1)
        +
        m.group(2)
    )


    dt = datetime.strptime(
        date_string,
        "%Y%m%d%H%M%S",
    ).replace(
        tzinfo=timezone.utc
    )


    records.append(
        (
            dt,
            href,
        )
    )


records.sort(
    key=lambda q: q[0]
)


if not records:

    raise RuntimeError(
        "No timestamped MRMS hourly files "
        "were found in NOAA directory."
    )


print(
    f"Live hourly files found: "
    f"{len(records)}"
)

print(
    "Earliest live file: "
    f"{records[0][0].isoformat()}"
)

print(
    "Latest live file:   "
    f"{records[-1][0].isoformat()}"
)

print()


# ============================================================
# DOWNLOAD NEW FILES
# ============================================================

print(
    "Updating local MRMS archive..."
)


for i, (
    dt,
    filename,
) in enumerate(
    records,
    start=1,
):

    gz_path = (
        RAW_DIR
        /
        filename
    )


    grib_name = filename[
        :-3
    ]

    grib_path = (
        RAW_DIR
        /
        grib_name
    )


    # Already decompressed -> nothing to do.
    if grib_path.exists():

        continue


    # Download gzip if needed.
    if not gz_path.exists():

        url = (
            BASE_URL
            +
            filename
        )

        rr = requests.get(
            url,
            timeout=120,
        )

        rr.raise_for_status()

        gz_path.write_bytes(
            rr.content
        )


    # Decompress.
    with gzip.open(
        gz_path,
        "rb",
    ) as fin:

        with open(
            grib_path,
            "wb",
        ) as fout:

            shutil.copyfileobj(
                fin,
                fout,
            )


    # Remove gzip after successful expansion.
    gz_path.unlink(
        missing_ok=True
    )


print(
    "Local archive updated."
)
print()


# ============================================================
# FIND ALL LOCAL MRMS FILES
#
# Includes files retained from earlier Hero Dirt runs.
# ============================================================

local_pattern = re.compile(
    r"MRMS_MultiSensor_QPE_01H_Pass2_"
    r"00\.00_"
    r"(\d{8})-(\d{6})"
    r"\.grib2$"
)


local_records = []


for path in RAW_DIR.glob(
    "*.grib2"
):

    m = local_pattern.match(
        path.name
    )

    if not m:
        continue


    dt = datetime.strptime(
        m.group(1)
        +
        m.group(2),
        "%Y%m%d%H%M%S",
    ).replace(
        tzinfo=timezone.utc
    )


    local_records.append(
        (
            dt,
            path,
        )
    )


local_records.sort(
    key=lambda q: q[0]
)


print(
    f"Local hourly archive: "
    f"{len(local_records)} files"
)

print()


# ============================================================
# GRIB READER
# ============================================================

def read_mrms_grib(
    path,
):

    """
    Read one MRMS GRIB2 file with cfgrib.

    Returns:
        precipitation array
        latitude array
        longitude array
    """

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": ""
        },
    )


    # Usually only one precipitation variable is present.
    data_vars = list(
        ds.data_vars
    )


    if not data_vars:

        ds.close()

        raise RuntimeError(
            f"No data variable in {path.name}"
        )


    varname = data_vars[0]

    precip = (
        ds[varname]
        .values
        .astype(
            np.float32
        )
    )


    # MRMS latitude/longitude coordinates.
    lat = ds["latitude"].values

    lon = ds["longitude"].values


    ds.close()


    # --------------------------------------------------------
    # MRMS missing / special values
    #
    # Negative values are not physical hourly precipitation.
    # Treat them as missing/zero for QPE accumulation.
    # --------------------------------------------------------

    precip[
        ~np.isfinite(
            precip
        )
    ] = 0.0


    precip[
        precip < 0.0
    ] = 0.0


    return (
        precip,
        lat,
        lon,
    )


# ============================================================
# PROCESS HOURLY FIELDS
# ============================================================

hourly_times = []
hourly_fields = []


print(
    "Processing hourly MRMS fields..."
)


for i, (
    dt,
    path,
) in enumerate(
    local_records,
    start=1,
):

    try:

        precip, lat, lon = (
            read_mrms_grib(
                path
            )
        )

    except Exception as exc:

        print(
            f"  WARNING: skipping "
            f"{path.name}"
        )

        print(
            f"           {exc}"
        )

        continue


    # --------------------------------------------------------
    # MRMS native grid is regular lat/lon.
    #
    # Determine transform from coordinate centers.
    # --------------------------------------------------------

    if lat.ndim == 1:

        dy = float(
            np.abs(
                np.diff(
                    lat[:2]
                )
            )[0]
        )

        dx = float(
            np.abs(
                np.diff(
                    lon[:2]
                )
            )[0]
        )


        # Ensure raster rows are north -> south.
        if lat[0] < lat[-1]:

            precip = np.flipud(
                precip
            )

            lat = lat[::-1]


        # Convert longitude from 0-360 if necessary.
        if np.nanmax(
            lon
        ) > 180:

            lon = np.where(
                lon > 180,
                lon - 360,
                lon,
            )


        left = (
            float(
                np.min(
                    lon
                )
            )
            -
            dx / 2
        )

        top = (
            float(
                np.max(
                    lat
                )
            )
            +
            dy / 2
        )


        src_transform = from_origin(
            left,
            top,
            dx,
            dy,
        )


    else:

        raise RuntimeError(
            "Unexpected MRMS coordinate structure."
        )


    # --------------------------------------------------------
    # Reproject to Hero Dirt 200 m grid.
    #
    # Bilinear interpolation is acceptable here because
    # the MRMS source grid is much coarser than 200 m.
    # --------------------------------------------------------

    out = np.zeros(
        shape200,
        dtype=np.float32,
    )


    reproject(
        source=precip,
        destination=out,

        src_transform=src_transform,
        src_crs="EPSG:4326",

        dst_transform=transform200,
        dst_crs=TARGET_CRS,

        src_nodata=None,
        dst_nodata=0.0,

        resampling=Resampling.bilinear,

        num_threads=4,
    )


    out[
        ~np.isfinite(
            out
        )
    ] = 0.0


    out[
        out < 0
    ] = 0.0


    hourly_times.append(
        np.datetime64(
            dt.replace(
                tzinfo=None
            )
        )
    )


    hourly_fields.append(
        out
    )


    if (
        i % 12 == 0
        or
        i == len(
            local_records
        )
    ):

        print(
            f"  processed "
            f"{i} / "
            f"{len(local_records)}"
        )


if not hourly_fields:

    raise RuntimeError(
        "No MRMS hourly fields "
        "were processed successfully."
    )


hourly_times = np.array(
    hourly_times,
    dtype="datetime64[s]",
)


hourly_precip = np.stack(
    hourly_fields,
    axis=0,
).astype(
    np.float32
)


# ============================================================
# SORT / REMOVE DUPLICATES
# ============================================================

order = np.argsort(
    hourly_times
)

hourly_times = (
    hourly_times[
        order
    ]
)

hourly_precip = (
    hourly_precip[
        order,
        :,
        :
    ]
)


unique_times, unique_index = np.unique(
    hourly_times,
    return_index=True,
)


hourly_times = unique_times

hourly_precip = (
    hourly_precip[
        unique_index,
        :,
        :
    ]
)


# ============================================================
# SAVE HOURLY HISTORY
# ============================================================

np.savez_compressed(

    OUT_HOURLY,

    time=hourly_times,

    precip_mm=hourly_precip,

    x=x,
    y=y,

    resolution=np.float64(
        res
    ),

    epsg=np.int32(
        26911
    ),
)


# ============================================================
# BUILD 6-HOUR ACCUMULATIONS
#
# Use UTC six-hour blocks:
#
# 00-06
# 06-12
# 12-18
# 18-24
#
# A block is saved only if all six hourly fields are present.
# ============================================================

print()
print(
    "Building complete 6-hour accumulations..."
)


time_to_index = {
    t: i
    for i, t in enumerate(
        hourly_times
    )
}


start_hour = (
    hourly_times[0]
    .astype(
        "datetime64[h]"
    )
)


# Align to next 6-hour UTC boundary.
hour_number = int(
    (
        start_hour
        .astype(
            "datetime64[h]"
        )
        .astype(
            np.int64
        )
    )
)


aligned_hour_number = (
    (
        hour_number + 5
    )
    //
    6
    *
    6
)


block_start = np.datetime64(
    aligned_hour_number,
    "h",
)


last_hour = (
    hourly_times[-1]
    .astype(
        "datetime64[h]"
    )
)


six_times = []
six_precip = []


while (
    block_start
    +
    np.timedelta64(
        5,
        "h",
    )
    <= last_hour
):

    needed = np.array(
        [
            block_start
            +
            np.timedelta64(
                k,
                "h",
            )
            for k in range(6)
        ],
        dtype="datetime64[s]",
    )


    if all(
        t in time_to_index
        for t in needed
    ):

        inds = [
            time_to_index[
                t
            ]
            for t in needed
        ]


        P6 = np.sum(
            hourly_precip[
                inds,
                :,
                :
            ],
            axis=0,
        ).astype(
            np.float32
        )


        six_times.append(
            block_start
            +
            np.timedelta64(
                6,
                "h",
            )
        )


        six_precip.append(
            P6
        )


    block_start = (
        block_start
        +
        np.timedelta64(
            6,
            "h",
        )
    )


if six_precip:

    six_times = np.array(
        six_times,
        dtype="datetime64[s]",
    )

    six_precip = np.stack(
        six_precip,
        axis=0,
    )


    np.savez_compressed(

        OUT_6H,

        time=six_times,

        precip_mm=six_precip,

        x=x,
        y=y,

        resolution=np.float64(
            res
        ),

        epsg=np.int32(
            26911
        ),
    )


else:

    six_times = np.array(
        [],
        dtype="datetime64[s]",
    )

    six_precip = np.empty(
        (
            0,
            height,
            width,
        ),
        dtype=np.float32,
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "============================================"
)
print(
    " MRMS SUMMARY"
)
print(
    "============================================"
)
print()


print(
    f"Hourly fields: "
    f"{len(hourly_times)}"
)

print(
    f"Hourly start:  "
    f"{hourly_times[0]}"
)

print(
    f"Hourly end:    "
    f"{hourly_times[-1]}"
)


print()


domain_mean_hourly = np.mean(
    hourly_precip,
    axis=(1, 2),
)


print(
    f"Mean hourly precipitation "
    f"over domain:"
)

print(
    f"  min:  "
    f"{domain_mean_hourly.min():.3f} mm"
)

print(
    f"  max:  "
    f"{domain_mean_hourly.max():.3f} mm"
)


total_domain_mean = np.mean(
    np.sum(
        hourly_precip,
        axis=0,
    )
)


print(
    f"Mean accumulated precipitation "
    f"over available history:"
)

print(
    f"  {total_domain_mean:.2f} mm"
)


print()


print(
    f"Complete 6-hour blocks: "
    f"{len(six_times)}"
)


if len(
    six_times
) > 0:

    print(
        f"6-hour start/end time: "
        f"{six_times[0]}"
    )

    print(
        f"Latest 6-hour end time: "
        f"{six_times[-1]}"
    )


    mean6 = np.mean(
        six_precip,
        axis=(1, 2),
    )


    print(
        f"Largest domain-mean "
        f"6-hour accumulation:"
    )

    print(
        f"  {mean6.max():.2f} mm"
    )


print()
print(
    "Saved:"
)

print(
    f"  {OUT_HOURLY}"
)

if len(
    six_times
) > 0:

    print(
        f"  {OUT_6H}"
    )


print()
print(
    "============================================"
)
print(
    " HERO DIRT MRMS INGEST COMPLETE"
)
print(
    "============================================"
)
print()
