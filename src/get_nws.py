#!/usr/bin/env python3

"""
Hero Dirt Forecast
Retrieve NWS gridded forecast forcing for the San Gabriel Mountains.

Strategy
--------
1. Load Hero Dirt 200 m terrain/static grid.
2. Select 9 representative sample cells:
       west / central / east
   crossed with:
       low / middle / high elevation
3. Convert those sample cells to lat/lon.
4. Call NWS /points for each.
5. Retrieve forecastGridData.
6. Expand meteorological fields to hourly values.
7. Aggregate to fixed 6-hour UTC blocks.
8. Interpolate each 6-hour field across the Hero Dirt grid
   using inverse-distance weighting.

Forecast variables
------------------
temperature             deg C
relative_humidity       %
wind_speed              m/s
sky_cover               %
precipitation           mm per 6 h

Output
------
data/nws/processed/HeroDirt_NWS_6hourly.npz

Important
---------
This is our first regional implementation.

We are NOT yet applying an explicit elevation lapse-rate
correction. NWS gridpoint forecasts already contain some
terrain dependence, and we will evaluate whether an
additional elevation correction improves the Hero Dirt model.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
import time

import numpy as np
import requests

from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

STATIC_FILE = (
    ROOT
    / "static"
    / "model"
    / "HeroDirt_static_200m.npz"
)

OUT_DIR = (
    ROOT
    / "data"
    / "nws"
    / "processed"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_FILE = (
    OUT_DIR
    / "HeroDirt_NWS_6hourly.npz"
)


# ============================================================
# SETTINGS
# ============================================================

FORECAST_HOURS = 120

NWS_BASE = "https://api.weather.gov"

HEADERS = {
    "User-Agent":
        "HeroDirtForecast/0.1 "
        "(research mountain-bike trail forecast)",
    "Accept":
        "application/geo+json",
}

IDW_POWER = 2.0

REQUEST_PAUSE = 0.20


print()
print("============================================")
print(" HERO DIRT NWS FORECAST INGEST")
print("============================================")
print()


# ============================================================
# LOAD STATIC MODEL
# ============================================================

D = np.load(
    STATIC_FILE
)

x = D["x"]
y = D["y"]

elevation = D["elevation"]

physics_valid = D["physics_valid"].astype(
    bool
)

public_valid = D["public_valid"].astype(
    bool
)

ny, nx = elevation.shape


X, Y = np.meshgrid(
    x,
    y,
)


print("Hero Dirt grid:")
print(
    f"  size:       {nx} x {ny}"
)
print(
    f"  valid cells: "
    f"{public_valid.sum():,}"
)
print()


# ============================================================
# UTM -> WGS84
# ============================================================

to_ll = Transformer.from_crs(
    "EPSG:26911",
    "EPSG:4326",
    always_xy=True,
)


# ============================================================
# SELECT 9 REPRESENTATIVE SAMPLE POINTS
#
# Divide valid domain into three west-east sectors.
# In each sector choose cells near the 20th, 50th,
# and 80th elevation percentile.
# ============================================================

valid_indices = np.argwhere(
    public_valid
)

valid_x = X[
    public_valid
]

x_breaks = np.quantile(
    valid_x,
    [
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0,
    ],
)


sample_points = []

elev_quantiles = [
    0.20,
    0.50,
    0.80,
]


for ix_sector in range(3):

    xmin = x_breaks[
        ix_sector
    ]

    xmax = x_breaks[
        ix_sector + 1
    ]


    if ix_sector < 2:

        sector_mask = (
            public_valid
            &
            (X >= xmin)
            &
            (X < xmax)
        )

    else:

        sector_mask = (
            public_valid
            &
            (X >= xmin)
            &
            (X <= xmax)
        )


    rows, cols = np.where(
        sector_mask
    )


    zvals = elevation[
        sector_mask
    ]


    if len(
        zvals
    ) == 0:

        continue


    for iq, q in enumerate(
        elev_quantiles
    ):

        target_z = np.quantile(
            zvals,
            q,
        )


        diff = np.abs(
            elevation[
                rows,
                cols
            ]
            -
            target_z
        )


        k = int(
            np.argmin(
                diff
            )
        )


        r = rows[
            k
        ]

        c = cols[
            k
        ]


        xx = float(
            X[
                r,
                c
            ]
        )

        yy = float(
            Y[
                r,
                c
            ]
        )

        zz = float(
            elevation[
                r,
                c
            ]
        )


        lon, lat = to_ll.transform(
            xx,
            yy,
        )


        sample_points.append(
            {
                "row": int(r),
                "col": int(c),

                "x": xx,
                "y": yy,

                "elevation": zz,

                "lon": float(lon),
                "lat": float(lat),

                "sector": ix_sector,
                "elevation_class": iq,
            }
        )


print(
    f"NWS sample points selected: "
    f"{len(sample_points)}"
)

print()


for i, p in enumerate(
    sample_points,
    start=1,
):

    print(
        f"  {i:2d}: "
        f"lat={p['lat']:.4f}, "
        f"lon={p['lon']:.4f}, "
        f"elev={p['elevation']:.0f} m"
    )


print()


# ============================================================
# ISO-8601 DURATION PARSER
# ============================================================

def parse_duration_seconds(
    text
):

    """
    Parse simple ISO durations used by NWS,
    such as PT1H, PT6H, PT3H30M.
    """

    if text is None:

        return 3600


    m = re.fullmatch(
        r"P"
        r"(?:(\d+)D)?"
        r"(?:T"
        r"(?:(\d+)H)?"
        r"(?:(\d+)M)?"
        r"(?:(\d+)S)?"
        r")?",
        text,
    )


    if not m:

        return 3600


    days = int(
        m.group(1)
        or 0
    )

    hours = int(
        m.group(2)
        or 0
    )

    minutes = int(
        m.group(3)
        or 0
    )

    seconds = int(
        m.group(4)
        or 0
    )


    return (
        days * 86400
        +
        hours * 3600
        +
        minutes * 60
        +
        seconds
    )


# ============================================================
# VALID-TIME PARSER
# ============================================================

def parse_valid_time(
    valid_time
):

    """
    Example:
        2026-09-11T17:00:00+00:00/PT1H
    """

    start_text, duration_text = (
        valid_time.split(
            "/"
        )
    )


    start = datetime.fromisoformat(
        start_text
        .replace(
            "Z",
            "+00:00",
        )
    )


    if start.tzinfo is None:

        start = start.replace(
            tzinfo=timezone.utc
        )


    start = start.astimezone(
        timezone.utc
    )


    duration = timedelta(
        seconds=parse_duration_seconds(
            duration_text
        )
    )


    return (
        start,
        start + duration,
    )


# ============================================================
# UNIT CONVERSION
# ============================================================

def convert_value(
    value,
    uom,
    variable,
):

    if value is None:

        return np.nan


    value = float(
        value
    )


    u = (
        uom
        or ""
    ).lower()


    if variable == "temperature":

        # Usually already degC.
        if "degc" in u:

            return value

        if "degf" in u:

            return (
                value - 32.0
            ) * 5.0 / 9.0

        return value


    if variable == "relativeHumidity":

        return value


    if variable == "skyCover":

        return value


    if variable == "windSpeed":

        if "km_h-1" in u:

            return (
                value
                /
                3.6
            )

        if "m_s-1" in u:

            return value

        if "kt" in u:

            return (
                value
                *
                0.514444
            )

        return value


    if variable == "quantitativePrecipitation":

        # NWS grid QPF is normally mm.
        if "mm" in u:

            return value

        # Defensive conversion if meters appear.
        if "m" in u:

            return (
                value
                *
                1000.0
            )

        return value


    return value


# ============================================================
# CREATE FORECAST TIMELINE
#
# Begin at the latest available MRMS analysis time.
#
# The first forecast interval may therefore be shorter than
# 6 hours. It extends from the latest MRMS time to the next
# fixed 6-hour UTC boundary. All later intervals are 6 hours.
#
# This prevents precipitation forecast between the latest
# MRMS analysis and the next 6-hour boundary from being lost.
# ============================================================

MRMS_FILE = (
    ROOT
    / "data"
    / "mrms"
    / "processed"
    / "HeroDirt_MRMS_hourly.npz"
)

if not MRMS_FILE.exists():

    raise RuntimeError(
        f"Missing MRMS forcing file:\n{MRMS_FILE}"
    )


MRMS = np.load(
    MRMS_FILE
)

mrms_time = MRMS[
    "time"
].astype(
    "datetime64[s]"
)

latest_mrms_np = mrms_time[
    -1
]

latest_mrms_seconds = int(
    latest_mrms_np.astype(
        "datetime64[s]"
    ).astype(
        np.int64
    )
)

forecast_start = datetime.fromtimestamp(
    latest_mrms_seconds,
    tz=timezone.utc,
)


# Next fixed 6-hour UTC boundary strictly after forecast_start.

hour_number = int(
    forecast_start.timestamp()
    //
    3600
)

next_boundary_hour = (
    (
        hour_number
        //
        6
    )
    +
    1
) * 6

first_block_end = datetime.fromtimestamp(
    next_boundary_hour
    *
    3600,
    tz=timezone.utc,
)


block_starts = [
    forecast_start
]

block_ends = [
    first_block_end
]


while (
    block_ends[-1]
    <
    forecast_start
    +
    timedelta(
        hours=FORECAST_HOURS
    )
):

    block_starts.append(
        block_ends[-1]
    )

    block_ends.append(
        block_ends[-1]
        +
        timedelta(
            hours=6
        )
    )


nblock = len(
    block_starts
)


print(
    "6-hour forecast window:"
)

print(
    f"  start: "
    f"{block_starts[0].isoformat()}"
)

print(
    f"  end:   "
    f"{block_ends[-1].isoformat()}"
)

print()


# ============================================================
# FETCH NWS GRID DATA
# ============================================================

def get_json(
    url
):

    r = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )


    if r.status_code != 200:

        print()
        print(
            "NWS response:"
        )

        print(
            r.text[:2000]
        )

        print()


    r.raise_for_status()

    return r.json()


def fetch_point_forecast(
    point
):

    points_url = (
        f"{NWS_BASE}/points/"
        f"{point['lat']:.5f},"
        f"{point['lon']:.5f}"
    )


    info = get_json(
        points_url
    )


    props = info[
        "properties"
    ]


    grid_url = props[
        "forecastGridData"
    ]


    office = props.get(
        "gridId",
        "?"
    )

    grid_x = props.get(
        "gridX",
        np.nan
    )

    grid_y = props.get(
        "gridY",
        np.nan
    )


    grid = get_json(
        grid_url
    )


    return {
        "office":
            office,

        "grid_x":
            grid_x,

        "grid_y":
            grid_y,

        "grid_url":
            grid_url,

        "properties":
            grid[
                "properties"
            ],
    }


# ============================================================
# EXPAND CONTINUOUS FIELDS INTO 6-HOUR BLOCK MEANS
# ============================================================

def block_mean_field(
    prop,
    variable,
):

    values = prop.get(
        "values",
        []
    )

    uom = prop.get(
        "uom",
        ""
    )


    result = np.full(
        nblock,
        np.nan,
        dtype=np.float64,
    )


    for ib, (
        bs,
        be,
    ) in enumerate(
        zip(
            block_starts,
            block_ends,
        )
    ):

        weighted_sum = 0.0

        total_seconds = 0.0


        for record in values:

            value = convert_value(
                record.get(
                    "value"
                ),
                uom,
                variable,
            )


            if not np.isfinite(
                value
            ):

                continue


            rs, re_ = parse_valid_time(
                record[
                    "validTime"
                ]
            )


            overlap_start = max(
                bs,
                rs,
            )

            overlap_end = min(
                be,
                re_,
            )


            seconds = (
                overlap_end
                -
                overlap_start
            ).total_seconds()


            if seconds <= 0:

                continue


            weighted_sum += (
                value
                *
                seconds
            )

            total_seconds += (
                seconds
            )


        if total_seconds > 0:

            result[
                ib
            ] = (
                weighted_sum
                /
                total_seconds
            )


    return result


# ============================================================
# PRECIPITATION ACCUMULATION
#
# QPF values represent totals across their valid intervals.
# Allocate proportional amounts to any overlapping 6-h block.
# ============================================================

def block_precipitation(
    prop
):

    values = prop.get(
        "values",
        []
    )

    uom = prop.get(
        "uom",
        ""
    )


    result = np.zeros(
        nblock,
        dtype=np.float64,
    )


    valid_block = np.zeros(
        nblock,
        dtype=bool,
    )


    for ib, (
        bs,
        be,
    ) in enumerate(
        zip(
            block_starts,
            block_ends,
        )
    ):

        total = 0.0

        found = False


        for record in values:

            value = convert_value(
                record.get(
                    "value"
                ),
                uom,
                "quantitativePrecipitation",
            )


            if not np.isfinite(
                value
            ):

                continue


            rs, re_ = parse_valid_time(
                record[
                    "validTime"
                ]
            )


            interval_seconds = (
                re_
                -
                rs
            ).total_seconds()


            if interval_seconds <= 0:

                continue


            overlap_start = max(
                bs,
                rs,
            )

            overlap_end = min(
                be,
                re_,
            )


            overlap_seconds = (
                overlap_end
                -
                overlap_start
            ).total_seconds()


            if overlap_seconds <= 0:

                continue


            total += (
                value
                *
                overlap_seconds
                /
                interval_seconds
            )

            found = True


        if found:

            result[
                ib
            ] = total

            valid_block[
                ib
            ] = True


    result[
        ~valid_block
    ] = np.nan


    return result


# ============================================================
# PROCESS SAMPLE POINTS
# ============================================================

sample_temperature = []
sample_rh = []
sample_wind = []
sample_cloud = []
sample_precip = []


print(
    "Retrieving NWS forecasts..."
)

print()


for i, point in enumerate(
    sample_points,
    start=1,
):

    print(
        f"Point {i:2d}/{len(sample_points)} "
        f"(elev {point['elevation']:.0f} m)..."
    )


    try:

        F = fetch_point_forecast(
            point
        )

    except Exception as exc:

        print(
            f"  FAILED: {exc}"
        )

        sample_temperature.append(
            np.full(
                nblock,
                np.nan
            )
        )

        sample_rh.append(
            np.full(
                nblock,
                np.nan
            )
        )

        sample_wind.append(
            np.full(
                nblock,
                np.nan
            )
        )

        sample_cloud.append(
            np.full(
                nblock,
                np.nan
            )
        )

        sample_precip.append(
            np.full(
                nblock,
                np.nan
            )
        )

        continue


    P = F[
        "properties"
    ]


    point[
        "office"
    ] = F[
        "office"
    ]

    point[
        "grid_x"
    ] = F[
        "grid_x"
    ]

    point[
        "grid_y"
    ] = F[
        "grid_y"
    ]


    print(
        f"  NWS grid: "
        f"{F['office']} "
        f"{F['grid_x']},"
        f"{F['grid_y']}"
    )


    sample_temperature.append(
        block_mean_field(
            P[
                "temperature"
            ],
            "temperature",
        )
    )


    sample_rh.append(
        block_mean_field(
            P[
                "relativeHumidity"
            ],
            "relativeHumidity",
        )
    )


    sample_wind.append(
        block_mean_field(
            P[
                "windSpeed"
            ],
            "windSpeed",
        )
    )


    sample_cloud.append(
        block_mean_field(
            P[
                "skyCover"
            ],
            "skyCover",
        )
    )


    sample_precip.append(
        block_precipitation(
            P[
                "quantitativePrecipitation"
            ]
        )
    )


    time.sleep(
        REQUEST_PAUSE
    )


sample_temperature = np.array(
    sample_temperature,
    dtype=np.float64,
)

sample_rh = np.array(
    sample_rh,
    dtype=np.float64,
)

sample_wind = np.array(
    sample_wind,
    dtype=np.float64,
)

sample_cloud = np.array(
    sample_cloud,
    dtype=np.float64,
)

sample_precip = np.array(
    sample_precip,
    dtype=np.float64,
)


# ============================================================
# SAMPLE COORDINATES
# ============================================================

sample_x = np.array(
    [
        p["x"]
        for p in sample_points
    ],
    dtype=np.float64,
)

sample_y = np.array(
    [
        p["y"]
        for p in sample_points
    ],
    dtype=np.float64,
)

sample_elevation = np.array(
    [
        p["elevation"]
        for p in sample_points
    ],
    dtype=np.float64,
)

sample_lat = np.array(
    [
        p["lat"]
        for p in sample_points
    ],
    dtype=np.float64,
)

sample_lon = np.array(
    [
        p["lon"]
        for p in sample_points
    ],
    dtype=np.float64,
)


# ============================================================
# INVERSE-DISTANCE INTERPOLATION
# ============================================================

def interpolate_idw(
    sample_values,
):

    """
    sample_values:
        shape = (n_sample, n_time)

    Returns:
        shape = (n_time, ny, nx)
    """

    nt = sample_values.shape[
        1
    ]


    output = np.full(
        (
            nt,
            ny,
            nx,
        ),
        np.nan,
        dtype=np.float32,
    )


    distances = []


    for sx, sy in zip(
        sample_x,
        sample_y,
    ):

        dist = np.sqrt(
            (
                X - sx
            ) ** 2
            +
            (
                Y - sy
            ) ** 2
        )


        # Prevent divide-by-zero.
        dist = np.maximum(
            dist,
            1.0,
        )


        distances.append(
            dist
        )


    distances = np.stack(
        distances,
        axis=0,
    )


    base_weights = (
        1.0
        /
        distances
        **
        IDW_POWER
    )


    for it in range(
        nt
    ):

        values = sample_values[
            :,
            it
        ]


        valid_samples = np.isfinite(
            values
        )


        if not np.any(
            valid_samples
        ):

            continue


        w = base_weights[
            valid_samples,
            :,
            :
        ]


        v = values[
            valid_samples
        ][
            :,
            None,
            None,
        ]


        field = (
            np.sum(
                w
                *
                v,
                axis=0,
            )
            /
            np.sum(
                w,
                axis=0,
            )
        )


        field[
            ~physics_valid
        ] = np.nan


        output[
            it,
            :,
            :
        ] = field.astype(
            np.float32
        )


    return output


# ============================================================
# INTERPOLATE FULL FIELDS
# ============================================================

print()
print(
    "Interpolating NWS forcing "
    "across Hero Dirt grid..."
)


temperature = interpolate_idw(
    sample_temperature
)

relative_humidity = interpolate_idw(
    sample_rh
)

wind_speed = interpolate_idw(
    sample_wind
)

sky_cover = interpolate_idw(
    sample_cloud
)

precipitation = interpolate_idw(
    sample_precip
)


# ============================================================
# CLIP PHYSICAL RANGES
# ============================================================

relative_humidity = np.clip(
    relative_humidity,
    0.0,
    100.0,
)

sky_cover = np.clip(
    sky_cover,
    0.0,
    100.0,
)

wind_speed = np.maximum(
    wind_speed,
    0.0,
)

precipitation = np.maximum(
    precipitation,
    0.0,
)


# ============================================================
# TIME ARRAY
#
# Stored as end-of-6-hour-block time.
# ============================================================

forecast_time = np.array(
    [
        np.datetime64(
            t.replace(
                tzinfo=None
            ),
            "s",
        )
        for t in block_ends
    ],
    dtype="datetime64[s]",
)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(

    OUT_FILE,

    time=forecast_time,
    forecast_start=np.datetime64(
        forecast_start.replace(
            tzinfo=None
        ),
        "s",
    ),

    temperature_c=temperature,

    relative_humidity=relative_humidity,

    wind_speed_ms=wind_speed,

    sky_cover_percent=sky_cover,

    precipitation_mm=precipitation,

    sample_x=sample_x,
    sample_y=sample_y,

    sample_lat=sample_lat,
    sample_lon=sample_lon,

    sample_elevation=sample_elevation,

    sample_temperature_c=sample_temperature,

    sample_relative_humidity=sample_rh,

    sample_wind_speed_ms=sample_wind,

    sample_sky_cover_percent=sample_cloud,

    sample_precipitation_mm=sample_precip,

    x=x,
    y=y,

    epsg=np.int32(
        26911
    ),
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("============================================")
print(" NWS FORECAST SUMMARY")
print("============================================")
print()


print(
    f"6-hour blocks: "
    f"{len(forecast_time)}"
)

print(
    f"First block end: "
    f"{forecast_time[0]}"
)

print(
    f"Last block end:  "
    f"{forecast_time[-1]}"
)

print()


domain = public_valid


def domain_stats(
    A,
):

    vals = A[
        :,
        domain
    ]

    return (
        np.nanmin(
            vals
        ),
        np.nanmean(
            vals
        ),
        np.nanmax(
            vals
        ),
    )


tmin, tmean, tmax = domain_stats(
    temperature
)

rhmin, rhmean, rhmax = domain_stats(
    relative_humidity
)

wmin, wmean, wmax = domain_stats(
    wind_speed
)

cmin, cmean, cmax = domain_stats(
    sky_cover
)


domain_precip = np.nanmean(
    precipitation[
        :,
        domain
    ],
    axis=1,
)


print(
    "Temperature [C]:"
)

print(
    f"  range: "
    f"{tmin:.1f} "
    f"to "
    f"{tmax:.1f}"
)

print(
    f"  mean:  "
    f"{tmean:.1f}"
)

print()


print(
    "Relative humidity [%]:"
)

print(
    f"  range: "
    f"{rhmin:.1f} "
    f"to "
    f"{rhmax:.1f}"
)

print(
    f"  mean:  "
    f"{rhmean:.1f}"
)

print()


print(
    "Wind speed [m/s]:"
)

print(
    f"  range: "
    f"{wmin:.1f} "
    f"to "
    f"{wmax:.1f}"
)

print(
    f"  mean:  "
    f"{wmean:.1f}"
)

print()


print(
    "Sky cover [%]:"
)

print(
    f"  range: "
    f"{cmin:.1f} "
    f"to "
    f"{cmax:.1f}"
)

print(
    f"  mean:  "
    f"{cmean:.1f}"
)

print()


print(
    "Forecast precipitation:"
)

print(
    f"  domain-mean total: "
    f"{np.nansum(domain_precip):.2f} mm"
)

print(
    f"  wettest domain-mean "
    f"6-h block: "
    f"{np.nanmax(domain_precip):.2f} mm"
)

print()


print(
    "Saved:"
)

print(
    f"  {OUT_FILE}"
)

print()
print("============================================")
print(" HERO DIRT NWS INGEST COMPLETE")
print("============================================")
print()
