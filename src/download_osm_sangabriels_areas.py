#!/usr/bin/env python3

"""
Hero Dirt Forecast
Download named San Gabriel park / preserve / recreation areas
for area-first search.

This is separate from the trail geometry cache.

Final output
------------
data/osm/SanGabriels_trail_areas_overpass.json
"""

from pathlib import Path
import json
import time

import requests

import numpy as np
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

OSM_DIR = (
    ROOT
    / "data"
    / "osm"
)

TILE_DIR = (
    OSM_DIR
    / "sangabriels_area_tiles"
)

OUT_FILE = (
    OSM_DIR
    / "SanGabriels_trail_areas_overpass.json"
)

OSM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TILE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAN GABRIELS DOMAIN
# ============================================================

STATIC_FILE = (
    ROOT
    / "static"
    / "model"
    / "HeroDirt_static_200m.npz"
)

D = np.load(
    STATIC_FILE
)

grid_x = D["x"]
grid_y = D["y"]

to_ll = Transformer.from_crs(
    "EPSG:26911",
    "EPSG:4326",
    always_xy=True,
)

xmin = float(np.nanmin(grid_x))
xmax = float(np.nanmax(grid_x))
ymin = float(np.nanmin(grid_y))
ymax = float(np.nanmax(grid_y))

corners = [
    to_ll.transform(xmin, ymin),
    to_ll.transform(xmin, ymax),
    to_ll.transform(xmax, ymin),
    to_ll.transform(xmax, ymax),
]

lons = [
    c[0]
    for c in corners
]

lats = [
    c[1]
    for c in corners
]

# Small geographic padding outside the model domain.
PAD = 0.01

WEST = min(lons) - PAD
EAST = max(lons) + PAD
SOUTH = min(lats) - PAD
NORTH = max(lats) + PAD

print()
print("OSM download domain from operational model grid:")
print(f"  south: {SOUTH:.5f}")
print(f"  west:  {WEST:.5f}")
print(f"  north: {NORTH:.5f}")
print(f"  east:  {EAST:.5f}")
print()


# ============================================================
# TILING
# ============================================================

N_LON = 8
N_LAT = 6


lon_edges = [
    WEST
    +
    i
    *
    (EAST - WEST)
    /
    N_LON

    for i in range(
        N_LON + 1
    )
]


lat_edges = [
    SOUTH
    +
    i
    *
    (NORTH - SOUTH)
    /
    N_LAT

    for i in range(
        N_LAT + 1
    )
]


# ============================================================
# OVERPASS SETTINGS
# ============================================================

ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

HEADERS = {
    "User-Agent":
        "HeroDirtForecast/0.2 research prototype"
}

REQUEST_TIMEOUT = 240

RETRIES_PER_ENDPOINT = 2

PAUSE_AFTER_FAILURE = 10.0

PAUSE_BETWEEN_TILES = 2.0


# ============================================================
# QUERY
# ============================================================

def fetch_tile(
    south,
    west,
    north,
    east,
):

    bbox = (
        f"{south},"
        f"{west},"
        f"{north},"
        f"{east}"
    )

    query = f"""
    [out:json][timeout:120];

    (
      nwr
        ["name"]
        ["leisure"~"park|nature_reserve|recreation_ground"]
        ({bbox});

      nwr
        ["name"]
        ["boundary"="protected_area"]
        ({bbox});

      nwr
        ["name"]
        ["landuse"="recreation_ground"]
        ({bbox});
    );

    out tags center;
    """

    last_error = None

    for endpoint in ENDPOINTS:

        for attempt in range(
            1,
            RETRIES_PER_ENDPOINT + 1
        ):

            print(
                f"    endpoint: {endpoint}"
            )

            print(
                f"    attempt:  {attempt}"
            )

            try:

                response = requests.post(
                    endpoint,
                    data={
                        "data": query
                    },
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                return response.json()

            except Exception as exc:

                last_error = exc

                print(
                    f"      failed: {exc}"
                )

                time.sleep(
                    PAUSE_AFTER_FAILURE
                )

    raise RuntimeError(
        "All Overpass attempts failed."
    ) from last_error


# ============================================================
# TILE CACHE
# ============================================================

def tile_file(
    iy,
    ix,
):

    return (
        TILE_DIR
        /
        f"tile_y{iy:02d}_x{ix:02d}.json"
    )


# ============================================================
# DOWNLOAD
# ============================================================

ntiles = (
    N_LON
    *
    N_LAT
)

tile_number = 0


print()
print(
    "============================================"
)
print(
    " HERO DIRT CONNECTICUT AREA DOWNLOAD"
)
print(
    "============================================"
)
print()

print(
    f"Tiles: {N_LON} x {N_LAT} = {ntiles}"
)

print()


for iy in range(
    N_LAT
):

    for ix in range(
        N_LON
    ):

        tile_number += 1

        fn = tile_file(
            iy,
            ix,
        )

        south = lat_edges[
            iy
        ]

        north = lat_edges[
            iy + 1
        ]

        west = lon_edges[
            ix
        ]

        east = lon_edges[
            ix + 1
        ]

        print(
            f"Tile {tile_number}/{ntiles}"
        )

        if fn.exists():

            print(
                "  cached"
            )

            continue

        print(
            f"  south={south:.5f}"
        )

        print(
            f"  west ={west:.5f}"
        )

        print(
            f"  north={north:.5f}"
        )

        print(
            f"  east ={east:.5f}"
        )

        data = fetch_tile(
            south,
            west,
            north,
            east,
        )

        with open(
            fn,
            "w",
        ) as f:

            json.dump(
                data,
                f,
            )

        print(
            f"  received "
            f"{len(data.get('elements', []))}"
        )

        time.sleep(
            PAUSE_BETWEEN_TILES
        )


# ============================================================
# MERGE
# ============================================================

print()
print(
    "Merging cached tiles..."
)


all_elements = []


for iy in range(
    N_LAT
):

    for ix in range(
        N_LON
    ):

        fn = tile_file(
            iy,
            ix,
        )

        with open(
            fn,
            "r",
        ) as f:

            data = json.load(
                f
            )

        all_elements.extend(
            data.get(
                "elements",
                []
            )
        )


# ============================================================
# DEDUPLICATE
# ============================================================

unique = {}


for element in all_elements:

    element_type = element.get(
        "type"
    )

    element_id = element.get(
        "id"
    )

    if (
        element_type is None
        or
        element_id is None
    ):
        continue

    key = (
        element_type,
        int(
            element_id
        ),
    )

    unique[
        key
    ] = element


elements_unique = list(
    unique.values()
)


# ============================================================
# WRITE
# ============================================================

output = {
    "version": 0.6,
    "generator":
        "HeroDirt Connecticut area downloader",
    "elements":
        elements_unique,
}


with open(
    OUT_FILE,
    "w",
) as f:

    json.dump(
        output,
        f,
    )


print()
print(
    "============================================"
)
print(
    " CONNECTICUT AREA DOWNLOAD COMPLETE"
)
print(
    "============================================"
)
print()

print(
    f"Raw tiled objects: "
    f"{len(all_elements):,}"
)

print(
    f"Unique areas:      "
    f"{len(elements_unique):,}"
)

print()

print(
    "Saved:"
)

print(
    f"  {OUT_FILE}"
)
