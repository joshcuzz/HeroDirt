#!/usr/bin/env python3

"""
Hero Dirt Forecast
Download OSM trails / paths / tracks for the San Gabriel Mountains.

The San Gabriel domain is divided into small geographic tiles
to reduce Overpass request size and avoid gateway timeouts.

Each successful tile is cached separately so interrupted runs
can resume without repeating completed requests.

Final output
------------
data/osm/SanGabriels_all_trails_overpass.json
"""

from pathlib import Path
import json
import time

import requests


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
    / "sangabriels_all_trails_tiles"
)

OSM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TILE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_FILE = (
    OSM_DIR
    / "SanGabriels_all_trails_overpass.json"
)


# ============================================================
# SAN GABRIELS DOMAIN
#
# Slightly padded around operational Hero Dirt grid.
# ============================================================

SOUTH = 34.14
WEST = -118.24
NORTH = 34.31
EAST = -118.06


# ============================================================
# TILING
#
# Smaller tiles are substantially more reliable for dense
# OSM regions around Fairfield / New Haven / Hartford.
# ============================================================

N_LON = 4
N_LAT = 3


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
        "HeroDirtForecast/0.1 research prototype"
}

REQUEST_TIMEOUT = 240

RETRIES_PER_ENDPOINT = 2

PAUSE_BETWEEN_TILES = 4.0

PAUSE_AFTER_FAILURE = 15.0


# ============================================================
# QUERY ONE TILE
# ============================================================

def fetch_tile(
    south,
    west,
    north,
    east,
):

    query = f"""
    [out:json][timeout:120];

    (
      way
        ["highway"~"path|track|footway|cycleway"]
        ({south},{west},{north},{east});
    );

    out tags geom;
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
                    timeout=REQUEST_TIMEOUT,
                    headers=HEADERS,
                )


                response.raise_for_status()


                data = response.json()


                return data


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
# TILE CACHE NAME
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
# DOWNLOAD / RESUME TILES
# ============================================================

ntiles = (
    N_LON
    *
    N_LAT
)

tile_number = 0


print()
print("============================================")
print(" HERO DIRT CONNECTICUT OSM DOWNLOAD")
print("============================================")
print()

print(
    f"Tiles: {N_LON} x {N_LAT} "
    f"= {ntiles}"
)

print()


for iy in range(
    N_LAT
):

    south = lat_edges[
        iy
    ]

    north = lat_edges[
        iy + 1
    ]


    for ix in range(
        N_LON
    ):

        west = lon_edges[
            ix
        ]

        east = lon_edges[
            ix + 1
        ]


        tile_number += 1


        fn = tile_file(
            iy,
            ix,
        )


        print(
            f"Tile {tile_number}/{ntiles}"
        )

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


        # ----------------------------------------------------
        # Resume from existing successful tile
        # ----------------------------------------------------

        if fn.exists():

            try:

                with open(
                    fn,
                    "r",
                ) as f:

                    data = json.load(
                        f
                    )


                nelem = len(
                    data.get(
                        "elements",
                        []
                    )
                )


                print(
                    f"  cached ways: "
                    f"{nelem:,}"
                )

                print()

                continue


            except Exception:

                print(
                    "  Existing cache unreadable; "
                    "downloading again."
                )


        # ----------------------------------------------------
        # Download tile
        # ----------------------------------------------------

        data = fetch_tile(
            south,
            west,
            north,
            east,
        )


        elements = data.get(
            "elements",
            []
        )


        print(
            f"  returned ways: "
            f"{len(elements):,}"
        )


        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        with open(
            fn,
            "w",
        ) as f:

            json.dump(
                data,
                f,
            )


        print(
            f"  saved: {fn.name}"
        )

        print()


        time.sleep(
            PAUSE_BETWEEN_TILES
        )


# ============================================================
# MERGE ALL TILE CACHES
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


        if not fn.exists():

            raise RuntimeError(
                f"Missing tile cache: {fn}"
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
# DEDUPLICATE OSM WAYS
#
# Ways crossing tile boundaries may appear more than once.
# ============================================================

unique = {}


for element in all_elements:

    if element.get(
        "type"
    ) != "way":

        continue


    way_id = element.get(
        "id"
    )


    if way_id is None:

        continue


    unique[
        int(way_id)
    ] = element


elements_unique = list(
    unique.values()
)


# ============================================================
# WRITE FINAL OVERPASS-LIKE CACHE
# ============================================================

output = {
    "version": 0.6,
    "generator":
        "HeroDirt tiled Connecticut Overpass downloader",
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


# ============================================================
# SUMMARY
# ============================================================

print()
print("============================================")
print(" CONNECTICUT OSM DOWNLOAD COMPLETE")
print("============================================")
print()

print(
    f"Raw tiled ways: "
    f"{len(all_elements):,}"
)

print(
    f"Unique ways:    "
    f"{len(elements_unique):,}"
)

print()

print(
    "Saved:"
)

print(
    f"  {OUT_FILE}"
)

print()
