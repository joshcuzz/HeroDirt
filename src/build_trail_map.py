#!/usr/bin/env python3

"""
Hero Dirt Forecast
==================

Interactive trail-condition forecast for the San Gabriel Mountains.

Includes
--------
- Now / +24 / +48 / +72 / +96 / +120 h forecast layers
- Colored trail-condition segments
- Trail click popup
- Rider reporting from clicked trail location
- About / methods panel
- Link to Hero Dirt Explorer
- Mobile-responsive layout
- Trail-name search and zoom

Output
------
web/HeroDirt_trails.html
"""

from pathlib import Path
import json
import math

import numpy as np
import rasterio
from pyproj import Transformer
import folium


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

CURRENT_FILE = (
    ROOT
    / "output/current_v2_beta/HeroDirt_current_state.npz"
)

FORECAST_FILE = (
    ROOT
    / "output/forecast_v2_beta/HeroDirt_forecast_6hourly.npz"
)

GRID_TIF = (
    ROOT
    / "static/grid/SanGabriels_DEM_200m_UTM11.tif"
)

OSM_CACHE = (
    ROOT
    / "data/osm/SanGabriels_all_trails_overpass.json"
)

AREA_CACHE = (
    ROOT
    / "data/osm/SanGabriels_trail_areas_overpass.json"
)


OUTPUT_HTML = (
    ROOT
    / "web/HeroDirt_trails.html"
)

WEB_DATA_DIR = (
    ROOT
    / "web/data"
)

WEB_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SPACING_M = 100.0

FORECAST_HOURS = [
    0,
    24,
    48,
    72,
    96,
    120,
]


# ============================================================
# HERO DIRT CLASSES
# ============================================================

CLASS_NAMES = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


CLASS_COLORS = {
    1: "#9c5a08",
    2: "#d8ad58",
    3: "#7fc7c0",
    4: "#42a89e",
    5: "#3182bd",
    6: "#08519c",
}


# ============================================================
# LOAD MODEL
# ============================================================

C = np.load(
    CURRENT_FILE
)

FCST = np.load(
    FORECAST_FILE
)


x = C[
    "x"
].astype(float)

y = C[
    "y"
].astype(float)


current_F = C[
    "F"
].astype(float)

current_score = C[
    "score"
].astype(float)

current_condition = C[
    "condition"
].astype(int)

public = C[
    "public_valid"
].astype(bool)


current_time = C[
    "time"
].astype(
    "datetime64[s]"
)

forecast_time = FCST[
    "time"
].astype(
    "datetime64[s]"
)

forecast_F = FCST[
    "F"
].astype(float)

forecast_score = FCST[
    "score"
].astype(float)

forecast_condition = FCST[
    "condition"
].astype(int)


# ============================================================
# PROJECTIONS
# ============================================================

ll_to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:26911",
    always_xy=True,
)

utm_to_ll = Transformer.from_crs(
    "EPSG:26911",
    "EPSG:4326",
    always_xy=True,
)


# ============================================================
# DOMAIN
# ============================================================

with rasterio.open(
    GRID_TIF
) as src:

    b = src.bounds


west_lon, south_lat = utm_to_ll.transform(
    b.left,
    b.bottom,
)

east_lon, north_lat = utm_to_ll.transform(
    b.right,
    b.top,
)


# ============================================================
# LOAD OSM
# ============================================================

if not OSM_CACHE.exists():

    raise RuntimeError(
        f"Missing OSM cache:\n{OSM_CACHE}"
    )


with open(
    OSM_CACHE,
    "r",
) as f:

    osm = json.load(
        f
    )

if not AREA_CACHE.exists():

    raise RuntimeError(
        f"Missing area cache:\n{AREA_CACHE}"
    )


with open(
    AREA_CACHE,
    "r",
) as f:

    osm_areas = json.load(
        f
    )
# ============================================================
# GRID LOOKUP
# ============================================================

def nearest_grid_cell(
    lon,
    lat,
):

    E, N = ll_to_utm.transform(
        lon,
        lat,
    )


    c = int(
        np.argmin(
            np.abs(
                x - E
            )
        )
    )


    r = int(
        np.argmin(
            np.abs(
                y - N
            )
        )
    )


    return r, c


# ============================================================
# GEOMETRY
# ============================================================

def segment_length_m(
    lon1,
    lat1,
    lon2,
    lat2,
):

    E1, N1 = ll_to_utm.transform(
        lon1,
        lat1,
    )

    E2, N2 = ll_to_utm.transform(
        lon2,
        lat2,
    )


    return math.hypot(
        E2 - E1,
        N2 - N1,
    )


def densify_way(
    coords,
):

    if len(coords) < 2:

        return []


    dense = [
        coords[0]
    ]


    for i in range(
        len(coords) - 1
    ):

        lon1, lat1 = coords[i]

        lon2, lat2 = coords[
            i + 1
        ]


        length = segment_length_m(
            lon1,
            lat1,
            lon2,
            lat2,
        )


        n = max(
            1,
            int(
                math.ceil(
                    length
                    /
                    SAMPLE_SPACING_M
                )
            ),
        )


        for k in range(
            1,
            n + 1
        ):

            f = (
                k
                /
                n
            )


            lon = (
                lon1
                +
                f
                *
                (
                    lon2
                    -
                    lon1
                )
            )


            lat = (
                lat1
                +
                f
                *
                (
                    lat2
                    -
                    lat1
                )
            )


            dense.append(
                (
                    lon,
                    lat,
                )
            )


    return dense
def is_trail_like(tags):

    highway = tags.get("highway", "").strip().lower()
    footway = tags.get("footway", "").strip().lower()
    surface = tags.get("surface", "").strip().lower()
    bicycle = tags.get("bicycle", "").strip().lower()
    access = tags.get("access", "").strip().lower()

    if access in {
        "no",
        "private",
        "customers",
    }:
        return False

    if footway in {
        "sidewalk",
        "crossing",
        "access_aisle",
    }:
        return False

    if surface in {
        "asphalt",
        "paved",
        "concrete",
        "concrete:plates",
        "paving_stones",
    }:
        return False

    if highway == "path":
        return True

    if highway == "track":
        return True

    if highway == "footway":

        if surface in {
            "dirt",
            "earth",
            "ground",
            "unpaved",
            "gravel",
            "fine_gravel",
            "compacted",
            "grass",
            "sand",
            "rock",
            "wood",
        }:
            return True

        if bicycle in {
            "yes",
            "designated",
            "permissive",
        }:
            return True

        return False

    if highway == "cycleway":

        if surface in {
            "dirt",
            "earth",
            "ground",
            "unpaved",
            "gravel",
            "fine_gravel",
            "compacted",
        }:
            return True

        return False

    return False

# ============================================================
# PREPROCESS TRAILS
# ============================================================

trails = []


for element in osm.get(
    "elements",
    [],
):

    if element.get(
        "type"
    ) != "way":

        continue


    tags = element.get(
        "tags",
        {},
    )

    if not is_trail_like(tags):
        continue

    name = tags.get(
        "name",
        "",
    ).strip()


    geometry = element.get(
        "geometry",
        [],
    )

    if len(geometry) < 2:
       continue

    coords = [
        (
            p["lon"],
            p["lat"],
        )
        for p in geometry
    ]


    dense = densify_way(
        coords
    )


    samples = []


    for lon, lat in dense:

        r, c = nearest_grid_cell(
            lon,
            lat,
        )


        if not public[
            r,
            c
        ]:

            continue


        samples.append(
            {
                "lon":
                    lon,

                "lat":
                    lat,

                "r":
                    r,

                "c":
                    c,
            }
        )

    if len(samples) < 2:
        continue

    surface = tags.get(
        "surface",
        "",
    ).lower()

    paved_surfaces = {
        "asphalt",
        "paved",
        "concrete",
        "concrete:plates",
        "paving_stones",
    }

    if surface in paved_surfaces:
        continue

    trails.append(
        {
            "name": name,
            "osm_id": element["id"],
            "highway": tags.get("highway", ""),
            "surface": surface,
            "bicycle": tags.get("bicycle", ""),
            "mtb_scale": tags.get("mtb:scale", ""),
            "samples": samples,
        }
    )
# ============================================================
# TRAIL SEARCH INDEX
# ============================================================

trail_search_index = {}


for trail in trails:

    name = trail[
        "name"
    ]

    if not name:
        continue

    lats = [
        p["lat"]
        for p in trail["samples"]
    ]

    lons = [
        p["lon"]
        for p in trail["samples"]
    ]


    if not lats:

        continue


    south = min(
        lats
    )

    north = max(
        lats
    )

    west = min(
        lons
    )

    east = max(
        lons
    )


    if name not in trail_search_index:

        trail_search_index[
            name
        ] = {
            "south":
                south,

            "north":
                north,

            "west":
                west,

            "east":
                east,
        }


    else:

        B = trail_search_index[
            name
        ]


        B["south"] = min(
            B["south"],
            south,
        )

        B["north"] = max(
            B["north"],
            north,
        )

        B["west"] = min(
            B["west"],
            west,
        )

        B["east"] = max(
            B["east"],
            east,
        )

# ============================================================
# AREA / PARK SEARCH INDEX
# ============================================================

area_search_index = {}


for element in osm_areas.get(
    "elements",
    [],
):

    tags = element.get(
        "tags",
        {},
    )

    name = tags.get(
        "name",
        "",
    ).strip()

    if not name:
        continue

    center = element.get(
        "center",
        {},
    )

    lat = center.get(
        "lat"
    )

    lon = center.get(
        "lon"
    )

    if (
        lat is None
        or lon is None
    ):
        continue

    key = name.lower()

    if key not in area_search_index:

        area_search_index[
            key
        ] = {
            "name": name,
            "lat": float(lat),
            "lon": float(lon),
            "count": 1,
        }

    else:

        A = area_search_index[
            key
        ]

        n = A[
            "count"
        ]

        A[
            "lat"
        ] = (
            A["lat"] * n
            + float(lat)
        ) / (
            n + 1
        )

        A[
            "lon"
        ] = (
            A["lon"] * n
            + float(lon)
        ) / (
            n + 1
        )

        A[
            "count"
        ] = n + 1

print()
print(
    "============================================"
)
print(
    " HERO DIRT TRAIL FORECAST"
)
print(
    "============================================"
)
print()

print(
    f"Usable named OSM ways: "
    f"{len(trails)}"
)

print(
    f"Unique searchable trail names: "
    f"{len(trail_search_index)}"
)

print()

print(
    f"Unique searchable areas: "
    f"{len(area_search_index)}"
)
# ============================================================
# FORECAST LOOKUP
# ============================================================

def get_state_for_hour(
    hour,
):

    if hour == 0:

        return (
            current_time,
            current_F,
            current_score,
            current_condition,
        )


    target_time = (
        current_time
        +
        np.timedelta64(
            hour,
            "h",
        )
    )


    k = int(
        np.argmin(
            np.abs(
                forecast_time
                -
                target_time
            )
        )
    )


    return (
        forecast_time[
            k
        ],

        forecast_F[
            k
        ],

        forecast_score[
            k
        ],

        forecast_condition[
            k
        ],
    )


# ============================================================
# CONDITION CHUNKS
# ============================================================

def make_condition_chunks(
    trail_samples,
    condition_array,
):

    samples = []


    for p in trail_samples:

        cid = int(
            condition_array[
                p["r"],
                p["c"]
            ]
        )


        if not (
            1 <= cid <= 6
        ):

            continue


        samples.append(
            {
                "lat":
                    p["lat"],

                "lon":
                    p["lon"],

                "class":
                    cid,
            }
        )


    if len(samples) < 2:

        return []


    chunks = []


    current_class = samples[
        0
    ][
        "class"
    ]


    current_coords = [
        (
            samples[
                0
            ][
                "lat"
            ],
            samples[
                0
            ][
                "lon"
            ],
        )
    ]


    for i in range(
        1,
        len(samples)
    ):

        point = samples[
            i
        ]


        coord = (
            point[
                "lat"
            ],
            point[
                "lon"
            ],
        )


        if (
            point[
                "class"
            ]
            ==
            current_class
        ):

            current_coords.append(
                coord
            )


        else:

            current_coords.append(
                coord
            )


            chunks.append(
                {
                    "class":
                        current_class,

                    "coords":
                        current_coords,
                }
            )


            previous = samples[
                i - 1
            ]


            current_coords = [
                (
                    previous[
                        "lat"
                    ],
                    previous[
                        "lon"
                    ],
                ),
                coord,
            ]


            current_class = (
                point[
                    "class"
                ]
            )


    chunks.append(
        {
            "class":
                current_class,

            "coords":
                current_coords,
        }
    )


    return chunks


# ============================================================
# CREATE MAP
# ============================================================

center_lat = (
    south_lat
    +
    north_lat
) / 2.0

center_lon = (
    west_lon
    +
    east_lon
) / 2.0


m = folium.Map(

    location=[
        center_lat,
        center_lon,
    ],

    zoom_start=10,

    tiles=None,

    control_scale=True,

    prefer_canvas=True,
)


# ============================================================
# MOBILE CSS
# ============================================================

mobile_css = """
<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0,
               maximum-scale=1.0,
               user-scalable=yes">

<style>

#hero-title-box,
#hero-search-box,
#hero-legend {
    box-sizing:border-box;
}


/* =========================================================
   PHONE
   ========================================================= */

@media (max-width:700px) {

    #hero-title-box {
        top:8px !important;
        left:8px !important;
        right:auto !important;
        width:185px !important;
        padding:7px 9px !important;
    }

    #hero-title-box b {
        font-size:14px !important;
    }

    #hero-title-box span {
        font-size:9px !important;
    }


    #hero-about-button {
        top:65px !important;
        left:8px !important;
        padding:6px 9px !important;
        font-size:11px !important;
    }


    #hero-explorer-button {
        top:65px !important;
        left:76px !important;
        padding:6px 9px !important;
        font-size:11px !important;
    }


    #hero-search-box {
        top:122px !important;
        left:8px !important;
	right:auto !important;
	width:220px !important;
      	max-width:calc(100vw - 16px) !important;
      	box-sizing:border-box !important;
    }


    #hero-trail-search {
        width:100% !important;
        box-sizing:border-box !important;
        padding:8px !important;
        font-size:14px !important;
    }


    #hero-search-results {
        width:100% !important;
        max-height:180px !important;
        overflow-y:auto !important;
    }


    #hero-legend {
        left:8px !important;
        bottom:58px !important;
        width:145px !important;
        padding:8px 10px !important;
        font-size:10px !important;
    }


    .leaflet-control-layers {
        max-width:175px !important;
        max-height:45vh !important;
        overflow-y:auto !important;
        font-size:11px !important;
    }


    .leaflet-popup-content-wrapper {
        max-width:90vw !important;
    }


    .leaflet-popup-content {
        margin:10px 12px !important;
        font-size:12px !important;
    }


    #hero-about-modal > div,
    #hero-report-modal > div {

        width:calc(100% - 24px) !important;
        max-width:none !important;
        max-height:calc(100vh - 24px) !important;

        margin:12px auto !important;

        padding:18px !important;

        box-sizing:border-box !important;

        overflow-y:auto !important;
    }

}


/* =========================================================
   SMALL PHONE
   ========================================================= */

@media (max-width:420px) {

    #hero-legend {
        width:128px !important;
        font-size:9px !important;
    }

}

</style>
"""


m.get_root().header.add_child(
    folium.Element(
        mobile_css
    )
)


# ============================================================
# BASEMAPS
# ============================================================

folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Topo_Map/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr="Tiles © Esri",

    name="Esri World Topo",

    overlay=False,

    control=True,

    show=True,

).add_to(
    m
)


folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}"
    ),

    attr="Tiles © Esri",

    name="Esri Satellite",

    overlay=False,

    control=True,

    show=False,

).add_to(
    m
)


# ============================================================
# FORECAST TRAIL LAYERS
# ============================================================

trail_line_metadata = {}

forecast_layers = []

layer_stats = []


for hour in FORECAST_HOURS:

    (
        valid_time,
        F_array,
        score_array,
        condition_array,
    ) = get_state_for_hour(
        hour
    )


    label = (
        "Now"
        if hour == 0
        else f"+{hour} h"
    )


    fg = folium.FeatureGroup(

        name=label,

        show=(
            hour == 0
        ),

        overlay=True,
    )


    forecast_layers.append(
        fg
    )


    chunk_count = 0

    geojson_features = []

    for trail in trails:

        samples = trail[
            "samples"
        ]


        Fs = []

        scores = []

        classes = []


        for p in samples:

            r = p[
                "r"
            ]

            c = p[
                "c"
            ]


            cid = int(
                condition_array[
                    r,
                    c
                ]
            )


            if not (
                1 <= cid <= 6
            ):

                continue


            Fs.append(
                float(
                    F_array[
                        r,
                        c
                    ]
                )
            )


            scores.append(
                float(
                    score_array[
                        r,
                        c
                    ]
                )
            )


            classes.append(
                cid
            )


        if len(classes) < 2:

            continue


        median_F = float(
            np.nanmedian(
                Fs
            )
        )


        median_score = float(
            np.nanmedian(
                scores
            )
        )


        chunks = make_condition_chunks(
            samples,
            condition_array,
        )


        for chunk in chunks:

            cid = chunk[
                "class"
            ]

            coords_lonlat = [
                [
                    float(lon),
                    float(lat),
                ]
                for lat, lon in chunk[
                    "coords"
                ]
            ]

            geojson_features.append(
                {
                    "type": "Feature",

                    "properties": {
                        "trail":
                            trail[
                                "name"
                            ],

                        "forecast_label":
                            label,

                        "valid_time":
                            str(
                                valid_time
                            ),

                        "local_condition":
                            CLASS_NAMES[
                                cid
                            ],

                        "median_F":
                            round(
                                median_F,
                                3,
                            ),

                        "median_score":
                            round(
                                median_score,
                                1,
                            ),

                        "color":
                            CLASS_COLORS[
                                cid
                            ],
                    },

                    "geometry": {
                        "type": "LineString",

                        "coordinates":
                            coords_lonlat,
                    },
                }
            )

            chunk_count += 1
    hour_tag = (
        "now"
        if hour == 0
        else f"plus{hour}"
    )

    geojson_name = (
        f"trails_{hour_tag}.geojson"
    )

    geojson_path = (
        WEB_DATA_DIR
        / geojson_name
    )

    with open(
        geojson_path,
        "w",
    ) as f:

        json.dump(
            {
                "type":
                    "FeatureCollection",

                "features":
                    geojson_features,
            },
            f,
            separators=(
                ",",
                ":",
            ),
        )

    fg.geojson_url = (
        f"data/{geojson_name}"
    )

    fg.add_to(
        m
    )


    layer_stats.append(
        (
            label,
            chunk_count,
            valid_time,
        )
    )


forecast_geojson_config = []

for fg in forecast_layers:

    forecast_geojson_config.append(
        {
            "layer":
                fg.get_name(),

            "url":
                fg.geojson_url,
        }
    )

forecast_geojson_js = json.dumps(
    forecast_geojson_config
)


# ============================================================
# TITLE
# ============================================================

title_html = f"""
<div
    id="hero-title-box"
    style="
        position:fixed;
        top:12px;
        left:55px;
        z-index:9999;
        background-color:rgba(255,255,255,0.94);
        padding:10px 15px;
        border-radius:7px;
        font-family:Arial;
        box-shadow:0px 1px 5px rgba(0,0,0,0.35);
    "
>

<b style="
    font-size:18px;
">
Hero Dirt Forecast
</b>

<br>

<span style="
    font-size:12px;
">
Analysis: {current_time} UTC
</span>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        title_html
    )
)


# ============================================================
# ABOUT BUTTON
# ============================================================

about_button_html = """
<button
    id="hero-about-button"
    style="
        position:fixed;
        top:15px;
        left:300px;
        z-index:10001;
        padding:8px 14px;
        border:1px solid #999;
        border-radius:6px;
        background:rgba(255,255,255,0.96);
        color:#222;
        font-family:Arial;
        font-size:13px;
        font-weight:bold;
        cursor:pointer;
        box-shadow:0px 1px 5px rgba(0,0,0,0.25);
    "
>
About
</button>
"""


m.get_root().html.add_child(
    folium.Element(
        about_button_html
    )
)


# ============================================================
# EXPLORER BUTTON
# ============================================================

explorer_button_html = """
<a
    id="hero-explorer-button"
    href="explorer.html"
    style="
        position:fixed;
        top:15px;
        left:390px;
        z-index:10001;
        padding:8px 14px;
        border:1px solid #999;
        border-radius:6px;
        background:rgba(255,255,255,0.96);
        color:#222;
        font-family:Arial;
        font-size:13px;
        font-weight:bold;
        text-decoration:none;
        cursor:pointer;
        box-shadow:0px 1px 5px rgba(0,0,0,0.25);
    "
>
Explorer
</a>
"""


m.get_root().html.add_child(
    folium.Element(
        explorer_button_html
    )
)


# ============================================================
# TRAIL SEARCH
# ============================================================

search_html = """
<div
    id="hero-search-box"
    style="
        position:fixed;
        top:105px;
        left:55px;
        width:220px;
        z-index:10002;
        font-family:Arial;
    "
>

<input
    id="hero-trail-search"
    type="text"
    placeholder="Search parks or trails..."
    autocomplete="off"
    style="
        width:220px;
        padding:8px 10px;
        border:1px solid #999;
        border-radius:6px;
        background:rgba(255,255,255,0.96);
        font-family:Arial;
        font-size:13px;
        box-shadow:0px 1px 5px rgba(0,0,0,0.25);
        outline:none;
    "
>

<div
    id="hero-search-results"
    style="
        display:none;
        background:white;
        border:1px solid #aaa;
        border-top:none;
        border-radius:0 0 6px 6px;
        box-shadow:0px 2px 5px rgba(0,0,0,0.25);
        width:260px;
        max-height:220px;
        overflow-y:auto;
    "
>
</div>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        search_html
    )
)


# ============================================================
# LEGEND
# ============================================================

legend_html = """
<div
    id="hero-legend"
    style="
        position:fixed;
        bottom:35px;
        left:35px;
        z-index:9998;
        background-color:rgba(255,255,255,0.94);
        padding:12px 15px;
        border-radius:7px;
        font-family:Arial;
        font-size:13px;
        box-shadow:0px 1px 5px rgba(0,0,0,0.35);
    "
>

<b>Trail Condition</b>

<br><br>

<span style="color:#9c5a08;">●</span>
Very Dry / Dusty<br>

<span style="color:#d8ad58;">●</span>
Dry / Firm<br>

<span style="color:#8fcfc8;">●</span>
Moist / Good<br>

<span style="color:#42a89e;">●</span>
Hero<br>

<span style="color:#3182bd;">●</span>
Wet<br>

<span style="color:#08519c;">●</span>
Too Wet

<br><br>

<span style="
    font-size:10px;
    color:#666;
">
Click a trail to inspect or report conditions.
</span>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# ABOUT MODAL
# ============================================================

about_modal_html = """
<div
    id="hero-about-modal"
    style="
        display:none;
        position:fixed;
        z-index:30000;
        left:0;
        top:0;
        width:100%;
        height:100%;
        background:rgba(0,0,0,0.45);
        font-family:Arial;
    "
>

<div
    style="
        background:white;
        width:640px;
        max-width:88%;
        max-height:82%;
        overflow-y:auto;
        margin:4% auto;
        padding:25px 30px;
        border-radius:10px;
        box-shadow:0px 3px 15px rgba(0,0,0,0.45);
        line-height:1.52;
        color:#222;
    "
>

<div style="
    display:flex;
    justify-content:space-between;
    align-items:center;
">

<div>

<b style="
    font-size:22px;
">
About Hero Dirt Forecast
</b>

<br>

<span style="
    color:#666;
    font-size:12px;
">
Experimental trail-moisture forecast for the San Gabriel Mountains
</span>

</div>

<span
    id="hero-about-close"
    style="
        cursor:pointer;
        font-size:28px;
        line-height:1;
    "
>
&times;
</span>

</div>


<hr style="
    border:none;
    border-top:1px solid #ddd;
    margin:16px 0;
">


<p>
Hero Dirt Forecast uses a
<b>three-layer soil-water model</b>
to estimate trail-surface moisture across the San Gabriel Mountains.
The model represents the
<b>trail tread (0–2 cm)</b>,
<b>shallow soil (2–5 cm)</b>,
and
<b>deeper soil (5–20 cm)</b>.
Water can infiltrate into the tread, move downward through the
soil profile, drain from the deeper layer, and exchange between
the shallow soil and trail surface through time.
</p>


<h3>Landscape information</h3>
<div style="
    background:#eef6f4;
    padding:12px 15px;
    border-radius:6px;
    margin:14px 0 18px 0;
    font-size:12px;
">

<b>v2 Beta</b>

<br><br>

This version uses a Penman-style atmospheric drying formulation
for the trail-tread layer during both the analysis and forecast
periods. Drying responds to temperature, humidity, wind,
solar geometry, cloud cover, terrain exposure, and vegetation.

<br><br>

The tread drying coefficient is currently provisional and is being
evaluated against field observations. Freeze-thaw processes are not
yet represented explicitly.

</div>
<p>
The model accounts for spatial differences in
<b>soil properties, terrain, and vegetation</b>.
Soil information includes sand and clay content,
field capacity, wilting point, saturation water content,
saturated hydraulic conductivity
(<b>K<sub>sat</sub></b>),
and
<b>Plasticity Index (PI)</b>.
Terrain information includes elevation, slope, and terrain
exposure, while vegetation includes tree canopy, shrub, and
herbaceous cover.
</p>


<h3>Observed and forecast weather</h3>

<p>
The regional soil-moisture state is constrained using
<b>NASA SMAP (Soil Moisture Active Passive)</b>
satellite observations.
SMAP provides a broad regional estimate of upper-soil moisture
and is not a direct measurement of the trail tread.
</p>

<p>
Recent rainfall is ingested from
<b>NOAA MRMS (Multi-Radar/Multi-Sensor System)</b>
precipitation analyses.
</p>

Future conditions are driven by
<b>National Weather Service (NWS)</b>
forecasts including
<b>air temperature, relative humidity, wind, cloud cover,
and precipitation</b>.

<h3>How the Hero Dirt score works</h3>

<p>
The Hero Dirt score is based primarily on modeled water content
in the upper 2 cm of soil.
Tread moisture is evaluated relative to the hydraulic
properties of the local soil.
</p>


<div style="
    text-align:center;
    font-family:serif;
    font-size:18px;
    margin:12px 0;
">
F =
(&theta;<sub>tread</sub> -
&theta;<sub>WP</sub>)
/
(&theta;<sub>FC</sub> -
&theta;<sub>WP</sub>)
</div>


<div style="
    background:#f5f5f5;
    padding:11px 14px;
    border-radius:6px;
    font-size:12px;
    margin-bottom:15px;
">

<b>where:</b>

<br><br>

<b>&theta;<sub>tread</sub></b> =
volumetric water content of the modeled 0–2 cm trail-tread layer

<br>

<b>&theta;<sub>WP</sub></b> =
local soil water content at wilting point

<br>

<b>&theta;<sub>FC</sub></b> =
local soil water content at field capacity

<br>

<b>F</b> =
dimensionless normalized hydraulic state

</div>


<p>
F = 0 corresponds approximately to wilting point and
F = 1 to field capacity.
</p>


<div style="
    text-align:center;
    font-family:serif;
    font-size:18px;
    margin:12px 0;
">
H =
100
&times;
S<sub>dry</sub>
&times;
S<sub>wet</sub>
&times;
S<sub>deep</sub>
</div>


<div style="
    background:#f5f5f5;
    padding:11px 14px;
    border-radius:6px;
    font-size:12px;
    margin-bottom:15px;
">

<b>where:</b>

<br><br>

<b>H</b> =
Hero Dirt score, approximately 0–100

<br>

<b>S<sub>dry</sub></b> =
penalty for excessively dry trail tread

<br>

<b>S<sub>wet</sub></b> =
penalty for excessively wet or deformable tread

<br>

<b>S<sub>deep</sub></b> =
penalty when deeper soil remains wet even after the immediate
surface begins drying

</div>


<h3>Why soil texture and plasticity matter</h3>

<p>
The wet-condition penalty depends in part on
<b>soil texture and soil plasticity</b>.
Texture describes the relative abundance of sand, silt, and clay,
which influence infiltration, drainage, and water retention.
</p>

<p>
Plasticity is represented using the
<b>Plasticity Index (PI)</b>,
an engineering measure of the range of water contents over which
fine-grained soil behaves plastically.
More plastic soils receive a stronger wet-condition penalty because
they are more likely to become soft or deformable as moisture rises.
</p>


<h3>Trail-condition classes</h3>

<p>
The modeled state is translated into six classes:
<b>Very Dry / Dusty, Dry / Firm, Moist / Good, Hero, Wet,</b>
and
<b>Too Wet</b>.
<b>Dry / Firm, Moist / Good, and Hero are generally favorable riding conditions.</b>
Current score relationships and thresholds are empirical and
will be evaluated against rider observations.
</p>

<div style="
    background:#eef3f4;
    padding:12px 15px;
    border-radius:6px;
    margin-top:19px;
    font-size:12px;
">

<b>Primary model inputs</b>

<br><br>

<b>SMAP</b> —
NASA Soil Moisture Active Passive satellite soil moisture

<br>

<b>MRMS</b> —
NOAA Multi-Radar/Multi-Sensor observed precipitation

<br>

<b>NWS</b> —
National Weather Service forecast temperature,
relative humidity, and 6-hour precipitation

<br>

<b>Landscape properties</b> —
soil hydraulic properties, texture and plasticity,
terrain, slope, and vegetation

</div>


<div style="
    background:#f3f3f3;
    padding:12px 15px;
    border-radius:6px;
    margin-top:14px;
    font-size:12px;
    color:#555;
">

<b>Experimental forecast</b>

<br><br>

Hero Dirt Forecast operates on an approximately
<b>200-m landscape grid</b>.
Landscape data therefore describe the environment surrounding the
trail rather than the exact engineered trail tread.

<br><br>

Trail construction, drainage, rock exposure, soil compaction,
canopy shading, microtopography, and localized precipitation may
cause actual conditions to differ from the model.

<br><br>

This is a <b>beta forecast</b>. Field validation is ongoing, and
individual trails may differ substantially from the surrounding
200-m model grid cell.

</div>

</div>
</div>
"""


m.get_root().html.add_child(
    folium.Element(
        about_modal_html
    )
)


# ============================================================
# REPORT MODAL
# ============================================================

report_modal_html = """
<div
    id="hero-report-modal"
    style="
        display:none;
        position:fixed;
        z-index:30000;
        left:0;
        top:0;
        width:100%;
        height:100%;
        background:rgba(0,0,0,0.45);
        font-family:Arial;
    "
>

<div
    style="
        background:white;
        width:380px;
        max-width:90%;
        margin:6% auto;
        padding:20px;
        border-radius:9px;
        box-shadow:0px 3px 15px rgba(0,0,0,0.45);
    "
>

<div style="
    display:flex;
    justify-content:space-between;
    align-items:center;
">

<div>

<b style="
    font-size:19px;
">
Report Trail Conditions
</b>

<br>

<span
    id="hero-report-trail-name"
    style="
        font-size:13px;
        color:#555;
    "
></span>

</div>

<span
    id="hero-report-close"
    style="
        cursor:pointer;
        font-size:25px;
    "
>
&times;
</span>

</div>


<hr style="
    border:none;
    border-top:1px solid #ddd;
    margin:15px 0;
">


<label>
Ride time
</label>

<br>

<input
    id="report-time"
    type="datetime-local"
    style="
        width:100%;
        box-sizing:border-box;
        margin-top:4px;
        margin-bottom:14px;
        padding:7px;
    "
>


<label>
How was the trail?
</label>

<br>

<select
    id="report-condition"
    style="
        width:100%;
        box-sizing:border-box;
        margin-top:4px;
        margin-bottom:14px;
        padding:7px;
    "
>

<option value="Very Dry">
Dusty / loose
</option>

<option value="Dry">
Dry but firm
</option>

<option value="Moist/Good">
Good
</option>

<option value="Hero">
Hero
</option>

<option value="Wet">
Damp but rideable
</option>

<option value="Too Wet">
Muddy / too wet
</option>

</select>


<label>
Comments
<span style="
    color:#777;
    font-size:11px;
">
(optional)
</span>
</label>

<br>

<textarea
    id="report-comment"
    rows="3"
    placeholder="e.g. pretty dry, but not dusty"
    style="
        width:100%;
        box-sizing:border-box;
        margin-top:4px;
        margin-bottom:14px;
        padding:7px;
    "
></textarea>


<label>
Name or initials
<span style="
    color:#777;
    font-size:11px;
">
(optional)
</span>
</label>

<br>

<input
    id="report-name"
    type="text"
    style="
        width:100%;
        box-sizing:border-box;
        margin-top:4px;
        margin-bottom:16px;
        padding:7px;
    "
>


<button
    id="report-download"
    style="
        width:100%;
        padding:11px;
        border:none;
        border-radius:6px;
        background:#333333;
        color:white;
        font-size:14px;
        font-weight:bold;
        cursor:pointer;
    "
>
Save Report
</button>


<p
    id="report-status"
    style="
        font-size:11px;
        color:#555;
        margin-top:10px;
        margin-bottom:0;
    "
></p>

</div>
</div>
"""


m.get_root().html.add_child(
    folium.Element(
        report_modal_html
    )
)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(
    m
)


# ============================================================
# JAVASCRIPT DATA
# ============================================================

metadata_js = json.dumps(
    trail_line_metadata
)

trail_search_js = json.dumps(
    trail_search_index
)

area_search_js = json.dumps(
    area_search_index
)

forecast_layer_names = [
    layer.get_name()
    for layer in forecast_layers
]


forecast_layer_js = (
    "["
    +
    ",".join(
        forecast_layer_names
    )
    +
    "]"
)


map_name = m.get_name()



geojson_loader_js = r"""
        var forecastGeoJSON =
            __FORECAST_GEOJSON_CONFIG__;


        function addExternalTrailLayer(
            targetGroup,
            url
        ) {

            fetch(url)

            .then(function(response) {

                if (!response.ok) {

                    throw new Error(
                        "Could not load "
                        + url
                    );

                }

                return response.json();

            })

            .then(function(data) {

                L.geoJSON(
                    data,
                    {
                        style:
                            function(feature) {

                                return {
                                    color:
                                        feature.properties.color,

                                    weight:
                                        5,

                                    opacity:
                                        0.90
                                };

                            },


                        onEachFeature:
                            function(
                                feature,
                                layer
                            ) {

                                var meta =
                                    feature.properties;


                                layer.bindTooltip(
                                    meta.trail
                                    +
                                    " — "
                                    +
                                    meta.local_condition
                                    +
                                    " ("
                                    +
                                    meta.forecast_label
                                    +
                                    ")"
                                );


                                layer.on(
                                    "click",
                                    function(e) {

                                        L.DomEvent.stopPropagation(
                                            e
                                        );


                                        var lat =
                                            e.latlng.lat;

                                        var lon =
                                            e.latlng.lng;


                                        var popupHTML =
                                            "<div style='font-family:Arial;min-width:245px;'>"
                                            +
                                            "<b style='font-size:15px;'>"
                                            +
                                            meta.trail
                                            +
                                            "</b>"
                                            +
                                            "<br><br>"
                                            +
                                            "<b>Forecast:</b> "
                                            +
                                            meta.forecast_label
                                            +
                                            "<br>"
                                            +
                                            "<b>Valid:</b> "
                                            +
                                            meta.valid_time
                                            +
                                            "<br><br>"
                                            +
                                            "<b>This segment:</b> "
                                            +
                                            meta.local_condition
                                            +
                                            "<br>"
                                            +
                                            "<b>Trail median F:</b> "
                                            +
                                            Number(
                                                meta.median_F
                                            ).toFixed(2)
                                            +
                                            "<br>"
                                            +
                                            "<b>Trail median score:</b> "
                                            +
                                            Number(
                                                meta.median_score
                                            ).toFixed(0)
                                            +
                                            "<br><br>"
                                            +
                                            "<button "
                                            +
                                            "id='hero-popup-report-button' "
                                            +
                                            "style='width:100%;padding:9px;border:none;border-radius:5px;background:#333;color:white;font-weight:bold;cursor:pointer;'>"
                                            +
                                            "Report conditions here"
                                            +
                                            "</button>"
                                            +
                                            "</div>";


                                        L.popup()
                                        .setLatLng(
                                            e.latlng
                                        )
                                        .setContent(
                                            popupHTML
                                        )
                                        .openOn(
                                            map
                                        );


                                        setTimeout(
                                            function() {

                                                var button =
                                                    document.getElementById(
                                                        "hero-popup-report-button"
                                                    );


                                                if (
                                                    button !== null
                                                ) {

                                                    button.onclick =
                                                        function() {

                                                            map.closePopup();

                                                            window.openHeroReport(
                                                                meta.trail,
                                                                lat,
                                                                lon
                                                            );

                                                        };

                                                }

                                            },
                                            30
                                        );

                                    }
                                );

                            }
                    }
                ).addTo(
                    targetGroup
                );

            })

            .catch(function(error) {

                console.error(
                    "Hero Dirt trail layer error:",
                    error
                );

            });

        }


        // ====================================================
        // LAZY-LOAD FORECAST TRAIL DATA
        // ====================================================

        function ensureForecastLayerLoaded(
            cfg
        ) {

            var targetGroup =
                window[
                    cfg.layer
                ];


            if (
                targetGroup === undefined
                ||
                cfg.loaded === true
                ||
                cfg.loading === true
            ) {

                return;

            }


            cfg.loading = true;

            addExternalTrailLayer(
                targetGroup,
                cfg.url
            );

            cfg.loaded = true;
            cfg.loading = false;

        }


        forecastGeoJSON.forEach(
            function(cfg) {

                cfg.loaded = false;
                cfg.loading = false;


                var targetGroup =
                    window[
                        cfg.layer
                    ];


                if (
                    targetGroup !== undefined
                    &&
                    map.hasLayer(
                        targetGroup
                    )
                ) {

                    ensureForecastLayerLoaded(
                        cfg
                    );

                }

            }
        );


        map.on(
            "overlayadd",
            function(e) {

                forecastGeoJSON.forEach(
                    function(cfg) {

                        var targetGroup =
                            window[
                                cfg.layer
                            ];


                        if (
                            targetGroup ===
                            e.layer
                        ) {

                            // Keep only one forecast dataset
                            // resident in memory at a time.
                            forecastGeoJSON.forEach(
                                function(otherCfg) {

                                    var otherGroup =
                                        window[
                                            otherCfg.layer
                                        ];


                                    if (
                                        otherGroup !== undefined
                                        &&
                                        otherGroup !== targetGroup
                                    ) {

                                        if (
                                            map.hasLayer(
                                                otherGroup
                                            )
                                        ) {

                                            map.removeLayer(
                                                otherGroup
                                            );

                                        }


                                        otherGroup.clearLayers();

                                        otherCfg.loaded = false;
                                        otherCfg.loading = false;

                                    }

                                }
                            );


                            ensureForecastLayerLoaded(
                                cfg
                            );

                        }

                    }
                );

            }
        );


        map.on(
            "overlayremove",
            function(e) {

                forecastGeoJSON.forEach(
                    function(cfg) {

                        var targetGroup =
                            window[
                                cfg.layer
                            ];


                        if (
                            targetGroup ===
                            e.layer
                        ) {

                            targetGroup.clearLayers();

                            cfg.loaded = false;
                            cfg.loading = false;

                        }

                    }
                );

            }
        );
"""

geojson_loader_js = (
    geojson_loader_js.replace(
        "__FORECAST_GEOJSON_CONFIG__",
        forecast_geojson_js,
    )
)



forecast_radio_js = r"""
        // ====================================================
        // FORECAST TIMES: MUTUALLY EXCLUSIVE RADIO BUTTONS
        // ====================================================

        function makeForecastControlsRadio() {

            var forecastNames = [
                "Now",
                "+24 h",
                "+48 h",
                "+72 h",
                "+96 h",
                "+120 h"
            ];


            var labels =
                document.querySelectorAll(
                    ".leaflet-control-layers-overlays label"
                );


            labels.forEach(
                function(label) {

                    var labelText =
                        label.textContent.trim();


                    if (
                        forecastNames.includes(
                            labelText
                        )
                    ) {

                        var input =
                            label.querySelector(
                                "input"
                            );


                        if (
                            input !== null
                        ) {

                            input.type =
                                "radio";

                            input.name =
                                "hero-forecast-time";

                        }

                    }

                }
            );

        }


        setTimeout(
            makeForecastControlsRadio,
            100
        );
"""

# ============================================================
# JAVASCRIPT
# ============================================================

page_js = f"""
<script>

document.addEventListener(
    "DOMContentLoaded",
    function() {{

        var map =
            {map_name};


        var trailMetadata =
            {metadata_js};


        var trailSearchIndex =
            {trail_search_js};
   
        var areaSearchIndex =
            {area_search_js};

        var forecastLayers =
            {forecast_layer_js};


{geojson_loader_js}

{forecast_radio_js}


        // ====================================================
        // TRAIL SEARCH
        // ====================================================

        var searchInput =
            document.getElementById(
                "hero-trail-search"
            );


        var searchResults =
            document.getElementById(
                "hero-search-results"
            );


        var allTrailNames =
            Object.keys(
                trailSearchIndex
            ).sort();


        var allAreaKeys =
            Object.keys(
                areaSearchIndex
            );


        function zoomToTrail(
            trailName
        ) {{

            var b =
                trailSearchIndex[
                    trailName
                ];

            if (!b) {{
                return;
            }}

            var latPad =
                Math.max(
                    (
                        b.north
                        -
                        b.south
                    )
                    *
                    0.20,
                    0.002
                );

            var lonPad =
                Math.max(
                    (
                        b.east
                        -
                        b.west
                    )
                    *
                    0.20,
                    0.002
                );

            map.fitBounds(
                [
                    [
                        b.south - latPad,
                        b.west - lonPad
                    ],
                    [
                        b.north + latPad,
                        b.east + lonPad
                    ]
                ]
            );

            searchInput.value =
                trailName;

            searchResults.style.display =
                "none";
        }}


        function zoomToArea(
            areaKey
        ) {{

            var a =
                areaSearchIndex[
                    areaKey
                ];

            if (!a) {{
                return;
            }}

            map.setView(
                [
                    a.lat,
                    a.lon
                ],
                15
            );

            searchInput.value =
                a.name;

            searchResults.style.display =
                "none";
        }}


        function makeSearchRow(
            label,
            subtitle,
            onclick
        ) {{

            var row =
                document.createElement(
                    "div"
                );

            row.style.padding =
                "8px 10px";

            row.style.cursor =
                "pointer";

            row.style.borderBottom =
                "1px solid #eee";

            row.onmouseenter =
                function() {{
                    row.style.background =
                        "#f2f2f2";
                }};

            row.onmouseleave =
                function() {{
                    row.style.background =
                        "white";
                }};

            var main =
                document.createElement(
                    "div"
                );

            main.innerText =
                label;

            main.style.fontWeight =
                "600";

            row.appendChild(
                main
            );

            if (subtitle) {{

                var sub =
                    document.createElement(
                        "div"
                    );

                sub.innerText =
                    subtitle;

                sub.style.fontSize =
                    "11px";

                sub.style.color =
                    "#777";

                sub.style.marginTop =
                    "2px";

                row.appendChild(
                    sub
                );
            }}

            row.onclick =
                onclick;

            return row;
        }}


        function updateTrailSearch() {{

            var query =
                searchInput
                .value
                .trim()
                .toLowerCase();

            searchResults.innerHTML =
                "";

            if (
                query.length
                <
                2
            ) {{

                searchResults.style.display =
                    "none";

                return;
            }}


            var areaMatches =
                allAreaKeys
                .filter(
                    function(key) {{

                        return areaSearchIndex[
                            key
                        ].name
                            .toLowerCase()
                            .includes(
                                query
                            );

                    }}
                )
                .slice(
                    0,
                    6
                );


            var trailMatches =
                allTrailNames
                .filter(
                    function(name) {{

                        return name
                            .toLowerCase()
                            .includes(
                                query
                            );

                    }}
                )
                .slice(
                    0,
                    6
                );


            if (
                areaMatches.length
                ===
                0
                &&
                trailMatches.length
                ===
                0
            ) {{

                searchResults.innerHTML =
                    "<div style='padding:8px;color:#777;'>"
                    +
                    "No matching park or trail"
                    +
                    "</div>";

                searchResults.style.display =
                    "block";

                return;
            }}


            areaMatches.forEach(
                function(key) {{

                    var a =
                        areaSearchIndex[
                            key
                        ];

                    var row =
                        makeSearchRow(
                            a.name,
                            "Park / riding area",
                            function() {{

                                zoomToArea(
                                    key
                                );

                            }}
                        );

                    searchResults.appendChild(
                        row
                    );

                }}
            );


            trailMatches.forEach(
                function(name) {{

                    var row =
                        makeSearchRow(
                            name,
                            "Trail",
                            function() {{

                                zoomToTrail(
                                    name
                                );

                            }}
                        );

                    searchResults.appendChild(
                        row
                    );

                }}
            );


            searchResults.style.display =
                "block";
        }}


        searchInput.addEventListener(
            "input",
            updateTrailSearch
        );


        searchInput.addEventListener(
            "keydown",
            function(e) {{

                if (
                    e.key
                    !==
                    "Enter"
                ) {{
                    return;
                }}

                var query =
                    searchInput
                    .value
                    .trim()
                    .toLowerCase();


                var exactArea =
                    allAreaKeys
                    .find(
                        function(key) {{

                            return areaSearchIndex[
                                key
                            ].name
                                .toLowerCase()
                                ===
                                query;

                        }}
                    );

                if (exactArea) {{

                    zoomToArea(
                        exactArea
                    );

                    return;
                }}


                var partialArea =
                    allAreaKeys
                    .find(
                        function(key) {{

                            return areaSearchIndex[
                                key
                            ].name
                                .toLowerCase()
                                .includes(
                                    query
                                );

                        }}
                    );

                if (partialArea) {{

                    zoomToArea(
                        partialArea
                    );

                    return;
                }}


                var exactTrail =
                    allTrailNames
                    .find(
                        function(name) {{

                            return name
                                .toLowerCase()
                                ===
                                query;

                        }}
                    );

                if (exactTrail) {{

                    zoomToTrail(
                        exactTrail
                    );

                    return;
                }}


                var partialTrail =
                    allTrailNames
                    .find(
                        function(name) {{

                            return name
                                .toLowerCase()
                                .includes(
                                    query
                                );

                        }}
                    );

                if (partialTrail) {{

                    zoomToTrail(
                        partialTrail
                    );
                }}

            }}
        );

        L.DomEvent.disableClickPropagation(
            document.getElementById(
                "hero-search-box"
            )
        );


        // ====================================================
        // ABOUT
        // ====================================================

        document
        .getElementById(
            "hero-about-button"
        )
        .onclick =
            function() {{

                document
                .getElementById(
                    "hero-about-modal"
                )
                .style.display =
                    "block";

            }};


        document
        .getElementById(
            "hero-about-close"
        )
        .onclick =
            function() {{

                document
                .getElementById(
                    "hero-about-modal"
                )
                .style.display =
                    "none";

            }};


        document
        .getElementById(
            "hero-about-modal"
        )
        .addEventListener(
            "click",
            function(e) {{

                if (
                    e.target === this
                ) {{

                    this.style.display =
                        "none";

                }}

            }}
        );


        // ====================================================
        // FORECAST LAYER EXCLUSIVITY
        // ====================================================

        var changingForecastLayer =
            false;


        map.on(
            "overlayadd",
            function(e) {{

                if (
                    forecastLayers.indexOf(
                        e.layer
                    )
                    === -1
                ) {{

                    return;

                }}


                if (
                    changingForecastLayer
                ) {{

                    return;

                }}


                changingForecastLayer =
                    true;


                forecastLayers.forEach(
                    function(layer) {{

                        if (
                            layer !== e.layer
                            &&
                            map.hasLayer(
                                layer
                            )
                        ) {{

                            map.removeLayer(
                                layer
                            );

                        }}

                    }}
                );


                changingForecastLayer =
                    false;

            }}
        );


        // ====================================================
        // RIDER REPORT STATE
        // ====================================================

        var reportTrail =
            null;

        var reportLat =
            null;

        var reportLon =
            null;


        window.openHeroReport =
            function(
                trail,
                lat,
                lon
            ) {{

                reportTrail =
                    trail;

                reportLat =
                    lat;

                reportLon =
                    lon;


                document
                .getElementById(
                    "hero-report-trail-name"
                )
                .innerText =
                    trail;


                var now =
                    new Date();


                var localNow =
                    new Date(
                        now.getTime()
                        -
                        now.getTimezoneOffset()
                        *
                        60000
                    );


                document
                .getElementById(
                    "report-time"
                )
                .value =
                    localNow
                    .toISOString()
                    .slice(
                        0,
                        16
                    );


                document
                .getElementById(
                    "report-status"
                )
                .innerText =
                    "";


                document
                .getElementById(
                    "hero-report-modal"
                )
                .style.display =
                    "block";

            }};


        // ====================================================
        // TRAIL CLICK HANDLERS
        // ====================================================

        for (
            var variableName
            in trailMetadata
        ) {{

            var layer =
                window[
                    variableName
                ];


            if (
                layer === undefined
            ) {{

                continue;

            }}


            (
                function(
                    trailLayer,
                    meta
                ) {{

                    trailLayer.on(
                        "click",
                        function(e) {{

                            L.DomEvent.stopPropagation(
                                e
                            );


                            var lat =
                                e.latlng.lat;

                            var lon =
                                e.latlng.lng;


                            var trailName =
                                meta.trail;


                            var popupHTML =
                                "<div style='"
                                +
                                    "font-family:Arial;"
                                +
                                    "min-width:245px;"
                                +
                                "'>"
                                +

                                "<b style='font-size:15px;'>"
                                +
                                trailName
                                +
                                "</b>"
                                +

                                "<br><br>"
                                +

                                "<b>Forecast:</b> "
                                +
                                meta.forecast_label
                                +

                                "<br>"
                                +

                                "<b>Valid:</b> "
                                +
                                meta.valid_time
                                +

                                "<br><br>"
                                +

                                "<b>This segment:</b> "
                                +
                                meta.local_condition
                                +

                                "<br>"
                                +

                                "<b>Trail median F:</b> "
                                +
                                Number(
                                    meta.median_F
                                ).toFixed(
                                    2
                                )
                                +

                                "<br>"
                                +

                                "<b>Trail median score:</b> "
                                +
                                Number(
                                    meta.median_score
                                ).toFixed(
                                    0
                                )
                                +

                                "<br><br>"
                                +

                                "<button "
                                +
                                "id='hero-popup-report-button' "
                                +
                                "style='"
                                +
                                    "width:100%;"
                                +
                                    "padding:9px;"
                                +
                                    "border:none;"
                                +
                                    "border-radius:5px;"
                                +
                                    "background:#333;"
                                +
                                    "color:white;"
                                +
                                    "font-weight:bold;"
                                +
                                    "cursor:pointer;"
                                +
                                "'>"
                                +

                                "Report conditions here"

                                +
                                "</button>"
                                +

                                "</div>";


                            L.popup()
                            .setLatLng(
                                e.latlng
                            )
                            .setContent(
                                popupHTML
                            )
                            .openOn(
                                map
                            );


                            setTimeout(
                                function() {{

                                    var button =
                                        document
                                        .getElementById(
                                            "hero-popup-report-button"
                                        );


                                    if (
                                        button !== null
                                    ) {{

                                        button.onclick =
                                            function() {{

                                                map.closePopup();


                                                window.openHeroReport(
                                                    trailName,
                                                    lat,
                                                    lon
                                                );

                                            }};

                                    }}

                                }},
                                30
                            );

                        }}
                    );

                }}
            )(
                layer,
                trailMetadata[
                    variableName
                ]
            );

        }}


        // ====================================================
        // CLOSE REPORT
        // ====================================================

        document
        .getElementById(
            "hero-report-close"
        )
        .onclick =
            function() {{

                document
                .getElementById(
                    "hero-report-modal"
                )
                .style.display =
                    "none";

            }};


        document
        .getElementById(
            "hero-report-modal"
        )
        .addEventListener(
            "click",
            function(e) {{

                if (
                    e.target === this
                ) {{

                    this.style.display =
                        "none";

                }}

            }}
        );


        // ====================================================
        // SAVE REPORT JSON
        // ====================================================

        document
        .getElementById(
            "report-download"
        )
        .onclick =
            function() {{

                var observationLocal =
                    document
                    .getElementById(
                        "report-time"
                    )
                    .value;


                var condition =
                    document
                    .getElementById(
                        "report-condition"
                    )
                    .value;


                var comment =
                    document
                    .getElementById(
                        "report-comment"
                    )
                    .value
                    .trim();


                var reporter =
                    document
                    .getElementById(
                        "report-name"
                    )
                    .value
                    .trim();


                if (
                    reportTrail === null
                    ||
                    reportLat === null
                    ||
                    reportLon === null
                ) {{

                    document
                    .getElementById(
                        "report-status"
                    )
                    .innerText =
                        "Please select a trail again.";

                    return;

                }}


                if (
                    observationLocal === ""
                ) {{

                    document
                    .getElementById(
                        "report-status"
                    )
                    .innerText =
                        "Please provide the ride time.";

                    return;

                }}


                var observationDate =
                    new Date(
                        observationLocal
                    );


                var payload = {{

                    schema_version:
                        1,

                    trail:
                        reportTrail,

                    latitude:
                        reportLat,

                    longitude:
                        reportLon,

                    observation_time_local:
                        observationLocal,

                    observation_time_utc:
                        observationDate
                        .toISOString(),

                    timezone_offset_minutes:
                        observationDate
                        .getTimezoneOffset(),

                    observed_category:
                        condition,

                    rider_description:
                        comment,

                    source:
                        reporter !== ""
                        ?
                        reporter
                        :
                        "rider report",

                    submitted_at_utc:
                        new Date()
                        .toISOString()

                }};


		var supabaseUrl =
                    "https://ldyeqmgdtaiaeciakmvi.supabase.co";


                var supabaseKey =
                    "sb_publishable_ra-ze-B0wgx9KaltrcwaZA_jcNbr3iS";


                var reportRecord = {{

                    observation_utc:
                        observationDate
                        .toISOString(),

                    trail_name:
                        reportTrail,

                    latitude:
                        reportLat,

                    longitude:
                        reportLon,

                    condition:
                        condition,

                    comments:
                        comment,

                    initials:
                        reporter !== ""
                        ?
                        reporter
                        :
                        null

                }};


                document
                .getElementById(
                    "report-status"
                )
                .innerText =
                    "Submitting report...";


                fetch(
                    supabaseUrl
                    +
                    "/rest/v1/trail_reports",
                    {{

                        method:
                            "POST",

                        headers:
                        {{

                            "apikey":
                                supabaseKey,

                            "Authorization":
                                "Bearer "
                                +
                                supabaseKey,

                            "Content-Type":
                                "application/json",

                            "Prefer":
                                "return=minimal"

                        }},

                        body:
                            JSON.stringify(
                                reportRecord
                            )

                    }}
                )
                .then(
                    function(response) {{

                        if (
                            !response.ok
                        ) {{

                            return response
                            .text()
                            .then(
                                function(text) {{

                                    throw new Error(
                                        text
                                    );

                                }}
                            );

                        }}


                        document
                        .getElementById(
                            "report-status"
                        )
                        .innerText =
                            "Report submitted — thank you!";


                        setTimeout(
                            function() {{

                                document
                                .getElementById(
                                    "hero-report-modal"
                                )
                                .style.display =
                                    "none";

                            }},
                            1200
                        );

                    }}
                )
                .catch(
                    function(error) {{

                        console.error(
                            "Hero Dirt report submission failed:",
                            error
                        );


                        document
                        .getElementById(
                            "report-status"
                        )
                        .innerText =
                            "Report could not be submitted. Please try again.";

                    }}
                );
            }};

    }}
);

</script>
"""


m.get_root().html.add_child(
    folium.Element(
        page_js
    )
)


# ============================================================
# FIT DOMAIN
# ============================================================

m.fit_bounds(
    [
        [
            south_lat,
            west_lon,
        ],
        [
            north_lat,
            east_lon,
        ],
    ]
)


# ============================================================
# SAVE
# ============================================================

m.save(
    OUTPUT_HTML
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "Forecast layers:"
)

for (
    label,
    chunks,
    valid_time,
) in layer_stats:

    print(
        f"  {label:6s} "
        f"{chunks:6d} chunks   "
        f"valid {valid_time}"
    )


print()

print(
    f"Clickable trail segments:"
)

print(
    f"  {len(trail_line_metadata)}"
)

print()

print(
    f"Searchable trail names:"
)

print(
    f"  {len(trail_search_index)}"
)

print()

print(
    "Included:"
)

print(
    "  mobile-responsive layout"
)

print(
    "  trail search + zoom"
)

print(
    "  trail-click rider reporting"
)

print(
    "  About panel"
)

print(
    "  Explorer navigation"
)

print()

print(
    "Saved:"
)

print(
    f"  {OUTPUT_HTML}"
)

print()

print(
    "============================================"
)
print(
    " HERO DIRT MAP COMPLETE"
)
print(
    "============================================"
)
