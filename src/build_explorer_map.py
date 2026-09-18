#!/usr/bin/env python3

"""
Hero Dirt Explorer
------------------

Interactive science / diagnostic map showing the spatial
information that feeds Hero Dirt Forecast.

Features
--------
Basemaps:
    Esri World Topo
    Esri Satellite

Science layers:
    Terrain
    Soils
    Vegetation
    Current hydrology
    Recent MRMS precipitation

Behavior:
    - Esri basemap always remains underneath.
    - Science rasters are semi-transparent overlays.
    - Only one science raster is active at a time.
    - Masks may be independently stacked.
    - Dynamic color bar changes with active science layer.
    - Clicking the map reports:
          active-layer value
          elevation
          slope
          current F
          Hero score
          current Hero Dirt class

Output:
    web/HeroDirt_explorer.html
"""

from pathlib import Path
import json

import numpy as np
import rasterio

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

import folium

import matplotlib.pyplot as plt
from matplotlib.colors import (
    Normalize,
    to_hex,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "HeroDirt"

STATIC_FILE = (
    ROOT
    / "static/model/HeroDirt_static_200m.npz"
)

PLASTICITY_FILE = (
    ROOT
    / "static/soils/HeroDirt_plasticity_200m.npz"
)

CURRENT_FILE = (
    ROOT
    / "output/current_v2_beta/HeroDirt_current_state.npz"
)

GRID_TIF = (
    ROOT
    / "static/grid/SanGabriels_DEM_200m_UTM11.tif"
)

WEB_DIR = (
    ROOT
    / "web"
)

WEB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_HTML = (
    WEB_DIR
    / "HeroDirt_explorer.html"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

SCIENCE_OPACITY = 0.35

PRECIP_OPACITY = 0.40

MASK_OPACITY = 0.12


# ============================================================
# LOAD DATA
# ============================================================

S = np.load(
    STATIC_FILE
)

P = np.load(
    PLASTICITY_FILE
)

C = np.load(
    CURRENT_FILE
)


# ============================================================
# ARRAY HELPER
# ============================================================

def get_array(
    archive,
    names,
):

    for name in names:

        if name in archive.files:

            return archive[
                name
            ].astype(float)

    raise KeyError(
        f"Could not find any of {names}\n"
        f"Available keys:\n{archive.files}"
    )


# ============================================================
# STATIC FIELDS
# ============================================================

elevation = get_array(
    S,
    [
        "elevation",
        "dem",
    ],
)

slope = get_array(
    S,
    [
        "slope",
    ],
)

southness = get_array(
    S,
    [
        "southness",
    ],
)

sand = get_array(
    S,
    [
        "sand",
    ],
)

clay = get_array(
    S,
    [
        "clay",
    ],
)

ksat = get_array(
    S,
    [
        "ksat",
    ],
)

theta_fc = get_array(
    S,
    [
        "theta_fc",
    ],
)

theta_wp = get_array(
    S,
    [
        "theta_wp",
    ],
)

theta_sat = get_array(
    S,
    [
        "theta_sat",
    ],
)

tree = get_array(
    S,
    [
        "tree_fraction",
    ],
)

shrub = get_array(
    S,
    [
        "shrub_cover_fraction",
    ],
)

herb = get_array(
    S,
    [
        "herb_cover_fraction",
    ],
)

physics = get_array(
    S,
    [
        "physics_valid",
    ],
).astype(bool)

public = get_array(
    S,
    [
        "public_valid",
    ],
).astype(bool)


plasticity_index = get_array(
    P,
    [
        "plasticity_index",
        "pi",
        "pi_r",
    ],
)


# ============================================================
# CURRENT MODEL
# ============================================================

tread = get_array(
    C,
    [
        "tread_theta",
        "surface_theta",
    ],
)

theta05 = get_array(
    C,
    [
        "theta_0_5cm",
    ],
)

F = get_array(
    C,
    [
        "F",
    ],
)

score = get_array(
    C,
    [
        "score_physics",
        "score",
        "hero_score",
    ],
)

condition = get_array(
    C,
    [
        "condition",
        "hero_class",
    ],
).astype(int)


if "analysis_precip_mm" in C.files:

    recent_precip = C[
        "analysis_precip_mm"
    ].astype(float)

else:

    recent_precip = np.zeros_like(
        F
    )


model_time = str(
    C[
        "time"
    ]
)


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


# ============================================================
# GRID GEOMETRY
# ============================================================

with rasterio.open(
    GRID_TIF
) as src:

    src_crs = src.crs

    src_transform = src.transform

    src_width = src.width

    src_height = src.height

    src_bounds = src.bounds


# ============================================================
# WGS84 DESTINATION GRID
# ============================================================

dst_crs = "EPSG:4326"


(
    dst_transform,
    dst_width,
    dst_height,
) = calculate_default_transform(

    src_crs,
    dst_crs,

    src_width,
    src_height,

    *src_bounds,
)


west = dst_transform.c

north = dst_transform.f

pixel_lon = dst_transform.a

pixel_lat = dst_transform.e


east = (
    west
    +
    pixel_lon
    *
    dst_width
)


south = (
    north
    +
    pixel_lat
    *
    dst_height
)


bounds = [
    [
        south,
        west,
    ],
    [
        north,
        east,
    ],
]


center_lat = (
    south
    +
    north
) / 2.0

center_lon = (
    west
    +
    east
) / 2.0


# ============================================================
# REPROJECT ARRAY
# ============================================================

def reproject_array(
    A,
    categorical=False,
):

    out = np.full(
        (
            dst_height,
            dst_width,
        ),
        np.nan,
        dtype=np.float32,
    )


    method = (
        Resampling.nearest
        if categorical
        else Resampling.bilinear
    )


    reproject(

        source=A.astype(
            np.float32
        ),

        destination=out,

        src_transform=src_transform,

        src_crs=src_crs,

        dst_transform=dst_transform,

        dst_crs=dst_crs,

        src_nodata=np.nan,

        dst_nodata=np.nan,

        resampling=method,
    )


    return out


# ============================================================
# REPROJECT CORE ARRAYS ONCE
# ============================================================

print()
print(
    "Reprojecting Explorer fields..."
)


elevation_ll = reproject_array(
    elevation
)

slope_ll = reproject_array(
    slope
)

southness_ll = reproject_array(
    southness
)

sand_ll = reproject_array(
    sand
)

clay_ll = reproject_array(
    clay
)


log_ksat = np.full(
    ksat.shape,
    np.nan,
    dtype=float,
)


good_ksat = (
    np.isfinite(
        ksat
    )
    &
    (
        ksat > 0
    )
)


log_ksat[
    good_ksat
] = np.log10(
    ksat[
        good_ksat
    ]
)


log_ksat_ll = reproject_array(
    log_ksat
)


theta_fc_ll = reproject_array(
    theta_fc
)

theta_wp_ll = reproject_array(
    theta_wp
)

theta_sat_ll = reproject_array(
    theta_sat
)

plasticity_ll = reproject_array(
    plasticity_index
)

tree_ll = reproject_array(
    tree
)

shrub_ll = reproject_array(
    shrub
)

herb_ll = reproject_array(
    herb
)

tread_ll = reproject_array(
    tread
)

theta05_ll = reproject_array(
    theta05
)

F_ll = reproject_array(
    F
)

score_ll = reproject_array(
    score
)

precip_ll = reproject_array(
    recent_precip
)


condition_ll = reproject_array(
    condition.astype(float),
    categorical=True,
)


physics_ll = reproject_array(
    physics.astype(float),
    categorical=True,
)

public_ll = reproject_array(
    public.astype(float),
    categorical=True,
)


# ============================================================
# ROBUST DISPLAY RANGE
# ============================================================

def robust_range(
    A,
    mask=physics,
    low=2,
    high=98,
):

    values = A[
        mask
        &
        np.isfinite(
            A
        )
    ]


    return (
        float(
            np.percentile(
                values,
                low,
            )
        ),
        float(
            np.percentile(
                values,
                high,
            )
        ),
    )


elev_min, elev_max = robust_range(
    elevation
)

slope_min, slope_max = robust_range(
    slope
)

sand_min, sand_max = robust_range(
    sand
)

clay_min, clay_max = robust_range(
    clay
)

ksat_min, ksat_max = robust_range(
    log_ksat
)

pi_min, pi_max = robust_range(
    plasticity_index
)


precip_values = recent_precip[
    physics
    &
    np.isfinite(
        recent_precip
    )
]


if len(
    precip_values
) > 0:

    precip_max = float(
        np.nanpercentile(
            precip_values,
            99.5,
        )
    )

else:

    precip_max = 1.0


precip_max = max(
    precip_max,
    1.0,
)


# ============================================================
# COLORIZE ALREADY-REPROJECTED ARRAY
# ============================================================

def make_rgba_ll(
    A_ll,
    cmap_name,
    vmin,
    vmax,
):

    norm = Normalize(
        vmin=vmin,
        vmax=vmax,
        clip=True,
    )


    cmap = plt.get_cmap(
        cmap_name
    )


    rgba = cmap(
        norm(
            np.nan_to_num(
                A_ll,
                nan=vmin,
            )
        )
    )


    valid = (
        np.isfinite(
            A_ll
        )
        &
        (
            physics_ll
            >
            0.5
        )
    )


    rgba[
        ...,
        3
    ] = np.where(
        valid,
        1.0,
        0.0,
    )


    return rgba


# ============================================================
# COLOR BAR HELPER
# ============================================================

def cmap_gradient(
    cmap_name,
    n=9,
):

    cmap = plt.get_cmap(
        cmap_name
    )


    colors = [
        to_hex(
            cmap(
                i
                /
                (
                    n - 1
                )
            )
        )
        for i in range(
            n
        )
    ]


    return (
        "linear-gradient(to right,"
        +
        ",".join(
            colors
        )
        +
        ")"
    )


# ============================================================
# SCIENCE LAYER DEFINITIONS
# ============================================================

science_specs = [

    {
        "name":
            "Terrain: Elevation",

        "array":
            elevation_ll,

        "cmap":
            "terrain",

        "vmin":
            elev_min,

        "vmax":
            elev_max,

        "units":
            "m",

        "decimals":
            0,

        "label":
            "Elevation",
    },

    {
        "name":
            "Terrain: Slope",

        "array":
            slope_ll,

        "cmap":
            "magma",

        "vmin":
            0.0,

        "vmax":
            max(
                40.0,
                slope_max,
            ),

        "units":
            "°",

        "decimals":
            1,

        "label":
            "Slope",
    },

    {
        "name":
            "Terrain: Southness",

        "array":
            southness_ll,

        "cmap":
            "coolwarm",

        "vmin":
            -1.0,

        "vmax":
            1.0,

        "units":
            "",

        "decimals":
            2,

        "label":
            "Southness",
    },

    {
        "name":
            "Soil: Sand %",

        "array":
            sand_ll,

        "cmap":
            "YlOrBr",

        "vmin":
            sand_min,

        "vmax":
            sand_max,

        "units":
            "%",

        "decimals":
            1,

        "label":
            "Sand",
    },

    {
        "name":
            "Soil: Clay %",

        "array":
            clay_ll,

        "cmap":
            "OrRd",

        "vmin":
            clay_min,

        "vmax":
            clay_max,

        "units":
            "%",

        "decimals":
            1,

        "label":
            "Clay",
    },

    {
        "name":
            "Soil: log10 Ksat",

        "array":
            log_ksat_ll,

        "cmap":
            "viridis",

        "vmin":
            ksat_min,

        "vmax":
            ksat_max,

        "units":
            "log10(µm/s)",

        "decimals":
            2,

        "label":
            "Ksat",
    },

    {
        "name":
            "Soil: Field Capacity",

        "array":
            theta_fc_ll,

        "cmap":
            "Blues",

        "vmin":
            0.05,

        "vmax":
            0.35,

        "units":
            "m³/m³",

        "decimals":
            3,

        "label":
            "Field capacity",
    },

    {
        "name":
            "Soil: Wilting Point",

        "array":
            theta_wp_ll,

        "cmap":
            "Purples",

        "vmin":
            0.02,

        "vmax":
            0.20,

        "units":
            "m³/m³",

        "decimals":
            3,

        "label":
            "Wilting point",
    },

    {
        "name":
            "Soil: Saturation",

        "array":
            theta_sat_ll,

        "cmap":
            "GnBu",

        "vmin":
            0.25,

        "vmax":
            0.60,

        "units":
            "m³/m³",

        "decimals":
            3,

        "label":
            "Saturation",
    },

    {
        "name":
            "Soil: Plasticity Index",

        "array":
            plasticity_ll,

        "cmap":
            "plasma",

        "vmin":
            pi_min,

        "vmax":
            pi_max,

        "units":
            "",

        "decimals":
            1,

        "label":
            "Plasticity index",
    },

    {
        "name":
            "Vegetation: Tree Canopy",

        "array":
            tree_ll,

        "cmap":
            "Greens",

        "vmin":
            0.0,

        "vmax":
            1.0,

        "units":
            "fraction",

        "decimals":
            2,

        "label":
            "Tree canopy",
    },

    {
        "name":
            "Vegetation: Shrub Cover",

        "array":
            shrub_ll,

        "cmap":
            "YlGn",

        "vmin":
            0.0,

        "vmax":
            1.0,

        "units":
            "fraction",

        "decimals":
            2,

        "label":
            "Shrub cover",
    },

    {
        "name":
            "Vegetation: Herbaceous Cover",

        "array":
            herb_ll,

        "cmap":
            "summer",

        "vmin":
            0.0,

        "vmax":
            1.0,

        "units":
            "fraction",

        "decimals":
            2,

        "label":
            "Herbaceous cover",
    },

    {
        "name":
            "Current: Tread Moisture",

        "array":
            tread_ll,

        "cmap":
            "Blues",

        "vmin":
            0.03,

        "vmax":
            0.30,

        "units":
            "m³/m³",

        "decimals":
            3,

        "label":
            "Tread moisture",
    },

    {
        "name":
            "Current: 0-5 cm Moisture",

        "array":
            theta05_ll,

        "cmap":
            "viridis",

        "vmin":
            0.03,

        "vmax":
            0.30,

        "units":
            "m³/m³",

        "decimals":
            3,

        "label":
            "0-5 cm moisture",
    },

    {
        "name":
            "Current: Hydraulic State F",

        "array":
            F_ll,

        "cmap":
            "viridis",

        "vmin":
            0.0,

        "vmax":
            1.2,

        "units":
            "",

        "decimals":
            2,

        "label":
            "Hydraulic state F",
    },

    {
        "name":
            "Current: Hero Dirt Score",

        "array":
            score_ll,

        "cmap":
            "viridis",

        "vmin":
            0.0,

        "vmax":
            100.0,

        "units":
            "",

        "decimals":
            0,

        "label":
            "Hero Dirt score",
    },

    {
        "name":
            "Recent MRMS Precipitation",

        "array":
            precip_ll,

        "cmap":
            "Blues",

        "vmin":
            0.0,

        "vmax":
            precip_max,

        "units":
            "mm",

        "decimals":
            1,

        "label":
            "Recent precipitation",

        "opacity":
            PRECIP_OPACITY,
    },
]


# ============================================================
# CREATE MAP
# ============================================================

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
# ADD SCIENCE LAYERS
# ============================================================

science_layers = []

layer_metadata = {}


for spec in science_specs:

    rgba = make_rgba_ll(

        spec[
            "array"
        ],

        spec[
            "cmap"
        ],

        spec[
            "vmin"
        ],

        spec[
            "vmax"
        ],
    )


    opacity = spec.get(
        "opacity",
        SCIENCE_OPACITY,
    )


    show = (
        spec[
            "name"
        ]
        ==
        "Current: Hydraulic State F"
    )


    layer = folium.raster_layers.ImageOverlay(

        image=rgba,

        bounds=bounds,

        name=spec[
            "name"
        ],

        opacity=opacity,

        interactive=False,

        cross_origin=False,

        zindex=5,

        show=show,

        overlay=True,

        control=True,
    )


    layer.add_to(
        m
    )


    science_layers.append(
        layer
    )


    layer_metadata[
        layer.get_name()
    ] = {

        "name":
            spec[
                "name"
            ],

        "label":
            spec[
                "label"
            ],

        "units":
            spec[
                "units"
            ],

        "vmin":
            float(
                spec[
                    "vmin"
                ]
            ),

        "vmax":
            float(
                spec[
                    "vmax"
                ]
            ),

        "decimals":
            int(
                spec[
                    "decimals"
                ]
            ),

        "gradient":
            cmap_gradient(
                spec[
                    "cmap"
                ]
            ),
    }


# ============================================================
# MASKS
# ============================================================

physics_rgba = np.zeros(
    (
        dst_height,
        dst_width,
        4,
    ),
    dtype=float,
)


physics_rgba[
    ...,
    0
] = 1.0

physics_rgba[
    ...,
    1
] = 0.0

physics_rgba[
    ...,
    2
] = 0.0

physics_rgba[
    ...,
    3
] = np.where(
    physics_ll > 0.5,
    1.0,
    0.0,
)


folium.raster_layers.ImageOverlay(

    image=physics_rgba,

    bounds=bounds,

    name="Mask: Physics Valid",

    opacity=MASK_OPACITY,

    show=False,

    overlay=True,

    control=True,

).add_to(
    m
)


public_rgba = np.zeros(
    (
        dst_height,
        dst_width,
        4,
    ),
    dtype=float,
)


public_rgba[
    ...,
    0
] = 0.0

public_rgba[
    ...,
    1
] = 0.3

public_rgba[
    ...,
    2
] = 1.0

public_rgba[
    ...,
    3
] = np.where(
    public_ll > 0.5,
    1.0,
    0.0,
)


folium.raster_layers.ImageOverlay(

    image=public_rgba,

    bounds=bounds,

    name="Mask: Public Valid",

    opacity=MASK_OPACITY,

    show=False,

    overlay=True,

    control=True,

).add_to(
    m
)


# ============================================================
# SUNSET RIDGE VALIDATION MARKER
# ============================================================

folium.CircleMarker(

    location=[
        34.21775,
        -118.12699,
    ],

    radius=6,

    color="#111111",

    weight=2,

    fill=True,

    fill_color="#ffffff",

    fill_opacity=1.0,

    tooltip="Sunset Ridge validation observation",

    popup=folium.Popup(
        """
        <b>Sunset Ridge Trail</b>
        <br><br>
        Rider report:
        <br>
        "Pretty dry, but not dusty."
        <br><br>
        Observed: Dry
        <br>
        Model F: 0.220
        <br>
        Hero score: 41
        <br>
        Model class: Dry
        """,
        max_width=300,
    ),

).add_to(
    m
)


# ============================================================
# TITLE
# ============================================================

title_html = f"""
<div style="
    position:fixed;
    top:12px;
    left:55px;
    z-index:9999;
    background-color:rgba(255,255,255,0.94);
    padding:10px 15px;
    border-radius:7px;
    font-family:Arial;
    box-shadow:0px 1px 5px rgba(0,0,0,0.35);
">

<b style="
    font-size:18px;
">
Hero Dirt Explorer
</b>

<br>

<span style="
    font-size:12px;
">
Science + model inputs
<br>
Current model: {model_time} UTC
</span>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        title_html
    )
)

forecast_button_html = """
<a
    href="./"
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
        text-decoration:none;
        cursor:pointer;
        box-shadow:0px 1px 5px rgba(0,0,0,0.25);
    "
>
Forecast
</a>
"""

m.get_root().html.add_child(
    folium.Element(
        forecast_button_html
    )
)

# ============================================================
# INFO PANEL
# ============================================================

info_html = """
<div style="
    position:fixed;
    bottom:35px;
    left:35px;
    z-index:9999;
    background-color:rgba(255,255,255,0.92);
    padding:11px 14px;
    width:245px;
    border-radius:7px;
    font-family:Arial;
    font-size:12px;
    box-shadow:0px 1px 5px rgba(0,0,0,0.35);
">

<b>Hero Dirt Explorer</b>

<br><br>

Choose one science layer from the upper-right menu.

<br><br>

Click anywhere in the model domain to inspect the
selected variable and current Hero Dirt state.

<br><br>

<b>Useful scales</b>
<br>
F = 0 at WP
<br>
F = 1 at FC
<br>
Southness: -1 north, +1 south
<br>
Ksat shown as log10(µm/s)

<br><br>

<span style="
    color:#666;
    font-size:10px;
">
Many Hero Dirt parameters remain empirical.
Spatial inputs are not direct measurements of
trail tread at 200-m resolution.
</span>

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        info_html
    )
)


# ============================================================
# DYNAMIC LEGEND CONTAINER
# ============================================================

legend_html = """
<div
    id="hero-dirt-dynamic-legend"
    style="
        position:fixed;
        bottom:35px;
        right:35px;
        width:250px;
        z-index:9999;
        background-color:rgba(255,255,255,0.94);
        padding:12px 14px;
        border-radius:7px;
        font-family:Arial;
        font-size:12px;
        box-shadow:0px 1px 5px rgba(0,0,0,0.35);
    "
>
</div>
"""


m.get_root().html.add_child(
    folium.Element(
        legend_html
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
#
# We use the same WGS84 rasters displayed on the map,
# so click values line up exactly with the visible overlay.
# ============================================================

def js_flat_array(
    A,
    decimals=4,
):

    flat = np.nan_to_num(
        A,
        nan=-9999.0,
        posinf=-9999.0,
        neginf=-9999.0,
    )


    flat = np.round(
        flat,
        decimals
    )


    return flat.ravel().tolist()


js_science_data = {}


for layer, spec in zip(
    science_layers,
    science_specs,
):

    js_science_data[
        layer.get_name()
    ] = js_flat_array(
        spec[
            "array"
        ],
        max(
            spec[
                "decimals"
            ],
            3,
        ),
    )


js_elevation = js_flat_array(
    elevation_ll,
    1,
)

js_slope = js_flat_array(
    slope_ll,
    1,
)

js_F = js_flat_array(
    F_ll,
    3,
)

js_score = js_flat_array(
    score_ll,
    1,
)

js_condition = js_flat_array(
    condition_ll,
    0,
)


# ============================================================
# JAVASCRIPT
# ============================================================

map_name = m.get_name()


science_names_js = [
    layer.get_name()
    for layer in science_layers
]


default_layer_name = None


for layer, spec in zip(
    science_layers,
    science_specs,
):

    if (
        spec[
            "name"
        ]
        ==
        "Current: Hydraulic State F"
    ):

        default_layer_name = layer.get_name()

        break


if default_layer_name is None:

    default_layer_name = science_names_js[
        0
    ]


javascript = f"""
<script>

document.addEventListener(
    "DOMContentLoaded",
    function() {{

        var map = {map_name};


        // ====================================================
        // SCIENCE LAYERS
        // ====================================================

        var scienceLayers = [
            {",".join(science_names_js)}
        ];


        var scienceMetadata =
            {json.dumps(layer_metadata)};


        var scienceData =
            {json.dumps(js_science_data)};


        var elevationData =
            {json.dumps(js_elevation)};


        var slopeData =
            {json.dumps(js_slope)};


        var fData =
            {json.dumps(js_F)};


        var scoreData =
            {json.dumps(js_score)};


        var conditionData =
            {json.dumps(js_condition)};


        var nRows = {dst_height};

        var nCols = {dst_width};


        var west = {west};

        var north = {north};

        var pixelLon = {pixel_lon};

        var pixelLat = {pixel_lat};


        var activeScienceLayer =
            {default_layer_name};


        var changingScienceLayer = false;


        // ====================================================
        // HERO CLASS NAMES
        // ====================================================

        var classNames = {{
            1: "Very Dry",
            2: "Dry",
            3: "Moist/Good",
            4: "Hero",
            5: "Wet",
            6: "Too Wet"
        }};


        // ====================================================
        // VALUE FORMATTER
        // ====================================================

        function formatValue(
            value,
            decimals,
            units
        ) {{

            if (
                value === undefined
                ||
                value <= -9998
                ||
                !isFinite(value)
            ) {{

                return "No data";

            }}


            var text =
                Number(value).toFixed(
                    decimals
                );


            if (
                units !== ""
            ) {{

                text += " " + units;

            }}


            return text;

        }}


        // ====================================================
        // DYNAMIC LEGEND
        // ====================================================

        function updateLegend() {{

            var legend =
                document.getElementById(
                    "hero-dirt-dynamic-legend"
                );


            var meta =
                scienceMetadata[
                    activeScienceLayer._leaflet_id
                        ? activeScienceLayer.getPane
                        : ""
                ];


            // Metadata keys correspond to JavaScript variable
            // names, not Leaflet IDs. Find current variable.

            var variableName = null;


            scienceLayers.forEach(
                function(layer) {{

                    if (
                        layer === activeScienceLayer
                    ) {{

                        for (
                            var key
                            in scienceMetadata
                        ) {{

                            if (
                                window[key]
                                === layer
                            ) {{

                                variableName = key;

                            }}

                        }}

                    }}

                }}
            );


            if (
                variableName === null
            ) {{

                return;

            }}


            meta =
                scienceMetadata[
                    variableName
                ];


            legend.innerHTML =
                "<b>"
                +
                meta.label
                +
                "</b>"
                +
                "<br>"
                +
                "<span style='color:#666;'>"
                +
                meta.units
                +
                "</span>"
                +
                "<div style='"
                +
                    "margin-top:8px;"
                +
                    "height:16px;"
                +
                    "width:220px;"
                +
                    "background:"
                +
                    meta.gradient
                +
                ";"
                +
                    "border:1px solid #999;"
                +
                "'></div>"
                +
                "<div style='"
                +
                    "width:220px;"
                +
                    "display:flex;"
                +
                    "justify-content:space-between;"
                +
                    "margin-top:3px;"
                +
                "'>"
                +
                "<span>"
                +
                Number(meta.vmin).toFixed(
                    meta.decimals
                )
                +
                "</span>"
                +
                "<span>"
                +
                Number(meta.vmax).toFixed(
                    meta.decimals
                )
                +
                "</span>"
                +
                "</div>";

        }}


        // ====================================================
        // FIND JS VARIABLE NAME FOR LEAFLET LAYER
        // ====================================================

        function scienceVariableName(
            layer
        ) {{

            for (
                var key
                in scienceMetadata
            ) {{

                if (
                    window[key]
                    === layer
                ) {{

                    return key;

                }}

            }}


            return null;

        }}


        // ====================================================
        // SCIENCE-LAYER EXCLUSIVITY
        // ====================================================

        map.on(
            "overlayadd",
            function(e) {{

                if (
                    scienceLayers.indexOf(
                        e.layer
                    )
                    === -1
                ) {{

                    return;

                }}


                if (
                    changingScienceLayer
                ) {{

                    return;

                }}


                changingScienceLayer = true;


                scienceLayers.forEach(
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


                activeScienceLayer =
                    e.layer;


                updateLegend();


                changingScienceLayer = false;

            }}
        );


        // ====================================================
        // CLICK INSPECTOR
        // ====================================================

        map.on(
            "click",
            function(e) {{

                var lon =
                    e.latlng.lng;

                var lat =
                    e.latlng.lat;


                var col =
                    Math.floor(
                        (
                            lon
                            -
                            west
                        )
                        /
                        pixelLon
                    );


                var row =
                    Math.floor(
                        (
                            lat
                            -
                            north
                        )
                        /
                        pixelLat
                    );


                if (
                    row < 0
                    ||
                    row >= nRows
                    ||
                    col < 0
                    ||
                    col >= nCols
                ) {{

                    return;

                }}


                var index =
                    row
                    *
                    nCols
                    +
                    col;


                var variableName =
                    scienceVariableName(
                        activeScienceLayer
                    );


                if (
                    variableName === null
                ) {{

                    return;

                }}


                var meta =
                    scienceMetadata[
                        variableName
                    ];


                var value =
                    scienceData[
                        variableName
                    ][
                        index
                    ];


                var elev =
                    elevationData[
                        index
                    ];


                var slp =
                    slopeData[
                        index
                    ];


                var fval =
                    fData[
                        index
                    ];


                var hscore =
                    scoreData[
                        index
                    ];


                var cid =
                    Math.round(
                        conditionData[
                            index
                        ]
                    );


                var conditionName =
                    classNames[
                        cid
                    ]
                    ||
                    "No data";


                var popupHTML =
                    "<div style='"
                    +
                        "font-family:Arial;"
                    +
                        "min-width:220px;"
                    +
                    "'>"
                    +

                    "<b>"
                    +
                    meta.label
                    +
                    "</b>"
                    +
                    "<br>"
                    +
                    formatValue(
                        value,
                        meta.decimals,
                        meta.units
                    )
                    +

                    "<hr style='"
                    +
                        "border:none;"
                    +
                        "border-top:1px solid #ddd;"
                    +
                    "'>"
                    +

                    "<b>Current Hero Dirt</b>"
                    +
                    "<br>"
                    +

                    "Condition: "
                    +
                    conditionName
                    +
                    "<br>"
                    +

                    "F: "
                    +
                    formatValue(
                        fval,
                        2,
                        ""
                    )
                    +
                    "<br>"
                    +

                    "Score: "
                    +
                    formatValue(
                        hscore,
                        0,
                        ""
                    )
                    +

                    "<br><br>"
                    +

                    "<b>Terrain</b>"
                    +
                    "<br>"
                    +

                    "Elevation: "
                    +
                    formatValue(
                        elev,
                        0,
                        "m"
                    )
                    +
                    "<br>"
                    +

                    "Slope: "
                    +
                    formatValue(
                        slp,
                        1,
                        "°"
                    )
                    +

                    "<br><br>"
                    +

                    "<span style='"
                    +
                        "font-size:10px;"
                    +
                        "color:#777;"
                    +
                    "'>"
                    +

                    lat.toFixed(5)
                    +
                    ", "
                    +
                    lon.toFixed(5)
                    +

                    "</span>"
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

            }}
        );


        // ====================================================
        // INITIAL LEGEND
        // ====================================================

        updateLegend();

    }}
);

</script>
"""


m.get_root().html.add_child(
    folium.Element(
        javascript
    )
)


# ============================================================
# FIT DOMAIN
# ============================================================

m.fit_bounds(
    bounds
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

print()
print(
    "============================================"
)
print(
    " HERO DIRT EXPLORER COMPLETE"
)
print(
    "============================================"
)
print()

print(
    f"Current model:"
)

print(
    f"  {model_time}"
)

print()

print(
    "Features:"
)

print(
    "  Dynamic science-layer color bar"
)

print(
    "  Click-to-query active science variable"
)

print(
    "  Click popup also reports F, Hero score,"
)

print(
    "  condition, elevation and slope"
)

print()

print(
    f"Science opacity:"
)

print(
    f"  {SCIENCE_OPACITY:.2f}"
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
    "Open:"
)

print(
    f"  open {OUTPUT_HTML}"
)

print()

print(
    "============================================"
)
