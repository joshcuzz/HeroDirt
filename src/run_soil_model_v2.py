#!/usr/bin/env python3

"""
Hero Dirt Forecast
Operational three-layer soil-moisture model.

Layers
------
Tread:
    0-2 cm
    Z = 20 mm

Shallow:
    2-5 cm
    Z = 30 mm

Deep:
    5-20 cm
    Z = 150 mm

Architecture
------------
Rain
  -> tread
  -> shallow
  -> deep

with mass-conserving upward moisture exchange:

    shallow -> tread

when the shallow layer is wetter than the tread.

The upward exchange relaxes the tread/shallow moisture
difference with a provisional timescale:

    tau = 6 hours

Analysis
--------
Initial upper-5-cm moisture is constrained by latest SMAP.

MRMS hourly precipitation is replayed from the SMAP observation
time to the beginning of the NWS forecast.

Rainy intervals:
    5-minute internal timestep

Dry intervals:
    <= 1-hour internal timestep

Forecast
--------
NWS 6-hour fields drive the model forward to 120 h.

Important
---------
Physical soil properties:
    theta_wp
    theta_fc
    theta_sat
    SSURGO Ksat
    texture

Empirical / provisional Hero Dirt parameters:
    effective tread Ksat fraction
    shallow/deep transfer scaling
    upward-exchange timescale
    ET coefficients
    score thresholds

These empirical parameters remain subject to calibration.
"""

from pathlib import Path

import numpy as np
import rasterio

from pyproj import Transformer

from hero_score import calculate_hero_score
from penman_drying import compute_penman_et_block

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

SMAP_FILE = (
    ROOT
    / "data/smap/processed/HeroDirt_SMAP_latest.npz"
)

MRMS_FILE = (
    ROOT
    / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)

NWS_FILE = (
    ROOT
    / "data/nws/processed/HeroDirt_NWS_6hourly.npz"
)

GRID_TIF = (
    ROOT
    / "static/grid/SanGabriels_DEM_200m_UTM11.tif"
)


CURRENT_DIR = (
    ROOT
    / "output/current_v2_beta"
)

FORECAST_DIR = (
    ROOT
    / "output/forecast_v2_beta"
)

CURRENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LAYER THICKNESSES [mm]
# ============================================================

ZTREAD = 20.0

ZSHALLOW = 30.0

ZDEEP = 150.0


# ============================================================
# HYDROLOGY PARAMETERS
# ============================================================

# Effective intact-trail infiltration conductivity:
#
# K_eff = KSAT_EFFICIENCY * SSURGO Ksat
#
# Provisional empirical value.

KSAT_EFFICIENCY = 0.10


# Dry-soil enhancement of surface infiltration capacity.

DRY_BOOST = 2.0


# Reference shallow -> deep transfer [mm/day].
#
# Spatially modified using Ksat and texture.

SHALLOW_TRANSFER_BASE_DAY = 64.0


# Deep drainage reference [mm/day].

DEEP_DRAIN_BASE_DAY = 7.0


# Upward tread/shallow equilibration timescale [hours].
#
# Selected from sensitivity tests:
#
# 2, 4, 6, 8, 12 h
#
# 6 h provides intermediate coupling while preserving
# post-storm tread recovery.

UPWARD_TAU_HOURS = 6.0


# ============================================================
# ET / DRYING PARAMETERS [mm/day]
# ============================================================

ET_TREAD_BASE_DAY = 2.5

ET_SHALLOW_BASE_DAY = 0.7

ET_DEEP_BASE_DAY = 0.8

# ============================================================
# V2 TREAD DRYING
# ============================================================

USE_PENMAN_TREAD_ET = True

# Provisional trail-tread scaling applied to Penman potential ET.
# We will test 0.50, 0.75, and 1.00.
K_TREAD = 0.50

# ============================================================
# INTERNAL TIMESTEPS
# ============================================================

RAIN_SUBSTEP_HOURS = (
    5.0
    /
    60.0
)

DRY_SUBSTEP_HOURS = 1.0


# ============================================================
# LOAD DATA
# ============================================================

S = np.load(
    STATIC_FILE
)

P = np.load(
    PLASTICITY_FILE
)

SMAP = np.load(
    SMAP_FILE
)

MRMS = np.load(
    MRMS_FILE
)

NWS = np.load(
    NWS_FILE
)


# ============================================================
# BASIC GRID
# ============================================================

x = S[
    "x"
].astype(float)

y = S[
    "y"
].astype(float)


ny = len(y)

nx = len(x)

# ============================================================
# GRID LATITUDE / LONGITUDE
# ============================================================

X, Y = np.meshgrid(
    x,
    y,
)


transformer = Transformer.from_crs(
    "EPSG:26911",
    "EPSG:4326",
    always_xy=True,
)


lon, lat = transformer.transform(
    X,
    Y,
)


lat_rad = np.deg2rad(
    lat
)

physics = S[
    "physics_valid"
].astype(bool)

public = S[
    "public_valid"
].astype(bool)


# ============================================================
# SOIL PROPERTIES
# ============================================================

wp = S[
    "theta_wp"
].astype(float)

fc = S[
    "theta_fc"
].astype(float)

sat = S[
    "theta_sat"
].astype(float)


sand = (
    S[
        "sand"
    ].astype(float)
    /
    100.0
)

clay = (
    S[
        "clay"
    ].astype(float)
    /
    100.0
)


ksat = S[
    "ksat"
].astype(float)


PI = P[
    "plasticity_index"
].astype(float)


# ============================================================
# TERRAIN / VEGETATION
# ============================================================

slope = S[
    "slope"
].astype(float)

northness = S[
    "northness"
].astype(float)

eastness = S[
    "eastness"
].astype(float)

southness = S[
    "southness"
].astype(float)

elevation = S[
    "elevation"
].astype(float)


tree = S[
    "tree_fraction"
].astype(float)

shrub = S[
    "shrub_cover_fraction"
].astype(float)

herb = S[
    "herb_cover_fraction"
].astype(float)
# ============================================================
# STATIC FACTORS
# ============================================================

texture_factor = (
    1.0
    +
    0.50
    *
    sand
    -
    0.80
    *
    clay
)


texture_factor = np.clip(
    texture_factor,
    0.5,
    1.5,
)


# ------------------------------------------------------------
# Normalize Ksat to 5-95 percentile range
# ------------------------------------------------------------

logk = np.full(
    ksat.shape,
    np.nan,
    dtype=float,
)


good_k = (
    physics
    &
    np.isfinite(
        ksat
    )
    &
    (
        ksat > 0
    )
)


logk[
    good_k
] = np.log10(
    ksat[
        good_k
    ]
)


klo = np.nanpercentile(
    logk[
        good_k
    ],
    5,
)

khi = np.nanpercentile(
    logk[
        good_k
    ],
    95,
)


Knorm = np.clip(
    (
        logk
        -
        klo
    )
    /
    (
        khi
        -
        klo
    ),
    0.0,
    1.0,
)


Kfactor = (
    0.6
    +
    0.8
    *
    Knorm
)


# ------------------------------------------------------------
# Spatial shallow -> deep transfer rate
# ------------------------------------------------------------

shallow_transfer_rate = (
    SHALLOW_TRANSFER_BASE_DAY
    *
    Kfactor
    *
    texture_factor
)


shallow_transfer_rate = np.clip(
    shallow_transfer_rate,
    16.0,
    128.0,
)


# ------------------------------------------------------------
# Deep drainage
# ------------------------------------------------------------

deep_drain_rate = (
    DEEP_DRAIN_BASE_DAY
    *
    Kfactor
    *
    texture_factor
)


deep_drain_rate = np.clip(
    deep_drain_rate,
    2.0,
    20.0,
)


# ------------------------------------------------------------
# Drying modifiers
# ------------------------------------------------------------

aspect_factor = np.clip(
    1.0
    +
    0.35
    *
    southness,
    0.65,
    1.35,
)


vegetation_factor = np.clip(
    1.0
    -
    0.55
    *
    tree
    -
    0.30
    *
    shrub
    -
    0.15
    *
    herb,
    0.45,
    1.0,
)


# ============================================================
# EFFECTIVE SURFACE KSAT
#
# SSURGO ksat_r:
#     micrometers / second
#
# 1 um/s = 3.6 mm/hour
# ============================================================

ksat_mm_h = (
    3.6
    *
    ksat
)


Keff_mm_h = (
    KSAT_EFFICIENCY
    *
    ksat_mm_h
)


# ============================================================
# HELPER FOR NPZ KEYS
# ============================================================

def get_first_key(
    archive,
    names,
):

    for name in names:

        if name in archive.files:

            return archive[
                name
            ]


    raise KeyError(
        "Could not find any of these keys:\n"
        f"{names}\n\n"
        "Available keys are:\n"
        f"{archive.files}"
    )


# ============================================================
# DYNAMIC INPUTS
# ============================================================

mrms_time = get_first_key(
    MRMS,
    [
        "time",
        "times",
    ],
).astype(
    "datetime64[s]"
)


mrms_precip = get_first_key(
    MRMS,
    [
        "precip_mm",
        "precipitation_mm",
        "precip",
    ],
).astype(float)


nws_time = get_first_key(
    NWS,
    [
        "time",
        "times",
    ],
).astype(
    "datetime64[s]"
)


nws_temp = get_first_key(
    NWS,
    [
        "temperature_c",
        "temperature",
    ],
).astype(float)


nws_rh = get_first_key(
    NWS,
    [
        "relative_humidity",
        "relative_humidity_percent",
        "rh",
    ],
).astype(float)


nws_wind = get_first_key(
    NWS,
    [
        "wind_speed_ms",
        "wind_speed",
        "wind",
    ],
).astype(float)


nws_cloud = get_first_key(
    NWS,
    [
        "sky_cover_percent",
        "cloud_cover_percent",
        "sky_cover",
        "cloud",
    ],
).astype(float)


nws_precip = get_first_key(
    NWS,
    [
        "precip_mm",
        "precipitation_mm",
        "quantitative_precipitation_mm",
        "qpf_mm",
        "precipitation",
    ],
).astype(float)


# ============================================================
# SMAP INITIALIZATION
# ============================================================

smap_mean = float(
    SMAP[
        "mean_soil_moisture"
    ]
)


smap_time = SMAP[
    "observation_time_utc"
].astype(
    "datetime64[s]"
)


fc_mean = np.nanmean(
    fc[
        physics
    ]
)


theta05 = (
    smap_mean
    +
    0.35
    *
    (
        fc
        -
        fc_mean
    )
)


# Force regional mean to SMAP mean.

mean_before = np.nanmean(
    theta05[
        physics
    ]
)


theta05 += (
    smap_mean
    -
    mean_before
)


theta05 = np.clip(
    theta05,
    wp,
    sat,
)


theta05[
    ~physics
] = np.nan


# Initial tread and shallow states together represent SMAP
# upper-5-cm moisture.

tread = theta05.copy()

shallow = theta05.copy()


# Existing deep initialization retained.

deep = (
    wp
    +
    0.70
    *
    (
        fc
        -
        wp
    )
)


deep = np.maximum(
    deep,
    theta05
    -
    0.01,
)


deep = np.clip(
    deep,
    wp,
    sat,
)


deep[
    ~physics
] = np.nan


# ============================================================
# HERO SCORE
# ============================================================

def score_state(
    tread,
    shallow,
    deep,
    valid_mask=public,
):

    subsurface = (
        ZSHALLOW
        *
        shallow
        +
        ZDEEP
        *
        deep
    ) / (
        ZSHALLOW
        +
        ZDEEP
    )


    result = calculate_hero_score(

        surf=tread,

        deep=subsurface,

        theta_wp=wp,

        theta_fc=fc,

        theta_sat=sat,

        plasticity_index=PI,

        public_valid=valid_mask,
    )


    return (
        result[
            "score"
        ],

        result[
            "condition"
        ],

        result[
            "F"
        ],

        subsurface,
    )


# ============================================================
# UPWARD SHALLOW -> TREAD EXCHANGE
# ============================================================

def upward_exchange(
    Wt,
    Ws,
    dt_hours,
):

    """
    Mass-conserving moisture equilibration.

    Only operates when:

        theta_shallow > theta_tread

    Moisture difference relaxes exponentially with
    timescale UPWARD_TAU_HOURS.
    """


    theta_t = (
        Wt
        /
        ZTREAD
    )


    theta_s = (
        Ws
        /
        ZSHALLOW
    )


    delta = np.maximum(
        theta_s
        -
        theta_t,
        0.0,
    )


    # Effective two-layer storage depth.

    Zeq = (
        ZTREAD
        *
        ZSHALLOW
        /
        (
            ZTREAD
            +
            ZSHALLOW
        )
    )


    fraction = (
        1.0
        -
        np.exp(
            -dt_hours
            /
            UPWARD_TAU_HOURS
        )
    )


    Qup = (
        delta
        *
        Zeq
        *
        fraction
    )


    # Never remove shallow water below WP.

    shallow_available = np.maximum(
        Ws
        -
        wp
        *
        ZSHALLOW,
        0.0,
    )


    Qup = np.minimum(
        Qup,
        shallow_available,
    )


    # Never fill tread beyond saturation.

    tread_space = np.maximum(
        sat
        *
        ZTREAD
        -
        Wt,
        0.0,
    )


    Qup = np.minimum(
        Qup,
        tread_space,
    )


    Qup = np.maximum(
        Qup,
        0.0,
    )


    Qup[
        ~physics
    ] = 0.0


    Wt += Qup

    Ws -= Qup


    return (
        Wt,
        Ws,
        Qup,
    )


# ============================================================
# ONE MODEL SUBSTEP
# ============================================================
def model_substep(
    tread,
    shallow,
    deep,
    rain,
    dt_hours,
    temperature,
    rh,
    wind,
    cloud,
    penman_tread_et=None,
):

    dt_day = (
        dt_hours
        /
        24.0
    )


    # ========================================================
    # STORAGE [mm]
    # ========================================================

    Wt = (
        tread
        *
        ZTREAD
    )

    Ws = (
        shallow
        *
        ZSHALLOW
    )

    Wd = (
        deep
        *
        ZDEEP
    )


    Wt_wp = (
        wp
        *
        ZTREAD
    )

    Wt_fc = (
        fc
        *
        ZTREAD
    )

    Wt_sat = (
        sat
        *
        ZTREAD
    )


    Ws_wp = (
        wp
        *
        ZSHALLOW
    )

    Ws_fc = (
        fc
        *
        ZSHALLOW
    )

    Ws_sat = (
        sat
        *
        ZSHALLOW
    )


    Wd_wp = (
        wp
        *
        ZDEEP
    )

    Wd_fc = (
        fc
        *
        ZDEEP
    )

    Wd_sat = (
        sat
        *
        ZDEEP
    )


    # ========================================================
    # RAIN
    # ========================================================

    rain = np.where(
        np.isfinite(
            rain
        ),
        np.maximum(
            rain,
            0.0,
        ),
        0.0,
    )


    Wt += rain


    # ========================================================
    # TREAD -> SHALLOW INFILTRATION
    # ========================================================

    Se = np.clip(
        (
            tread
            -
            wp
        )
        /
        (
            sat
            -
            wp
        ),
        0.0,
        1.0,
    )


    moisture_multiplier = (
        1.0
        +
        DRY_BOOST
        *
        (
            1.0
            -
            Se
        )
    )


    infiltration_rate = (
        Keff_mm_h
        *
        moisture_multiplier
        *
        texture_factor
    )


    infiltration_capacity = (
        infiltration_rate
        *
        dt_hours
    )


    tread_available = np.maximum(
        Wt
        -
        Wt_fc,
        0.0,
    )


    shallow_space = np.maximum(
        Ws_sat
        -
        Ws,
        0.0,
    )


    tread_to_shallow = np.minimum(
        infiltration_capacity,
        tread_available,
    )


    tread_to_shallow = np.minimum(
        tread_to_shallow,
        shallow_space,
    )


    tread_to_shallow = np.maximum(
        tread_to_shallow,
        0.0,
    )


    tread_to_shallow[
        ~physics
    ] = 0.0


    Wt -= tread_to_shallow

    Ws += tread_to_shallow


    # ========================================================
    # SATURATION-EXCESS RUNOFF
    # ========================================================

    runoff = np.maximum(
        Wt
        -
        Wt_sat,
        0.0,
    )


    runoff[
        ~physics
    ] = 0.0


    Wt = np.minimum(
        Wt,
        Wt_sat,
    )


    # ========================================================
    # SHALLOW -> DEEP
    # ========================================================

    shallow_excess = np.clip(
        (
            Ws
            -
            Ws_fc
        )
        /
        (
            Ws_sat
            -
            Ws_fc
        ),
        0.0,
        1.0,
    )


    shallow_capacity = (
        shallow_transfer_rate
        *
        dt_day
    )


    shallow_to_deep = (
        shallow_capacity
        *
        shallow_excess
    )


    shallow_to_deep = np.minimum(
        shallow_to_deep,
        np.maximum(
            Ws
            -
            Ws_fc,
            0.0,
        ),
    )


    shallow_to_deep = np.minimum(
        shallow_to_deep,
        np.maximum(
            Wd_sat
            -
            Wd,
            0.0,
        ),
    )


    shallow_to_deep = np.maximum(
        shallow_to_deep,
        0.0,
    )


    shallow_to_deep[
        ~physics
    ] = 0.0


    Ws -= shallow_to_deep

    Wd += shallow_to_deep


    # ========================================================
    # DEEP DRAINAGE
    # ========================================================

    deep_excess = np.clip(
        (
            Wd
            -
            Wd_fc
        )
        /
        (
            Wd_sat
            -
            Wd_fc
        ),
        0.0,
        1.0,
    )


    deep_drainage = (
        deep_drain_rate
        *
        dt_day
        *
        deep_excess
    )


    deep_drainage = np.minimum(
        deep_drainage,
        np.maximum(
            Wd
            -
            Wd_fc,
            0.0,
        ),
    )


    deep_drainage = np.maximum(
        deep_drainage,
        0.0,
    )


    deep_drainage[
        ~physics
    ] = 0.0


    Wd -= deep_drainage


    # ========================================================
    # UPWARD SHALLOW -> TREAD EXCHANGE
    # ========================================================

    (
        Wt,
        Ws,
        upward_flux,
    ) = upward_exchange(
        Wt,
        Ws,
        dt_hours,
    )


    # ========================================================
    # METEOROLOGICAL DRYING
    # ========================================================

    Tfactor = np.clip(
        (
            temperature
            -
            5.0
        )
        /
        20.0,
        0.2,
        1.5,
    )


    RHfactor = np.clip(
        1.0
        -
        rh
        /
        100.0,
        0.10,
        1.0,
    )


    wind_factor = np.clip(
        0.7
        +
        0.15
        *
        wind,
        0.7,
        1.4,
    )


    cloud_factor = np.clip(
        1.0
        -
        0.50
        *
        cloud
        /
        100.0,
        0.5,
        1.0,
    )


    # --------------------------------------------------------
    # Tread ET
    # --------------------------------------------------------

    tread_moisture = np.clip(
        (
            Wt
            -
            Wt_wp
        )
        /
        (
            Wt_fc
            -
            Wt_wp
        ),
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Tread ET
    #
    # v1 fallback:
    #     empirical Hero Dirt drying formulation
    #
    # v2:
    #     Penman potential evaporation supplied for this
    #     substep, scaled for trail tread and by available
    #     tread moisture.
    # --------------------------------------------------------

    if (
        USE_PENMAN_TREAD_ET
        and
        penman_tread_et is not None
    ):

        ET_tread = (
            K_TREAD
            *
            penman_tread_et
            *
            tread_moisture
        )

    else:

        ET_tread = (
            ET_TREAD_BASE_DAY
            *
            dt_day
            *
            Tfactor
            *
            RHfactor
            *
            wind_factor
            *
            cloud_factor
            *
            aspect_factor
            *
            vegetation_factor
            *
            tread_moisture
        )


    ET_tread = np.minimum(
        ET_tread,
        np.maximum(
            Wt
            -
            Wt_wp,
            0.0,
        ),
    )


    ET_tread[
        ~physics
    ] = 0.0


    Wt -= ET_tread


    # --------------------------------------------------------
    # Shallow ET
    # --------------------------------------------------------

    shallow_moisture = np.clip(
        (
            Ws
            -
            Ws_wp
        )
        /
        (
            Ws_fc
            -
            Ws_wp
        ),
        0.0,
        1.0,
    )


    ET_shallow = (
        ET_SHALLOW_BASE_DAY
        *
        dt_day
        *
        Tfactor
        *
        RHfactor
        *
        wind_factor
        *
        shallow_moisture
    )


    ET_shallow = np.minimum(
        ET_shallow,
        np.maximum(
            Ws
            -
            Ws_wp,
            0.0,
        ),
    )


    ET_shallow[
        ~physics
    ] = 0.0


    Ws -= ET_shallow


    # --------------------------------------------------------
    # Deep ET
    # --------------------------------------------------------

    deep_moisture = np.clip(
        (
            Wd
            -
            Wd_wp
        )
        /
        (
            Wd_fc
            -
            Wd_wp
        ),
        0.0,
        1.0,
    )


    ET_deep = (
        ET_DEEP_BASE_DAY
        *
        dt_day
        *
        Tfactor
        *
        RHfactor
        *
        wind_factor
        *
        deep_moisture
    )


    ET_deep = np.minimum(
        ET_deep,
        np.maximum(
            Wd
            -
            Wd_wp,
            0.0,
        ),
    )


    ET_deep[
        ~physics
    ] = 0.0


    Wd -= ET_deep


    # ========================================================
    # BACK TO THETA
    # ========================================================

    tread = np.clip(
        Wt
        /
        ZTREAD,
        wp,
        sat,
    )


    shallow = np.clip(
        Ws
        /
        ZSHALLOW,
        wp,
        sat,
    )


    deep = np.clip(
        Wd
        /
        ZDEEP,
        wp,
        sat,
    )


    tread[
        ~physics
    ] = np.nan

    shallow[
        ~physics
    ] = np.nan

    deep[
        ~physics
    ] = np.nan


    return (
        tread,
        shallow,
        deep,
        tread_to_shallow,
        shallow_to_deep,
        deep_drainage,
        upward_flux,
        ET_tread,
        ET_shallow,
        ET_deep,
        runoff,
    )


# ============================================================
# ADVANCE AN INTERVAL
# ============================================================
def advance_interval(
    tread,
    shallow,
    deep,
    total_rain,
    total_hours,
    temperature,
    rh,
    wind,
    cloud,
    penman_tread_et=None,
):

    if total_hours <= 0:

        zeros = np.zeros(
            tread.shape,
            dtype=float,
        )

        return (
            tread,
            shallow,
            deep,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        )


    finite_rain = np.where(
        np.isfinite(
            total_rain
        ),
        total_rain,
        0.0,
    )


    has_rain = (
        np.max(
            finite_rain
        )
        >
        1.0e-8
    )


    if has_rain:

        max_dt = (
            RAIN_SUBSTEP_HOURS
        )

    else:

        max_dt = (
            DRY_SUBSTEP_HOURS
        )


    nsteps = max(
        1,
        int(
            np.ceil(
                total_hours
                /
                max_dt
            )
        ),
    )


    dt = (
        total_hours
        /
        nsteps
    )


    rain_step = (
        finite_rain
        /
        nsteps
    )

    if penman_tread_et is None:

        penman_step = None

    else:

        penman_step = (
            penman_tread_et
            /
            nsteps
        )

    totals = [
        np.zeros(
            tread.shape,
            dtype=float,
        )
        for _ in range(
            8
        )
    ]


    for _ in range(
        nsteps
    ):

        (
            tread,
            shallow,
            deep,
            ts,
            sd,
            drainage,
            qup,
            et_t,
            et_s,
            et_d,
            runoff,
        ) = model_substep(

            tread,
            shallow,
            deep,

            rain=rain_step,

            dt_hours=dt,

            temperature=temperature,

            rh=rh,

            wind=wind,

            cloud=cloud,

            penman_tread_et=penman_step,
        )


        totals[0] += ts

        totals[1] += sd

        totals[2] += drainage

        totals[3] += qup

        totals[4] += et_t

        totals[5] += et_s

        totals[6] += et_d

        totals[7] += runoff


    return (
        tread,
        shallow,
        deep,
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals[4],
        totals[5],
        totals[6],
        totals[7],
    )


# ============================================================
# ANALYSIS PERIOD
# ============================================================

if "forecast_start" in NWS.files:

    forecast_start = NWS[
        "forecast_start"
    ].astype(
        "datetime64[s]"
    )

else:

    # Backward compatibility with older forcing files
    # whose first interval was always exactly 6 hours.

    forecast_start = (
        nws_time[
            0
        ]
        -
        np.timedelta64(
            6,
            "h",
        )
    )


analysis_indices = np.where(
    (
        mrms_time
        >
        smap_time
    )
    &
    (
        mrms_time
        <=
        forecast_start
    )
)[0]


print()
print(
    "============================================"
)
print(
    " HERO DIRT THREE-LAYER OPERATIONAL MODEL"
)
print(
    "============================================"
)
print()


print(
    f"SMAP observation: "
    f"{smap_time}"
)

print(
    f"Forecast start:   "
    f"{forecast_start}"
)

print(
    f"MRMS fields:      "
    f"{len(analysis_indices)}"
)

print(
    f"Upward tau:       "
    f"{UPWARD_TAU_HOURS:.1f} h"
)

print()


# ============================================================
# ANALYSIS MET PROXY
#
# Known limitation:
# first NWS forecast field is still used as the atmospheric
# drying proxy during the observed MRMS period.
# ============================================================

analysis_T = nws_temp[
    0
]

analysis_RH = nws_rh[
    0
]

analysis_wind = nws_wind[
    0
]

analysis_cloud = nws_cloud[
    0
]


# ============================================================
# ANALYSIS ACCUMULATORS
# ============================================================

analysis_total_rain = np.zeros(
    tread.shape,
    dtype=float,
)

analysis_total_runoff = np.zeros(
    tread.shape,
    dtype=float,
)

analysis_total_upward = np.zeros(
    tread.shape,
    dtype=float,
)


# ============================================================
# PRE-MRMS DRY GAP
# ============================================================

if len(
    analysis_indices
) > 0:

    first_mrms_time = mrms_time[
        analysis_indices[
            0
        ]
    ]


    pre_hours = (
        first_mrms_time
        -
        smap_time
    ).astype(
        "timedelta64[s]"
    ).astype(
        float
    ) / 3600.0


    pre_hours = max(
        pre_hours
        -
        1.0,
        0.0,
    )


    if pre_hours > 0:

        zero_rain = np.zeros(
            tread.shape,
            dtype=float,
        )
        pre_end_time = (
            first_mrms_time
            -
            np.timedelta64(
                1,
                "h",
            )
        )


        pre_penman_et = compute_penman_et_block(
            block_end=pre_end_time,

            block_hours=pre_hours,

            temperature=analysis_T,

            rh=analysis_RH,

            wind=analysis_wind,

            cloud=analysis_cloud,

            elevation=elevation,

            lat_rad=lat_rad,

            lon_deg=lon,

            slope_deg=slope,

            eastness=eastness,

            northness=northness,

            tree=tree,

            shrub=shrub,

            herb=herb,

            physics=physics,
        )

        (
            tread,
            shallow,
            deep,
            ts,
            sd,
            drainage,
            qup,
            et_t,
            et_s,
            et_d,
            runoff,
        ) = advance_interval(

            tread,
            shallow,
            deep,

            total_rain=zero_rain,

            total_hours=pre_hours,

            temperature=analysis_T,

            rh=analysis_RH,

            wind=analysis_wind,

            cloud=analysis_cloud,
            
            penman_tread_et=pre_penman_et,
          )


        analysis_total_upward += qup


# ============================================================
# HOURLY MRMS REPLAY
# ============================================================

for idx in analysis_indices:

    rain = mrms_precip[
        idx
    ]
    analysis_penman_et = compute_penman_et_block(
        block_end=mrms_time[
            idx
        ],

        block_hours=1.0,

        temperature=analysis_T,

        rh=analysis_RH,

        wind=analysis_wind,

        cloud=analysis_cloud,

        elevation=elevation,

        lat_rad=lat_rad,

        lon_deg=lon,

        slope_deg=slope,

        eastness=eastness,

        northness=northness,

        tree=tree,

        shrub=shrub,

        herb=herb,

        physics=physics,
    )

    (
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        qup,
        et_t,
        et_s,
        et_d,
        runoff,
    ) = advance_interval(

        tread,
        shallow,
        deep,

        total_rain=rain,

        total_hours=1.0,

        temperature=analysis_T,

        rh=analysis_RH,

        wind=analysis_wind,

        cloud=analysis_cloud,
        penman_tread_et=analysis_penman_et,

    )


    analysis_total_rain += np.where(
        np.isfinite(
            rain
        ),
        np.maximum(
            rain,
            0.0,
        ),
        0.0,
    )


    analysis_total_runoff += runoff

    analysis_total_upward += qup


# ============================================================
# GAP TO FORECAST START
# ============================================================

if len(
    analysis_indices
) > 0:

    last_analysis_time = mrms_time[
        analysis_indices[
            -1
        ]
    ]

else:

    last_analysis_time = (
        smap_time
    )


final_gap_hours = (
    forecast_start
    -
    last_analysis_time
).astype(
    "timedelta64[s]"
).astype(
    float
) / 3600.0


if final_gap_hours > 0:

    zero_rain = np.zeros(
        tread.shape,
        dtype=float,
    )
    final_penman_et = compute_penman_et_block(
        block_end=forecast_start,

        block_hours=final_gap_hours,

        temperature=analysis_T,

        rh=analysis_RH,

        wind=analysis_wind,

        cloud=analysis_cloud,

        elevation=elevation,

        lat_rad=lat_rad,

        lon_deg=lon,

        slope_deg=slope,

        eastness=eastness,

        northness=northness,

        tree=tree,

        shrub=shrub,

        herb=herb,

        physics=physics,
    )

    (
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        qup,
        et_t,
        et_s,
        et_d,
        runoff,
    ) = advance_interval(

        tread,
        shallow,
        deep,

        total_rain=zero_rain,

        total_hours=final_gap_hours,

        temperature=analysis_T,

        rh=analysis_RH,

        wind=analysis_wind,

        cloud=analysis_cloud,

        penman_tread_et=final_penman_et,

    )


    analysis_total_upward += qup


# ============================================================
# CURRENT SCORE
# ============================================================

(
    current_score,
    current_condition,
    current_F,
    current_subsurface,
) = score_state(

    tread,
    shallow,
    deep,
)


# Full physics-domain score for the science Explorer.
# Operational rider-facing products continue to use
# current_score, which remains masked by public_valid.

(
    current_score_physics,
    _,
    _,
    _,
) = score_state(

    tread,
    shallow,
    deep,
    valid_mask=physics,
)


current_theta05 = (
    ZTREAD
    *
    tread
    +
    ZSHALLOW
    *
    shallow
) / (
    ZTREAD
    +
    ZSHALLOW
)

# Freeze the analysis state before forecast integration.
# The live tread/shallow/deep arrays are modified in-place
# during the forecast loop.

current_tread = tread.copy()

current_shallow = shallow.copy()

current_deep = deep.copy()
# ============================================================
# SAVE CURRENT NPZ
# ============================================================

CURRENT_NPZ = (
    CURRENT_DIR
    / "HeroDirt_current_state.npz"
)


np.savez_compressed(

    CURRENT_NPZ,

    time=forecast_start,

    x=x.astype(
        np.float32
    ),

    y=y.astype(
        np.float32
    ),

    # --------------------------------------------------------
    # Backward-compatible aliases
    # --------------------------------------------------------

    surface_theta=tread.astype(
        np.float32
    ),

    deep_theta=current_subsurface.astype(
        np.float32
    ),

    F=current_F.astype(
        np.float32
    ),

    score=current_score.astype(
        np.float32
    ),

    score_physics=current_score_physics.astype(
        np.float32
    ),

    hero_score=current_score.astype(
        np.float32
    ),

    condition=current_condition.astype(
        np.int8
    ),

    hero_class=current_condition.astype(
        np.int8
    ),

    # --------------------------------------------------------
    # Three-layer fields
    # --------------------------------------------------------

    tread_theta=tread.astype(
        np.float32
    ),

    shallow_theta=shallow.astype(
        np.float32
    ),

    physical_deep_theta=deep.astype(
        np.float32
    ),

    theta_0_5cm=current_theta05.astype(
        np.float32
    ),

    subsurface_theta=current_subsurface.astype(
        np.float32
    ),

    analysis_precip_mm=
        analysis_total_rain.astype(
            np.float32
        ),

    analysis_runoff_mm=
        analysis_total_runoff.astype(
            np.float32
        ),

    analysis_upward_exchange_mm=
        analysis_total_upward.astype(
            np.float32
        ),

    shallow_transfer_rate_mm_day=
        shallow_transfer_rate.astype(
            np.float32
        ),

    effective_ksat_mm_h=
        Keff_mm_h.astype(
            np.float32
        ),

    upward_tau_hours=
        np.float32(
            UPWARD_TAU_HOURS
        ),

    public_valid=public,
)


# ============================================================
# GEOTIFF WRITER
# ============================================================

with rasterio.open(
    GRID_TIF
) as src:

    base_profile = (
        src.profile.copy()
    )


def write_float_tif(
    path,
    A,
):

    profile = (
        base_profile.copy()
    )


    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
    )


    out = A.astype(
        np.float32
    )


    out = np.where(
        np.isfinite(
            out
        )
        &
        public,
        out,
        -9999.0,
    )


    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            out,
            1,
        )


def write_class_tif(
    path,
    A,
):

    profile = (
        base_profile.copy()
    )


    profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="deflate",
    )


    out = np.where(
        public,
        A,
        0,
    ).astype(
        np.uint8
    )


    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            out,
            1,
        )


# ============================================================
# CURRENT TIFFS
# ============================================================

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_surface_theta.tif",
    tread,
)

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_tread_theta.tif",
    tread,
)

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_shallow_theta.tif",
    shallow,
)

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_deep_theta.tif",
    deep,
)

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_theta_0_5cm.tif",
    current_theta05,
)

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_F.tif",
    current_F,
)

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_score.tif",
    current_score,
)

write_float_tif(
    CURRENT_DIR
    / "HeroDirt_analysis_runoff.tif",
    analysis_total_runoff,
)

write_class_tif(
    CURRENT_DIR
    / "HeroDirt_class.tif",
    current_condition,
)


# ============================================================
# FORECAST STORAGE
# ============================================================

nt = len(
    nws_time
)


forecast_tread = np.full(
    (
        nt,
        ny,
        nx,
    ),
    np.nan,
    dtype=np.float32,
)


forecast_shallow = np.full(
    (
        nt,
        ny,
        nx,
    ),
    np.nan,
    dtype=np.float32,
)


forecast_deep = np.full(
    (
        nt,
        ny,
        nx,
    ),
    np.nan,
    dtype=np.float32,
)


forecast_theta05 = np.full(
    (
        nt,
        ny,
        nx,
    ),
    np.nan,
    dtype=np.float32,
)


forecast_subsurface = np.full(
    (
        nt,
        ny,
        nx,
    ),
    np.nan,
    dtype=np.float32,
)


forecast_F = np.full(
    (
        nt,
        ny,
        nx,
    ),
    np.nan,
    dtype=np.float32,
)


forecast_score = np.full(
    (
        nt,
        ny,
        nx,
    ),
    np.nan,
    dtype=np.float32,
)


forecast_condition = np.zeros(
    (
        nt,
        ny,
        nx,
    ),
    dtype=np.int8,
)


forecast_runoff = np.zeros(
    (
        nt,
        ny,
        nx,
    ),
    dtype=np.float32,
)


forecast_upward = np.zeros(
    (
        nt,
        ny,
        nx,
    ),
    dtype=np.float32,
)


# ============================================================
# FORECAST INTEGRATION
# ============================================================

previous_time = (
    forecast_start
)


for k in range(
    nt
):

    hours = (
        nws_time[
            k
        ]
        -
        previous_time
    ).astype(
        "timedelta64[s]"
    ).astype(
        float
    ) / 3600.0


    if hours <= 0:

        hours = 6.0


    rain = nws_precip[
        k
    ]

    # ========================================================
    # V2 PENMAN TREAD DRYING
    #
    # Potential evaporation [mm] over this forecast interval.
    #
    # This replaces the empirical v1 tread-drying demand
    # during the forecast only.
    #
    # The analysis / MRMS replay still uses the original
    # Hero Dirt drying formulation.
    # ========================================================

    penman_et = compute_penman_et_block(
        block_end=nws_time[
            k
        ],

        block_hours=hours,

        temperature=nws_temp[
            k
        ],

        rh=nws_rh[
            k
        ],

        wind=nws_wind[
            k
        ],

        cloud=nws_cloud[
            k
        ],

        elevation=elevation,

        lat_rad=lat_rad,

        lon_deg=lon,

        slope_deg=slope,

        eastness=eastness,

        northness=northness,

        tree=tree,

        shrub=shrub,

        herb=herb,

        physics=physics,
    )

    (
        tread,
        shallow,
        deep,
        ts,
        sd,
        drainage,
        qup,
        et_t,
        et_s,
        et_d,
        runoff,
    ) = advance_interval(

        tread,
        shallow,
        deep,

        total_rain=rain,

        total_hours=hours,

        temperature=nws_temp[
            k
        ],

        rh=nws_rh[
            k
        ],

        wind=nws_wind[
            k
        ],

        cloud=nws_cloud[
            k
        ],

        penman_tread_et=penman_et,
    )

    (
        score,
        condition,
        F,
        subsurface,
    ) = score_state(

        tread,
        shallow,
        deep,
    )


    theta05_now = (
        ZTREAD
        *
        tread
        +
        ZSHALLOW
        *
        shallow
    ) / (
        ZTREAD
        +
        ZSHALLOW
    )


    forecast_tread[
        k
    ] = tread.astype(
        np.float32
    )


    forecast_shallow[
        k
    ] = shallow.astype(
        np.float32
    )


    forecast_deep[
        k
    ] = deep.astype(
        np.float32
    )


    forecast_theta05[
        k
    ] = theta05_now.astype(
        np.float32
    )


    forecast_subsurface[
        k
    ] = subsurface.astype(
        np.float32
    )


    forecast_F[
        k
    ] = F.astype(
        np.float32
    )


    forecast_score[
        k
    ] = score.astype(
        np.float32
    )


    forecast_condition[
        k
    ] = condition.astype(
        np.int8
    )


    forecast_runoff[
        k
    ] = runoff.astype(
        np.float32
    )


    forecast_upward[
        k
    ] = qup.astype(
        np.float32
    )


    previous_time = nws_time[
        k
    ]


# ============================================================
# SAVE FORECAST
# ============================================================

FORECAST_NPZ = (
    FORECAST_DIR
    / "HeroDirt_forecast_6hourly.npz"
)


np.savez_compressed(

    FORECAST_NPZ,

    time=nws_time,

    x=x.astype(
        np.float32
    ),

    y=y.astype(
        np.float32
    ),

    # Backward-compatible aliases

    surface_theta=forecast_tread,

    deep_theta=forecast_subsurface,

    F=forecast_F,

    score=forecast_score,

    hero_score=forecast_score,

    condition=forecast_condition,

    hero_class=forecast_condition,

    # Three-layer fields

    tread_theta=forecast_tread,

    shallow_theta=forecast_shallow,

    physical_deep_theta=forecast_deep,

    theta_0_5cm=forecast_theta05,

    subsurface_theta=forecast_subsurface,

    precip_mm=nws_precip.astype(
        np.float32
    ),

    runoff_mm=forecast_runoff,

    upward_exchange_mm=forecast_upward,

    upward_tau_hours=
        np.float32(
            UPWARD_TAU_HOURS
        ),

    public_valid=public,
)


# ============================================================
# SUMMARY HELPERS
# ============================================================

def masked_mean(
    A,
):

    return np.nanmean(
        A[
            public
        ]
    )


# ============================================================
# CURRENT SUMMARY
# ============================================================

print(
    "Current three-layer state:"
)

print(
    f"  tread theta:      "
    f"{masked_mean(current_tread):.4f}"
)

print(
    f"  shallow theta:    "
    f"{masked_mean(current_shallow):.4f}"
)

print(
    f"  deep theta:       "
    f"{masked_mean(current_deep):.4f}"
)

print(
    f"  combined 0-5 cm:  "
    f"{masked_mean(current_theta05):.4f}"
)

print(
    f"  tread F:          "
    f"{masked_mean(current_F):.3f}"
)

print(
    f"  Hero score:       "
    f"{masked_mean(current_score):.1f}"
)

print()


print(
    "Analysis water:"
)

print(
    f"  domain-mean rain:   "
    f"{masked_mean(analysis_total_rain):.3f} mm"
)

print(
    f"  domain-mean runoff: "
    f"{masked_mean(analysis_total_runoff):.3f} mm"
)

print(
    f"  mean upward flux:   "
    f"{masked_mean(analysis_total_upward):.3f} mm"
)

print()


print(
    "Current classes:"
)


class_names = {
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


for class_id in range(
    1,
    7
):

    count = np.count_nonzero(
        public
        &
        (
            current_condition
            ==
            class_id
        )
    )


    pct = (
        100.0
        *
        count
        /
        np.count_nonzero(
            public
        )
    )


    print(
        f"  {class_names[class_id]:10s}: "
        f"{pct:5.1f}%"
    )


print()


# ============================================================
# FORECAST SUMMARY
# ============================================================

for target_hours in [
    24,
    48,
    72,
    96,
    120,
]:

    target_time = (
        forecast_start
        +
        np.timedelta64(
            target_hours,
            "h",
        )
    )


    k = int(
        np.argmin(
            np.abs(
                nws_time
                -
                target_time
            )
        )
    )


    print(
        f"+{target_hours:3d} h:"
        f" tread={masked_mean(forecast_tread[k]):.4f}"
        f" F={masked_mean(forecast_F[k]):.3f}"
        f" score={masked_mean(forecast_score[k]):.1f}"
    )


print()


print(
    "Saved:"
)

print(
    f"  {CURRENT_NPZ}"
)

print(
    f"  {FORECAST_NPZ}"
)

print()


print(
    "============================================"
)
print(
    " THREE-LAYER OPERATIONAL RUN COMPLETE"
)
print(
    "============================================"
)
print()
