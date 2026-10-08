#!/usr/bin/env python3

"""
Hero Dirt score v0.3

Hydraulic state
---------------
F = (theta - theta_wp) / (theta_fc - theta_wp)

F = 0 : wilting point
F = 1 : field capacity

Plasticity
----------
SSURGO plasticity index (PI) is used only as a measure of
wet-condition mechanical sensitivity.

It is NOT used as a direct moisture threshold because
Atterberg limits apply to the remolded fine fraction rather
than intact rocky trail tread.

Empirical Hero Dirt calibration parameters are explicitly
identified below.
"""

import numpy as np


# ============================================================
# HERO DIRT EMPIRICAL CALIBRATION PARAMETERS
# ============================================================

# Dry-side suitability is essentially unrestricted once
# the soil reaches this fraction of the WP-to-FC interval.
F_DRY_FULL = 0.50


# Wet-condition sensitivity begins before field capacity.
F_WET_START = 0.80


# Wet-deformation penalty strengths.
WET_BASE_STRENGTH = 1.5
WET_PI_STRENGTH = 3.0


# Deep-layer wetness penalty.
DEEP_WET_STRENGTH = 0.50


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "Masked",
    1: "Very Dry",
    2: "Dry",
    3: "Moist/Good",
    4: "Hero",
    5: "Wet",
    6: "Too Wet",
}


# ============================================================
# UTILITIES
# ============================================================

def smoothstep(x):

    x = np.clip(
        x,
        0.0,
        1.0,
    )

    return (
        x
        *
        x
        *
        (
            3.0
            -
            2.0
            *
            x
        )
    )


# ============================================================
# SCORE
# ============================================================

def calculate_hero_score(
    surf,
    deep,
    theta_wp,
    theta_fc,
    theta_sat,
    plasticity_index,
    public_valid,
):

    """
    Calculate Hero Dirt v0.3 score and directional condition.

    Parameters
    ----------
    surf
        Surface volumetric water content.

    deep
        Deep-layer volumetric water content.

    theta_wp
        Wilting-point volumetric water content.

    theta_fc
        Field-capacity volumetric water content.

    theta_sat
        Saturated volumetric water content.

    plasticity_index
        SSURGO PI [%].

    public_valid
        Boolean public-product mask.

    Returns
    -------
    score
        Continuous 0-100 Hero Dirt suitability score.

    condition
        Directional class:
            1 Very Dry
            2 Dry
            3 Moist/Good
            4 Hero
            5 Wet
            6 Too Wet

    F
        Surface hydraulic state relative to WP->FC.

    Fdeep
        Deep hydraulic state relative to WP->FC.

    pi_sensitivity
        Bounded mechanical wetness sensitivity.

    dry_suitability
        Dry-side score component.

    wet_suitability
        Plasticity-aware wet-side component.

    deep_suitability
        Deep-layer wetness component.
    """

    # --------------------------------------------------------
    # Hydraulic coordinates
    # --------------------------------------------------------

    denom = (
        theta_fc
        -
        theta_wp
    )


    F = np.full(
        surf.shape,
        np.nan,
        dtype=np.float64,
    )

    Fdeep = np.full_like(
        F,
        np.nan,
    )


    valid_denom = (
        np.isfinite(denom)
        &
        (denom > 0.0)
    )


    valid_surface = (
        valid_denom
        &
        np.isfinite(surf)
    )


    valid_deep = (
        valid_denom
        &
        np.isfinite(deep)
    )


    F[
        valid_surface
    ] = (
        surf[
            valid_surface
        ]
        -
        theta_wp[
            valid_surface
        ]
    ) / denom[
        valid_surface
    ]


    Fdeep[
        valid_deep
    ] = (
        deep[
            valid_deep
        ]
        -
        theta_wp[
            valid_deep
        ]
    ) / denom[
        valid_deep
    ]


    # --------------------------------------------------------
    # Dry-side suitability
    #
    # This is deliberately broad.
    #
    # WP is poor.
    # Suitability rises smoothly toward F=0.5.
    # Above F=0.5 there is no additional dry penalty.
    #
    # F_DRY_FULL remains an empirical Hero Dirt parameter.
    # --------------------------------------------------------

    dry_coordinate = (
        F
        /
        F_DRY_FULL
    )


    dry_suitability = smoothstep(
        dry_coordinate
    )


    # --------------------------------------------------------
    # Mechanical wet sensitivity from PI
    #
    # Nonplastic soil gets sensitivity = 0.
    #
    # PI/(PI+7) is a bounded empirical transformation,
    # NOT a soil-mechanics equation.
    # --------------------------------------------------------

    PI = np.asarray(
        plasticity_index,
        dtype=np.float64,
    )


    pi_sensitivity = np.zeros(
        PI.shape,
        dtype=np.float64,
    )


    valid_pi = (
        np.isfinite(PI)
        &
        (PI > 0.0)
    )


    pi_sensitivity[
        valid_pi
    ] = (
        PI[
            valid_pi
        ]
        /
        (
            PI[
                valid_pi
            ]
            +
            7.0
        )
    )


    pi_sensitivity = np.clip(
        pi_sensitivity,
        0.0,
        1.0,
    )


    # --------------------------------------------------------
    # Approach to field capacity
    # --------------------------------------------------------

    wet_pre_fc = (
        F
        -
        F_WET_START
    ) / (
        1.0
        -
        F_WET_START
    )


    wet_pre_fc = np.clip(
        wet_pre_fc,
        0.0,
        1.0,
    )


    # --------------------------------------------------------
    # Actual field-capacity excess toward saturation
    # --------------------------------------------------------

    sat_range = (
        theta_sat
        -
        theta_fc
    )


    above_fc = np.zeros(
        surf.shape,
        dtype=np.float64,
    )


    valid_sat_range = (
        np.isfinite(sat_range)
        &
        (sat_range > 0.0)
        &
        np.isfinite(surf)
    )


    above_fc[
        valid_sat_range
    ] = (
        surf[
            valid_sat_range
        ]
        -
        theta_fc[
            valid_sat_range
        ]
    ) / sat_range[
        valid_sat_range
    ]


    above_fc = np.clip(
        above_fc,
        0.0,
        1.0,
    )


    # --------------------------------------------------------
    # Combined wetness state
    #
    # Only a modest penalty approaching FC.
    # Stronger penalty occurs once water exceeds FC.
    # --------------------------------------------------------

    wet_state = (
        0.35
        *
        wet_pre_fc
        +
        above_fc
    )


    wet_strength = (
        WET_BASE_STRENGTH
        +
        WET_PI_STRENGTH
        *
        pi_sensitivity
    )


    wet_suitability = np.exp(
        -wet_strength
        *
        wet_state
    )


    # --------------------------------------------------------
    # Deep-layer wetness
    # --------------------------------------------------------

    deep_pre_fc = (
        Fdeep
        -
        0.90
    ) / (
        1.0
        -
        0.90
    )


    deep_pre_fc = np.clip(
        deep_pre_fc,
        0.0,
        1.0,
    )


    deep_above_fc = np.zeros(
        deep.shape,
        dtype=np.float64,
    )


    valid_deep_sat = (
        np.isfinite(sat_range)
        &
        (sat_range > 0.0)
        &
        np.isfinite(deep)
    )


    deep_above_fc[
        valid_deep_sat
    ] = (
        deep[
            valid_deep_sat
        ]
        -
        theta_fc[
            valid_deep_sat
        ]
    ) / sat_range[
        valid_deep_sat
    ]


    deep_above_fc = np.clip(
        deep_above_fc,
        0.0,
        1.0,
    )


    deep_suitability = np.exp(
        -DEEP_WET_STRENGTH
        *
        (
            0.25
            *
            deep_pre_fc
            +
            deep_above_fc
        )
    )


    # --------------------------------------------------------
    # Combined 0-100 score
    # --------------------------------------------------------

    score = (
        100.0
        *
        dry_suitability
        *
        wet_suitability
        *
        deep_suitability
    )


    score = np.clip(
        score,
        0.0,
        100.0,
    )


    score[
        ~public_valid
    ] = np.nan


    # --------------------------------------------------------
    # Directional condition class
    #
    # Important:
    # classes depend on moisture direction, not score alone.
    #
    # Thresholds remain Hero Dirt calibration choices.
    # --------------------------------------------------------

    condition = np.zeros(
        surf.shape,
        dtype=np.uint8,
    )


    condition[
        public_valid
        &
        (F < 0.07)
    ] = 1


    condition[
        public_valid
        &
        (F >= 0.07)
        &
        (F < 0.40)
    ] = 2


    condition[
        public_valid
        &
        (F >= 0.40)
        &
        (F < 0.65)
    ] = 3


    condition[
        public_valid
        &
        (F >= 0.65)
        &
        (F < F_WET_START)
    ] = 4


    condition[
        public_valid
        &
        (F >= F_WET_START)
        &
        (F < 1.0)
    ] = 5


    condition[
        public_valid
        &
        (F >= 1.0)
    ] = 6


    return {
        "score":
            score,

        "condition":
            condition,

        "F":
            F,

        "Fdeep":
            Fdeep,

        "pi_sensitivity":
            pi_sensitivity,

        "dry_suitability":
            dry_suitability,

        "wet_suitability":
            wet_suitability,

        "deep_suitability":
            deep_suitability,
    }
