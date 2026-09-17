#!/usr/bin/env python3

import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

SOLAR_CONSTANT = 1361.0       # W m-2
ALBEDO_TRAIL = 0.18           # provisional bare-soil value
STEFAN_BOLTZMANN = 2.043e-10  # MJ K-4 m-2 h-1


# ============================================================
# METEOROLOGICAL HELPERS
# ============================================================

def saturation_vapor_pressure(T):
    """
    Saturation vapor pressure [kPa].
    """

    return (
        0.6108
        *
        np.exp(
            17.27 * T
            /
            (T + 237.3)
        )
    )


def slope_sat_vapor_curve(T):
    """
    Slope of saturation vapor-pressure curve [kPa / deg C].
    """

    es = saturation_vapor_pressure(T)

    return (
        4098.0
        *
        es
        /
        (T + 237.3)**2
    )


def atmospheric_pressure(elevation):
    """
    Atmospheric pressure [kPa]
    from elevation [m].
    """

    return (
        101.3
        *
        (
            (
                293.0
                -
                0.0065 * elevation
            )
            /
            293.0
        )**5.26
    )


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def solar_state(
    timestamp,
    lat_rad,
    lon_deg,
):
    """
    Return local solar unit-vector components and
    extraterrestrial normal irradiance.

    Components are in local:
        east
        north
        up
    """

    day = timestamp.astype(
        "datetime64[D]"
    )

    year = timestamp.astype(
        "datetime64[Y]"
    )

    doy = (
        day
        -
        year
    ).astype(int) + 1


    seconds_today = (
        timestamp
        -
        day
    ).astype(
        "timedelta64[s]"
    ).astype(float)


    hour_utc = (
        seconds_today
        /
        3600.0
    )


    gamma = (
        2.0
        *
        np.pi
        /
        365.0
        *
        (
            doy
            -
            1
            +
            (
                hour_utc
                -
                12.0
            )
            /
            24.0
        )
    )


    decl = (
        0.006918
        -
        0.399912 * np.cos(gamma)
        +
        0.070257 * np.sin(gamma)
        -
        0.006758 * np.cos(2.0 * gamma)
        +
        0.000907 * np.sin(2.0 * gamma)
        -
        0.002697 * np.cos(3.0 * gamma)
        +
        0.00148 * np.sin(3.0 * gamma)
    )


    eqtime = (
        229.18
        *
        (
            0.000075
            +
            0.001868 * np.cos(gamma)
            -
            0.032077 * np.sin(gamma)
            -
            0.014615 * np.cos(2.0 * gamma)
            -
            0.040849 * np.sin(2.0 * gamma)
        )
    )


    utc_minutes = (
        hour_utc
        *
        60.0
    )


    solar_minutes = (
        utc_minutes
        +
        eqtime
        +
        4.0 * lon_deg
    )


    hour_angle = np.deg2rad(
        solar_minutes / 4.0
        -
        180.0
    )


    cos_decl = np.cos(
        decl
    )

    sin_decl = np.sin(
        decl
    )


    sun_east = (
        -cos_decl
        *
        np.sin(
            hour_angle
        )
    )


    sun_north = (
        sin_decl
        *
        np.cos(
            lat_rad
        )
        -
        cos_decl
        *
        np.cos(
            hour_angle
        )
        *
        np.sin(
            lat_rad
        )
    )


    sun_up = (
        sin_decl
        *
        np.sin(
            lat_rad
        )
        +
        cos_decl
        *
        np.cos(
            hour_angle
        )
        *
        np.cos(
            lat_rad
        )
    )


    distance_factor = (
        1.0
        +
        0.033
        *
        np.cos(
            2.0
            *
            np.pi
            *
            doy
            /
            365.0
        )
    )


    extraterrestrial_normal = (
        SOLAR_CONSTANT
        *
        distance_factor
    )


    return (
        sun_east,
        sun_north,
        sun_up,
        extraterrestrial_normal,
    )


# ============================================================
# PENMAN TREAD DRYING
# ============================================================

def compute_penman_et_block(
    block_end,
    block_hours,
    temperature,
    rh,
    wind,
    cloud,
    elevation,
    lat_rad,
    lon_deg,
    slope_deg,
    eastness,
    northness,
    tree,
    shrub,
    herb,
    physics,
):
    """
    Compute potential Penman-Monteith evaporation [mm]
    over one forecast block.

    Meteorological inputs are treated as representative
    of the full block.

    Solar geometry is integrated using hourly midpoint samples.

    Returns
    -------
    ET_block : ndarray
        Potential evaporation over block [mm].
    """

    # --------------------------------------------------------
    # Vapor pressure
    # --------------------------------------------------------

    es = saturation_vapor_pressure(
        temperature
    )

    ea = (
        rh
        /
        100.0
        *
        es
    )

    vpd = np.maximum(
        es - ea,
        0.0,
    )


    delta = slope_sat_vapor_curve(
        temperature
    )


    pressure = atmospheric_pressure(
        elevation
    )


    psychrometric = (
        0.000665
        *
        pressure
    )


    # --------------------------------------------------------
    # Terrain normal
    # --------------------------------------------------------

    slope_rad = np.deg2rad(
        slope_deg
    )


    normal_east = (
        np.sin(
            slope_rad
        )
        *
        eastness
    )


    normal_north = (
        np.sin(
            slope_rad
        )
        *
        northness
    )


    normal_up = np.cos(
        slope_rad
    )


    # --------------------------------------------------------
    # Canopy transmission
    #
    # Retain existing Hero Dirt relationship for now.
    # --------------------------------------------------------

    canopy_transmission = np.clip(
        1.0
        -
        0.55 * tree
        -
        0.30 * shrub
        -
        0.15 * herb,
        0.45,
        1.0,
    )


    # --------------------------------------------------------
    # Cloud transmission
    #
    # Also retained provisionally from v1.
    # --------------------------------------------------------

    cloud_fraction = np.clip(
        cloud
        /
        100.0,
        0.0,
        1.0,
    )


    cloud_transmission = np.clip(
        1.0
        -
        0.50
        *
        cloud_fraction,
        0.50,
        1.0,
    )


    # --------------------------------------------------------
    # Clear-sky atmospheric transmissivity
    # --------------------------------------------------------

    clear_transmission = (
        0.75
        +
        2.0e-5
        *
        elevation
    )


    clear_transmission = np.clip(
        clear_transmission,
        0.65,
        0.85,
    )


    # --------------------------------------------------------
    # Integrate over hourly midpoint samples
    # --------------------------------------------------------

    nsamp = max(
        1,
        int(
            round(
                block_hours
            )
        ),
    )


    dt_hours = (
        block_hours
        /
        nsamp
    )


    ET_total = np.zeros(
        physics.shape,
        dtype=float,
    )


    for j in range(
        nsamp
    ):

        hours_before_end = (
            block_hours
            -
            (
                j
                +
                0.5
            )
            *
            dt_hours
        )


        sample_seconds = int(
            round(
                hours_before_end
                *
                3600.0
            )
        )


        sample_time = (
            block_end
            -
            np.timedelta64(
                sample_seconds,
                "s",
            )
        )


        (
            sun_e,
            sun_n,
            sun_u,
            R_normal,
        ) = solar_state(
            sample_time,
            lat_rad,
            lon_deg,
        )


        incidence = (
            normal_east
            *
            sun_e
            +
            normal_north
            *
            sun_n
            +
            normal_up
            *
            sun_u
        )


        incidence = np.where(
            (
                sun_u > 0.0
            )
            &
            (
                incidence > 0.0
            ),
            incidence,
            0.0,
        )


        # ----------------------------------------------------
        # Incoming terrain-plane shortwave [W m-2]
        # ----------------------------------------------------

        Ra_topo_W = (
            R_normal
            *
            incidence
        )


        Rs_W = (
            Ra_topo_W
            *
            clear_transmission
            *
            cloud_transmission
            *
            canopy_transmission
        )


        # Convert W m-2 -> MJ m-2 over this substep
        Rs = (
            Rs_W
            *
            dt_hours
            *
            3600.0
            /
            1.0e6
        )


        # ----------------------------------------------------
        # Net shortwave
        # ----------------------------------------------------

        Rns = (
            1.0
            -
            ALBEDO_TRAIL
        ) * Rs


        # ----------------------------------------------------
        # Approximate net longwave
        # ----------------------------------------------------

        Tk = (
            temperature
            +
            273.16
        )


        emissivity_term = np.clip(
            0.34
            -
            0.14
            *
            np.sqrt(
                np.maximum(
                    ea,
                    0.0,
                )
            ),
            0.05,
            0.34,
        )


        cloud_longwave_factor = np.clip(
            1.0
            -
            0.50
            *
            cloud_fraction,
            0.50,
            1.0,
        )


        Rnl_hourly = (
            STEFAN_BOLTZMANN
            *
            Tk**4
            *
            emissivity_term
            *
            cloud_longwave_factor
        )


        Rnl = (
            Rnl_hourly
            *
            dt_hours
        )


        # ----------------------------------------------------
        # Net radiation
        # ----------------------------------------------------

        Rn = (
            Rns
            -
            Rnl
        )


        # ----------------------------------------------------
        # Soil heat flux
        # ----------------------------------------------------

        daytime = (
            sun_u > 0.0
        )


        G = np.where(
            daytime,
            0.10 * Rn,
            0.50 * Rn,
        )


        # ----------------------------------------------------
        # Hourly FAO-56 style Penman-Monteith
        #
        # Scale aerodynamic constant by timestep duration.
        # ----------------------------------------------------

        numerator = (
            0.408
            *
            delta
            *
            (
                Rn - G
            )
            +
            psychrometric
            *
            (
                37.0
                /
                (
                    temperature
                    +
                    273.0
                )
            )
            *
            wind
            *
            vpd
            *
            dt_hours
        )


        denominator = (
            delta
            +
            psychrometric
            *
            (
                1.0
                +
                0.34
                *
                wind
            )
        )


        ET_step = np.where(
            denominator > 0.0,
            numerator / denominator,
            0.0,
        )


        ET_step = np.maximum(
            ET_step,
            0.0,
        )


        ET_step[
            ~physics
        ] = 0.0


        ET_total += ET_step


    return ET_total
