#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path.home() / "HeroDirt"

MRMS_FILE = (
    ROOT
    / "data/mrms/processed/HeroDirt_MRMS_hourly.npz"
)

NWS_FILE = (
    ROOT
    / "data/nws/processed/HeroDirt_NWS_6hourly.npz"
)

SMAP_FILE = (
    ROOT
    / "data/smap/processed/HeroDirt_SMAP_latest.npz"
)


# ============================================================
# LOAD
# ============================================================

mrms = np.load(MRMS_FILE)
nws = np.load(NWS_FILE)
smap = np.load(SMAP_FILE)


print()
print("============================================")
print(" HERO DIRT SOCAL FORCING CHECK")
print("============================================")


# ============================================================
# MRMS — LAST 24 HOURS
# ============================================================

mt = mrms["time"]
mp = mrms["precip_mm"]

latest_mrms = mt[-1]

start_24h = (
    latest_mrms
    -
    np.timedelta64(
        24,
        "h",
    )
)

ii = np.where(
    (mt > start_24h)
    &
    (mt <= latest_mrms)
)[0]

obs24 = np.nansum(
    mp[ii],
    axis=0,
)

mean24 = np.nanmean(obs24)
median24 = np.nanmedian(obs24)
max24 = np.nanmax(obs24)


print()
print("MRMS OBSERVED — LAST 24 HOURS")
print("--------------------------------")

print(
    f"Start: {start_24h}"
)

print(
    f"End:   {latest_mrms}"
)

print(
    f"Hourly fields: {len(ii)}"
)

print()

print(
    f"Domain mean: {mean24:6.2f} mm"
    f"  ({mean24 / 25.4:.2f} in)"
)

print(
    f"Median:      {median24:6.2f} mm"
    f"  ({median24 / 25.4:.2f} in)"
)

print(
    f"Maximum:     {max24:6.2f} mm"
    f"  ({max24 / 25.4:.2f} in)"
)


# ============================================================
# NWS FORECAST PRECIPITATION
# ============================================================

nt = nws["time"]
nppt = nws["precipitation_mm"]

forecast_total = np.nansum(
    nppt,
    axis=0,
)

mean_fcst = np.nanmean(
    forecast_total
)

median_fcst = np.nanmedian(
    forecast_total
)

max_fcst = np.nanmax(
    forecast_total
)


print()
print("NWS FORECAST PRECIPITATION")
print("--------------------------------")

print(
    f"First block: {nt[0]}"
)

print(
    f"Last block:  {nt[-1]}"
)

print(
    f"Blocks:      {len(nt)}"
)

print()

print(
    f"Domain mean total: {mean_fcst:6.2f} mm"
    f"  ({mean_fcst / 25.4:.2f} in)"
)

print(
    f"Median total:      {median_fcst:6.2f} mm"
    f"  ({median_fcst / 25.4:.2f} in)"
)

print(
    f"Maximum total:     {max_fcst:6.2f} mm"
    f"  ({max_fcst / 25.4:.2f} in)"
)


print()
print("FORECAST PRECIPITATION BY INTERVAL")
print("--------------------------------")

for t, p in zip(
    nt,
    nppt,
):

    good = np.isfinite(p)

    if not np.any(good):
        continue

    mean_block = np.nanmean(p)
    max_block = np.nanmax(p)

    frac_01 = (
        100.0
        *
        np.sum(
            good
            &
            (p > 0.1)
        )
        /
        np.sum(good)
    )

    frac_10 = (
        100.0
        *
        np.sum(
            good
            &
            (p > 1.0)
        )
        /
        np.sum(good)
    )

    if max_block > 0.01:

        print()
        print(f"End: {t}")

        print(
            f"  mean:       {mean_block:6.2f} mm"
            f"  ({mean_block / 25.4:.2f} in)"
        )

        print(
            f"  maximum:    {max_block:6.2f} mm"
            f"  ({max_block / 25.4:.2f} in)"
        )

        print(
            f"  area >0.1:  {frac_01:5.1f}%"
        )

        print(
            f"  area >1.0:  {frac_10:5.1f}%"
        )


# ============================================================
# SMAP
# ============================================================

obs_date = smap[
    "observation_date"
]

obs_time = smap[
    "observation_time_utc"
]

sm = smap[
    "soil_moisture"
].astype(float)

good = np.isfinite(
    sm
)


print()
print("LATEST SMAP PASS")
print("--------------------------------")

print(
    f"Observation date: {obs_date}"
)

print(
    f"Observation time: {obs_time}"
)

print(
    f"Valid cells:      {np.sum(good)}"
)

print()

print(
    f"Mean:    {float(smap['mean_soil_moisture']):.4f} m3/m3"
)

print(
    f"Median:  {float(smap['median_soil_moisture']):.4f} m3/m3"
)

print(
    f"Std dev: {float(smap['std_soil_moisture']):.4f} m3/m3"
)

print(
    f"Minimum: {float(smap['min_soil_moisture']):.4f} m3/m3"
)

print(
    f"Maximum: {float(smap['max_soil_moisture']):.4f} m3/m3"
)


# ============================================================
# SMAP CELL-TIME RANGE
# ============================================================

cell_time = smap[
    "cell_observation_time_utc"
]

print()

print(
    "Cell observation range:"
)

print(
    f"  earliest: {np.min(cell_time)}"
)

print(
    f"  latest:   {np.max(cell_time)}"
)


print()
print("============================================")
print(" FORCING CHECK COMPLETE")
print("============================================")
