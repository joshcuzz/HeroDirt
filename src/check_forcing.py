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


# ============================================================
# MRMS CONTINUITY CHECK
# ============================================================

mt = np.asarray(
    mrms["time"],
    dtype="datetime64[s]",
)

smap_time = np.datetime64(
    str(smap["observation_time_utc"])
).astype("datetime64[s]")

RESTART_FILE = ROOT / "restart" / "HeroDirt_analysis_restart.npz"
if RESTART_FILE.exists():
    with np.load(RESTART_FILE, allow_pickle=False) as prior:
        if "time" not in prior.files:
            raise RuntimeError("Restart file missing analysis time")
        smap_time = np.asarray(prior["time"]).astype("datetime64[s]")[()]
    print(f"Continuity checked since saved state {smap_time}")
else:
    print(f"Cold-start continuity checked since SMAP {smap_time}")

forecast_start = np.asarray(
    nws["forecast_start"]
).astype("datetime64[s]")

if forecast_start.ndim != 0:
    raise RuntimeError(
        "Expected a single NWS forecast start time."
    )

forecast_start = forecast_start[()]

# No requirement that NWS has reissued since the saved soil analysis.
# The analysis advances to the latest hourly MRMS observation.
if not len(mt):
    raise RuntimeError("No MRMS observations")
if np.any(np.isnat(mt)) or np.any(np.diff(mt) <= np.timedelta64(0, 's')):
    raise RuntimeError("MRMS timestamps must be strictly increasing")
if mt[-1] < smap_time:
    raise RuntimeError("Saved state is newer than available MRMS observations")
if np.asarray(nws["time"]).astype("datetime64[s]")[-1] <= mt[-1]:
    raise RuntimeError("NWS forecast expires before latest MRMS analysis")

analysis_end = mt[-1]

# MRMS timestamps represent the end of each hourly
# precipitation accumulation.
#
# Include the hour containing the SMAP observation
# and all subsequent hours through forecast_start.

expected_start = (
    smap_time.astype("datetime64[h]")
    + np.timedelta64(1, "h")
)

expected_end = analysis_end.astype("datetime64[h]")

expected = np.arange(
    expected_start,
    expected_end + np.timedelta64(1, "h"),
    np.timedelta64(1, "h"),
).astype("datetime64[s]")

if len(np.unique(mt)) != len(mt):
    raise RuntimeError(
        "Duplicate MRMS hourly timestamps detected."
    )

available = mt[
    (mt >= expected_start)
    & (mt <= expected_end)
]

missing = np.setdiff1d(
    expected,
    available,
)

if len(missing) > 0:
    print("\nMissing MRMS hourly observations:")
    for t in missing:
        print("  ", t)

    raise RuntimeError(
        f"MRMS analysis incomplete: "
        f"{len(missing)} missing hourly fields."
    )

print()
print("MRMS CONTINUITY CHECK PASSED")
print(f"Analysis origin:  {smap_time}")
print(f"MRMS analysis:    {analysis_end}")
print(f"NWS issue start:  {forecast_start}")
print(f"Required hours:   {len(expected)}")
print(f"Available hours:  {len(available)}")
print()

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
