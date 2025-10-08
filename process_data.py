# -*- coding: utf-8 -*-
"""
Created on Oct  2025

@author: Ebrahim Eslami (e.eslami@gmail.com)
"""

"""
process_data.py
-----------------
Processes ERA5 reanalysis data (monthly means) to derive:
  • Calendar-based summer metrics (JJA)
  • Weather-driven summer metrics (variable onset/end)
  • Outputs: mean temperature, precipitation, humidity, and summer length

Author: Ebrahim Eslami
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
ERA5_FILE = os.path.join("data_raw", "era5", "era5_gulf_monthly_1979_2024.nc")
OUTPUT_DIR = os.path.join("data_processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seasonal and threshold parameters
CALENDAR_SUMMER_MONTHS = [6, 7, 8]  # JJA
TEMP_THRESHOLD_STD = 0.35            # sensitivity of temperature-based summer (0.35σ above that year’s mean)
HUMIDITY_THRESHOLD_STD = 0.25        # sensitivity of dewpoint-based summer (0.25σ fo capturing high-humidity months)



# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def kelvin_to_celsius(arr):
    return arr - 273.15


def open_era5_any(path: str):
    """
    Open ERA5 file automatically detecting NetCDF or GRIB formats.
    """
    for engine in ("netcdf4", "h5netcdf", "cfgrib"):
        try:
            ds = xr.open_dataset(path, engine=engine)
            print(f"✅ Opened ERA5 with engine={engine}")
            return ds
        except Exception:
            continue
    raise RuntimeError(f"Unable to open {path} with available engines.")



    def process_era5_load(era5_file=ERA5_FILE):
        """Load ERA5 NetCDF file and extract needed variables."""
        print(f"==> Loading ERA5 file: {era5_file}")
        try:
            ds = xr.open_dataset(era5_file, engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(era5_file, engine="h5netcdf")

        # Variables
        t2m = kelvin_to_celsius(ds["t2m"])        # °C
        tp = ds["tp"] * 1000                      # mm
        d2m = kelvin_to_celsius(ds["d2m"]) if "d2m" in ds else None

        # Spatial average
        ds_mean = xr.Dataset()
        ds_mean["t2m"] = t2m.mean(dim=["latitude", "longitude"])
        ds_mean["tp"] = tp.mean(dim=["latitude", "longitude"])
        if d2m is not None:
            ds_mean["d2m"] = d2m.mean(dim=["latitude", "longitude"])

        # Convert to DataFrame
        df = ds_mean.to_dataframe().reset_index()
        df["year"] = pd.to_datetime(df["time"]).dt.year
        df["month"] = pd.to_datetime(df["time"]).dt.month
        print(f"✅ ERA5 monthly data loaded: {len(df)} records")

        return df


# ============================================================
# APPROACH 1: CALENDAR-BASED SUMMER (JJA)
# ============================================================
def process_calendar_based_summer(df):
    """
    Compute metrics for fixed calendar-based summer (June–July–August).
    Converts ERA5 units to °C and mm, and uses day-weighted averages.
    Outputs summer length in days (≈92).
    """
    print("\n==> Computing calendar-based summer metrics (June–July–August)...")

    # --- Convert Kelvin → °C for temperature variables ---
    df["t2m_C"] = df["t2m"] - 273.15
    df["d2m_C"] = df["d2m"] - 273.15

    # --- Detect precipitation variable automatically ---
    precip_col = None
    for cand in ["tp", "total_precipitation", "precip", "mtpr"]:
        if cand in df.columns:
            precip_col = cand
            break
    if precip_col is None:
        print("⚠️ Warning: No precipitation variable found in ERA5 dataset. Setting to 0 mm.")
        df["precip_mm"] = 0.0
    else:
        df["precip_mm"] = df[precip_col] * 1000  # convert m → mm

    # --- Assign number of days per month ---
    month_days = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                  7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    df["month_days"] = df["month"].map(month_days)

    # --- Filter for summer months (June–August) ---
    summer_months = [6, 7, 8]
    summer = df[df["month"].isin(summer_months)].copy()

    # --- Weighted aggregation (by number of days per month) ---
    def weighted_avg(x, w):
        return np.average(x, weights=w) if len(x) > 0 else np.nan

    results = []
    for year, g in summer.groupby("year"):
        w = g["month_days"]
        mean_temp = weighted_avg(g["t2m_C"], w)
        mean_dew = weighted_avg(g["d2m_C"], w)
        total_precip = np.sum(g["precip_mm"])
        summer_length_days = 92  # total days in JJA (92)
        results.append([year, mean_temp, mean_dew, total_precip, summer_length_days])

    summer_df = pd.DataFrame(
        results,
        columns=[
            "year",
            "mean_temp_C",
            "mean_dewpoint_C",
            "total_precip_mm",
            "summer_length_days",
        ],
    )

    # --- Save output ---
    out_csv = os.path.join(OUTPUT_DIR, "summer_calendar_summary.csv")
    summer_df.to_csv(out_csv, index=False)
    print(f"✅ Calendar-based summer metrics saved to: {out_csv}")
    print(f"👉 Example output (first 5 rows):\n{summer_df.head()}\n")

    return summer_df




# ============================================================
# APPROACH 2: WEATHER-DRIVEN SUMMER
# ============================================================
def process_weather_driven_summer(df,
                                  BASELINE_START=1961,
                                  BASELINE_END=1990,
                                  PERCENTILE=75,
                                  SMOOTH_WINDOW=3):
    """
    Compute weather-driven summer metrics following the Lin & Wang (2022) method,
    adapted for monthly ERA5 data with 3-month smoothing.

    Steps:
      1. Convert ERA5 variables to physical units (°C, mm)
      2. Aggregate to regional (year, month) means
      3. Compute baseline T{PERCENTILE} threshold from 1961–1990
      4. Smooth monthly temperatures using a centered 3-month running mean
      5. Define summer as consecutive months above threshold (≥2 consecutive months)
      6. Compute onset, end, total summer days, and weighted means

    Output → data_processed/summer_weather_summary.csv
    Columns:
        year, onset_month, end_month, summer_length_days,
        mean_temp_C, mean_dewpoint_C, total_precip_mm
    """

    print(f"\n==> Computing weather-driven summer metrics (Lin & Wang {PERCENTILE}th percentile, smoothed)...")

    # --- 1️⃣ Convert to °C and mm ---
    df = df.copy()
    df["t2m_C"] = df["t2m"] - 273.15
    df["d2m_C"] = (df["d2m"] - 273.15) if "d2m" in df.columns else np.nan

    # Precipitation
    precip_col = next((c for c in ["tp", "total_precipitation", "precip", "mtpr"] if c in df.columns), None)
    if precip_col:
        df["precip_mm"] = df[precip_col] * 1000.0
    else:
        df["precip_mm"] = 0.0

    # Days per month
    month_days = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                  7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    df["month_days"] = df["month"].map(month_days)

    # --- 2️⃣ Aggregate by (year, month) to avoid grid duplication ---
    monthly = (
        df.groupby(["year", "month"], as_index=False)
          .agg(
              t2m_C=("t2m_C", "mean"),
              d2m_C=("d2m_C", "mean"),
              precip_mm=("precip_mm", "mean"),
              month_days=("month_days", "first")
          )
    )

    # --- 3️⃣ Baseline T{PERCENTILE} threshold ---
    baseline = monthly[(monthly["year"] >= BASELINE_START) & (monthly["year"] <= BASELINE_END)]
    Tth = np.percentile(baseline["t2m_C"], PERCENTILE)
    print(f"🌡️ Baseline {PERCENTILE}th percentile threshold: {Tth:.2f} °C (from {BASELINE_START}-{BASELINE_END})")

    results = []

    # --- 4️⃣ Process each year individually ---
    for year, g in monthly.groupby("year"):
        g = g.sort_values("month").reset_index(drop=True)

        # Smooth monthly temps with centered moving average (3-month)
        g["t2m_smooth"] = g["t2m_C"].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

        # Flag months above threshold
        g["above_Tth"] = g["t2m_smooth"] > Tth

        # --- Detect onset (≥2 consecutive above-threshold months) ---
        onset_month = np.nan
        for i in range(len(g) - 1):
            if g.loc[i, "above_Tth"] and g.loc[i + 1, "above_Tth"]:
                onset_month = int(g.loc[i, "month"])
                break

        # --- Detect end (≥2 consecutive below-threshold months after onset) ---
        end_month = np.nan
        if not np.isnan(onset_month):
            for j in range(i + 2, len(g) - 1):
                if (not g.loc[j, "above_Tth"]) and (not g.loc[j + 1, "above_Tth"]):
                    end_month = int(g.loc[j, "month"])
                    break

        # --- Compute summer metrics ---
        if np.isnan(onset_month):
            summer_days = 0
            mean_temp = mean_dew = total_precip = np.nan
        else:
            if np.isnan(end_month):
                end_month = int(g["month"].max())

            active = g[(g["month"] >= onset_month) & (g["month"] <= end_month)]
            w = active["month_days"]

            summer_days = w.sum()
            mean_temp = np.average(active["t2m_C"], weights=w)
            mean_dew = np.average(active["d2m_C"], weights=w) if active["d2m_C"].notna().any() else np.nan
            total_precip = np.sum(active["precip_mm"] * (active["month_days"] / active["month_days"].sum()))

        results.append([
            year, onset_month, end_month,
            summer_days, mean_temp, mean_dew, total_precip
        ])

    # --- 5️⃣ Build results dataframe ---
    summer_weather_df = pd.DataFrame(
        results,
        columns=[
            "year", "onset_month", "end_month",
            "summer_length_days", "mean_temp_C",
            "mean_dewpoint_C", "total_precip_mm",
        ],
    )

    # --- 6️⃣ Save ---
    out_csv = os.path.join(OUTPUT_DIR, "summer_weather_summary.csv")
    summer_weather_df.to_csv(out_csv, index=False)
    print(f"✅ Weather-driven summer metrics saved to: {out_csv}")
    print("📊 Example output:")
    print(summer_weather_df.head(10))

    return summer_weather_df


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """Main ERA5 processing pipeline: load, process, and compare summer metrics."""

    import glob
    print("==> Searching for ERA5 decade files in data_raw/era5/")

    # === 1️⃣ Locate all unzipped ERA5 NetCDF files ===
    files = sorted(glob.glob("data_raw/era5/era5_gulf_*_unzipped.nc"))
    if not files:
        raise FileNotFoundError("❌ No ERA5 files found in data_raw/era5/. Please run download_data.py first.")

    print(f"📦 Found {len(files)} ERA5 blocks to merge:")
    for f in files:
        print("   •", os.path.basename(f))

    # === 2️⃣ Load and concatenate all ERA5 datasets ===
    datasets = [open_era5_any(f) for f in files]
    ds = xr.concat(datasets, dim="valid_time", join="override")
    print(f"✅ ERA5 dataset merged successfully with {len(ds.valid_time)} timesteps.\n")

    # === 3️⃣ Convert to DataFrame ===
    df = ds.to_dataframe().reset_index()
    df["year"] = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    print(f"📊 Converted ERA5 dataset to dataframe ({len(df):,} records, {df['year'].nunique()} years).")
    print(list(df.columns))

    # === 4️⃣ Compute calendar-based summer metrics (June–July–August) ===
    cal_summer = process_calendar_based_summer(df)

    # === 5️⃣ Compute weather-driven summer metrics (temperature anomaly-based) ===
    dyn_summer = process_weather_driven_summer(df)

    # === 6️⃣ Combine both approaches for comparison ===
    merged = cal_summer.merge(
        dyn_summer[["year", "summer_length_days", "mean_temp_C", "mean_dewpoint_C", "total_precip_mm"]],
        on="year",
        how="left",
        suffixes=("_calendar", "_weather"),
    )

    # === 7️⃣ Save final combined summary ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_out = os.path.join(OUTPUT_DIR, "summer_comparison_summary.csv")
    merged.to_csv(merged_out, index=False)

    print(f"\n✅ Combined comparison file saved to: {merged_out}")
    print("\n--- Preview ---")
    print(merged.head())

    # === 8️⃣ Optional: Save processed ERA5 as compact NetCDF ===
    compact_nc = os.path.join("data_processed", "era5_gulf_merged_processed.nc")
    os.makedirs("data_processed", exist_ok=True)
    ds.to_netcdf(compact_nc)
    print(f"💾 Merged ERA5 dataset saved to: {compact_nc}")



# ============================================================
# RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    main()

