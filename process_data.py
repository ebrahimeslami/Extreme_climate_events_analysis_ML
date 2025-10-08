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
TEMP_THRESHOLD_STD = 1.0  # anomaly threshold for weather-driven summer
HUMIDITY_THRESHOLD_STD = 0.5  # optional humidity factor (dewpoint)


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
    """Compute metrics for fixed June–August summer."""
    print("\n==> Computing calendar-based summer metrics (JJA)")

    summer_df = (
        df[df["month"].isin(CALENDAR_SUMMER_MONTHS)]
        .groupby("year")
        .agg(
            mean_temp_C=("t2m", "mean"),
            mean_precip_mm=("tp", "mean"),
            mean_dewpoint_C=("d2m", "mean"),
        )
        .reset_index()
    )

    summer_df["summer_length"] = len(CALENDAR_SUMMER_MONTHS)  # fixed 3 months
    out_csv = os.path.join(OUTPUT_DIR, "summer_calendar_summary.csv")
    summer_df.to_csv(out_csv, index=False)
    print(f"✅ Calendar-based summer metrics saved to: {out_csv}")

    return summer_df


# ============================================================
# APPROACH 2: WEATHER-DRIVEN SUMMER
# ============================================================
def process_weather_driven_summer(df):
    """
    Compute dynamically determined summer based on T2m and D2m anomalies.
    Defines summer onset and end per year using anomaly thresholds.
    """
    print("\n==> Computing weather-driven summer metrics")

    # ---- Baseline (1979–2000)
    baseline = df[(df["year"] >= 1979) & (df["year"] <= 2000)]
    mean_t2m = baseline["t2m"].mean()
    std_t2m = baseline["t2m"].std()
    mean_d2m = baseline["d2m"].mean() if "d2m" in df else np.nan
    std_d2m = baseline["d2m"].std() if "d2m" in df else np.nan

    df["t2m_anomaly"] = df["t2m"] - mean_t2m
    if "d2m" in df:
        df["d2m_anomaly"] = df["d2m"] - mean_d2m
    else:
        df["d2m_anomaly"] = np.nan

    results = []
    for year in tqdm(sorted(df["year"].unique()), desc="Processing years"):
        d = df[df["year"] == year]

        # Identify "summer-like" months (temp > mean + 1σ, dewpoint optional)
        hot_condition = d["t2m_anomaly"] > TEMP_THRESHOLD_STD * std_t2m
        if not np.isnan(mean_d2m):
            humid_condition = d["d2m_anomaly"] > HUMIDITY_THRESHOLD_STD * std_d2m
            summer_mask = hot_condition & humid_condition
        else:
            summer_mask = hot_condition

        if summer_mask.any():
            onset_month = int(d.loc[summer_mask, "month"].min())
            end_month = int(d.loc[summer_mask, "month"].max())
            summer_length = end_month - onset_month + 1
            summer_t = d.loc[summer_mask, "t2m"].mean()
            summer_p = d.loc[summer_mask, "tp"].mean()
            summer_d = d.loc[summer_mask, "d2m"].mean()
        else:
            onset_month = end_month = np.nan
            summer_length = 0
            summer_t = summer_p = summer_d = np.nan

        results.append(
            [
                year,
                onset_month,
                end_month,
                summer_length,
                summer_t,
                summer_p,
                summer_d,
            ]
        )

    summer_weather_df = pd.DataFrame(
        results,
        columns=[
            "year",
            "onset_month",
            "end_month",
            "summer_length",
            "mean_temp_C",
            "mean_precip_mm",
            "mean_dewpoint_C",
        ],
    )

    out_csv = os.path.join(OUTPUT_DIR, "summer_weather_summary.csv")
    summer_weather_df.to_csv(out_csv, index=False)
    print(f"✅ Weather-driven summer metrics saved to: {out_csv}")

    return summer_weather_df


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    import glob
    import pandas as pd
    import datetime as dt

    print("==> Searching for ERA5 decade files in data_raw/era5/")
    files = sorted(glob.glob("data_raw/era5/era5_gulf_*_unzipped.nc"))
    if not files:
        raise FileNotFoundError("❌ No ERA5 files found in data_raw/era5/. Please run download_data.py first.")

    print(f"📦 Found {len(files)} ERA5 blocks to merge:")
    for f in files:
        print("   •", os.path.basename(f))

    # === 1️⃣ Load and merge all ERA5 decade files ===
    start_time = dt.datetime.now()
    datasets = [open_era5_any(f) for f in files]
    ds = xr.concat(datasets, dim="valid_time", join="override")
    print(f"✅ ERA5 dataset merged successfully with {len(ds.valid_time)} timesteps.")

    # === 2️⃣ Convert to dataframe for downstream analysis ===
    df = ds.to_dataframe().reset_index()
    df["year"] = pd.to_datetime(df["valid_time"]).dt.year
    df["month"] = pd.to_datetime(df["valid_time"]).dt.month
    print(df.columns.tolist())  # 👈 check variable names here

    if "time" in df.columns and "valid_time" not in df.columns:
        df.rename(columns={"time": "valid_time"}, inplace=True)
    df["year"] = pd.to_datetime(df["valid_time"]).dt.year
    df["month"] = pd.to_datetime(df["valid_time"]).dt.month  # ✅ NEW LINE
    print(f"📊 Converted ERA5 dataset to dataframe ({len(df):,} records, {df.year.nunique()} years).")

    # ---- Ensure precipitation column exists ----
    if "tp" not in df.columns:
        candidates = [c for c in df.columns if "precip" in c.lower() or c.lower().startswith("tp")]
        if candidates:
            df["tp"] = df[candidates[0]]
            print(f"💧 Using {candidates[0]} as precipitation (aliased to 'tp').")
        else:
            df["tp"] = 0.0
            print("⚠️ No precipitation variable found. Setting 'tp' to 0.0 placeholder.")

    # === 3️⃣ Calendar-based summer metrics (JJA) ===
    print("\n==> Computing calendar-based summer metrics (June–July–August)...")
    cal_summer = process_calendar_based_summer(df)

        # === 4️⃣ Weather-driven summer metrics (temperature-based) ===
    print("\n==> Computing weather-driven summer metrics (threshold-based)...")
    dyn_summer = process_weather_driven_summer(df)

    # === 5️⃣ Combine both approaches for comparison ===
    merged = cal_summer.merge(
        dyn_summer[["year", "summer_length"]],
        on="year",
        how="left",
        suffixes=("_calendar", "_weather"),
    )

    # === 6️⃣ Save combined results ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_out = os.path.join(OUTPUT_DIR, "summer_comparison_summary.csv")
    merged.to_csv(merged_out, index=False)

    runtime = (dt.datetime.now() - start_time).total_seconds() / 60
    print(f"\n✅ Combined comparison file saved to: {merged_out}")
    print(f"⏱️  Processing completed in {runtime:.2f} minutes.")

    # === 7️⃣ Optional metadata snapshot ===
    meta_file = os.path.join(OUTPUT_DIR, "run_metadata.txt")
    with open(meta_file, "w") as f:
        f.write(f"ERA5 Gulf Summer Study\n")
        f.write(f"Generated: {dt.datetime.now().isoformat()}\n")
        f.write(f"Files merged: {len(files)}\n")
        for fpath in files:
            f.write(f" - {os.path.basename(fpath)}\n")
    print(f"🗂️  Metadata written to: {meta_file}")

    # === 8️⃣ Preview a few rows ===
    print("\n--- Preview ---")
    print(merged.head(10))



# ============================================================
# RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    main()

