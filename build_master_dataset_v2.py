# -*- coding: utf-8 -*-
"""
build_master_dataset_v2.py
--------------------------
Creates unified, machine-learning-ready datasets combining:
  • ERA5 summer metrics (calendar + weather-driven) from process_data.py outputs
  • NOAA Storm Events yearly counts (details CSVs)
  • Derived features: lags, rolling means, simple trends, ratios, anomalies
  • Granular outputs: national, state-level, and optional metro-level

Outputs:
  data_processed/
    ├─ master_summer_extremes_national.csv
    ├─ master_summer_extremes_by_state.csv
    ├─ master_summer_extremes_by_metro.csv           [optional if mapping provided]
    ├─ master_summer_extremes_features_metadata.json

Author: Ebrahim Eslami
Date: 2025
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
# Input files/folders
SUMMER_FILE = os.path.join("data_processed", "summer_comparison_summary.csv")
STORMS_DIR  = os.path.join("data_raw", "storm_events")   # folder with StormEvents_details-*.csv.gz (or .csv)

# Optional: County→Metro mapping to enable metro-level outputs.
# Provide a CSV with at least columns: STATE, CZ_NAME (or COUNTY), METRO
# A good start is the US OMB CBSA/CSA mapping merged to county—custom for your study.
COUNTY_TO_METRO_CSV = os.path.join("data_raw", "county_to_metro_mapping.csv")  # set to existing file or leave None
USE_METRO = os.path.exists(COUNTY_TO_METRO_CSV)

# Years coverage sanity (used for trimming final tables to overlapping years)
MIN_YEAR = 1979
MAX_YEAR = 2024

# Rolling window for smoothed features (years)
ROLL_YRS = 5

# Hazard category mapping (NOAA → your categories)
CATEGORY_MAP = {
    "HURRICANE": "Hurricane",
    "TROPICAL STORM": "Hurricane",
    "FLASH FLOOD": "Flood",
    "FLOOD": "Flood",
    "HEAT": "Heat",
    "EXCESSIVE HEAT": "Heat",
    "TORNADO": "Tornado",
    "DROUGHT": "Drought",
}
CATS = ["Hurricane", "Flood", "Heat", "Tornado", "Drought"]


# ============================================================
# HELPERS
# ============================================================
def _print_shape(df, name):
    print(f"    {name:<35}  rows={len(df):>6}, cols={len(df.columns):>3}")

def _normalize_columns(df, id_cols=("year",), add_suffix="_norm"):
    """Min-max normalize numeric columns (except ids)."""
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    for col in numeric:
        if col in id_cols:
            continue
        cmin, cmax = out[col].min(), out[col].max()
        out[col + add_suffix] = 0.0 if cmax <= cmin else (out[col] - cmin) / (cmax - cmin)
    return out

def _add_lags_rolls_trends(df, cols_for_features, group_cols=None, roll=ROLL_YRS):
    """
    Add lag-1, rolling mean, and first difference trend for selected columns.
    If group_cols provided (e.g., ['STATE'] or ['METRO']), apply per group.
    """
    out = df.copy()
    if group_cols is None:
        group_cols = []

    def _augment(g):
        g = g.sort_values("year").copy()
        for c in cols_for_features:
            if c not in g.columns:
                continue
            g[f"{c}_lag1"] = g[c].shift(1)
            g[f"{c}_roll{roll}"] = g[c].rolling(roll, center=True, min_periods=1).mean()
            g[f"{c}_trend"] = g[c].diff()
        return g

    if group_cols:
        out = out.groupby(group_cols, group_keys=False).apply(_augment)
    else:
        out = _augment(out)

    return out

def _ensure_event_cols(df, cats=CATS):
    """Add missing hazard columns set to zero."""
    out = df.copy()
    for c in cats:
        if c not in out.columns:
            out[c] = 0
    return out

def _event_agg(storms, group_cols):
    """
    Aggregate NOAA Storm Events to counts per year (+optional region).
    Returns df[year, (region...), Hurricane, Flood, Heat, Tornado, Drought]
    """
    # Normalize
    s = storms.copy()
    s["BEGIN_DATE_TIME"] = pd.to_datetime(s["BEGIN_DATE_TIME"], errors="coerce")
    s["year"] = s["BEGIN_DATE_TIME"].dt.year
    s["EVENT_TYPE"] = s["EVENT_TYPE"].astype(str).str.upper().str.strip()
    s["Category"] = s["EVENT_TYPE"].map(CATEGORY_MAP)
    s = s[s["Category"].notna()]
    s = s[s["year"].between(MIN_YEAR, MAX_YEAR)]

    # NOAA StormEvents county/zone name columns vary slightly; remap to a generic CZ_NAME if present
    if "CZ_NAME" in s.columns:
        s["CZ_NAME"] = s["CZ_NAME"].astype(str).str.upper().str.strip()

    # Aggregate counts
    grp = s.groupby(group_cols + ["Category"]).size().unstack(fill_value=0).reset_index()
    grp = _ensure_event_cols(grp, CATS)
    return grp


# ============================================================
# LOAD DATA
# ============================================================
print("==> Loading ERA5 summer metrics...")
summer = pd.read_csv(SUMMER_FILE)
# Harmonize column names to expected forms (safe rename)
summer = summer.rename(columns=lambda c: c.strip())
summer = summer.rename(columns={
    "mean_temp_c_calendar": "mean_temp_C_calendar",
    "mean_dewpoint_c_calendar": "mean_dewpoint_C_calendar",
    "total_precip_mm_calendar": "total_precip_mm_calendar",
    "mean_temp_c_weather": "mean_temp_C_weather",
    "mean_dewpoint_c_weather": "mean_dewpoint_C_weather",
    "total_precip_mm_weather": "total_precip_mm_weather",
})
# Keep year range overlap
summer = summer[summer["year"].between(MIN_YEAR, MAX_YEAR)].drop_duplicates(subset=["year"])
_print_shape(summer, "summer (input)")

print("==> Loading NOAA Storm Events (details)...")
storm_files = [f for f in os.listdir(STORMS_DIR) if f.lower().endswith((".csv", ".csv.gz"))]
if not storm_files:
    raise FileNotFoundError("❌ No StormEvents files found in data_raw/storm_events/")
storms_list = []
for f in tqdm(storm_files, desc="  reading"):
    path = os.path.join(STORMS_DIR, f)
    try:
        storms_list.append(pd.read_csv(path, compression="infer", low_memory=False))
    except Exception as e:
        print(f"   ⚠️ skipped {f}: {e}")
storms = pd.concat(storms_list, ignore_index=True)
_print_shape(storms, "storms (raw)")

# ============================================================
# NATIONAL-LEVEL MERGE
# ============================================================
print("\n==> Building national-level dataset...")
nat_events = _event_agg(storms, group_cols=["year"])
nat = summer.merge(nat_events, on="year", how="left").fillna(0)

# Derived features (totals, ratios, diffs)
nat["total_events"] = nat[["Hurricane", "Flood", "Heat", "Tornado", "Drought"]].sum(axis=1)
nat["heat_ratio"]  = nat["Heat"]  / (nat["total_events"] + 1e-6)
nat["flood_ratio"] = nat["Flood"] / (nat["total_events"] + 1e-6)
nat["temp_diff_C"]     = nat["mean_temp_C_weather"]     - nat["mean_temp_C_calendar"]
nat["dewpoint_diff_C"] = nat["mean_dewpoint_C_weather"] - nat["mean_dewpoint_C_calendar"]
nat["precip_diff_mm"]  = nat["total_precip_mm_weather"] - nat["total_precip_mm_calendar"]
nat["length_diff_days"] = nat["summer_length_days_weather"] - nat["summer_length_days_calendar"]

# Lags / rolls / trends
nat = _add_lags_rolls_trends(
    nat,
    cols_for_features=[
        "summer_length_days_weather",
        "mean_temp_C_weather",
        "mean_dewpoint_C_weather",
        "total_precip_mm_weather",
        "total_events", "Heat", "Flood", "Hurricane", "Tornado", "Drought"
    ],
    group_cols=None,
    roll=ROLL_YRS
)

# Normalized copy
nat_out = _normalize_columns(nat, id_cols=("year",))
_print_shape(nat_out, "national (final)")

# ============================================================
# STATE-LEVEL MERGE
# ============================================================
print("\n==> Building state-level dataset...")
if "STATE" not in storms.columns:
    print("   ⚠️ 'STATE' column not found in StormEvents. Skipping state-level outputs.")
    state_out = None
else:
    state_events = _event_agg(storms, group_cols=["year", "STATE"])
    # For climate features at state level, we currently have region-average only (from ERA5 area box).
    # We can still join national climate metrics to every state per year (same climate features repeated),
    # OR if you later compute state-resolved ERA5 metrics, merge those here instead.
    state = state_events.merge(summer, on="year", how="left").fillna(0)

    # Derived features
    state["total_events"] = state[["Hurricane", "Flood", "Heat", "Tornado", "Drought"]].sum(axis=1)
    state["heat_ratio"]  = state["Heat"]  / (state["total_events"] + 1e-6)
    state["flood_ratio"] = state["Flood"] / (state["total_events"] + 1e-6)
    state["temp_diff_C"]     = state["mean_temp_C_weather"]     - state["mean_temp_C_calendar"]
    state["dewpoint_diff_C"] = state["mean_dewpoint_C_weather"] - state["mean_dewpoint_C_calendar"]
    state["precip_diff_mm"]  = state["total_precip_mm_weather"] - state["total_precip_mm_calendar"]
    state["length_diff_days"] = state["summer_length_days_weather"] - state["summer_length_days_calendar"]

    # Lags / rolls / trends per state
    state = _add_lags_rolls_trends(
        state,
        cols_for_features=[
            "summer_length_days_weather",
            "mean_temp_C_weather",
            "mean_dewpoint_C_weather",
            "total_precip_mm_weather",
            "total_events", "Heat", "Flood", "Hurricane", "Tornado", "Drought"
        ],
        group_cols=["STATE"],
        roll=ROLL_YRS
    )

    # Normalized per state (note: normalization across all states together—ok for ML; or do per state if desired)
    state_out = _normalize_columns(state, id_cols=("year",))
    _print_shape(state_out, "state (final)")

# ============================================================
# METRO-LEVEL MERGE (optional)
# ============================================================
print("\n==> Building metro-level dataset...")
metro_out = None
if USE_METRO and "STATE" in storms.columns:
    try:
        mapping = pd.read_csv(COUNTY_TO_METRO_CSV)
        # Normalize keys
        if "CZ_NAME" in mapping.columns:
            mapping["CZ_NAME"] = mapping["CZ_NAME"].astype(str).str.upper().str.strip()
        if "COUNTY" in mapping.columns and "CZ_NAME" not in mapping.columns:
            mapping["CZ_NAME"] = mapping["COUNTY"].astype(str).upper().str.strip()
        mapping["STATE"] = mapping["STATE"].astype(str).str.upper().str.strip()
        mapping = mapping.dropna(subset=["STATE", "CZ_NAME", "METRO"])

        s2 = storms.copy()
        s2["BEGIN_DATE_TIME"] = pd.to_datetime(s2["BEGIN_DATE_TIME"], errors="coerce")
        s2["year"] = s2["BEGIN_DATE_TIME"].dt.year
        s2["EVENT_TYPE"] = s2["EVENT_TYPE"].astype(str).str.upper().str.strip()
        s2["Category"] = s2["EVENT_TYPE"].map(CATEGORY_MAP)
        s2 = s2[s2["Category"].notna()]
        s2 = s2[s2["year"].between(MIN_YEAR, MAX_YEAR)]

        # Normalize and join to metro mapping
        if "CZ_NAME" in s2.columns:
            s2["CZ_NAME"] = s2["CZ_NAME"].astype(str).str.upper().str.strip()
        else:
            # If county/zone name missing, skip metro build
            raise KeyError("CZ_NAME column missing in StormEvents; cannot map to METRO.")

        s2["STATE"] = s2["STATE"].astype(str).str.upper().str.strip()
        s2 = s2.merge(mapping[["STATE", "CZ_NAME", "METRO"]].drop_duplicates(), on=["STATE", "CZ_NAME"], how="left")
        s2 = s2[s2["METRO"].notna()]  # keep only matched metros

        metro_events = _event_agg(s2, group_cols=["year", "METRO"])

        # Merge climate (national metrics repeated to each metro — replace later with metro-resolved ERA5 if you compute it)
        metro = metro_events.merge(summer, on="year", how="left").fillna(0)

        # Derived features
        metro["total_events"] = metro[["Hurricane", "Flood", "Heat", "Tornado", "Drought"]].sum(axis=1)
        metro["heat_ratio"]  = metro["Heat"]  / (metro["total_events"] + 1e-6)
        metro["flood_ratio"] = metro["Flood"] / (metro["total_events"] + 1e-6)
        metro["temp_diff_C"]     = metro["mean_temp_C_weather"]     - metro["mean_temp_C_calendar"]
        metro["dewpoint_diff_C"] = metro["mean_dewpoint_C_weather"] - metro["mean_dewpoint_C_calendar"]
        metro["precip_diff_mm"]  = metro["total_precip_mm_weather"] - metro["total_precip_mm_calendar"]
        metro["length_diff_days"] = metro["summer_length_days_weather"] - metro["summer_length_days_calendar"]

        # Lags/rolls/trends per METRO
        metro = _add_lags_rolls_trends(
            metro,
            cols_for_features=[
                "summer_length_days_weather",
                "mean_temp_C_weather",
                "mean_dewpoint_C_weather",
                "total_precip_mm_weather",
                "total_events", "Heat", "Flood", "Hurricane", "Tornado", "Drought"
            ],
            group_cols=["METRO"],
            roll=ROLL_YRS
        )

        metro_out = _normalize_columns(metro, id_cols=("year",))
        _print_shape(metro_out, "metro (final)")

    except Exception as e:
        print(f"   ⚠️ Metro-level build skipped: {e}")
else:
    print("   ℹ️ Metro-level build disabled (no mapping or 'STATE' not present).")

# ============================================================
# SAVE OUTPUTS
# ============================================================
print("\n==> Saving outputs...")
os.makedirs("data_processed", exist_ok=True)
nat_path   = os.path.join("data_processed", "master_summer_extremes_national.csv")
state_path = os.path.join("data_processed", "master_summer_extremes_by_state.csv")
metro_path = os.path.join("data_processed", "master_summer_extremes_by_metro.csv")
meta_path  = os.path.join("data_processed", "master_summer_extremes_features_metadata.json")

nat_out.to_csv(nat_path, index=False)
print(f"  ✅ {nat_path}")
if state_out is not None:
    state_out.to_csv(state_path, index=False)
    print(f"  ✅ {state_path}")
if metro_out is not None:
    metro_out.to_csv(metro_path, index=False)
    print(f"  ✅ {metro_path}")

# Feature metadata (names, units, notes)
feature_notes = {
    "ids": ["year", "STATE", "METRO"],
    "climate_calendar": {
        "mean_temp_C_calendar": "JJA temp (°C), day-weighted",
        "mean_dewpoint_C_calendar": "JJA dewpoint (°C), day-weighted",
        "total_precip_mm_calendar": "JJA total precip (mm)",
        "summer_length_days_calendar": "Fixed JJA length in days (~92)",
    },
    "climate_weather": {
        "mean_temp_C_weather": "Weather-driven summer mean temp (°C), weighted by summer days",
        "mean_dewpoint_C_weather": "Weather-driven summer mean dewpoint (°C)",
        "total_precip_mm_weather": "Weather-driven summer total precip (mm)",
        "summer_length_days_weather": "Weather-driven summer length (days), Lin & Wang-based",
        "onset_month_weather": "Computed in process_data (if exported)",
        "end_month_weather": "Computed in process_data (if exported)",
    },
    "hazards": {c: f"NOAA StormEvents yearly count ({c})" for c in CATS},
    "derived": {
        "total_events": "Sum of major hazard counts",
        "heat_ratio": "Heat / total_events",
        "flood_ratio": "Flood / total_events",
        "temp_diff_C": "mean_temp_C_weather - mean_temp_C_calendar",
        "dewpoint_diff_C": "mean_dewpoint_C_weather - mean_dewpoint_C_calendar",
        "precip_diff_mm": "total_precip_mm_weather - total_precip_mm_calendar",
        "length_diff_days": "summer_length_days_weather - summer_length_days_calendar",
    },
    "temporal_features": {
        "lag1": "Previous year value (per group if state/metro)",
        f"roll{ROLL_YRS}": f"Centered {ROLL_YRS}-yr rolling mean",
        "trend": "First difference year-over-year",
    },
    "normalization": "Columns with suffix _norm are min-max normalized across the dataset.",
    "years": {"min": MIN_YEAR, "max": MAX_YEAR},
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(feature_notes, f, indent=2)
print(f"  ✅ {meta_path}")

print("\n🎉 Done. ML-ready datasets are in data_processed/.")
