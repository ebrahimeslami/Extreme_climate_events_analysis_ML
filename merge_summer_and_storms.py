# -*- coding: utf-8 -*-
"""
merge_summer_and_storms.py
--------------------------
Merges NOAA Storm Events data with processed summer metrics to create a
complete dataset linking summer length, temperature, and extreme events.

Author: Ebrahim Eslami
Date: 2025
"""

import os
import pandas as pd
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================
SUMMER_FILE = os.path.join("data_processed", "summer_comparison_summary.csv")
STORMS_DIR = os.path.join("data_raw", "storm_events")  # folder with NOAA csv.gz files
OUTPUT_FILE = os.path.join("data_processed", "summer_storms_merged.csv")

# ============================================================
# LOAD SUMMER METRICS
# ============================================================
summer = pd.read_csv(SUMMER_FILE)
print(f"✅ Loaded summer metrics: {len(summer)} records ({summer['year'].min()}–{summer['year'].max()})")

# ============================================================
# LOAD STORM EVENTS
# ============================================================
print(f"📂 Loading NOAA Storm Events from {STORMS_DIR} ...")
storm_files = [f for f in os.listdir(STORMS_DIR) if f.endswith(".csv") or f.endswith(".csv.gz")]
if not storm_files:
    raise FileNotFoundError("❌ No storm event files found in data_raw/storm_events/")

storms_list = []
for f in tqdm(storm_files, desc="Reading storm event files"):
    path = os.path.join(STORMS_DIR, f)
    try:
        df = pd.read_csv(path, compression="infer", low_memory=False)
        storms_list.append(df)
    except Exception as e:
        print(f"⚠️ Skipped {f} due to error: {e}")

storms = pd.concat(storms_list, ignore_index=True)
print(f"✅ Loaded {len(storms):,} storm event records")

# ============================================================
# CLEAN AND CATEGORIZE EVENTS
# ============================================================
storms["BEGIN_DATE_TIME"] = pd.to_datetime(storms["BEGIN_DATE_TIME"], errors="coerce")
storms["year"] = storms["BEGIN_DATE_TIME"].dt.year

# Normalize event names
storms["EVENT_TYPE"] = storms["EVENT_TYPE"].str.upper().str.strip()

category_map = {
    "HURRICANE": "Hurricane",
    "TROPICAL STORM": "Hurricane",
    "FLASH FLOOD": "Flood",
    "FLOOD": "Flood",
    "HEAT": "Heat",
    "EXCESSIVE HEAT": "Heat",
    "TORNADO": "Tornado",
    "DROUGHT": "Drought",
}

storms["Category"] = storms["EVENT_TYPE"].map(category_map)
storms = storms[storms["Category"].notna()]  # keep only relevant ones

# ============================================================
# COUNT EVENTS PER YEAR
# ============================================================
events_by_year = storms.groupby(["year", "Category"]).size().unstack(fill_value=0).reset_index()

# Ensure all expected columns exist
for cat in ["Hurricane", "Flood", "Heat", "Tornado", "Drought"]:
    if cat not in events_by_year.columns:
        events_by_year[cat] = 0

print(f"✅ Aggregated yearly event counts ({len(events_by_year)} years)")

# ============================================================
# MERGE WITH SUMMER METRICS
# ============================================================
merged = summer.merge(events_by_year, on="year", how="left")
merged = merged.fillna(0)

# ============================================================
# SAVE
# ============================================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
merged.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Merged dataset saved to: {OUTPUT_FILE}")
print("📊 Example:")
print(merged.head())
