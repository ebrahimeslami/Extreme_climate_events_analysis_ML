"""
Analyze relationships between summer duration and extreme events
Author: Ebrahim Eslami
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip

# === Resolve project paths automatically ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data_raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data_processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_output", "analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === File paths ===
SUMMER_FILE = os.path.join(DATA_PROCESSED, "summer_comparison_summary.csv")
STORM_DIR = os.path.join(DATA_RAW, "storm_events")
#os.makedirs(STORM_DIR, exist_ok=True)

# === 1️⃣ Load data ===
print("==> Loading summer metrics...")
if not os.path.exists(SUMMER_FILE):
    raise FileNotFoundError(
        f"❌ Could not find {SUMMER_FILE}\n"
        "Please run process_data.py first to generate this file."
    )

summer = pd.read_csv(SUMMER_FILE)
# === Normalize column names for compatibility ===
summer.columns = [c.strip().lower() for c in summer.columns]
rename_map = {
    "summer_length_weather": "summer_length_days_weather",
    "summer_length_calendar": "summer_length_days_calendar",
}
summer.rename(columns=rename_map, inplace=True)

print(f"✅ Loaded summer metrics: {len(summer)} records")

print("==> Loading NOAA Storm Events data...")
if not os.path.exists(STORM_DIR):
    raise FileNotFoundError(
        f"❌ Could not find {STORM_DIR}\n"
        "Please ensure storm event CSVs are downloaded in data_raw/storm_events/\n/"
    )

# --- Support both .csv and .csv.gz files ---
storm_files = [
    f for f in os.listdir(STORM_DIR)
    if f.endswith(".csv") or f.endswith(".csv.gz")
]

if not storm_files:
    raise FileNotFoundError(f"❌ No storm CSV or CSV.GZ files found in {STORM_DIR}")

df_list = []
for f in storm_files:
    path = os.path.join(STORM_DIR, f)
    print(f"📥 Loading {os.path.basename(f)} ...")
    try:
        if f.endswith(".gz"):
            df = pd.read_csv(path, compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(path, low_memory=False)
        df_list.append(df)
    except Exception as e:
        print(f"⚠️ Skipping {f}: {e}")

storms = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(storms):,} storm event records\n")


# === 2️⃣ Clean & preprocess ===
storms["BEGIN_DATE_TIME"] = pd.to_datetime(storms["BEGIN_DATE_TIME"], errors="coerce")
storms["year"] = storms["BEGIN_DATE_TIME"].dt.year
storms = storms.dropna(subset=["year"])

# --- Focus on Gulf and adjacent coastal states ---
GULF_STATES = [
    "TEXAS", "LOUISIANA", "MISSISSIPPI", "ALABAMA", "FLORIDA",
    "GEORGIA", "SOUTH CAROLINA"
]
storms = storms[storms["STATE"].isin(GULF_STATES)]

# === 3️⃣ Metro-area classification ===
def classify_region(row):
    cz = str(row.get("CZ_NAME", "")).lower()
    state = row["STATE"]
    # --- Major Texas Metros ---
    if any(k in cz for k in ["houston", "harris", "galveston", "brazoria"]):
        return "Houston Metro"
    if any(k in cz for k in ["dallas", "tarrant", "fort worth"]):
        return "Dallas-Fort Worth Metro"
    if any(k in cz for k in ["austin", "travis", "williamson"]):
        return "Austin Metro"
    if any(k in cz for k in ["san antonio", "bexar"]):
        return "San Antonio Metro"

    # --- Louisiana / Mississippi / Alabama ---
    if any(k in cz for k in ["new orleans", "jefferson", "st bernard"]):
        return "New Orleans Metro"
    if any(k in cz for k in ["baton rouge", "east baton rouge"]):
        return "Baton Rouge Metro"
    if any(k in cz for k in ["gulfport", "biloxi", "harrison"]):
        return "Gulfport-Biloxi Metro"
    if any(k in cz for k in ["mobile", "baldwin"]):
        return "Mobile Metro"

    # --- Florida / Georgia / Carolinas ---
    if any(k in cz for k in ["pensacola", "escambia"]):
        return "Pensacola Metro"
    if any(k in cz for k in ["tallahassee", "leon"]):
        return "Tallahassee Metro"
    if any(k in cz for k in ["tampa", "st. petersburg", "hillsborough", "pinellas"]):
        return "Tampa Bay Metro"
    if any(k in cz for k in ["orlando", "orange", "seminole"]):
        return "Orlando Metro"
    if any(k in cz for k in ["miami", "dade", "broward", "palm beach"]):
        return "Miami-Ft Lauderdale Metro"
    if any(k in cz for k in ["atlanta", "fulton", "dekalb"]):
        return "Atlanta Metro"
    if any(k in cz for k in ["charleston"]):
        return "Charleston Metro"

    # Default: use state name
    return state.title()

storms["REGION"] = storms.apply(classify_region, axis=1)

# === 4️⃣ Extreme event categories ===
extreme_keywords = {
    "Hurricane": ["Hurricane", "Tropical Storm"],
    "Flood": ["Flood", "Flash Flood"],
    "Heat": ["Heat", "Excessive Heat"],
    "Tornado": ["Tornado"],
    "Drought": ["Drought"],
}

def count_extremes(df):
    out = []
    for year, g in df.groupby("year"):
        rec = {"year": year}
        for cat, kws in extreme_keywords.items(): # To avoid overcounting
            mask = g["EVENT_TYPE"].str.contains("|".join(kws), case=False, na=False)
            rec[cat] = g.loc[mask, "EVENT_ID"].nunique() if "EVENT_ID" in g.columns else mask.sum()
        out.append(rec)
    return pd.DataFrame(out)


# === 5️⃣ Overall Gulf analysis ===
overall = count_extremes(storms)
merged_all = pd.merge(summer, overall, on="year", how="inner")
merged_all.to_csv(os.path.join(OUTPUT_DIR, "gulf_overall.csv"), index=False)

plt.figure(figsize=(8, 6))
sns.heatmap(merged_all.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Gulf Region: Correlation between Summer Metrics & Extreme Events")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corr_gulf_overall.png"))
plt.close()
print("✅ Saved overall Gulf correlation heatmap.")

# === 6️⃣ Regional analysis (States + Metros) ===
regions = sorted(storms["REGION"].unique())
print(f"==> Analyzing {len(regions)} regions...")

for region in regions:
    df_r = storms[storms["REGION"] == region]
    counts = count_extremes(df_r)
    merged = pd.merge(summer, counts, on="year", how="inner")

    region_name = region.replace(" ", "_").replace("-", "_").lower()
    out_csv = os.path.join(OUTPUT_DIR, f"{region_name}_summary.csv")
    merged.to_csv(out_csv, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(merged["year"], merged["summer_length_days_weather"], color="darkred",
             label="Weather Summer Length (days)")
    plt.plot(merged["year"], merged["Hurricane"], color="navy", label="Hurricanes")
    plt.title(f"{region}: Summer Length vs Hurricane Events")
    plt.xlabel("Year")
    plt.ylabel("Days / Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{region_name}_trend.png"))
    plt.close()

print("✅ Completed all state and metro analyses.")

# === 7️⃣ Gulf-wide summer trend ===
plt.figure(figsize=(10, 6))
plt.plot(merged_all["year"], merged_all["summer_length_days_calendar"], label="Calendar Summer", color="orange")
plt.plot(merged_all["year"], merged_all["summer_length_days_calendar"], label="Weather-Driven Summer", color="red")
plt.title("Gulf Region: Trend in Summer Length (1979–2024)")
plt.xlabel("Year")
plt.ylabel("Days")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "summer_length_gulf_trend.png"))
plt.close()

print("\n🎯 All analyses (overall, state, and metro) saved in:", OUTPUT_DIR)
