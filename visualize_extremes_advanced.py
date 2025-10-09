# -*- coding: utf-8 -*-
"""
visualize_extremes_advanced.py
--------------------------------
Advanced visualization toolkit for exploring the relationships between
summer length, temperature, humidity, and extreme weather across regions.

Features:
  • Time-series with adaptive scaling
  • Trend analysis (Theil–Sen slope)
  • Regional and correlation panels
  • Dual-axis visualization for multi-scale variables
  • Rolling (5-year) trend smoothing for clarity

Author: Ebrahim Eslami
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import theilslopes

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = os.path.join("data_processed", "summer_storms_merged.csv")
OUTPUT_DIR = os.path.join("figures", "advanced_relationships")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid", context="talk", palette="crest")

# ============================================================
# LOAD DATA
# ============================================================
# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df)} records from {DATA_FILE}")
print("Columns:", list(df.columns))

# Drop missing values where necessary
df = df.dropna(subset=["summer_length_days_weather", "mean_temp_C_weather"])

# Ensure extreme-event columns exist (create placeholders if missing)
event_cols = ["Hurricane", "Flood", "Heat", "Tornado", "Drought"]
for col in event_cols:
    if col not in df.columns:
        df[col] = 0  # placeholder if not merged yet

# Compute total events
df["total_events"] = df[event_cols].sum(axis=1)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def plot_dual_axis(df, x, y1, y2, label1, label2, title, filename):
    """
    Plot with two y-axes for variables of very different scales.
    Automatically adjusts limits and color balance.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    c1, c2 = "darkorange", "steelblue"

    ax1.plot(df[x], df[y1], color=c1, lw=2, label=label1)
    ax2.plot(df[x], df[y2], color=c2, lw=2, linestyle="--", label=label2)

    # Auto-adjust y-scale visibility
    ax1.set_ylim(df[y1].min() * 0.9, df[y1].max() * 1.1)
    ax2.set_ylim(df[y2].min() * 0.9, df[y2].max() * 1.1)

    ax1.set_xlabel("Year")
    ax1.set_ylabel(label1, color=c1)
    ax2.set_ylabel(label2, color=c2)
    plt.title(title)
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"💾 Saved {filename}")


def compute_theilsen_trend(x, y):
    """Compute Theil–Sen slope and intercept."""
    try:
        slope, intercept, _, _ = theilslopes(y, x)
        return slope, intercept
    except Exception:
        return np.nan, np.nan


# ============================================================
# 1️⃣ Rolling averages & smoothed time-series
# ============================================================
df_sorted = df.sort_values("year")
df_sorted["summer_length_rolling"] = df_sorted["summer_length_days_weather"].rolling(5, center=True).mean()
df_sorted["mean_temp_rolling"] = df_sorted["mean_temp_C_weather"].rolling(5, center=True).mean()
df_sorted["events_rolling"] = df_sorted["total_events"].rolling(5, center=True).mean()

plt.figure(figsize=(12, 6))
plt.plot(df_sorted["year"], df_sorted["summer_length_rolling"], color="darkorange", lw=2, label="Summer Length (5-yr mean)")
plt.plot(df_sorted["year"], df_sorted["mean_temp_rolling"], color="firebrick", lw=2, label="Mean Temp (°C, 5-yr mean)")
plt.plot(df_sorted["year"], df_sorted["events_rolling"] * 5, color="royalblue", lw=2, label="Extreme Events ×5 (5-yr mean)")
plt.title("⏳ Smoothed 5-Year Trends: Summer Length, Temperature, and Extremes")
plt.xlabel("Year")
plt.ylabel("Value (°C or days)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "smoothed_trends.png"), dpi=300)
plt.close()


# ============================================================
# 2️⃣ Dual-axis plots for multi-scale relationships
# ============================================================
plot_dual_axis(df_sorted, "year",
               "summer_length_days_weather", "Heat",
               "Summer Length (days)", "Heat Events",
               "Summer Length vs Heat Events", "dual_summer_heat.png")

plot_dual_axis(df_sorted, "year",
               "mean_temp_C_weather", "Flood",
               "Mean Temperature (°C)", "Flood Events",
               "Temperature vs Flood Events", "dual_temp_flood.png")

plot_dual_axis(df_sorted, "year",
               "mean_dewpoint_C_weather", "Hurricane",
               "Mean Dewpoint (°C)", "Hurricane Events",
               "Humidity vs Hurricanes", "dual_dew_hurricane.png")


# ============================================================
# 3️⃣ Trend estimation (Theil–Sen slopes)
# ============================================================
metrics = [
    "summer_length_days_weather",
    "mean_temp_C_weather",
    "mean_dewpoint_C_weather",
    "total_precip_mm_weather",
    "total_events"
]

trends = []
for col in metrics:
    slope, intercept = compute_theilsen_trend(df_sorted["year"], df_sorted[col])
    trends.append({"variable": col, "slope_per_year": slope, "intercept": intercept})

trend_df = pd.DataFrame(trends)
print("\n📈 Theil–Sen trend estimates:")
print(trend_df)
trend_df.to_csv(os.path.join(OUTPUT_DIR, "trend_summary.csv"), index=False)


# ============================================================
# 4️⃣ Correlation matrix with heat events highlighted
# ============================================================
corr_cols = [
    "summer_length_days_weather",
    "mean_temp_C_weather",
    "mean_dewpoint_C_weather",
    "total_precip_mm_weather",
    "Hurricane", "Flood", "Heat", "Tornado", "Drought"
]
corr_df = df[corr_cols].corr(method="spearman")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("🔗 Spearman Correlation Between Summer Metrics & Extreme Events")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()


# ============================================================
# 5️⃣ Regional trends and boxplots (if region/state column exists)
# ============================================================
if "region" in df.columns or "state" in df.columns:
    region_col = "region" if "region" in df.columns else "state"

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=region_col, y="summer_length_days_weather", palette="coolwarm")
    plt.title("📦 Distribution of Weather-driven Summer Length by Region")
    plt.ylabel("Summer Length (days)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_summer_length_by_region.png"), dpi=300)
    plt.close()

    # Correlation per region (between length and each hazard)
    region_corrs = []
    for region, sub in df.groupby(region_col):
        corr = sub[["summer_length_days_weather", "Heat", "Flood", "Hurricane", "Drought"]].corr(method="spearman")
        row = {"region": region}
        for c in ["Heat", "Flood", "Hurricane", "Drought"]:
            row[c] = corr.loc["summer_length_days_weather", c]
        region_corrs.append(row)

    corr_region_df = pd.DataFrame(region_corrs).set_index("region")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_region_df, annot=True, cmap="coolwarm", center=0)
    plt.title("Regional Correlations Between Summer Length & Extreme Events")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "regional_correlation_heatmap.png"), dpi=300)
    plt.close()


# ============================================================
# 6️⃣ Multi-panel summary: all variables vs total events
# ============================================================
vars_to_plot = ["summer_length_days_weather", "mean_temp_C_weather",
                "mean_dewpoint_C_weather", "total_precip_mm_weather"]

fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 3.5 * len(vars_to_plot)))
for i, v in enumerate(vars_to_plot):
    ax1 = axes[i]
    ax2 = ax1.twinx()
    ax1.plot(df_sorted["year"], df_sorted[v], color="darkorange", lw=2)
    ax2.bar(df_sorted["year"], df_sorted["total_events"], color="royalblue", alpha=0.3)
    ax1.set_ylabel(v.replace("_", " "), color="darkorange")
    ax2.set_ylabel("Extreme Events", color="royalblue")
    ax1.set_title(f"{v.replace('_', ' ').title()} vs Extreme Events")
    ax1.set_xlabel("Year")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "multi_panel_summer_vs_extremes.png"), dpi=300)
plt.close()


print(f"\n✅ Advanced visualizations saved to: {OUTPUT_DIR}")
