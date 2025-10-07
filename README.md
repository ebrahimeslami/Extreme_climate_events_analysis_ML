# Extreme_climate_events_analysis_ML
Automated Python tool for downloading NOAA and ERA5 climate datasets for the U.S. Gulf Coast region (1979–2024).
# 🌤️ Climate Analysis Data Downloader

**Author:** Ebrahim Eslami, PhD  
**License:** MIT  
**Python:** 3.13+

A lightweight, modular Python tool for downloading official climate and extreme-event datasets
for the **U.S. Gulf Coast region (1979–2024)**.  
The tool automatically retrieves:

- **NOAA nClimDiv:** monthly temperature and precipitation
- **NOAA HURDAT2:** Atlantic hurricane tracks and intensities
- **NOAA Storm Events:** flood, rainfall, and severe weather reports
- **Copernicus ERA5:** monthly reanalysis (2 m temperature and total precipitation)

---

## 🧩 Features
- One-command quickstart: `python download_data.py --quickstart`
- Regionally subset ERA5 data (Gulf Coast bounding box)
- Handles storage-efficient downloads with progress bars
- Works cross-platform (Windows / macOS / Linux)
- Fully open and reproducible

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Climate_analysis_downloader.git
cd Climate_analysis_downloader
