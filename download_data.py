# -*- coding: utf-8 -*-
"""
Created on Oct  2025

@author: Ebrahim Eslami (e.eslami@gmail.com)
"""

"""
Lightweight data downloader for:
- NOAA nClimDiv (monthly TMAX, TMIN, PCP) - very small
- NOAA HURDAT2 (Atlantic hurricanes) - tiny
- NOAA Storm Events (filter by Gulf states + chosen years) - moderate
- (Optional) ERA5 monthly means (Gulf subset) via CDS API - small-ish

Storage-aware:
- Skips existing files
- Streams large downloads
- Lets you choose which datasets and years

Usage examples:
  python download_data.py --all --years 1990-2024
  python download_data.py --nclimdiv
  python download_data.py --storm 2000,2005,2017,2020
  python download_data.py --era5 --era5-years 1979-2024

Gulf states: TX, LA, MS, AL, FL
"""

import os
import re
import sys
import gzip
import glob
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from dateutil import parser as dateparser
import requests
from tqdm import tqdm

import zipfile
import shutil
import xarray as xr

def ensure_unzipped_nc(filepath: str) -> str:
    """
    Detects whether a CDS 'NetCDF' file is actually a zipped archive
    or a GRIB file mislabeled as .nc. Extracts and validates the usable file.
    Returns the path of the usable file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    # 1. Unzip if necessary
    if zipfile.is_zipfile(filepath):
        print(f"⚠️  File '{os.path.basename(filepath)}' is zipped. Extracting...")
        extract_dir = os.path.dirname(filepath)
        with zipfile.ZipFile(filepath) as z:
            members = z.namelist()
            inner_nc = next((m for m in members if m.endswith(".nc")), members[0])
            z.extract(inner_nc, extract_dir)
            new_path = os.path.join(extract_dir, os.path.basename(inner_nc))
            final_path = filepath.replace(".nc", "_unzipped.nc")
            shutil.move(new_path, final_path)
        filepath = final_path
        print(f"✅  Extracted to: {filepath}")

    # 2. Validate file size
    size_mb = os.path.getsize(filepath) / 1e6
    if size_mb < 5:
        raise RuntimeError(f"❌ Downloaded file too small ({size_mb:.1f} MB) — likely incomplete.")

    return filepath

# ===== Seamless ERA5 detection =====
def _home_dir():
    import os
    return os.path.expanduser("~")

def cds_available():
    """Return (has_cdsapi, has_cds_config_path)."""
    try:
        import cdsapi  # noqa: F401
        has_cdsapi = True
    except Exception:
        has_cdsapi = False
    import os
    rc_path = os.path.join(_home_dir(), ".cdsapirc")
    return has_cdsapi, os.path.isfile(rc_path)

# -------------------------
# Config / constants
# -------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data_raw"
DATA_RAW.mkdir(exist_ok=True, parents=True)

NCLIMDIV_DIR = DATA_RAW / "nclimdiv"
HURDAT2_DIR  = DATA_RAW / "hurdat2"
STORMS_DIR   = DATA_RAW / "storm_events"
ERA5_DIR     = DATA_RAW / "era5"

for p in (NCLIMDIV_DIR, HURDAT2_DIR, STORMS_DIR, ERA5_DIR):
    p.mkdir(exist_ok=True, parents=True)

GULF_STATES = ["TEXAS","LOUISIANA","MISSISSIPPI","ALABAMA","FLORIDA"]

# NOAA directory indexes
NCLIMDIV_BASE = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"
# ✅ Correct current bulk CSV directory for NOAA Storm Events
STORMS_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
HURDAT2_BASE  = "https://www.nhc.noaa.gov/data/hurdat/"

# Filenames in the climdiv directory change by date stamp; we’ll auto-pick latest by prefix.
NCLIMDIV_PREFIXES = {
    "tmax": "climdiv-tmaxdv-v1.0.0-",
    "tmin": "climdiv-tmindv-v1.0.0-",
    "pcp" : "climdiv-pcpndv-v1.0.0-",
}

# -------------------------
# Helpers
# -------------------------
def stream_download(url: str, out_path: Path, chunk=1<<14):
    """Download with streaming and a progress bar."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name
        ) as bar:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)
                    bar.update(len(b))
        tmp.rename(out_path)

STORMS_BASE_FALLBACK = "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

def list_http_files(base_url: str) -> list[str]:
    """Return file names from an Apache-style index, with retry and long timeout."""
    import requests, re, time
    max_tries = 5
    for attempt in range(max_tries):
        try:
            resp = requests.get(base_url, timeout=120)  # 2-minute timeout
            resp.raise_for_status()
            files = re.findall(r'href="([^"?][^"]+)"', resp.text)
            return [f for f in files if not f.startswith("?") and not f.startswith("/")]
        except requests.exceptions.ReadTimeout:
            wait = 10 * (attempt + 1)
            print(f"⏳ Timeout on attempt {attempt+1}/{max_tries}. Retrying in {wait}s…")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            print(f"⚠️  HTTP error on attempt {attempt+1}: {e}")
            time.sleep(10)
    raise SystemExit(f"❌ Failed to fetch file list from {base_url} after {max_tries} tries.")


def ensure_years_list(years_arg: str) -> List[int]:
    """
    Convert "1990-2024" or "1990,1995,2000" to a sorted list of ints.
    Default to 1979-2024 if none supplied.
    """
    if not years_arg:
        return list(range(1979, datetime.utcnow().year + 1))
    if "-" in years_arg:
        a, b = years_arg.split("-")
        return list(range(int(a), int(b) + 1))
    return sorted(set(int(y) for y in years_arg.split(",")))


# -------------------------
# Downloaders
# -------------------------
def download_nclimdiv():
    """
    Get latest climdiv tmax/tmin/pcp fixed-width monthly files (nationwide).
    These are small (<~20 MB each). We’ll store them raw and you can filter Gulf states later.
    """
    print("\n==> Downloading NOAA nClimDiv (monthly TMAX, TMIN, PCP)")
    files = list_http_files(NCLIMDIV_BASE)
    for key, prefix in NCLIMDIV_PREFIXES.items():
        # Find latest filename for this prefix by choosing max lexicographically on trailing date string
        matches = sorted([f for f in files if f.startswith(prefix)])
        if not matches:
            print(f"  !! No files found for {key} at {NCLIMDIV_BASE}")
            continue
        latest = matches[-1]
        url = NCLIMDIV_BASE + latest
        out = NCLIMDIV_DIR / latest
        if out.exists():
            print(f"  - {latest} already exists, skipping.")
        else:
            print(f"  - Downloading {latest} …")
            stream_download(url, out)
    print("==> nClimDiv done.")


def download_hurdat2():
    """
    Grab the latest HURDAT2 Atlantic basin file from the directory by pattern.
    Filenames look like: hurdat2-1851-2023-052624.txt (date code changes).
    """
    print("\n==> Downloading HURDAT2 (Atlantic hurricanes)")
    files = list_http_files(HURDAT2_BASE)
    atl = sorted([f for f in files if f.startswith("hurdat2-") and f.endswith(".txt")])
    if not atl:
        print("  !! No hurdat2 file found.")
        return
    latest = atl[-1]
    url = HURDAT2_BASE + latest
    out = HURDAT2_DIR / latest
    if out.exists():
        print(f"  - {latest} already exists, skipping.")
    else:
        print(f"  - Downloading {latest} …")
        stream_download(url, out)
    print("==> HURDAT2 done.")


def download_storm_events(years: List[int], gulf_only=True):
    """
    Download Storm Events details files per year (gzipped CSV).
    We auto-select the file that matches pattern 'StormEvents_details-ftp_v1.0_d{YEAR}_*.csv.gz'
    Note: You’ll filter by Gulf states later (TX, LA, MS, AL, FL) to keep dataframes small.
    """
    print("\n==> Downloading NOAA Storm Events (details)")
    files = list_http_files(STORMS_BASE)

    def pick_year_file(yr: int) -> str:
        pattern = re.compile(rf"StormEvents_details-ftp_v1\.0_d{yr}_.*\.csv\.gz")
        cand = [f for f in files if pattern.fullmatch(f)]
        # pick latest by c-timestamp (sort lexicographically works here)
        return sorted(cand)[-1] if cand else ""

    for yr in years:
        fname = pick_year_file(yr)
        if not fname:
            print(f"  !! No details file found for {yr}")
            continue
        url = STORMS_BASE + fname
        out = STORMS_DIR / fname
        if out.exists():
            print(f"  - {fname} already exists, skipping.")
        else:
            print(f"  - Downloading {fname} …")
            stream_download(url, out)

    print("==> Storm Events done.")


# ============================================================
# ERA5 Monthly Download Function
# ============================================================
def download_era5_monthly(years=None):
    """
    Download ERA5 monthly-mean reanalysis for Gulf region (1979–2024)
    with auto-splitting, unzip, and GRIB/NetCDF handling.
    """
    import cdsapi
    os.makedirs("data_raw/era5", exist_ok=True)
    outdir = "data_raw/era5"
    dataset = "reanalysis-era5-single-levels-monthly-means"

    if years is None:
        years = list(range(1979, 2025))

    c = cdsapi.Client()

    # Split by decade to avoid timeouts
    for start in range(years[0], years[-1] + 1, 10):
        end = min(start + 9, years[-1])
        yr_block = [str(y) for y in range(start, end + 1)]
        outfile = os.path.join(outdir, f"era5_gulf_{start}_{end}.nc")

        if os.path.exists(outfile.replace(".nc", "_unzipped.nc")):
            print(f"⏩ {outfile} already processed, skipping.")
            continue

        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "surface_pressure",
                "total_precipitation"
            ],
            "year": yr_block,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "format": "netcdf",   # may still return GRIB
            "area": [32.5, -98, 24, -80],  # Gulf region
        }

        print(f"\n📦 Downloading ERA5 {start}-{end} ...")
        c.retrieve(dataset, request, outfile)

        # Check if zipped or small, then fix
        try:
            valid_file = ensure_unzipped_nc(outfile)
            print(f"✅ Valid ERA5 block saved: {valid_file}")
        except Exception as e:
            print(f"⚠️  ERA5 block {start}-{end} failed integrity check: {e}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Download lightweight climate & extremes datasets for Gulf Coast study."
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="Download nClimDiv + HURDAT2 + Storm Events (no ERA5 unless --era5 is also set).")

    ap.add_argument("--quickstart", action="store_true",
                    help="One-shot: download ALL datasets for 1979–2024. Includes ERA5 automatically if cdsapi + .cdsapirc found.")

    ap.add_argument("--nclimdiv", action="store_true", help="Download nClimDiv (monthly Tmax/Tmin/Precip).")
    ap.add_argument("--hurdat2", action="store_true", help="Download HURDAT2 (Atlantic hurricanes).")
    ap.add_argument("--storm", nargs="?", const="", help="Download Storm Events details for given years (e.g., 1990-2024 or 2005,2017).")
    ap.add_argument("--era5", action="store_true", help="Also download ERA5 monthly (Gulf subset) via CDS API.")
    ap.add_argument("--years", dest="years", default="", help="Default for storm; used if --storm without an explicit years list.")
    ap.add_argument("--era5-years", dest="era5_years", default="1979-2024", help="Year range for ERA5 monthly (default 1979-2024).")
    return ap.parse_args()



def main():
    args = parse_args()
    # ===== Smart defaults / Quickstart =====
    if args.quickstart:
        # One-button: everything for 1979–2024; include ERA5 if available
        args.all = True
        args.years = "1979-2024"
        has_cds, has_rc = cds_available()
        args.era5 = bool(has_cds and has_rc)
        args.era5_years = "1979-2024"
        print(f"[Quickstart] Years: {args.years} | ERA5: {'ON' if args.era5 else 'OFF (cdsapi/.cdsapirc not found)'}")

    # If truly no flags, default to all (1979–2024) and ERA5 if set up
    if (not args.all and not args.nclimdiv and not args.hurdat2
        and args.storm is None and not args.era5 and not args.quickstart):
        print("No flags provided — defaulting to ALL datasets for 1979–2024.")
        args.all = True
        args.years = "1979-2024"
        has_cds, has_rc = cds_available()
        args.era5 = bool(has_cds and has_rc)
        args.era5_years = "1979-2024"
        print(f"[Default] ERA5: {'ON' if args.era5 else 'OFF (cdsapi/.cdsapirc not found)'}")

    # Determine what to download
    do_nclim = args.all or args.nclimdiv
    do_hurd  = args.all or args.hurdat2
    do_storm = args.all or (args.storm is not None)
    do_era5  = args.era5

    if not (do_nclim or do_hurd or do_storm or do_era5):
        print("Nothing to do. Use --all or pick datasets, e.g., --nclimdiv --hurdat2 --storm 1990-2024")
        sys.exit(0)

    # Storm years
    if args.storm is None:
        storm_years = []
    elif args.storm == "":
        storm_years = ensure_years_list(args.years or "1979-2024")
    else:
        storm_years = ensure_years_list(args.storm)

    # ERA5 years
    era5_years = range(1979, 2025)
    download_era5_monthly(era5_years)

    # Execute
    if do_nclim:
        download_nclimdiv()
    if do_hurd:
        download_hurdat2()
    if do_storm:
        if not storm_years:
            storm_years = ensure_years_list("1979-2024")
        download_storm_events(storm_years)
    if do_era5:
        download_era5_monthly(era5_years)

    print("\nAll requested downloads completed.")
    print(f"Data saved under: {DATA_RAW.resolve()}")


if __name__ == "__main__":
    # >>> CUSTOM: set your working directory (change path if needed)
    import os
    target_dir = r"D:\Business\Climate_analysis"
    if os.path.isdir(target_dir):
        os.chdir(target_dir)
        print(f"Working directory changed to: {target_dir}")
    else:
        print(f"⚠️ Directory not found: {target_dir}. Using current directory instead.")
    # <<< END CUSTOM

    main()
