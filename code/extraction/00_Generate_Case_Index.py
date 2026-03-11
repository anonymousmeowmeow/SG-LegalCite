"""
00_Generate_Case_Index.py
=========================
Generates the master index of Singapore Supreme Court judgment URLs
by probing eLitigation (https://www.elitigation.sg) for valid case pages.

Output: singapore_cases_complete_<TIMESTAMP>.xlsx
Columns: Year, Court_Type, Case_Number, URL, Full_Reference

Logic:
- Iterates years 2000–2025, court types, and case numbers (starting from 1)
- Confirms each URL exists by loading the page (HTTP 200 + content check)
- Stops iterating case numbers for a given (year, court_type) after
  MAX_CONSECUTIVE_FAILURES consecutive misses
- Saves progress incrementally so the script can be safely interrupted
  and resumed

Usage:
    pip install requests pandas openpyxl tqdm
    python 00_Generate_Case_Index.py

    # To resume after interruption, just re-run — it picks up from the
    # progress file automatically.
"""

import requests
import urllib3
import pandas as pd
import json
import os
import time
from datetime import datetime
from tqdm import tqdm

# Suppress SSL warnings (needed on some macOS environments)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

START_YEAR = 2000
END_YEAR   = 2025

# Court types in order of typical volume (largest first for faster feedback)
COURT_TYPES = ["SGHC", "SGCA", "SGHCF", "SGHCR", "SGCAI"]

# Stop iterating case numbers after this many consecutive 404/failures
MAX_CONSECUTIVE_FAILURES = 10

# Max case number to try before giving up regardless (safety ceiling)
MAX_CASE_NUMBER = 9999

# Seconds to wait between requests (be polite to the server)
REQUEST_DELAY = 0.3

# Retry attempts per URL before counting as a failure
MAX_RETRIES = 3

# Output / progress files
TIMESTAMP       = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE     = f"singapore_cases_complete_{TIMESTAMP}.csv"
PROGRESS_FILE   = "case_index_progress.json"

# ─────────────────────────── HELPERS ─────────────────────────────────────────

BASE_URL = "https://www.elitigation.sg/gd/s/{year}_{court}_{num}"

def build_url(year: int, court: str, num: int) -> str:
    return BASE_URL.format(year=year, court=court, num=num)

def build_reference(year: int, court: str, num: int) -> str:
    return f"[{year}] {court} {num}"

def page_exists(url: str) -> bool:
    """
    Returns True if the URL resolves to a real judgment page.
    Checks for HTTP 200 and that the response body is not an
    eLitigation 'not found' / error page.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15, allow_redirects=True, verify=False)
            if resp.status_code == 200:
                # eLitigation sometimes returns 200 for missing cases with
                # an error message in the body — filter those out
                body_lower = resp.text.lower()
                if ("no results found" in body_lower
                        or "page not found" in body_lower
                        or "not available" in body_lower
                        or "citation for this case has been reassigned" in body_lower
                        or len(resp.text) < 500):   # suspiciously short page
                    pass  # treat as failure, retry
                else:
                    return True
            # For any non-200 or failed content check, retry
        except requests.RequestException:
            pass
        time.sleep(REQUEST_DELAY * (attempt + 1))   # back-off on retry
    return False

# ─────────────────────────── PROGRESS ────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_progress(year: int, court: str, case_num: int, records: list):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "year":     year,
            "court":    court,
            "case_num": case_num,
            "timestamp": datetime.now().isoformat(),
            "records_so_far": len(records)
        }, f, indent=2)

def clear_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

def save_records(records: list):
    """Incrementally save collected records to CSV."""
    if not records:
        return
    df = pd.DataFrame(records, columns=["Year", "Court_Type", "Case_Number",
                                         "URL", "Full_Reference"])
    df.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SG-LEGALCITE — CASE INDEX GENERATOR")
    print("=" * 65)

    records     = []
    start_year  = START_YEAR
    start_court_idx = 0
    start_case  = 1

    # Check for previous progress
    prev = load_progress()
    if prev:
        print(f"\nPrevious progress found:")
        print(f"  Last position : {prev['year']} / {prev['court']} / case #{prev['case_num']}")
        print(f"  Records so far: {prev['records_so_far']}")
        print(f"  Timestamp     : {prev['timestamp']}")
        print(f"\nOptions:")
        print(f"  1. Resume from last position")
        print(f"  2. Start fresh")
        choice = input("\nEnter choice (1/2): ").strip()
        if choice == "1":
            start_year      = prev["year"]
            start_court_idx = COURT_TYPES.index(prev["court"]) if prev["court"] in COURT_TYPES else 0
            start_case      = prev["case_num"] + 1
            # We can't recover the in-memory records from a previous run,
            # but the xlsx was saved incrementally — load it
            if os.path.exists(OUTPUT_FILE):
                existing_df = pd.read_csv(OUTPUT_FILE)
                records = existing_df.to_dict("records")
                print(f"  Loaded {len(records)} existing records from {OUTPUT_FILE}")
            else:
                # Find latest output file
                csv_files = [f for f in os.listdir(".")
                              if f.startswith("singapore_cases_complete_") and f.endswith(".csv")]
                if csv_files:
                    latest = sorted(csv_files)[-1]
                    existing_df = pd.read_csv(latest)
                    records = existing_df.to_dict("records")
                    print(f"  Loaded {len(records)} existing records from {latest}")
        else:
            clear_progress()
            print("  Starting fresh.")

    total_found = len(records)
    print(f"\nStarting from year {start_year}, court {COURT_TYPES[start_court_idx]}, case #{start_case}")
    print(f"Already have {total_found} records.\n")

    try:
        for year in range(start_year, END_YEAR + 1):
            ct_start = start_court_idx if year == start_year else 0

            for court in COURT_TYPES[ct_start:]:
                consecutive_fails = 0
                last_found_num = 0
                cs_start = start_case if (year == start_year and court == COURT_TYPES[ct_start]) else 1

                desc = f"{year} {court:6s}"
                for case_num in tqdm(range(cs_start, MAX_CASE_NUMBER + 1), desc=desc, leave=False):
                    url = build_url(year, court, case_num)

                    if page_exists(url):
                        records.append({
                            "Year":           year,
                            "Court_Type":     court,
                            "Case_Number":    case_num,
                            "URL":            url,
                            "Full_Reference": build_reference(year, court, case_num)
                        })
                        consecutive_fails = 0
                        last_found_num = case_num
                        total_found += 1

                        # Save incrementally every 50 new records
                        if total_found % 50 == 0:
                            save_records(records)
                            save_progress(year, court, case_num, records)
                    else:
                        consecutive_fails += 1
                        if consecutive_fails >= MAX_CONSECUTIVE_FAILURES:
                            ct_count = sum(1 for r in records if r['Year']==year and r['Court_Type']==court)
                            tqdm.write(f"  ✓ {year} {court}: stopped at #{case_num} "
                                       f"({consecutive_fails} consecutive misses) — "
                                       f"{ct_count} cases found, last valid: #{last_found_num}")
                            break

                    time.sleep(REQUEST_DELAY)

                save_progress(year, court, case_num, records)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user — saving progress...")

    # Final save
    save_records(records)
    clear_progress()

    print("\n" + "=" * 65)
    print("COMPLETE")
    print("=" * 65)
    print(f"Total cases found : {len(records)}")
    print(f"Output file       : {OUTPUT_FILE}")

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"\nBreakdown by court type:")
        print(df.groupby("Court_Type").size().to_string())
        print(f"\nYear range: {df['Year'].min()} – {df['Year'].max()}")


if __name__ == "__main__":
    main()
