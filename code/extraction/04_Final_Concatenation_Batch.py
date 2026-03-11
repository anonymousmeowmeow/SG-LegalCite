"""
04_Final_Concatenation_Batch.py
================================
Final step — adds Case Name, Current Court Level, and Precedential Weight
to the dataset.

Reads  : 03_Fact_Query_Output.csv
Outputs: 04_SG_LegalCite_Final.csv

What this script adds:
- Case Name      : scraped from eLitigation page of each unique Judgment_URL
- Current Court Level : derived from Court_Type column (no scraping needed)
- Court Level    : derived from Cited Case URL if available, else from
                   existing Court Level column from Step 2
- Precedential Weight : Binding / Comity / Persuasive based on court hierarchy

Usage:
    pip install requests pandas beautifulsoup4 tqdm
    python 04_Final_Concatenation_Batch.py
"""

import pandas as pd
import requests
import re
import os
import json
import time
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

INPUT_FILE       = "03_Fact_Query_Output.csv"
OUTPUT_FILE      = "04_SG_LegalCite_Final.csv"
CASE_NAME_CACHE  = "04_case_name_cache.json"   # cache so we don't re-scrape

REQUEST_DELAY    = 0.3
SAVE_EVERY       = 50

# ─────────────────────────── COURT LEVEL MAPPINGS ────────────────────────────

# Derive Current Court Level directly from Court_Type column
COURT_TYPE_TO_LEVEL = {
    "SGCA":  "Singapore Court of Appeal",
    "SGCAI": "Singapore Court of Appeal International",
    "SGHC":  "Singapore High Court",
    "SGHCF": "Singapore High Court Family Division",
    "SGHCR": "Singapore High Court Registrar",
    "SGDC":  "Singapore District Court",
    "SGMC":  "Singapore District Court",
}

# Court hierarchy for precedential weight calculation
COURT_HIERARCHY = {
    "Singapore Court of Appeal":               4,
    "Singapore Court of Appeal International": 4,
    "Singapore High Court":                    3,
    "Singapore High Court Family Division":    3,
    "Singapore High Court Registrar":          2,
    "Singapore District Court":                1,
    "Singapore Court Level Undetermined":      0,
    "Foreign Court":                           0,
}

# ─────────────────────────── CASE NAME SCRAPING ──────────────────────────────

def extract_style_1(soup):
    header = soup.select_one("h2.title") or soup
    title    = header.select_one("span.caseTitle")
    citation = header.select_one("span.Citation, span.NCitation")
    if title and citation:
        return f"{title.get_text(' ', strip=True)} {citation.get_text(strip=True)}"
    return None

def extract_style_3(soup):
    title    = soup.find("div", class_="HN-CaseName")
    citation = soup.find("div", class_="HN-NeutralCit")
    if title and citation:
        return f"{title.get_text(' ', strip=True)} {citation.get_text(strip=True)}"
    return None

def extract_role_based(soup, url):
    roles = {"Appellant": [], "Respondent": [], "Plaintiff": [],
             "Defendant": [], "Claimants": [], "Intervener": []}
    for role_tag in soup.find_all("span", class_="font-italic"):
        role = role_tag.get_text(strip=True)
        if role in roles:
            name_table = role_tag.find_parent("div").find_previous_sibling("table")
            if name_table:
                name = name_table.get_text(" ", strip=True)
                if name:
                    roles[role].append(name)
    left  = roles["Appellant"] or roles["Plaintiff"] or roles["Claimants"]
    right = roles["Respondent"] or roles["Defendant"]
    if not (left and right):
        return None
    year_match = re.search(r'/(\d{4})_', url)
    year = year_match.group(1) if year_match else "Unknown"
    case_name = f"{' & '.join(left)} v {' & '.join(right)}"
    if roles["Intervener"]:
        case_name += f" ({' & '.join(roles['Intervener'])}, intervener)"
    case_name += f" [{year}]"
    return case_name

def scrape_case_name(url):
    try:
        r = requests.get(url, timeout=20, verify=False)
        soup = BeautifulSoup(r.text, "html.parser")
        for method in [extract_style_1, extract_style_3]:
            result = method(soup)
            if result:
                return result
        result = extract_role_based(soup, url)
        if result:
            return result
        return "Unknown Case Name"
    except Exception as e:
        return f"Error: {str(e)[:50]}"

# ─────────────────────────── COURT LEVEL HELPERS ─────────────────────────────

def court_type_to_level(court_type):
    """Derive Current Court Level from Court_Type column."""
    return COURT_TYPE_TO_LEVEL.get(str(court_type).strip().upper(),
                                   "Singapore Court Level Undetermined")

def infer_cited_court_level(cited_case_url):
    """Infer court level of a cited case from its URL."""
    if not isinstance(cited_case_url, str):
        return "Singapore Court Level Undetermined"
    url_upper = cited_case_url.upper()
    if "ELITIGATION.SG" not in url_upper:
        return "Foreign Court"
    if "SGCAI" in url_upper:
        return "Singapore Court of Appeal International"
    if "SGCA" in url_upper:
        return "Singapore Court of Appeal"
    if "SGHCR" in url_upper:
        return "Singapore High Court Registrar"
    if "SGHCF" in url_upper:
        return "Singapore High Court Family Division"
    if "SGHC" in url_upper:
        return "Singapore High Court"
    if "SGDC" in url_upper or "SGMC" in url_upper:
        return "Singapore District Court"
    return "Singapore Court Level Undetermined"

def determine_precedential_weight(cited_level, current_level):
    """Return Binding, Comity, or Persuasive."""
    if not isinstance(cited_level, str) or not isinstance(current_level, str):
        return "Persuasive"
    cited_level   = cited_level.strip()
    current_level = current_level.strip()
    if cited_level in ["Singapore Court Level Undetermined", "Foreign Court"]:
        return "Persuasive"
    cited_rank   = COURT_HIERARCHY.get(cited_level, 0)
    current_rank = COURT_HIERARCHY.get(current_level, 0)
    if cited_rank > current_rank:
        return "Binding"
    if cited_rank == current_rank and cited_rank > 0:
        return "Comity"
    return "Persuasive"

# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SG-LEGALCITE — STEP 4: FINAL CONCATENATION (BATCH)")
    print("=" * 65)

    # Load input
    print(f"\nReading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows       : {len(df)}")

    # ── 1. Current Court Level (from Court_Type, no scraping) ────────────────
    print("\nDeriving Current Court Level from Court_Type...")
    df["Current Court Level"] = df["Court_Type"].apply(court_type_to_level)

    # ── 2. Case Name (scrape from eLitigation, cached) ───────────────────────
    print("\nScraping Case Names from eLitigation...")

    # Load cache
    cache = {}
    if os.path.exists(CASE_NAME_CACHE):
        with open(CASE_NAME_CACHE) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached case names.")

    unique_urls = df["Judgment_URL"].unique() if "Judgment_URL" in df.columns else df["URL"].unique()
    url_col     = "Judgment_URL" if "Judgment_URL" in df.columns else "URL"
    to_scrape   = [u for u in unique_urls if u not in cache]
    print(f"Need to scrape   : {len(to_scrape)} new URLs")

    try:
        for i, url in enumerate(tqdm(to_scrape, desc="Case Names")):
            cache[url] = scrape_case_name(url)
            time.sleep(REQUEST_DELAY)
            if (i + 1) % SAVE_EVERY == 0:
                with open(CASE_NAME_CACHE, "w") as f:
                    json.dump(cache, f)
                tqdm.write(f"  💾 Cache saved ({len(cache)} entries)")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted — saving cache...")

    with open(CASE_NAME_CACHE, "w") as f:
        json.dump(cache, f)

    df["Case Name"] = df[url_col].map(lambda u: cache.get(u, "Unknown Case Name"))

    # ── 3. Precedential Weight ────────────────────────────────────────────────
    print("\nCalculating Precedential Weight...")

    # Use existing Court Level column if present, else derive from URL
    if "Court Level" in df.columns:
        df["Precedential Weight"] = df.apply(
            lambda row: determine_precedential_weight(
                row["Court Level"], row["Current Court Level"]), axis=1)
    else:
        tqdm.write("  ⚠ No 'Court Level' column found — Precedential Weight set to Persuasive")
        df["Precedential Weight"] = "Persuasive"

    # ── 4. Reorder columns and save ──────────────────────────────────────────
    # Desired column order
    desired = [
        "Judgment_URL", "Judgment_Reference", "Year", "Court_Type", "Case_Number",
        "Case Name", "Current Court Level",
        "Fact_Query",
        "Cited Case", "Paragraph",
        "Key Principles Illustrated", "Issue", "Issue Group",
        "Court Level", "Precedential Weight"
    ]
    # Keep only columns that exist, then append any extras at the end
    ordered = [c for c in desired if c in df.columns]
    extras  = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("COMPLETE")
    print("=" * 65)
    print(f"Total rows  : {len(df)}")
    print(f"Columns     : {list(df.columns)}")
    print(f"Output      : {OUTPUT_FILE}")

    if "Precedential Weight" in df.columns:
        print(f"\nPrecedential Weight distribution:")
        print(df["Precedential Weight"].value_counts().to_string())


if __name__ == "__main__":
    main()
