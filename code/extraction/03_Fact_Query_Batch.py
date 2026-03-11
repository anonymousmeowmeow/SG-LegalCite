"""
03_Extract_of_Facts_Batch.py
============================
Scrapes the Facts/Background section from each unique judgment URL
and generates a 2-3 sentence lawyer-style summary using DeepSeek.

Reads  : 02_Deepseek_Output.csv
Outputs: 03_Extract_of_Facts_Output.csv

Logic:
- Extracts unique Judgment_URLs from input CSV
- Scrapes Facts/Background section from each judgment page
- Sends text to DeepSeek for lawyer-style summarisation
- Retries failed cases using paragraph fallback
- Maps summaries back to all rows sharing the same Judgment_URL
- Resume-safe: skips already-processed URLs

Usage:
    pip install requests pandas beautifulsoup4 tqdm
    python 03_Extract_of_Facts_Batch.py
"""

import pandas as pd
import requests
import re
import time
import os
import gc
import json
import shutil
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

API_KEY          = "YOUR_DEEPSEEK_API_KEY"
INPUT_FILE       = "02_Deepseek_Output.csv"
OUTPUT_FILE      = "03_Extract_of_Facts_Output.csv"
RAW_TEXT_FILE    = "03_raw_scraped_texts.csv"
BACKUP_FILE      = "03_raw_scraped_texts_backup.csv"
LOG_FILE         = "03_extraction_log.csv"

MODEL            = "deepseek-chat"
MAX_INPUT_TOKENS = 3000
MAX_OUTPUT_TOKENS = 512
SLEEP_BETWEEN    = 1
SAVE_EVERY       = 50

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ─────────────────────────── PROMPT ──────────────────────────────────────────

PROMPT_TEMPLATE = """Rewrite the key facts of this case as a lawyer would describe their client's situation to a colleague — conversational, practical, and focused on what matters legally.

Rules:
- Write 2-3 sentences maximum
- Use phrases like "my client", "the other party", "the company" instead of full names
- Focus on: what happened and what's at stake
- Do NOT include legal questions or ask anything
- Write as if you are a lawyer explaining the situation, not summarising a judgment
- Do NOT start with any preamble like "Of course", "Sure", "Based on the provided text"
- Start directly with the facts

Here are examples of good lawyer-style fact descriptions:

Example 1: My client signed a 2-year clause preventing them from working for any competitor after leaving the company.
Example 2: My client transferred money by mistake to the wrong company and there's no contractual relationship between them.
Example 3: My client is going through a divorce after 15 years together. Her husband earned more but she was a stay-at-home mum taking care of three kids.
Example 4: A new bubble tea shop opened down the road with a name and cup design that customers keep mixing up with my client's brand.
Example 5: My client's customers have been accidentally buying a competitor's product thinking it was ours.
Example 6: The shareholders want to sue a director for losses from a failed project.
Example 7: We have medical reports showing our client's mental state was significantly impaired at the time of the offence.
Example 8: The document production exercise will cost our client over $200,000 and it seems disproportionate to what's actually at stake.

If the text does not contain enough information to rewrite, respond with: "Insufficient information to summarize facts."

Text from Judgment:
{facts_section}"""

# ─────────────────────────── SCRAPING ────────────────────────────────────────

def find_section_heading(soup, keywords):
    for span in soup.find_all("span"):
        if span.string:
            text = span.string.strip().lower()
            if any(k in text for k in keywords):
                return span
    for div in soup.find_all("div", class_=lambda c: c and "Judg-Heading" in c):
        if any(k in div.get_text(strip=True).lower() for k in keywords):
            return div
    for p in soup.find_all("p", class_=lambda c: c and "Judg-Heading" in c):
        if any(k in p.get_text(strip=True).lower() for k in keywords):
            return p
    return None


def extract_section_text(start_element):
    section_text = []
    if not start_element:
        return section_text
    for tag in start_element.find_all_next():
        if tag.name == "span":
            heading = tag.get_text(strip=True)
            if heading and heading.istitle() and 2 <= len(heading.split()) <= 8:
                break
        if tag.name in ["div", "p"] and tag.get("class"):
            if "Judg-Heading" in " ".join(tag.get("class", [])):
                break
        if tag.name in ["p", "div", "span"]:
            text = tag.get_text(strip=True)
            if text and len(text) > 20:
                section_text.append(text)
    return section_text


def extract_fallback_paragraphs(soup, max_paragraphs=15):
    fallback_text = []
    judg_paragraphs = soup.find_all(
        ["p", "div"],
        class_=lambda c: c and any(x in str(c) for x in ["Judg-1", "Judg-2", "Judg-3"])
    )
    if judg_paragraphs:
        for p in judg_paragraphs[:max_paragraphs]:
            text = p.get_text(strip=True)
            if text and len(text) > 50 and not text.isupper():
                fallback_text.append(text)
    if not fallback_text:
        for p in soup.find_all("p")[:max_paragraphs * 2]:
            text = p.get_text(strip=True)
            if text and len(text) > 100:
                fallback_text.append(text)
            if len(fallback_text) >= max_paragraphs:
                break
    return fallback_text


def extract_facts_section(url, retry_mode=False):
    """Scrape Facts/Background section. Returns (text, method)."""
    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        del response
    except Exception as e:
        return "", "ERROR_FETCH"

    facts_text = []
    method = ""

    if retry_mode:
        facts_text = extract_fallback_paragraphs(soup, max_paragraphs=15)
        method = "PARAGRAPH_FALLBACK_RETRY"
        if not facts_text:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 100:
                    facts_text.append(text)
                if len(facts_text) >= 15:
                    break
            method = "RAW_PARAGRAPH_RETRY" if facts_text else "NO_TEXT_FOUND"
    else:
        start = find_section_heading(soup, ["facts", "background", "introduction", "dispute", "background facts"])
        if start:
            method = "PRIMARY_HEADING"
            facts_text = extract_section_text(start)
        else:
            start = find_section_heading(soup, [
                "brief facts", "factual background", "the facts", "material facts",
                "undisputed facts", "agreed facts", "procedural history", "overview"
            ])
            if start:
                method = "ALTERNATIVE_HEADING"
                facts_text = extract_section_text(start)
            else:
                facts_text = extract_fallback_paragraphs(soup, max_paragraphs=15)
                method = "PARAGRAPH_FALLBACK" if facts_text else "NO_TEXT_FOUND"

    return " ".join(facts_text), method

# ─────────────────────────── DEEPSEEK ────────────────────────────────────────

def remove_markdown(text):
    return re.sub(r'(\*\*|\*|__|~~|`)', '', text).strip()


def call_deepseek(facts_section, retries=3, backoff=5):
    words = facts_section.split()
    if len(words) > MAX_INPUT_TOKENS:
        facts_section = " ".join(words[:MAX_INPUT_TOKENS])

    prompt = PROMPT_TEMPLATE.format(facts_section=facts_section).strip()

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=HEADERS,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": MAX_OUTPUT_TOKENS
                },
                timeout=60,
                verify=False
            )
            if resp.status_code == 400 and "Content Exists Risk" in resp.text:
                return "CONTENT_BLOCKED - DeepSeek content filter triggered"
            resp.raise_for_status()
            return remove_markdown(resp.json()["choices"][0]["message"]["content"].strip())
        except Exception as e:
            wait = backoff * attempt
            tqdm.write(f"    ❌ API Error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return "ERROR - All retries failed"


def extract_facts_summary(raw_text):
    if not raw_text or not raw_text.strip():
        return "Facts summary not available - no Facts/Background section found in judgment."
    if len(raw_text.strip()) < 100:
        return "Facts summary not available - extracted text too short for meaningful summary."
    return call_deepseek(raw_text)

# ─────────────────────────── PROGRESS ────────────────────────────────────────

def save_raw_progress(results):
    rows = [{"URL": url, "Summary": v[0], "Raw_Text": v[1], "Method": v[2]}
            for url, v in results.items()]
    df = pd.DataFrame(rows)
    temp = RAW_TEXT_FILE + ".tmp"
    df.to_csv(temp, index=False)
    if os.path.exists(RAW_TEXT_FILE):
        shutil.copy2(RAW_TEXT_FILE, BACKUP_FILE)
    os.replace(temp, RAW_TEXT_FILE)
    log_rows = [{"URL": url, "Method": v[2], "Text_Length": len(v[1])}
                for url, v in results.items()]
    pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)
    tqdm.write(f"  💾 Progress saved ({len(results)} URLs)")

# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SG-LEGALCITE — STEP 3: EXTRACT OF FACTS (BATCH)")
    print("=" * 65)

    # Load input
    print(f"\nReading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    unique_urls = df["Judgment_URL"].unique() if "Judgment_URL" in df.columns else df["URL"].unique()
    url_col     = "Judgment_URL" if "Judgment_URL" in df.columns else "URL"
    total = len(unique_urls)
    print(f"Total rows       : {len(df)}")
    print(f"Unique judgments : {total}")

    # Resume from existing raw text file
    results = {}
    progress_src = None
    if os.path.exists(RAW_TEXT_FILE):
        progress_src = RAW_TEXT_FILE
    elif os.path.exists(BACKUP_FILE):
        tqdm.write("⚠ Main progress file not found, loading from backup...")
        progress_src = BACKUP_FILE

    if progress_src:
        try:
            existing = pd.read_csv(progress_src)
            for _, row in existing.iterrows():
                results[row["URL"]] = (row.get("Summary", ""), row.get("Raw_Text", ""), row.get("Method", ""))
            print(f"Resumed: {len(results)}/{total} URLs already processed.")
        except Exception as e:
            print(f"⚠ Could not load progress: {e}. Starting fresh.")
    else:
        print("No existing progress. Starting from scratch.")

    # ── PASS 1: Main extraction ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PASS 1: Main extraction")
    print(f"{'='*65}")

    processed = 0
    try:
        for i, url in enumerate(tqdm(unique_urls, desc="Scraping")):
            if url in results:
                continue

            raw_text, method = extract_facts_section(url)
            summary = extract_facts_summary(raw_text)
            results[url] = (summary, raw_text, method)
            processed += 1

            time.sleep(SLEEP_BETWEEN)

            if processed % SAVE_EVERY == 0:
                save_raw_progress(results)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted — saving progress...")

    save_raw_progress(results)
    print(f"\nPass 1 done. Processed {processed} new URLs.")

    # ── PASS 2: Retry failures ───────────────────────────────────────────────
    problem_urls = [url for url, v in results.items()
                    if any(x in str(v[0]) for x in ["ERROR", "not available", "Insufficient", "CONTENT_BLOCKED"])
                    or not v[0]]

    if problem_urls:
        print(f"\n{'='*65}")
        print(f"PASS 2: Retrying {len(problem_urls)} failed URLs")
        print(f"{'='*65}")

        fixed = 0
        still_broken = 0

        try:
            for url in tqdm(problem_urls, desc="Retrying"):
                raw_text, method = extract_facts_section(url, retry_mode=True)
                summary = extract_facts_summary(raw_text)
                results[url] = (summary, raw_text, method)

                if any(x in str(summary) for x in ["ERROR", "not available", "Insufficient", "CONTENT_BLOCKED"]):
                    still_broken += 1
                else:
                    fixed += 1

                gc.collect()
                time.sleep(SLEEP_BETWEEN)

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted — saving progress...")

        save_raw_progress(results)
        print(f"\nPass 2 done. Fixed: {fixed} | Still broken: {still_broken}")

    # ── FINAL: Map summaries back to all rows and save ───────────────────────
    print(f"\n{'='*65}")
    print("Mapping summaries to all rows...")
    url_to_summary  = {url: v[0] for url, v in results.items()}
    url_to_raw_text = {url: v[1] for url, v in results.items()}
    df["Fact_Query"] = df[url_col].map(lambda u: url_to_summary.get(u, ""))
    df.to_csv(OUTPUT_FILE, index=False)

    good = df["Extract of Facts"].str.strip().ne("").sum()
    print(f"\n{'='*65}")
    print("COMPLETE")
    print(f"{'='*65}")
    print(f"Total rows      : {len(df)}")
    print(f"With facts      : {good} ({good/len(df)*100:.1f}%)")
    print(f"Output          : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
