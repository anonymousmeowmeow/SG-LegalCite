"""
02_Deepseek_Chat_Batch.py
=========================
Batch extraction of Key Principles, Issue Group, and Issue from cited case
paragraphs using DeepSeek Chat API.

Reads  : 01_Extracted_Cases_Output.csv
Outputs: 02_Deepseek_Output.csv

Logic:
- Iterates every row in the input CSV
- Sends the Paragraph text to DeepSeek Chat (T=0) to extract 3 fields
- Appends results to output CSV incrementally
- Skips rows already processed (resume-safe)
- Saves progress every SAVE_EVERY rows

Usage:
    pip install requests pandas tqdm
    python 02_Deepseek_Chat_Batch.py
"""

import pandas as pd
import requests
import json
import os
import time
import urllib3
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

API_KEY        = "YOUR_DEEPSEEK_API_KEY"
INPUT_FILE     = "01_Extracted_Cases_Output.csv"
OUTPUT_FILE    = "02_Deepseek_Output.csv"
PROGRESS_FILE  = "02_deepseek_progress.json"

PROMPT_FILE    = "prompt_with_paragraphs_FINAL.txt"
MODEL          = "deepseek-chat"
TEMPERATURE    = 0
MAX_TOKENS     = 4000
RETRY_COUNT    = 3
SLEEP_BETWEEN  = 1    # seconds between API calls
SAVE_EVERY     = 50   # save progress every N rows

# ─────────────────────────── PROMPT ──────────────────────────────────────────

def load_prompt_template() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(cited_case: str, paragraph_text: str, template: str) -> str:
    # Append the new input paragraph to the prompt template
    return template.strip() + f"\n{cited_case}\t{paragraph_text}"

# ─────────────────────────── API ─────────────────────────────────────────────

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def call_deepseek(cited_case: str, paragraph_text: str, template: str) -> str:
    prompt = build_prompt(cited_case, paragraph_text, template)
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}]
    }

    for attempt in range(RETRY_COUNT):
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=180,
                verify=False
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                tqdm.write(f"  ⚠ Rate limited — waiting 60s (attempt {attempt+1}/{RETRY_COUNT})")
                time.sleep(60)
            else:
                tqdm.write(f"  ✗ API error {resp.status_code}: {resp.text[:100]}")
                time.sleep(5)
        except requests.exceptions.Timeout:
            tqdm.write(f"  ✗ Timeout (attempt {attempt+1}/{RETRY_COUNT})")
            if attempt < RETRY_COUNT - 1:
                time.sleep(30 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            tqdm.write(f"  ✗ Request failed: {e}")
            if attempt < RETRY_COUNT - 1:
                time.sleep(5)

    return None

def parse_tsv_response(response_text: str) -> dict:
    """Parse TSV response: Cited Case\tParagraph\tKey Principles Illustrated\tIssue\tIssue Group"""
    empty = {"Key Principles Illustrated": "", "Issue": "", "Issue Group": ""}
    if not response_text:
        return empty
    try:
        lines = [l for l in response_text.strip().splitlines() if l.strip()]
        # Skip header row if present
        data_lines = [l for l in lines if not l.lower().startswith("cited case")]
        if not data_lines:
            return empty
        # Take first data row
        parts = data_lines[0].split("\t")
        if len(parts) >= 5:
            return {
                "Key Principles Illustrated": parts[2].strip(),
                "Issue":                      parts[3].strip(),
                "Issue Group":                parts[4].strip()
            }
        elif len(parts) >= 3:
            # Fallback: fewer columns
            return {
                "Key Principles Illustrated": parts[2].strip() if len(parts) > 2 else "",
                "Issue":                      parts[3].strip() if len(parts) > 3 else "",
                "Issue Group":                parts[4].strip() if len(parts) > 4 else ""
            }
        return empty
    except Exception as e:
        tqdm.write(f"  ✗ TSV parse error: {e}")
        return empty

# ─────────────────────────── PROGRESS ────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_progress(last_index: int, processed: int, failed: int):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_index": last_index,
            "processed":  processed,
            "failed":     failed
        }, f, indent=2)

# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SG-LEGALCITE — STEP 2: DEEPSEEK EXTRACTION (BATCH)")
    print("=" * 65)

    # Load input
    print(f"\nReading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")

    # Add output columns if missing
    for col in ["Key Principles Illustrated", "Issue", "Issue Group"]:
        if col not in df.columns:
            df[col] = ""

    # Resume logic — load existing output if present
    start_idx  = 0
    processed  = 0
    failed     = 0
    first_write = True

    prev = load_progress()
    if prev and os.path.exists(OUTPUT_FILE):
        print(f"\nResuming from row {prev['last_index'] + 1} "
              f"({prev['processed']} processed, {prev['failed']} failed)")
        start_idx  = prev["last_index"] + 1
        processed  = prev["processed"]
        failed     = prev["failed"]
        first_write = False
    else:
        # Fresh start — clear output file
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)

    # Load prompt template
    print(f"Loading prompt: {PROMPT_FILE}")
    prompt_template = load_prompt_template()

    # Process rows
    rows_to_process = df.iloc[start_idx:]

    try:
        for i, (idx, row) in enumerate(tqdm(rows_to_process.iterrows(),
                                             total=len(rows_to_process),
                                             desc="DeepSeek")):
            paragraph = str(row.get("Paragraph", "")).strip()
            cited     = str(row.get("Cited Case", ""))
            ref       = str(row.get("Judgment_Reference", ""))

            if not paragraph:
                tqdm.write(f"  ⚠ Skipping {ref} / {cited}: empty paragraph")
                result = {"Key Principles Illustrated": "", "Issue": "", "Issue Group": ""}
                failed += 1
            else:
                response_text = call_deepseek(cited, paragraph, prompt_template)
                result        = parse_tsv_response(response_text)

                if result["Key Principles Illustrated"].strip() == "":
                    tqdm.write(f"  ✗ Failed: {ref} / {cited}")
                    failed += 1
                else:
                    processed += 1

            # Write row to output CSV
            out_row = row.to_dict()
            out_row.update(result)
            out_df = pd.DataFrame([out_row])
            out_df.to_csv(OUTPUT_FILE, mode="a", header=first_write, index=False)
            first_write = False

            time.sleep(SLEEP_BETWEEN)

            # Save progress periodically
            abs_idx = start_idx + i
            if (i + 1) % SAVE_EVERY == 0:
                save_progress(abs_idx, processed, failed)
                tqdm.write(f"  💾 Progress saved ({processed} processed, {failed} failed)")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted — progress saved.")

    save_progress(start_idx + len(rows_to_process) - 1, processed, failed)

    print("\n" + "=" * 65)
    print("COMPLETE")
    print("=" * 65)
    print(f"Processed : {processed}")
    print(f"Failed    : {failed}")
    print(f"Output    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
