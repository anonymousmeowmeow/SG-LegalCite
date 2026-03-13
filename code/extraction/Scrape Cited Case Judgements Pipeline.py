"""
SG-LegalCite: Cited Case Judgment Scraping Pipeline
=====================================================
Scrapes full judgment text for cited cases found in Singapore court judgments.

Pipeline stages:
  1. Initial scrape    — search + scrape all unique cited cases via Serper API
  2. Rescrape          — retry cases with wrong/junk content
  3. Quality check     — detect and log junk entries
  4. Final merge       — merge rescrape results into clean dataset
  5. Verify and merge  — final deduplication and output

Usage:
  python scrape_pipeline.py [--stage STAGE]

  --stage 1   Run initial scrape only
  --stage 2   Run rescrape only
  --stage 3   Run quality check only
  --stage 4   Run final merge only
  --stage 5   Run verify and merge only
  (no flag)   Run all stages in sequence

Requirements:
  pip install requests beautifulsoup4 tqdm PyPDF2
"""

import os, csv, sys, io, gc, re, json, time, random, argparse, shutil
import urllib3
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import Counter
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
csv.field_size_limit(sys.maxsize)

# ============================================================================
# CONFIG — update paths and API key here
# ============================================================================
LOCAL_DIR        = '/Users/shannon/LocalScraping'
SERPER_API_KEY   = 'cfce968bd391bac240720eaf65b6dd47aeb5221a'

# Input
INPUT_FILE       = os.path.join(LOCAL_DIR, 'COMBINED_ALL_CASES_FINAL.csv')
OLD_BACKUP       = os.path.join(LOCAL_DIR, '20260206_152451_auto_23615_CITED_CASE_JUDGMENTS_CHECKPOINT.csv')

# Stage outputs
SCRAPE_OUTPUT    = os.path.join(LOCAL_DIR, 'SCRAPE_OUTPUT.csv')
SCRAPE_BACKUP    = os.path.join(LOCAL_DIR, 'SCRAPE_OUTPUT_BACKUP.csv')
WRONG_CSV        = os.path.join(LOCAL_DIR, 'WRONG_CASE_ENTRIES_v2.csv')
RESCRAPE_OUTPUT  = os.path.join(LOCAL_DIR, 'RESCRAPE4_WRONG.csv')
RESCRAPE_BACKUP  = os.path.join(LOCAL_DIR, 'RESCRAPE4_WRONG_BACKUP.csv')
JUNK_CSV         = os.path.join(LOCAL_DIR, 'JUNK_ENTRIES_v2.csv')
FINAL_CLEAN      = os.path.join(LOCAL_DIR, 'FINAL_CLEAN_JUDGMENTS.csv')
FINAL_OLD        = os.path.join(LOCAL_DIR, 'FINAL_DATASET.csv')
FINAL_V2         = os.path.join(LOCAL_DIR, 'FINAL_DATASET_v2.csv')
FINAL_OUTPUT     = os.path.join(LOCAL_DIR, 'FINAL_CITED_CASE_JUDGMENTS.csv')

MAX_JUDGMENT_CHARS  = 500_000
MAX_DOWNLOAD_BYTES  = 2 * 1024 * 1024

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7',
}

GOOD_SITES = [
    'bailii.org', 'commonlii.org', 'austlii.edu.au', 'elitigation.sg',
    'canlii.org', 'worldlii.org', 'lawnet.sg', 'e-lawresources.co.uk',
    'caselaw.nationalarchives.gov.uk', 'iclr.co.uk', 'swarb.co.uk',
    'singaporelawwatch.sg', 'judiciary.uk', 'supremecourt.uk', 'lawteacher.net',
]

BAD_SITES = [
    'scribd.com', 'vlex', 'academia.edu', 'coursehero', 'studocu',
    'quizlet', 'chegg', 'westlaw', 'lexisnexis', 'heinonline',
    'jstor', 'ssrn', 'youtube', 'facebook', 'twitter', 'linkedin',
    'justis.com', 'google.com', 'casemine.com', 'archive.org',
    'dokumen.pub', 'trove.nla.gov.au', 'fraser.stlouisfed.org',
    'opencasebook.org', 'onlinelibrary.wiley.com', 'jusmundi.com',
    'www.sec.gov', 'www.ecfr.gov', 'federalregister.gov',
    'pdfcoffee.com', 'www.i-law.com', 'i-law.com', 'researchgate.net',
    'www.cambridge.org', 'academic.oup.com', 'www.oxbridgenotes.co.uk',
    'www.lawjournals.co.uk', 'search.informit.org', 'www.yumpu.com',
    'financialremediesjournal.com', 'supremetoday.ai',
    'search.proquest.com', 'journals.sagepub.com', 'www.uscourts.gov',
]

STRICT_JUNK = [
    'jus mundi', 'jus ai', 'arbitration intelligence',
    'report dmca', 'download file',
    'securities and exchange commission', 'form 10-k', 'form n-px',
    'annual report of proxy',
    'accept cookies', 'cookie settings',
    'oral abstracts of the', 'poster abstracts of the',
    'learning objectives', 'exam question',
]

SHORT_JUNK = [
    'sign up to', 'create an account', 'subscribe to',
    'cookie policy', 'we use cookies',
    'this content is only available',
    'access denied', 'please log in to view',
    'you need to sign in',
]

JUNK_DOMAINS = [
    'jusmundi.com', 'dokumen.pub', 'www.sec.gov',
    'www.ecfr.gov', 'federalregister.gov',
    'opencasebook.org', 'fraser.stlouisfed.org',
]

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def serper_search(case_name, num_results=10):
    url = 'https://google.serper.dev/search'
    payload = json.dumps({'q': f'{case_name} judgment full text', 'num': num_results})
    headers_api = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        print(f'    [Serper] {case_name[:60]}...', end=' ', flush=True)
        r = requests.post(url, headers=headers_api, data=payload, verify=False, timeout=30)
        print(f'→ {r.status_code}', flush=True)
        if r.status_code != 200:
            return []
        results = r.json().get('organic', [])
        urls = [item['link'] for item in results
                if not any(bad in item['link'].lower() for bad in BAD_SITES)]
        priority = [u for u in urls if any(g in u.lower() for g in GOOD_SITES)]
        others   = [u for u in urls if u not in priority]
        final    = priority + others
        print(f'    {len(final)} URLs ({len(priority)} priority)', flush=True)
        return final
    except Exception as e:
        print(f'    Serper error: {type(e).__name__}', flush=True)
        return []


def extract_pdf_text(raw_bytes):
    if PyPDF2 is None:
        return None
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        text = '\n\n'.join(pages)
        return text if len(text) >= 500 else None
    except Exception as e:
        print(f'    PDF error: {str(e)[:50]}')
        return None


def scrape_judgment(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, verify=False, stream=True)
        r.raise_for_status()
        chunks = []
        total_bytes = 0
        content_type = r.headers.get('Content-Type', '').lower()

        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                total_bytes += len(chunk)
                if total_bytes >= MAX_DOWNLOAD_BYTES:
                    break

        raw = b''.join(chunks)
        del chunks

        # PDF
        if 'pdf' in content_type or raw[:4] == b'%PDF':
            text = extract_pdf_text(raw)
            del raw
            return (text[:MAX_JUDGMENT_CHARS] if text else None), url

        # HTML
        html = raw.decode('utf-8', errors='replace')
        del raw
        soup = BeautifulSoup(html, 'html.parser')
        del html
        for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                         'aside', 'form', 'button', 'select']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        del soup

        if len(text) < 200:
            return None, url

        # Basic junk check
        first_lower = text[:2000].lower()
        if any(p in first_lower for p in ['log in', 'login', 'sign in',
                                           'javascript is required', 'captcha',
                                           'access denied', 'verify you are human']):
            return None, url

        return text[:MAX_JUDGMENT_CHARS], url

    except Exception as e:
        print(f'    Scrape error ({url[:50]}): {type(e).__name__}')
        return None, url


def get_judgment(case_name):
    urls = serper_search(case_name)
    for url in urls:
        print(f'    [Scrape] {url[:70]}...', end=' ', flush=True)
        text, final_url = scrape_judgment(url)
        if text and len(text) >= 200:
            print(f'✓ {len(text):,} chars', flush=True)
            return text, final_url
        print('✗', flush=True)
    return None, None


def read_case_names(filepath):
    cases = set()
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 50:
        return cases
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        text = raw.replace(b'\x00', b'').decode('utf-8', errors='replace')
        del raw
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        case_idx = next((header.index(n) for n in ['Cited Case', 'Case'] if n in header), 0)
        for row in reader:
            if len(row) > case_idx and row[case_idx]:
                cases.add(row[case_idx].strip())
        del text
    except Exception as e:
        print(f'  Warning reading {os.path.basename(filepath)}: {e}')
    return cases


def read_results_file(filepath):
    results = {}
    with open(filepath, 'rb') as f:
        raw = f.read()
    text = raw.replace(b'\x00', b'').decode('utf-8', errors='replace')
    del raw
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    case_idx     = next((header.index(n) for n in ['Cited Case', 'Case'] if n in header), 0)
    judgment_idx = next((header.index(n) for n in ['Judgment of Cited Case', 'Judgment'] if n in header), 1)
    url_idx      = next((header.index(n) for n in ['Source URL', 'URL'] if n in header), 2)
    for row in reader:
        try:
            case_name = row[case_idx].strip() if len(row) > case_idx else ''
            judgment  = row[judgment_idx] if len(row) > judgment_idx else ''
            url       = row[url_idx] if len(row) > url_idx else ''
            if not case_name:
                continue
            if case_name not in results:
                results[case_name] = (judgment, url)
            else:
                old_j = results[case_name][0]
                if (not old_j or old_j == 'NOT FOUND' or old_j.startswith('ERROR')) \
                        and judgment and judgment != 'NOT FOUND':
                    results[case_name] = (judgment, url)
        except:
            continue
    del text
    gc.collect()
    return results


# ============================================================================
# STAGE 1: INITIAL SCRAPE
# ============================================================================

def stage1_scrape():
    print('\n' + '=' * 60)
    print('STAGE 1: Initial Scrape')
    print('=' * 60)

    # Load all unique cited cases
    all_cases = set()
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        case_idx = header.index('Cited Case')
        for row in reader:
            if len(row) > case_idx and row[case_idx]:
                all_cases.add(row[case_idx])
    clean_cases = [c for c in all_cases if len(c) < 150]
    print(f'Total unique cited cases: {len(all_cases):,} | Clean: {len(clean_cases):,}')

    # Find already-processed cases
    processed = set()
    all_csvs = sorted([f for f in os.listdir(LOCAL_DIR)
                       if f.endswith('.csv') and f != os.path.basename(INPUT_FILE)])
    print(f'\nScanning {len(all_csvs)} existing result files...')
    for fname in all_csvs:
        fp = os.path.join(LOCAL_DIR, fname)
        cases = read_case_names(fp)
        processed.update(cases)
        gc.collect()
    print(f'Already done: {len(processed):,}')

    to_process = [c for c in clean_cases if c not in processed]
    print(f'Remaining: {len(to_process):,}\n')

    if not to_process:
        print('Nothing to scrape. Skipping stage 1.')
        return

    # Init output file
    if not os.path.exists(SCRAPE_OUTPUT) or os.path.getsize(SCRAPE_OUTPUT) == 0:
        with open(SCRAPE_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Cited Case', 'Judgment of Cited Case', 'Source URL'])

    batch = found = 0
    try:
        for i, case_name in enumerate(tqdm(to_process, desc='Scraping')):
            try:
                print(f'\n--- [{i+1}] {case_name[:70]} ---', flush=True)
                judgment, url = get_judgment(case_name)
                with open(SCRAPE_OUTPUT, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        case_name,
                        judgment if judgment else 'NOT FOUND',
                        url if url else 'NOT FOUND'
                    ])
                if judgment:
                    found += 1
                batch += 1
                del judgment, url
                if batch % 10 == 0:
                    gc.collect()
                    print(f'\n[Progress] {batch:,} done | Found: {found:,} ({found/batch*100:.1f}%)', flush=True)
                if batch % 1000 == 0:
                    shutil.copy2(SCRAPE_OUTPUT, SCRAPE_BACKUP)
                    print(f'💾 Backup saved ({batch:,})', flush=True)
                time.sleep(random.uniform(0.3, 0.6))
            except Exception as e:
                print(f'    ERROR: {e}', flush=True)
                with open(SCRAPE_OUTPUT, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([case_name, f'ERROR: {e}', 'ERROR'])
                batch += 1
    except KeyboardInterrupt:
        print('\n⚠️ Stopped by user.')
    finally:
        if os.path.exists(SCRAPE_OUTPUT):
            shutil.copy2(SCRAPE_OUTPUT, SCRAPE_BACKUP)
        print(f'\n✓ Stage 1 complete: {batch:,} processed | {found:,} found ({found/max(batch,1)*100:.1f}%)')


# ============================================================================
# STAGE 2: RESCRAPE WRONG/JUNK CASES
# ============================================================================

def stage2_rescrape():
    print('\n' + '=' * 60)
    print('STAGE 2: Rescrape Wrong/Junk Cases')
    print('=' * 60)

    if not os.path.exists(WRONG_CSV):
        print(f'⚠️ {WRONG_CSV} not found. Run quality check first.')
        return

    # Load wrong cases (skip elitigation.sg — trusted source)
    wrong_cases = []
    skipped_elit = 0
    with open(WRONG_CSV, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0]:
                continue
            url = row[1] if len(row) > 1 else ''
            if 'elitigation.sg' in url:
                skipped_elit += 1
                continue
            wrong_cases.append(row[0].strip())

    print(f'Total wrong: {len(wrong_cases) + skipped_elit:,} | Skipped (elitigation trusted): {skipped_elit:,} | To re-scrape: {len(wrong_cases):,}')

    # Check already done
    already_done = read_case_names(RESCRAPE_OUTPUT)
    final_list = [c for c in wrong_cases if c not in already_done]
    print(f'Already done: {len(already_done):,} | Remaining: {len(final_list):,}\n')

    if not final_list:
        print('Nothing to rescrape. Skipping stage 2.')
        return

    if not os.path.exists(RESCRAPE_OUTPUT) or os.path.getsize(RESCRAPE_OUTPUT) == 0:
        with open(RESCRAPE_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Cited Case', 'Judgment of Cited Case', 'Source URL'])

    batch = found = 0
    try:
        for i, case_name in enumerate(tqdm(final_list, desc='Re-scraping')):
            try:
                print(f'\n--- [{i+1}] {case_name[:70]} ---', flush=True)
                judgment, url = get_judgment(case_name)
                with open(RESCRAPE_OUTPUT, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        case_name,
                        judgment if judgment else 'NOT FOUND',
                        url if url else 'NOT FOUND'
                    ])
                if judgment:
                    found += 1
                batch += 1
                del judgment, url
                if batch % 10 == 0:
                    gc.collect()
                    print(f'\n[Progress] {batch:,} done | Found: {found:,} ({found/batch*100:.1f}%)', flush=True)
                if batch % 500 == 0:
                    shutil.copy2(RESCRAPE_OUTPUT, RESCRAPE_BACKUP)
                    print(f'💾 Backup saved ({batch:,})', flush=True)
                time.sleep(random.uniform(0.3, 0.6))
            except Exception as e:
                print(f'    ERROR: {e}', flush=True)
                with open(RESCRAPE_OUTPUT, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([case_name, f'ERROR: {e}', 'ERROR'])
                batch += 1
    except KeyboardInterrupt:
        print('\n⚠️ Stopped by user.')
    finally:
        if os.path.exists(RESCRAPE_OUTPUT):
            shutil.copy2(RESCRAPE_OUTPUT, RESCRAPE_BACKUP)
        print(f'\n✓ Stage 2 complete: {batch:,} processed | {found:,} found ({found/max(batch,1)*100:.1f}%)')


# ============================================================================
# STAGE 3: QUALITY CHECK
# ============================================================================

def stage3_quality_check():
    print('\n' + '=' * 60)
    print('STAGE 3: Quality Check')
    print('=' * 60)

    input_file = FINAL_CLEAN if os.path.exists(FINAL_CLEAN) else SCRAPE_OUTPUT
    print(f'Scanning {os.path.basename(input_file)}...\n')

    junk_entries = []
    good_count = not_found_count = total = 0

    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            total += 1
            if total % 10000 == 0:
                print(f'  ...{total:,} rows', flush=True)
            case_name = row[0] if row else ''
            judgment  = row[1] if len(row) > 1 else ''
            url       = row[2] if len(row) > 2 else ''

            if judgment in ('NOT FOUND', 'NOT SCRAPED') or judgment.startswith('ERROR'):
                not_found_count += 1
                continue

            first_2000 = judgment[:2000].lower()
            j_len = len(judgment)
            is_junk = False
            reason = ''

            for domain in JUNK_DOMAINS:
                if domain in url.lower():
                    is_junk = True
                    reason = f'Junk domain: {domain}'
                    break

            if not is_junk:
                for p in STRICT_JUNK:
                    if p in first_2000:
                        is_junk = True
                        reason = f'Strict pattern: "{p}"'
                        break

            if not is_junk and j_len < 5000:
                for p in SHORT_JUNK:
                    if p in first_2000:
                        is_junk = True
                        reason = f'Short ({j_len} chars) + "{p}"'
                        break

            if not is_junk:
                if '%PDF-' in judgment[:20] or '\ufffd\ufffd\ufffd' in judgment[:200]:
                    is_junk = True
                    reason = 'Binary/PDF content'

            if is_junk:
                junk_entries.append((case_name, url, reason, j_len))
            else:
                good_count += 1

    reason_counts = Counter()
    for _, _, reason, _ in junk_entries:
        key = reason if not reason.startswith('Short') else 'Short entry + suspicious'
        reason_counts[key] += 1

    domain_counts = Counter()
    for _, url, _, _ in junk_entries:
        try:
            domain_counts[url.split('/')[2]] += 1
        except:
            pass

    print(f'\n{"=" * 60}')
    print(f'QUALITY CHECK RESULTS')
    print(f'{"=" * 60}')
    print(f'Total:         {total:,}')
    print(f'Good:          {good_count:,} ({good_count/total*100:.1f}%)')
    print(f'NOT FOUND:     {not_found_count:,} ({not_found_count/total*100:.1f}%)')
    print(f'Junk:          {len(junk_entries):,} ({len(junk_entries)/total*100:.1f}%)')
    print(f'\n--- By reason ---')
    for reason, count in reason_counts.most_common(20):
        print(f'  {reason}: {count}')
    print(f'\n--- By domain ---')
    for domain, count in domain_counts.most_common(15):
        print(f'  {domain}: {count}')

    with open(JUNK_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Cited Case', 'Source URL', 'Reason', 'Length'])
        for entry in junk_entries:
            writer.writerow(entry)

    size_kb = os.path.getsize(JUNK_CSV) / 1024
    print(f'\n✓ Saved: {os.path.basename(JUNK_CSV)} ({size_kb:.0f} KB, {len(junk_entries):,} entries)')


# ============================================================================
# STAGE 4: FINAL MERGE
# ============================================================================

def stage4_final_merge():
    print('\n' + '=' * 60)
    print('STAGE 4: Final Merge')
    print('=' * 60)

    if not os.path.exists(RESCRAPE_OUTPUT):
        print(f'⚠️ {RESCRAPE_OUTPUT} not found. Run stage 2 first.')
        return
    if not os.path.exists(FINAL_OLD):
        print(f'⚠️ {FINAL_OLD} not found.')
        return

    # Load rescrape results
    print('Loading rescrape results...')
    rescrape4 = {}
    with open(RESCRAPE_OUTPUT, 'rb') as f:
        raw = f.read()
    text = raw.replace(b'\x00', b'').decode('utf-8', errors='replace')
    del raw
    for row in csv.reader(io.StringIO(text)):
        try:
            case_name = row[0].strip()
            judgment  = row[1] if len(row) > 1 else ''
            url       = row[2] if len(row) > 2 else ''
            if not case_name:
                continue
            is_good = (judgment and judgment != 'NOT FOUND' and
                       not judgment.startswith('ERROR') and
                       '%PDF-' not in judgment[:20] and len(judgment) >= 100)
            if is_good:
                if case_name not in rescrape4 or len(judgment) > len(rescrape4[case_name][0]):
                    rescrape4[case_name] = (judgment, url)
            elif case_name not in rescrape4:
                rescrape4[case_name] = (judgment, url)
        except:
            continue
    del text
    gc.collect()

    good = sum(1 for j, _ in rescrape4.values()
               if j and j != 'NOT FOUND' and not j.startswith('ERROR') and len(j) >= 100)
    print(f'Rescrape: {len(rescrape4):,} loaded | {good:,} good | {len(rescrape4) - good:,} NOT FOUND')

    # Load wrong case list
    wrong_set = set()
    if os.path.exists(WRONG_CSV):
        with open(WRONG_CSV, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                url = row[1] if len(row) > 1 else ''
                if 'elitigation.sg' not in url:
                    wrong_set.add(row[0].strip())
    print(f'Wrong cases to replace: {len(wrong_set):,}')

    # Merge
    print('Merging...')
    total = replaced = kept_good = still_bad = 0

    with open(FINAL_OLD, 'r', encoding='utf-8', errors='replace') as fin, \
         open(FINAL_V2,  'w', newline='', encoding='utf-8') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        next(reader)
        writer.writerow(['Cited Case', 'Judgment of Cited Case', 'Source URL'])
        for row in reader:
            total += 1
            if total % 10000 == 0:
                print(f'  ...{total:,} rows', flush=True)
            try:
                case_name = row[0].strip() if row else ''
                judgment  = row[1] if len(row) > 1 else ''
                url       = row[2] if len(row) > 2 else ''
                if case_name in wrong_set and case_name in rescrape4:
                    new_j, new_url = rescrape4[case_name]
                    if new_j and new_j != 'NOT FOUND' and not new_j.startswith('ERROR') and len(new_j) >= 100:
                        writer.writerow([case_name, new_j, new_url])
                        replaced += 1
                    else:
                        writer.writerow([case_name, judgment, url])
                        kept_good += 1
                elif judgment in ('NOT FOUND', 'NOT SCRAPED'):
                    writer.writerow([case_name, 'NOT FOUND', 'NOT FOUND'])
                    still_bad += 1
                else:
                    writer.writerow([case_name, judgment, url])
                    kept_good += 1
            except:
                continue

    size_mb = os.path.getsize(FINAL_V2) / 1024 / 1024
    print(f'\n{"=" * 60}')
    print(f'FINAL MERGE REPORT')
    print(f'{"=" * 60}')
    print(f'Total:           {total:,}')
    print(f'Kept original:   {kept_good:,}')
    print(f'Replaced wrong:  {replaced:,}')
    print(f'Still NOT FOUND: {still_bad:,}')
    print(f'Found rate:      {(kept_good + replaced)/max(total,1)*100:.1f}%')
    print(f'Output:          {os.path.basename(FINAL_V2)} ({size_mb:.0f} MB)')


# ============================================================================
# STAGE 5: VERIFY AND MERGE
# ============================================================================

def stage5_verify_merge():
    print('\n' + '=' * 60)
    print('STAGE 5: Verify and Final Merge')
    print('=' * 60)

    # Load target cases
    target_cases = set()
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        case_idx = header.index('Cited Case')
        for row in reader:
            if len(row) > case_idx and row[case_idx]:
                target_cases.add(row[case_idx])
    clean_targets = {c for c in target_cases if len(c) < 150}
    print(f'Target cases: {len(target_cases):,} | Clean: {len(clean_targets):,}')

    # Merge all result sources
    all_results = {}
    sources = [f for f in [OLD_BACKUP, SCRAPE_OUTPUT, RESCRAPE_OUTPUT] if os.path.exists(f)]
    for src in sources:
        size_mb = os.path.getsize(src) / 1024 / 1024
        print(f'Loading {os.path.basename(src)} ({size_mb:.0f} MB)...', end=' ', flush=True)
        results = read_results_file(src)
        print(f'{len(results):,} cases')
        for case_name, (judgment, url) in results.items():
            if case_name not in all_results:
                all_results[case_name] = (judgment, url)
            else:
                old_j = all_results[case_name][0]
                if (not old_j or old_j == 'NOT FOUND' or old_j.startswith('ERROR')) \
                        and judgment and judgment != 'NOT FOUND' and not judgment.startswith('ERROR'):
                    all_results[case_name] = (judgment, url)
        del results
        gc.collect()
    print(f'\nTotal unique scraped: {len(all_results):,}')

    # Verification stats
    matched = found_judgment = not_found_j = errors = 0
    not_scraped = []
    judgment_lengths = []

    for case_name in clean_targets:
        if case_name in all_results:
            matched += 1
            j, url = all_results[case_name]
            if j and j != 'NOT FOUND' and not j.startswith('ERROR'):
                found_judgment += 1
                judgment_lengths.append(len(j))
            elif j and j.startswith('ERROR'):
                errors += 1
            else:
                not_found_j += 1
        else:
            not_scraped.append(case_name)

    print(f'\n{"=" * 60}')
    print(f'VERIFICATION REPORT')
    print(f'{"=" * 60}')
    print(f'Target (clean):     {len(clean_targets):,}')
    print(f'Matched:            {matched:,} ({matched/len(clean_targets)*100:.1f}%)')
    print(f'Not scraped:        {len(not_scraped):,}')
    print(f'Judgment found:     {found_judgment:,} ({found_judgment/max(matched,1)*100:.1f}% of matched)')
    print(f'NOT FOUND:          {not_found_j:,}')
    print(f'Errors:             {errors:,}')

    if judgment_lengths:
        judgment_lengths.sort()
        n = len(judgment_lengths)
        print(f'\n--- Judgment Length Stats ---')
        print(f'Average: {sum(judgment_lengths)/n:,.0f} chars')
        print(f'Median:  {judgment_lengths[n//2]:,} chars')
        print(f'Min:     {judgment_lengths[0]:,} chars')
        print(f'Max:     {judgment_lengths[-1]:,} chars')

    # Write final output
    print(f'\nWriting {os.path.basename(FINAL_OUTPUT)}...')
    written = 0
    with open(FINAL_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Cited Case', 'Judgment of Cited Case', 'Source URL'])
        for case_name in sorted(clean_targets):
            if case_name in all_results:
                j, url = all_results[case_name]
                writer.writerow([case_name, j, url])
            else:
                writer.writerow([case_name, 'NOT SCRAPED', 'NOT SCRAPED'])
            written += 1

    size_mb = os.path.getsize(FINAL_OUTPUT) / 1024 / 1024
    print(f'✓ {os.path.basename(FINAL_OUTPUT)}: {written:,} cases | {size_mb:.0f} MB')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SG-LegalCite Judgment Scraping Pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5],
                        help='Run a single stage (1-5). Omit to run all stages.')
    args = parser.parse_args()

    print('SG-LegalCite Judgment Scraping Pipeline')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Working directory: {LOCAL_DIR}\n')

    stages = {
        1: stage1_scrape,
        2: stage2_rescrape,
        3: stage3_quality_check,
        4: stage4_final_merge,
        5: stage5_verify_merge,
    }

    if args.stage:
        stages[args.stage]()
    else:
        for i in range(1, 6):
            stages[i]()

    print(f'\n✅ Done: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
