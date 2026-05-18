# ============================================================
# SG-LegalCite Stage 2 Paragraph-Augmented Ablation V2: Generate Candidate Pools
# Task: [FACT] Fact [PARAGRAPH] Scrubbed Paragraph Window → Cited Case
# Pool: 1 correct case + 999 random negatives
# One pool per (fact, principle, cited_case) triple — IDENTICAL test set
#   to single_stage_pools_v2 to enable apples-to-apples comparison
# Uses GROUND TRUTH fields directly (no Stage 1 prediction)
# Run on CLUSTER
#
# This ablation answers Reviewer 3's question:
#   "Do the gains reflect doctrinal extraction, or just citation-proximal text?"
# By comparing this against the single-stage setting:
#   Fact + Principle (existing) vs Fact + Scrubbed Paragraph (this script)
#
# V2 NOTE: Deduplication is by (fact, principle, cited_case) — same as
# single_stage_pools_v2 — so the two settings evaluate on identical pools.
# The scrubbed_paragraph is taken from the first row matching each triple.
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────
INPUT_PATH  = "./COMBINED_ALL_CASES_FINAL_V2_with_scrubbed.csv"
OUTPUT_DIR  = "./"
POOL_SIZE   = 1000
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

# ── STEP 1: Load & clean data ─────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(INPUT_PATH, encoding='latin-1', on_bad_lines='skip')
df_clean = df[['URL', 'Extract of Facts', 'Key Principles Illustrated',
               'Paragraph_Scrubbed_Window', 'Cited Case']].copy()
df_clean.columns = ['url', 'fact', 'principle', 'scrubbed_paragraph', 'cited_case']
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean['fact'].str.len() > 20]
df_clean = df_clean[df_clean['principle'].str.len() > 20]
df_clean = df_clean[df_clean['scrubbed_paragraph'].str.len() > 20]
df_clean = df_clean[df_clean['cited_case'].str.len() > 3]
df_clean = df_clean[~df_clean['fact'].str.contains('ERROR|not available|Insufficient|CONTENT_BLOCKED', case=False, na=False)]
print(f"Clean rows: {len(df_clean)}")

# ── STEP 2: Same split as Stage 1 ─────────────────────────────
unique_urls = df_clean['url'].unique()
train_urls, temp_urls = train_test_split(unique_urls, test_size=0.2, random_state=42)
val_urls,   test_urls = train_test_split(temp_urls,  test_size=0.5, random_state=42)
test_df = df_clean[df_clean['url'].isin(test_urls)].reset_index(drop=True)
print(f"Test rows: {len(test_df)} | Test judgments: {len(test_urls)}")

# ── STEP 3: Build case universe ────────────────────────────────
all_cases  = df_clean['cited_case'].unique().tolist()
case_to_id = {c: i for i, c in enumerate(all_cases)}
id_to_case = {i: c for i, c in enumerate(all_cases)}
print(f"Total unique cited cases: {len(all_cases)}")

# ── STEP 4: Get unique (fact, principle, cited_case) triples ──
# IMPORTANT: deduplicate by the SAME key as single_stage_pools_v2
# so the test set is identical. Take the first scrubbed_paragraph
# associated with each triple.
print("\nBuilding (fact, principle, cited_case) triples (matching single_stage_pools_v2 dedup key)...")
triples = test_df.drop_duplicates(subset=['fact', 'principle', 'cited_case']).reset_index(drop=True)
triples = triples[['fact', 'principle', 'scrubbed_paragraph', 'cited_case']]
print(f"Unique (fact, principle, cited_case) triples: {len(triples)}")

# Stats on how many cited cases per (fact, principle) pair
pair_to_cases = defaultdict(set)
for _, row in triples.iterrows():
    pair_to_cases[(row['fact'], row['principle'])].add(row['cited_case'])
counts = [len(v) for v in pair_to_cases.values()]
print(f"Unique (fact, principle) pairs: {len(pair_to_cases)}")
print(f"Cited cases per pair — min: {min(counts)} | max: {max(counts)} | mean: {np.mean(counts):.2f}")
print(f"Pairs with 1 cited case:   {sum(1 for c in counts if c == 1)} ({100*sum(1 for c in counts if c == 1)/len(counts):.1f}%)")
print(f"Pairs with 2+ cited cases: {sum(1 for c in counts if c >= 2)} ({100*sum(1 for c in counts if c >= 2)/len(counts):.1f}%)")

# ── STEP 5: Generate pools ─────────────────────────────────────
print("\nGenerating paragraph-augmented V2 candidate pools...")
random.seed(RANDOM_SEED)
all_case_ids = set(range(len(all_cases)))
pools = {}

for pool_id, row in triples.iterrows():
    correct_case_id   = case_to_id[row['cited_case']]
    negative_pool     = list(all_case_ids - {correct_case_id})
    sampled_negatives = random.sample(negative_pool, POOL_SIZE - 1)
    full_pool         = [correct_case_id] + sampled_negatives
    random.shuffle(full_pool)

    pools[pool_id] = {
        "fact_text":               row['fact'],
        "principle_text":          row['principle'],
        "scrubbed_paragraph_text": row['scrubbed_paragraph'],
        "query_text":              f"[FACT] {row['fact']} [PARAGRAPH] {row['scrubbed_paragraph']}",
        "correct_case_id":         correct_case_id,
        "correct_case_name":       row['cited_case'],
        "pool":                    full_pool,
        "pool_size":               len(full_pool)
    }

print(f"Generated {len(pools)} pools")

# ── STEP 6: Save ──────────────────────────────────────────────
pool_path   = OUTPUT_DIR + "stage2_paragraph_pools_v2.json"
lookup_path = OUTPUT_DIR + "stage2_case_lookup.json"

with open(pool_path,   'w') as f: json.dump(pools,      f, indent=2)
with open(lookup_path, 'w') as f: json.dump(id_to_case, f, indent=2)

print(f"\n✓ Saved stage2_paragraph_pools_v2.json → {pool_path}")
print(f"✓ Saved stage2_case_lookup.json        → {lookup_path}")

# ── STEP 7: Sanity check ──────────────────────────────────────
sizes = [v['pool_size'] for v in pools.values()]
print(f"\n── Pool Statistics ───────────────────────────")
print(f"  Total pools:              {len(pools)}")
print(f"  Pool size — min/max/mean: {min(sizes)}/{max(sizes)}/{np.mean(sizes):.1f}")
print(f"  Correct cases per pool:   always 1")
print(f"  Negative cases per pool:  always 999")
print(f"  Dedup key:                (fact, principle, cited_case) — matches single_stage_pools_v2")
print(f"\nDone! Use stage2_paragraph_pools_v2.json for paragraph-augmented ablation evaluation.")
