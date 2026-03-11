# ============================================================
# SG-LegalCite Stage 2 Direct Baseline V2: Generate Candidate Pools
# Task: Facts → Cited Case directly (no key principles)
# Pool: 1 correct case + 999 random negatives
# Evaluated at (fact, cited_case) pair level
# One pool per unique (fact, cited_case) pair
# Run on CLUSTER
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
INPUT_PATH  = "./COMBINED_ALL_CASES_FINAL_V2.csv"
OUTPUT_DIR  = "./"
POOL_SIZE   = 1000
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

# ── STEP 1: Load & clean data ─────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(INPUT_PATH, encoding='latin-1', on_bad_lines='skip')
df_clean = df[['URL', 'Extract of Facts', 'Key Principles Illustrated', 'Cited Case']].copy()
df_clean.columns = ['url', 'fact', 'principle', 'cited_case']
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean['fact'].str.len() > 20]
df_clean = df_clean[df_clean['principle'].str.len() > 20]
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

# ── STEP 4: Check how many cited cases each fact has ──────────
print("\nChecking cited cases per fact...")
fact_to_cited_cases = defaultdict(set)
for _, row in test_df.iterrows():
    fact_to_cited_cases[row['fact']].add(row['cited_case'])

counts = [len(v) for v in fact_to_cited_cases.values()]
print(f"Unique test facts: {len(fact_to_cited_cases)}")
print(f"Cited cases per fact — min: {min(counts)} | max: {max(counts)} | mean: {np.mean(counts):.1f}")
print(f"Total (fact, cited_case) pairs: {sum(counts)}")

# ── STEP 5: Generate pools ─────────────────────────────────────
# One pool per (fact, cited_case) pair
# Query = Fact only (no principle)
# Pool = 1 correct cited case + 999 random negatives
print("\nGenerating direct V2 candidate pools...")
random.seed(RANDOM_SEED)
all_case_ids = set(range(len(all_cases)))
pools   = {}
pool_id = 0

for fact_text, cited_cases in fact_to_cited_cases.items():
    for correct_case in cited_cases:
        correct_case_id = case_to_id[correct_case]

        # 999 random negatives (excluding correct case)
        negative_pool     = list(all_case_ids - {correct_case_id})
        sampled_negatives = random.sample(negative_pool, POOL_SIZE - 1)

        full_pool = [correct_case_id] + sampled_negatives
        random.shuffle(full_pool)

        pools[pool_id] = {
            "fact_text":         fact_text,
            "correct_case_id":   correct_case_id,
            "correct_case_name": correct_case,
            "pool":              full_pool,
            "pool_size":         len(full_pool)
        }
        pool_id += 1

print(f"Generated {len(pools)} pools")

# ── STEP 6: Save ──────────────────────────────────────────────
pool_path   = OUTPUT_DIR + "stage2_direct_candidate_pools_v2.json"
lookup_path = OUTPUT_DIR + "stage2_case_lookup.json"

with open(pool_path,   'w') as f: json.dump(pools,      f, indent=2)
with open(lookup_path, 'w') as f: json.dump(id_to_case, f, indent=2)

print(f"\n✓ Saved stage2_direct_candidate_pools_v2.json → {pool_path}")
print(f"✓ Saved stage2_case_lookup.json                → {lookup_path}")

# ── STEP 7: Sanity check ──────────────────────────────────────
sizes = [v['pool_size'] for v in pools.values()]
print(f"\n── Pool Statistics ───────────────────────────")
print(f"  Total pools:              {len(pools)}")
print(f"  Pool size — min/max/mean: {min(sizes)}/{max(sizes)}/{np.mean(sizes):.1f}")
print(f"  Correct cases per pool:   always 1")
print(f"  Negative cases per pool:  always 999")
print(f"\nDone! Use stage2_direct_candidate_pools_v2.json for direct baseline V2 evaluation.")
