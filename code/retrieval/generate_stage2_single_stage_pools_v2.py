# ============================================================
# SG-LegalCite Single-Stage V2: Generate Candidate Pools
# Task: [FACT] Fact [PRINCIPLE] Ground Truth Principle → Cited Case
# Pool: 1 correct case + 999 random negatives
# One pool per (fact, principle, cited_case) triple
# Uses GROUND TRUTH principles directly (no Stage 1 prediction)
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

# ── STEP 4: Get unique (fact, principle, cited_case) triples ──
print("\nBuilding (fact, principle, cited_case) triples...")
triples = test_df[['fact', 'principle', 'cited_case']].drop_duplicates().reset_index(drop=True)
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
print("\nGenerating single-stage V2 candidate pools...")
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
        "fact_text":         row['fact'],
        "principle_text":    row['principle'],
        "query_text":        f"[FACT] {row['fact']} [PRINCIPLE] {row['principle']}",
        "correct_case_id":   correct_case_id,
        "correct_case_name": row['cited_case'],
        "pool":              full_pool,
        "pool_size":         len(full_pool)
    }

print(f"Generated {len(pools)} pools")

# ── STEP 6: Save ──────────────────────────────────────────────
pool_path   = OUTPUT_DIR + "stage2_single_stage_pools.json"
lookup_path = OUTPUT_DIR + "stage2_case_lookup.json"

with open(pool_path,   'w') as f: json.dump(pools,      f, indent=2)
with open(lookup_path, 'w') as f: json.dump(id_to_case, f, indent=2)

print(f"\n✓ Saved stage2_single_stage_pools.json → {pool_path}")
print(f"✓ Saved stage2_case_lookup.json         → {lookup_path}")

# ── STEP 7: Sanity check ──────────────────────────────────────
sizes = [v['pool_size'] for v in pools.values()]
print(f"\n── Pool Statistics ───────────────────────────")
print(f"  Total pools:              {len(pools)}")
print(f"  Pool size — min/max/mean: {min(sizes)}/{max(sizes)}/{np.mean(sizes):.1f}")
print(f"  Correct cases per pool:   always 1")
print(f"  Negative cases per pool:  always 999")
print(f"\nDone! Use stage2_single_stage_pools.json for single-stage evaluation.")
