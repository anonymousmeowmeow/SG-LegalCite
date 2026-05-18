# ============================================================
# SG-LegalCite Fact+Issue Ablation V1: Generate Candidate Pools
# Task: [FACT] Fact [ISSUE] Issue → Cited Case
# Pool: 1 correct case + 999 random negatives
# Dedup key: (fact, principle, cited_case) — matches single_stage_pools
#            so test set is identical to single-stage for apples-to-apples comparison
# Issue text taken from first row matching each triple.
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
df_clean = df[['URL', 'Extract of Facts', 'Issue Group', 'Issue',
               'Key Principles Illustrated', 'Cited Case']].copy()
df_clean.columns = ['url', 'fact', 'issue_group', 'issue', 'principle', 'cited_case']
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean['fact'].str.len() > 20]
df_clean = df_clean[df_clean['issue_group'].str.len() > 2]
df_clean = df_clean[df_clean['issue'].str.len() > 5]
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

# ── STEP 4: Dedup by (fact, principle, cited_case) to match single-stage ──
print("\nBuilding triples (dedup key = single_stage_pools' key for fair comparison)...")
triples = test_df.drop_duplicates(subset=['fact', 'principle', 'cited_case']).reset_index(drop=True)
triples = triples[['fact', 'issue', 'principle', 'cited_case']]
print(f"Unique (fact, principle, cited_case) triples: {len(triples)}")

# ── STEP 5: Generate pools ─────────────────────────────────────
print("\nGenerating fact+issue candidate pools...")
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
        "issue_text":        row['issue'],
        "principle_text":    row['principle'],  # stored for reference but NOT in query
        "query_text":        f"[FACT] {row['fact']} [ISSUE] {row['issue']}",
        "correct_case_id":   correct_case_id,
        "correct_case_name": row['cited_case'],
        "pool":              full_pool,
        "pool_size":         len(full_pool)
    }

print(f"Generated {len(pools)} pools")

# ── STEP 6: Save ──────────────────────────────────────────────
pool_path   = OUTPUT_DIR + "stage2_fact_issue_pools_v1.json"
lookup_path = OUTPUT_DIR + "stage2_case_lookup.json"

with open(pool_path,   'w') as f: json.dump(pools,      f, indent=2)
with open(lookup_path, 'w') as f: json.dump(id_to_case, f, indent=2)

print(f"\n✓ Saved stage2_fact_issue_pools_v1.json → {pool_path}")
print(f"✓ Saved stage2_case_lookup.json         → {lookup_path}")

# ── STEP 7: Sanity check ──────────────────────────────────────
sizes = [v['pool_size'] for v in pools.values()]
print(f"\n── Pool Statistics ───────────────────────────")
print(f"  Total pools:              {len(pools)}")
print(f"  Pool size — min/max/mean: {min(sizes)}/{max(sizes)}/{np.mean(sizes):.1f}")
print(f"  Correct cases per pool:   always 1")
print(f"  Negative cases per pool:  always 999")
print(f"  Dedup key:                (fact, principle, cited_case) — matches single_stage_pools")
print(f"\nDone! Use stage2_fact_issue_pools_v1.json for fact+issue ablation evaluation.")
