# ============================================================
# SG-LegalCite Fact+Issue Group Ablation — SBERT
# Task: [FACT] Fact + [ISSUE_GROUP] Issue Group
#        → Cited Case
# Model: sentence-transformers/all-mpnet-base-v2
# Pool: stage2_fact_issue_group_pools_v1.json (9979 pools)
# Zero-Shot + Fine-Tuned
# Purpose: Ablation R3 — fact+issue vs fact+principle baseline
#          Tests if issue_group alone (coarsest doctrinal category) provides retrieval signal
# ============================================================

import os
import subprocess
import sys

# v2: SKIP in-script pip install — packages already in ~/.local
# (compute nodes lack internet; pip install was a no-op that
#  also pulled triton 3.6 as a transitive dep, breaking torch)
print("Skipping in-script pip install (packages already on disk)")
print("✓ Using pre-installed packages from ~/.local")

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import datasets as st_datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ── CONFIG ────────────────────────────────────────────────────
MODEL_NAME    = "sentence-transformers/all-mpnet-base-v2"
POOL_PATH     = "./stage2_fact_issue_group_pools_v1.json"
LOOKUP_PATH   = "./stage2_case_lookup.json"
OUTPUT_DIR    = "./citation_rec_sbert_fact_issue_group_v1"
BATCH_SIZE    = 64
EPOCHS        = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
TOP_K_EVAL    = [1, 5, 10, 20]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv('COMBINED_ALL_CASES_FINAL_V2.csv', encoding='latin-1', on_bad_lines='skip')
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

unique_urls = df_clean['url'].unique()
train_urls, temp_urls = train_test_split(unique_urls, test_size=0.2, random_state=42)
val_urls,   test_urls = train_test_split(temp_urls,  test_size=0.5, random_state=42)
train_df = df_clean[df_clean['url'].isin(train_urls)].reset_index(drop=True)
val_df   = df_clean[df_clean['url'].isin(val_urls)].reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── LOAD POOLS ────────────────────────────────────────────────
print("\nLoading all-fields candidate pools v3...")
with open(POOL_PATH,   'r') as f: pools    = json.load(f)
with open(LOOKUP_PATH, 'r') as f: lookup   = json.load(f)
id_to_case = {int(k): v for k, v in lookup.items()}
print(f"✓ Loaded {len(pools)} pools")
print(f"✓ Loaded {len(id_to_case)} unique cited cases")


# ── ENCODE HELPER ─────────────────────────────────────────────
def encode_texts(model, texts, batch_size=128):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)


# ── METRICS ───────────────────────────────────────────────────
def compute_metrics(ranked_ids, correct_id, k_values):
    mrr = 0.0
    for rank, cid in enumerate(ranked_ids, 1):
        if cid == correct_id:
            mrr = 1.0 / rank
            break
    result = {'MRR': mrr, 'MAP': mrr}
    for k in k_values:
        hit               = int(correct_id in ranked_ids[:k])
        result[f'R@{k}']  = hit
        result[f'P@{k}']  = hit / k
        result[f'F1@{k}'] = (2*(hit/k)*hit)/((hit/k)+hit) if hit > 0 else 0.0
    return result


# ── EVALUATION FUNCTION ───────────────────────────────────────
def evaluate_all_fields(model, label):
    print(f"\n{'='*60}")
    print(f"  {label} — SBERT All Fields")
    print(f"  Query: [FACT] fact [ISSUE_GROUP] issue_group [ISSUE] issue [PRINCIPLE] principle")
    print(f"  Pool: 1 correct + 999 random | 9979 pools")
    print(f"{'='*60}")

    pool_ids     = list(pools.keys())
    unique_queries = list(set(pools[fid]['query_text'] for fid in pool_ids))
    query_to_idx   = {q: i for i, q in enumerate(unique_queries)}

    print(f"  Encoding {len(unique_queries)} unique queries (all fields)...")
    query_embs = encode_texts(model, unique_queries)

    needed_case_ids = sorted(set(cid for p in pools.values() for cid in p['pool']))
    case_texts      = [id_to_case[cid] for cid in needed_case_ids]
    cid_to_row      = {cid: idx for idx, cid in enumerate(needed_case_ids)}

    print(f"  Encoding {len(case_texts)} case candidates...")
    case_embs = encode_texts(model, case_texts)

    all_metrics        = {f'R@{k}': [] for k in TOP_K_EVAL}
    all_metrics.update({f'P@{k}':  [] for k in TOP_K_EVAL})
    all_metrics.update({f'F1@{k}': [] for k in TOP_K_EVAL})
    mrr_list, map_list = [], []

    for fid in pool_ids:
        pool_data  = pools[fid]
        correct_id = pool_data['correct_case_id']
        pool_ids_  = pool_data['pool']
        pool_rows  = [cid_to_row[cid] for cid in pool_ids_]
        pool_embs  = case_embs[pool_rows]
        query_emb  = query_embs[query_to_idx[pool_data['query_text']]]

        sims       = np.dot(pool_embs, query_emb)
        ranked_pos = np.argsort(sims)[::-1]
        ranked_ids = [pool_ids_[r] for r in ranked_pos]

        m = compute_metrics(ranked_ids, correct_id, TOP_K_EVAL)
        mrr_list.append(m['MRR'])
        map_list.append(m['MAP'])
        for k in TOP_K_EVAL:
            all_metrics[f'R@{k}'].append(m[f'R@{k}'])
            all_metrics[f'P@{k}'].append(m[f'P@{k}'])
            all_metrics[f'F1@{k}'].append(m[f'F1@{k}'])

    print(f"\n  MRR: {np.mean(mrr_list):.4f} | MAP: {np.mean(map_list):.4f}")
    print(f"  {'K':<5} {'Recall@K':<12} {'Precision@K':<14} {'F1@K'}")
    for k in TOP_K_EVAL:
        print(f"  {k:<5} {np.mean(all_metrics[f'R@{k}']):<12.4f} "
              f"{np.mean(all_metrics[f'P@{k}']):<14.4f} "
              f"{np.mean(all_metrics[f'F1@{k}']):.4f}")
    print(f"{'='*60}")

    return {
        'Model': label,
        'MRR':  round(np.mean(mrr_list), 4),
        'MAP':  round(np.mean(map_list), 4),
        **{f'R@{k}':  round(np.mean(all_metrics[f'R@{k}']),  4) for k in TOP_K_EVAL},
        **{f'P@{k}':  round(np.mean(all_metrics[f'P@{k}']),  4) for k in TOP_K_EVAL},
        **{f'F1@{k}': round(np.mean(all_metrics[f'F1@{k}']), 4) for k in TOP_K_EVAL},
    }


# ════════════════════════════════════════════════════════════
# PART 1: ZERO-SHOT EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 1: Zero-Shot SBERT — Fact+Issue Group Ablation")
print("="*60)

zs_model = SentenceTransformer(MODEL_NAME, device=device)
print(f"✓ Zero-shot model loaded: {MODEL_NAME}")

all_results = []
result      = evaluate_all_fields(zs_model, label="SBERT Zero-Shot — Fact+Issue Group")
all_results.append(result)
del zs_model


# ════════════════════════════════════════════════════════════
# PART 2: FINE-TUNING using ft_model.fit()
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2: Fine-Tuning SBERT — Fact+Issue Group → Cited Case")
print("="*60)

# Match dedup key to single_stage_pools (fact, principle, cited_case)
# so training distribution matches test distribution.
# Issue text taken from first row of each triple.
train_pairs = train_df.drop_duplicates(subset=['fact', 'principle', 'cited_case']).reset_index(drop=True)
train_pairs = train_pairs[['fact', 'issue_group', 'principle', 'cited_case']]
print(f"  Training pairs: {len(train_pairs)}")

train_examples = [InputExample(texts=[
    f"[FACT] {row['fact']} [ISSUE_GROUP] {row['issue_group']}",
    row['cited_case']
]) for _, row in train_pairs.iterrows()]
train_dataloader = st_datasets.NoDuplicatesDataLoader(train_examples, batch_size=BATCH_SIZE)

ft_model      = SentenceTransformer(MODEL_NAME, device=device)
train_loss_fn = losses.MultipleNegativesRankingLoss(ft_model)
warmup_steps  = int(len(train_dataloader) * EPOCHS * WARMUP_RATIO)

ft_model.fit(
    train_objectives=[(train_dataloader, train_loss_fn)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LEARNING_RATE},
    output_path=f"{OUTPUT_DIR}/sbert_fact_issue_group_model",
    show_progress_bar=True,
    save_best_model=True,
)
print(f"\n✓ Training complete. Model saved to {OUTPUT_DIR}/sbert_fact_issue_group_model")


# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3: Fine-Tuned SBERT — Fact+Issue Group Ablation")
print("="*60)

eval_model = SentenceTransformer(f"{OUTPUT_DIR}/sbert_fact_issue_group_model", device=device)
print("✓ Fine-tuned SBERT model loaded")

result = evaluate_all_fields(eval_model, label="SBERT Fine-Tuned — Fact+Issue Group")
all_results.append(result)


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  SBERT FACT+ISSUE GROUP ABLATION FINAL RESULTS")
print("  Zero-Shot vs Fine-Tuned")
print("  Query: [FACT] [ISSUE_GROUP] → Cited Case")
print("  Pool: 1 correct + 999 random | 9979 pools")
print("="*60)
summary_df = pd.DataFrame(all_results).set_index('Model')
print(summary_df.to_string())
summary_df.to_csv(f"{OUTPUT_DIR}/sbert_fact_issue_group_v1_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/sbert_fact_issue_group_v1_results.csv")
