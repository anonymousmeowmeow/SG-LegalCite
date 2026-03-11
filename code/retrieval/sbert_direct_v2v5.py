# ============================================================
# SG-LegalCite Stage 2 Direct Baseline — SBERT V2V4
# Task: Facts → Cited Case directly (no key principles)
# Model: sentence-transformers/all-mpnet-base-v2
# Pool: SAME as Legal BERT — stage2_direct_candidate_pools_v2.json
# 9942 pools — one per (fact, cited_case) pair
# Zero-Shot + Fine-Tuned
# ============================================================

import os
import subprocess
import sys

print("Installing required packages...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'sentence-transformers', 'numpy<2.0', 'datasets', '-q'])
print("✓ Packages installed")

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
POOL_PATH     = "./stage2_direct_candidate_pools_v2.json"
LOOKUP_PATH   = "./stage2_case_lookup.json"
OUTPUT_DIR    = "./citation_rec_sbert_direct_v2"
BATCH_SIZE    = 64
EPOCHS        = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
TOP_K_EVAL    = [1, 5, 10, 20]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv('COMBINED_ALL_CASES_FINAL_V2.csv', encoding='latin-1', on_bad_lines='skip')
df_clean = df[['URL', 'Extract of Facts', 'Key Principles Illustrated', 'Cited Case']].copy()
df_clean.columns = ['url', 'fact', 'principle', 'cited_case']
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean['fact'].str.len() > 20]
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
print("\nLoading direct V2 candidate pools (same as Legal BERT)...")
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
def evaluate_direct_v2(model, label):
    print(f"\n{'='*60}")
    print(f"  {label} — SBERT Direct Baseline V2 (Facts → Case)")
    print(f"  Pool: 1 correct + 999 random | 9942 pools (same as Legal BERT)")
    print(f"{'='*60}")

    pool_ids     = list(pools.keys())
    unique_facts = list(set(pools[fid]['fact_text'] for fid in pool_ids))
    fact_to_idx  = {f: i for i, f in enumerate(unique_facts)}

    print(f"  Encoding {len(unique_facts)} unique facts...")
    fact_embs = encode_texts(model, unique_facts)

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
        query_emb  = fact_embs[fact_to_idx[pool_data['fact_text']]]

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
print("  PART 1: Zero-Shot SBERT — Direct Baseline V2")
print("="*60)

zs_model = SentenceTransformer(MODEL_NAME, device=device)
print(f"✓ Zero-shot model loaded: {MODEL_NAME}")

all_results = []
result      = evaluate_direct_v2(zs_model, label="SBERT Zero-Shot")
all_results.append(result)
del zs_model


# ════════════════════════════════════════════════════════════
# PART 2: FINE-TUNING using ft_model.fit()
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2: Fine-Tuning SBERT — Facts → Cited Case directly")
print("="*60)

train_pairs = train_df[['fact', 'cited_case']].drop_duplicates().reset_index(drop=True)
print(f"  Training pairs: {len(train_pairs)}")

train_examples   = [InputExample(texts=[row['fact'], row['cited_case']])
                    for _, row in train_pairs.iterrows()]
train_dataloader = st_datasets.NoDuplicatesDataLoader(train_examples, batch_size=BATCH_SIZE)

ft_model      = SentenceTransformer(MODEL_NAME, device=device)
train_loss_fn = losses.MultipleNegativesRankingLoss(ft_model)
warmup_steps  = int(len(train_dataloader) * EPOCHS * WARMUP_RATIO)

ft_model.fit(
    train_objectives=[(train_dataloader, train_loss_fn)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LEARNING_RATE},
    output_path=f"{OUTPUT_DIR}/sbert_direct_v2_model",
    show_progress_bar=True,
    save_best_model=True,
)
print(f"\n✓ Training complete. Model saved to {OUTPUT_DIR}/sbert_direct_v2_model")


# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3: Fine-Tuned SBERT — Direct Baseline V2")
print("="*60)

eval_model = SentenceTransformer(f"{OUTPUT_DIR}/sbert_direct_v2_model", device=device)
print("✓ Fine-tuned SBERT model loaded")

result = evaluate_direct_v2(eval_model, label="SBERT Fine-Tuned")
all_results.append(result)


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  SBERT DIRECT BASELINE V2 FINAL RESULTS")
print("  Zero-Shot vs Fine-Tuned | Facts → Case (no principles)")
print("  Pool: 1 correct + 999 random | 9942 pools (same as Legal BERT)")
print("="*60)
summary_df = pd.DataFrame(all_results).set_index('Model')
print(summary_df.to_string())
summary_df.to_csv(f"{OUTPUT_DIR}/sbert_direct_v2_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/sbert_direct_v2_results.csv")
