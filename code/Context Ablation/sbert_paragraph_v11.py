# ============================================================
# SG-LegalCite Paragraph-Augmented Ablation — SBERT V11
# Task: [FACT] Fact + [PARAGRAPH] Scrubbed Paragraph Window → Cited Case
# Model: sentence-transformers/all-mpnet-base-v2
# Pool: stage2_paragraph_pools_v2.json (9979 pools)
#       — IDENTICAL pools to stage2_single_stage_pools.json
#       — dedup key: (fact, principle, cited_case) — same negatives, same gold
# Zero-Shot + Fine-Tuned
#
# Purpose: Reviewer 3 ablation — does the principle field add signal
# beyond raw citation-proximal text?
# Comparison: this run vs sbert_single_stage_v1 results.
#
# V8 KEY CHANGE: Token-matched truncation.
# The scrubbed paragraph is truncated to match the token count of
# the principle for each row. This ensures a fair comparison:
# both inputs get the same text budget (~70 tokens avg), so any
# performance difference reflects the quality of the content,
# not the quantity of text.
#
# V7 result (untruncated ~400 words): MRR 54.3% — unfair comparison
# because paragraph had 6x more text than principle (~70 tokens).
# V8 fixes this by matching the text budget per row.
#
# v4: environment handling copied from saullm_all_fields_v17.py
#     - flash_attn mock + blocker (broken system library)
#     - transformer_engine mock (triggers broken triton import)
#     - SKIP pip install (packages already on disk in ~/.local,
#       compute nodes have no internet anyway, and pip install
#       accidentally pulls in incompatible triton/torch versions)
# ============================================================

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Let PBS handle GPU assignment — do NOT set CUDA_VISIBLE_DEVICES manually

import subprocess
import sys
import time

# ============================================================================
# MOCK FLASH_ATTN — bypass broken system library
# (copied from saullm_all_fields_v17.py which runs successfully on this cluster)
# ============================================================================
print("=" * 60)
print("REMOVING BROKEN FLASH_ATTN + TRANSFORMER_ENGINE")
print("=" * 60)

try:
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'flash-attn', '-y',
                    '--break-system-packages'], capture_output=True, timeout=30)
    print("  Uninstalled flash-attn package")
except:
    print("  Could not uninstall flash-attn (may not be pip-installed)")

modules_to_remove = [k for k in list(sys.modules.keys()) if 'flash_attn' in k]
for mod in modules_to_remove:
    del sys.modules[mod]
print(f"  Removed {len(modules_to_remove)} existing flash_attn modules")

sys.path = [p for p in sys.path if 'flash_attn' not in p]

from types import ModuleType
from importlib.machinery import ModuleSpec
import importlib.abc
import importlib.util

class FlashAttnBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('flash_attn') or fullname == 'flash_attn_2_cuda':
            return importlib.util.spec_from_loader(fullname, FlashAttnLoader())
        return None

class FlashAttnLoader(importlib.abc.Loader):
    def create_module(self, spec): return sys.modules.get(spec.name)
    def exec_module(self, module): pass

sys.meta_path.insert(0, FlashAttnBlocker())

def dummy_func(*args, **kwargs):
    raise NotImplementedError("Flash attention not available")

def make_mock(name, attrs=None):
    mod = ModuleType(name)
    mod.__spec__ = ModuleSpec(name, FlashAttnLoader())
    mod.__path__ = []
    if attrs:
        for k, v in attrs.items(): setattr(mod, k, v)
    return mod

sys.modules['flash_attn']                      = make_mock('flash_attn', {'flash_attn_func': dummy_func, '__version__': '0.0.0'})
sys.modules['flash_attn.flash_attn_interface'] = make_mock('flash_attn.flash_attn_interface', {'flash_attn_func': dummy_func})
sys.modules['flash_attn.bert_padding']         = make_mock('flash_attn.bert_padding', {'unpad_input': dummy_func, 'pad_input': dummy_func})
sys.modules['flash_attn_2_cuda']               = make_mock('flash_attn_2_cuda')
print("✓ Flash attention mocked")

# Block transformer_engine — it transitively imports triton which is broken
# This is the actual root cause of the v3/v4 crashes (AttrsDescriptor import error)
# v5: catch-all blocker — mocks ANY transformer_engine.* import automatically
# so we never have to chase individual submodules again
class TEBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('transformer_engine'):
            return importlib.util.spec_from_loader(fullname, FlashAttnLoader())
        return None

sys.meta_path.insert(0, TEBlocker())

# Pre-populate the most common ones so they resolve immediately
for _te_mod in [
    'transformer_engine', 'transformer_engine.common',
    'transformer_engine.common.recipe', 'transformer_engine.pytorch',
    'transformer_engine.pytorch.module', 'transformer_engine.pytorch.fp8',
    'transformer_engine.pytorch.jit',
]:
    sys.modules[_te_mod] = make_mock(_te_mod)
print("✓ Transformer engine blocked (catch-all + pre-populated mocks)")

# ============================================================================
# SKIP PIP INSTALL — packages already on disk in ~/.local
# (compute nodes have no internet anyway, and pip install accidentally
#  pulls in incompatible triton/torch versions — see saullm_all_fields_v17.py)
# ============================================================================
print("\nSkipping in-script pip install (packages already on disk)")
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

# GPU diagnostic — verify we're actually using GPU, not silently falling back to CPU
print(f"\nCUDA_VISIBLE_DEVICES env: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

if not torch.cuda.is_available():
    print("\n*** ERROR: CUDA NOT AVAILABLE — job would run on CPU (50-100x slower) ***")
    print("*** Exiting to prevent wasting 68+ hours of CPU time ***")
    sys.exit(1)

device = torch.device('cuda')
print(f"\nDevice: {device}")
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ── CONFIG ────────────────────────────────────────────────────
MODEL_NAME    = "sentence-transformers/all-mpnet-base-v2"
CSV_PATH      = "./COMBINED_ALL_CASES_FINAL_V2_with_scrubbed.csv"
POOL_PATH     = "./stage2_paragraph_pools_v2.json"
LOOKUP_PATH   = "./stage2_case_lookup.json"
OUTPUT_DIR    = "./citation_rec_sbert_paragraph_v11"
BATCH_SIZE    = 32
EPOCHS        = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
TOP_K_EVAL    = [1, 5, 10, 20]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(CSV_PATH, encoding='latin-1', on_bad_lines='skip')
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

unique_urls = df_clean['url'].unique()
train_urls, temp_urls = train_test_split(unique_urls, test_size=0.2, random_state=42)
val_urls,   test_urls = train_test_split(temp_urls,  test_size=0.5, random_state=42)
train_df = df_clean[df_clean['url'].isin(train_urls)].reset_index(drop=True)
val_df   = df_clean[df_clean['url'].isin(val_urls)].reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── LOAD POOLS ────────────────────────────────────────────────
print("\nLoading paragraph V2 candidate pools (identical to single-stage pools)...")
with open(POOL_PATH,   'r') as f: pools    = json.load(f)
with open(LOOKUP_PATH, 'r') as f: lookup   = json.load(f)
id_to_case = {int(k): v for k, v in lookup.items()}
print(f"✓ Loaded {len(pools)} pools")
print(f"✓ Loaded {len(id_to_case)} unique cited cases")

# ── TOKEN-MATCHED TRUNCATION ──────────────────────────────────
# Truncate scrubbed_paragraph to match the token count of the principle
# for each pool, ensuring a fair text-budget comparison.
# We use whitespace tokenisation as a proxy for subword token count
# (principle avg = 69.9 tokens ≈ 54 words; the ratio is consistent enough).
def truncate_to_match(paragraph, principle):
    """Truncate paragraph to same word count as principle."""
    if not isinstance(paragraph, str) or not isinstance(principle, str):
        return paragraph or ''
    target_words = len(principle.split())
    para_words = paragraph.split()
    if len(para_words) <= target_words:
        return paragraph
    return ' '.join(para_words[:target_words])

# Apply truncation to all pool queries
print("\nApplying token-matched truncation to pool queries...")
trunc_lengths = []
for fid in pools:
    p = pools[fid]
    principle = p['principle_text']
    paragraph = p['scrubbed_paragraph_text']
    truncated = truncate_to_match(paragraph, principle)
    trunc_lengths.append(len(truncated.split()))
    # Overwrite the query_text with truncated version
    p['query_text'] = f"[FACT] {p['fact_text']} [PARAGRAPH] {truncated}"
    # Store truncated version for reference
    p['scrubbed_paragraph_truncated'] = truncated

print(f"✓ Truncated {len(pools)} pool queries")
print(f"  Truncated paragraph words — mean: {np.mean(trunc_lengths):.1f}, "
      f"median: {np.median(trunc_lengths):.1f}, "
      f"min: {min(trunc_lengths)}, max: {max(trunc_lengths)}")


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
def evaluate_paragraph(model, label):
    print(f"\n{'='*60}")
    print(f"  {label} — SBERT Paragraph (Fact + Scrubbed Paragraph → Case)")
    print(f"  Pool: 1 correct + 999 random | 9979 pools (same as Single Stage)")
    print(f"{'='*60}")

    pool_ids     = list(pools.keys())
    unique_queries = list(set(pools[fid]['query_text'] for fid in pool_ids))
    query_to_idx   = {q: i for i, q in enumerate(unique_queries)}

    print(f"  Encoding {len(unique_queries)} unique queries (fact + scrubbed paragraph)...")
    fact_embs = encode_texts(model, unique_queries)

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
        query_emb  = fact_embs[query_to_idx[pool_data['query_text']]]

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
print("  PART 1: Zero-Shot SBERT — Paragraph Ablation V11")
print("="*60)

zs_model = SentenceTransformer(MODEL_NAME, device=device)
print(f"✓ Zero-shot model loaded: {MODEL_NAME}")

all_results = []
result      = evaluate_paragraph(zs_model, label="SBERT Zero-Shot")
all_results.append(result)
del zs_model


# ════════════════════════════════════════════════════════════
# PART 2: FINE-TUNING using ft_model.fit()
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2: Fine-Tuning SBERT — Fact + Scrubbed Paragraph → Cited Case")
print("="*60)

train_pairs = train_df[['fact', 'principle', 'scrubbed_paragraph', 'cited_case']].drop_duplicates().reset_index(drop=True)
print(f"  Training pairs: {len(train_pairs)}")

# Apply same token-matched truncation to training queries
train_examples   = [InputExample(texts=[f"[FACT] {row['fact']} [PARAGRAPH] {truncate_to_match(row['scrubbed_paragraph'], row['principle'])}", row['cited_case']])
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
    output_path=f"{OUTPUT_DIR}/sbert_paragraph_model",
    show_progress_bar=True,
    save_best_model=True,
)
print(f"\n✓ Training complete. Model saved to {OUTPUT_DIR}/sbert_paragraph_model")


# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3: Fine-Tuned SBERT — Paragraph Ablation V11")
print("="*60)

eval_model = SentenceTransformer(f"{OUTPUT_DIR}/sbert_paragraph_model", device=device)
print("✓ Fine-tuned SBERT model loaded")

result = evaluate_paragraph(eval_model, label="SBERT Fine-Tuned")
all_results.append(result)


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  SBERT PARAGRAPH ABLATION V11 FINAL RESULTS")
print("  Zero-Shot vs Fine-Tuned | Fact + Scrubbed Paragraph → Case")
print("  Pool: 1 correct + 999 random | 9979 pools (same as Single Stage)")
print("="*60)
summary_df = pd.DataFrame(all_results).set_index('Model')
print(summary_df.to_string())
summary_df.to_csv(f"{OUTPUT_DIR}/sbert_paragraph_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/sbert_paragraph_results.csv")
