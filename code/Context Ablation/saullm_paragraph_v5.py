"""
SG-LegalCite Paragraph-Augmented Ablation V5 — SaulLM-7B (Mistral-based)
Task:   [FACT] Extract of Facts + [PARAGRAPH] Scrubbed Paragraph Window → Cited Case
Pool:   stage2_paragraph_pools_v2.json (9979 pools)
        — IDENTICAL pools to stage2_single_stage_pools.json
        — dedup key: (fact, principle, cited_case) — same negatives, same gold
Split:  By URL (same as ALL other models for fair comparison)

Purpose: Reviewer 3 ablation — does the principle field add signal
beyond raw citation-proximal text? Compare against single-stage results.

Training setup IDENTICAL to single-stage SaulLM:
  BATCH=64, EPOCHS=10, LR=2e-5, WARMUP=0.1
  Temperature: learnable, init=0.07
  Save best by val_loss
  pin_memory=True, num_workers=4

Decoder-specific (necessary differences, documented):
  QLoRA: 4-bit NF4 + LoRA (r=16, alpha=32, dropout=0.1)
  device_map={"": 0} (force all onto GPU 0, avoids CPU offload error)
  Last-token pooling + linear projection head (fine-tuned only)
  Gradient checkpointing

Zero-shot evaluation (v2 fix — Kai Dong):
  No LoRA adapters, no projection head.
  Raw pretrained weights only: last-token pool → L2-normalise.

Environment fixes (from sbert_paragraph_v7 + saullm_all_fields_v17):
  flash_attn mock + blocker (broken system library)
  catch-all transformer_engine blocker (broken triton import)
  ATTN_IMPL="sdpa" (Mistral-based)
  No hardcoded CUDA_VISIBLE_DEVICES (let PBS handle GPU assignment)
  Fail-fast GPU check (exit if no CUDA)
  Skip pip install (packages pre-installed in ~/.local)

To run: qsub run_saullm_paragraph_v1.pbs
"""

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
# MOCK FLASH_ATTN — bypass broken system library (inherited from v3)
# ============================================================================
print("=" * 60)
print("REMOVING BROKEN FLASH_ATTN")
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

# ============================================================================
# BLOCK TRANSFORMER_ENGINE — catch-all blocker for broken triton import chain
# (from sbert_paragraph_v7 — v3/v4 crashed on transformer_engine.common)
# ============================================================================
class TEBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('transformer_engine'):
            return importlib.util.spec_from_loader(fullname, FlashAttnLoader())
        return None

sys.meta_path.insert(0, TEBlocker())

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
#  pulls in incompatible triton/torch versions — see saullm_all_fields_v17)
# ============================================================================
print("\nSkipping in-script pip install (packages already on disk)")
print("✓ Using pre-installed packages from ~/.local")

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (AutoTokenizer, AutoModel, BitsAndBytesConfig,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"Transformers: {transformers.__version__} | PyTorch: {torch.__version__}")

# GPU diagnostic — fail fast if no CUDA
print(f"\nCUDA_VISIBLE_DEVICES env: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

if not torch.cuda.is_available():
    print("\n*** ERROR: CUDA NOT AVAILABLE — exiting to prevent CPU fallback ***")
    sys.exit(1)

device = torch.device('cuda')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)")

ATTN_IMPL = "sdpa"

# ============================================================================
# CONFIG — IDENTICAL to AdaptLLM and all other models
# ============================================================================
MODEL_NAME     = "Equall/Saul-Instruct-v1"
CSV_PATH       = "./COMBINED_ALL_CASES_FINAL_V2_with_scrubbed.csv"
POOL_PATH      = "./stage2_paragraph_pools_v2.json"
LOOKUP_PATH    = "./stage2_case_lookup.json"
OUTPUT_DIR     = "./citation_rec_saullm_paragraph_v5"

# --- Identical to ALL other models ---
BATCH_SIZE     = 16
EPOCHS         = 10
LEARNING_RATE  = 2e-5
WARMUP_RATIO   = 0.1
TEMP_INIT      = 0.07    # learnable temperature
MAX_LENGTH     = 512
TOP_K          = [1, 5, 10, 20]
NUM_WORKERS    = 4
PIN_MEMORY     = True

# --- Decoder-specific (necessary, documented) ---
EMBEDDING_DIM  = 4096    # Mistral hidden size
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.1
ENCODE_BATCH   = 32      # smaller for 7B model memory
EARLY_STOP     = 3       # stop if val_loss doesn't improve for 3 epochs

print(f"\n{'='*60}")
print(f"SaulLM-7B Paragraph — [FACT] Facts + [PARAGRAPH] Scrubbed Paragraph → Cited Case")
print(f"{'='*60}")
print(f"Model:      {MODEL_NAME}")
print(f"Pool:        stage2_paragraph_pools_v2.json (9979)")
print(f"Split:      by URL (identical to all models)")
print(f"Batch:      {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LEARNING_RATE}")
print(f"Warmup:     {WARMUP_RATIO} | Temp: learnable init={TEMP_INIT}")
print(f"Early Stop: patience={EARLY_STOP} (stops if no val_loss improvement)")
print(f"QLoRA:      r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Attention:  {ATTN_IMPL}")
print(f"GPU:        1x A100 (4-bit quantised, fits easily)")

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\nLoading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, encoding='latin-1', on_bad_lines='skip')
print(f"Raw rows: {len(df)}")

df_clean = df[['URL', 'Extract of Facts', 'Key Principles Illustrated',
               'Paragraph_Scrubbed_Window', 'Cited Case']].copy()
df_clean.columns = ['url', 'fact', 'principle', 'scrubbed_paragraph', 'cited_case']
df_clean = df_clean.dropna(subset=['fact', 'principle', 'scrubbed_paragraph', 'cited_case', 'url'])
df_clean = df_clean[df_clean['fact'].str.len() > 20]
df_clean = df_clean[df_clean['principle'].str.len() > 20]
df_clean = df_clean[df_clean['scrubbed_paragraph'].str.len() > 20]
df_clean = df_clean[df_clean['cited_case'].str.len() > 3]
df_clean = df_clean[~df_clean['fact'].str.contains(
    'ERROR|not available|Insufficient|CONTENT_BLOCKED', case=False, na=False)]
print(f"Clean rows: {len(df_clean)}")

# ── TOKEN-MATCHED TRUNCATION ──────────────────────────────────
# Truncate scrubbed_paragraph to match word count of principle per row.
# This ensures a fair comparison: both inputs get the same text budget
# (~54 words avg), so any performance difference reflects content quality,
# not quantity. (V2 had no truncation; SBERT V7 showed untruncated paragraph
# massively outperformed principle because it had 6x more text.)
def truncate_to_match(paragraph, principle):
    """Truncate paragraph to same word count as principle."""
    if not isinstance(paragraph, str) or not isinstance(principle, str):
        return str(paragraph or '')
    target_words = len(str(principle).split())
    para_words = str(paragraph).split()
    if len(para_words) <= target_words:
        return paragraph
    return ' '.join(para_words[:target_words])

df_clean['scrubbed_paragraph_truncated'] = df_clean.apply(
    lambda row: truncate_to_match(row['scrubbed_paragraph'], row['principle']), axis=1)

trunc_lens = df_clean['scrubbed_paragraph_truncated'].str.split().str.len()
print(f"Token-matched truncation applied:")
print(f"  Truncated paragraph words — mean: {trunc_lens.mean():.1f}, median: {trunc_lens.median():.1f}")

# Build query using TRUNCATED paragraph
df_clean['query'] = '[FACT] ' + df_clean['fact'].astype(str) + \
                    ' [PARAGRAPH] ' + df_clean['scrubbed_paragraph_truncated'].astype(str)

# Split by URL — identical to all other models
unique_urls = df_clean['url'].unique()
n           = len(unique_urls)
np.random.seed(42)
shuffled    = np.random.permutation(unique_urls)
train_urls  = set(shuffled[:int(0.8*n)])
val_urls    = set(shuffled[int(0.8*n):int(0.9*n)])
test_urls   = set(shuffled[int(0.9*n):])

train_df = df_clean[df_clean['url'].isin(train_urls)].reset_index(drop=True)
val_df   = df_clean[df_clean['url'].isin(val_urls)].reset_index(drop=True)
test_df  = df_clean[df_clean['url'].isin(test_urls)].reset_index(drop=True)

assert len(set(train_df['url']) & set(val_df['url']))  == 0, "Train/Val URL leakage!"
assert len(set(train_df['url']) & set(test_df['url'])) == 0, "Train/Test URL leakage!"
assert len(set(val_df['url'])   & set(test_df['url'])) == 0, "Val/Test URL leakage!"
print(f"✓ No URL leakage | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================================
# LOAD POOLS
# ============================================================================
print(f"\nLoading candidate pools...")
with open(POOL_PATH,   'r') as f: pools  = json.load(f)
with open(LOOKUP_PATH, 'r') as f: lookup = json.load(f)
id_to_case = {int(k): v for k, v in lookup.items()}
print(f"✓ {len(pools)} pools | {len(id_to_case)} unique cited cases")

# ============================================================================
# TOKENIZER
# ============================================================================
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_base_model():
    from peft import prepare_model_for_kbit_training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    # Force everything onto GPU 0 — with 4-bit quantisation, SaulLM-7B is ~4GB
    # which fits easily on one A100 (80GB). Using device_map="auto" with 1 GPU
    # can cause CPU offloading which bitsandbytes rejects.
    gpu_map = {"": 0}
    print(f"  Loading {MODEL_NAME} with {ATTN_IMPL} (device_map={gpu_map})...")
    t0 = time.time()
    try:
        m = AutoModel.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                      device_map=gpu_map, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation=ATTN_IMPL)
    except Exception as e:
        print(f"  ⚠️ {ATTN_IMPL} failed ({e}), falling back to eager")
        m = AutoModel.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                      device_map=gpu_map, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation="eager")
    print(f"  ✓ Base model loaded in {time.time()-t0:.1f}s")
    return prepare_model_for_kbit_training(m)

# ============================================================================
# DUAL ENCODER — last-token pooling (decoder architecture)
# ============================================================================
class SaulLMDualEncoder(nn.Module):
    def __init__(self, base_model, embedding_dim=4096, temp_init=0.07):
        super().__init__()
        self.base_model  = base_model
        first_dev        = next(iter(base_model.hf_device_map.values())) \
                           if hasattr(base_model, 'hf_device_map') else device
        self.projection  = nn.Linear(embedding_dim, embedding_dim).to(first_dev)
        self.temperature = nn.Parameter(torch.tensor(temp_init))

    def get_last_token_embedding(self, hidden_states, attention_mask):
        bs  = hidden_states.size(0)
        seq = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(bs, device=hidden_states.device), seq]

    def encode(self, input_ids, attention_mask):
        first_dev      = next(self.base_model.parameters()).device
        input_ids      = input_ids.to(first_dev)
        attention_mask = attention_mask.to(first_dev)
        out    = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True, return_dict=True)
        pooled = self.get_last_token_embedding(out.hidden_states[-1], attention_mask)
        pooled = pooled.to(self.projection.weight.device)
        return F.normalize(self.projection(pooled.float()), p=2, dim=-1)

    def forward(self, q_ids, q_mask, c_ids, c_mask):
        q_emb  = self.encode(q_ids, q_mask)
        c_emb  = self.encode(c_ids, c_mask)
        temp   = self.temperature.abs().clamp(min=1e-4)
        sim    = torch.matmul(q_emb, c_emb.T) / temp
        labels = torch.arange(sim.size(0), device=sim.device)
        loss   = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss, q_emb, c_emb

# ============================================================================
# ZERO-SHOT ENCODER — no LoRA, no projection (Kai Dong fix)
# Raw pretrained weights only: last-token pool → L2-normalise.
# LoRA and projection are fine-tuned-only components; including randomly
# initialised versions during zero-shot distorts the embedding geometry.
# ============================================================================
class SaulLMZeroShot(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def get_last_token_embedding(self, hidden_states, attention_mask):
        bs  = hidden_states.size(0)
        seq = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(bs, device=hidden_states.device), seq]

    def encode(self, input_ids, attention_mask):
        first_dev      = next(self.base_model.parameters()).device
        input_ids      = input_ids.to(first_dev)
        attention_mask = attention_mask.to(first_dev)
        out    = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True, return_dict=True)
        pooled = self.get_last_token_embedding(out.hidden_states[-1], attention_mask)
        return F.normalize(pooled.float(), p=2, dim=-1)   # no projection

# ============================================================================
# DATASET
# ============================================================================
class CitationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        # Query: [FACT] fact [PRINCIPLE] principle
        q_enc = self.tokenizer(str(row['query']),      truncation=True,
                               max_length=self.max_len, padding='max_length',
                               return_tensors='pt')
        c_enc = self.tokenizer(str(row['cited_case']), truncation=True,
                               max_length=self.max_len, padding='max_length',
                               return_tensors='pt')
        return {
            'q_ids':  q_enc['input_ids'].squeeze(0),
            'q_mask': q_enc['attention_mask'].squeeze(0),
            'c_ids':  c_enc['input_ids'].squeeze(0),
            'c_mask': c_enc['attention_mask'].squeeze(0),
        }

# ============================================================================
# POOL EVALUATION
# ============================================================================
def compute_pool_metrics(model, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Query: [FACT] fact [PARAGRAPH] scrubbed paragraph")
    print(f"  Pool: 1 correct + 999 random | 9979 pools")
    print(f"{'='*60}")

    pool_ids        = list(pools.keys())
    unique_queries  = list(set(
        '[FACT] ' + pools[fid]['fact_text'] + ' [PARAGRAPH] ' + truncate_to_match(pools[fid]['scrubbed_paragraph_text'], pools[fid]['principle_text'])
        for fid in pool_ids
    ))
    query_to_idx    = {q: i for i, q in enumerate(unique_queries)}
    needed_case_ids = sorted(set(cid for p in pools.values() for cid in p['pool']))
    cid_to_row      = {cid: i for i, cid in enumerate(needed_case_ids)}
    case_texts      = [id_to_case[cid] for cid in needed_case_ids]

    def encode_texts(texts):
        all_embs  = []
        model.eval()
        first_dev = next(model.base_model.parameters()).device
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), ENCODE_BATCH),
                          desc="    Encoding", leave=False):
                batch = texts[i:i+ENCODE_BATCH]
                enc   = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                                  padding='max_length', return_tensors='pt')
                emb   = model.encode(enc['input_ids'].to(first_dev),
                                     enc['attention_mask'].to(first_dev))
                all_embs.append(emb.cpu().float().numpy())
                if i % (ENCODE_BATCH*20) == 0: torch.cuda.empty_cache()
        return np.vstack(all_embs)

    print(f"  Encoding {len(unique_queries)} unique queries...")
    query_embs = encode_texts(unique_queries)
    print(f"  Encoding {len(case_texts)} case candidates...")
    case_embs  = encode_texts(case_texts)

    mrr_list, recalls, precs = [], {k: [] for k in TOP_K}, {k: [] for k in TOP_K}

    for fid in pool_ids:
        p          = pools[fid]
        correct_id = p['correct_case_id']
        p_ids      = p['pool']
        query_str  = '[FACT] ' + p['fact_text'] + ' [PARAGRAPH] ' + truncate_to_match(p['scrubbed_paragraph_text'], p['principle_text'])
        rows       = [cid_to_row[cid] for cid in p_ids]
        sims       = np.dot(case_embs[rows], query_embs[query_to_idx[query_str]])
        ranked     = [p_ids[r] for r in np.argsort(sims)[::-1]]

        for rank, cid in enumerate(ranked, 1):
            if cid == correct_id:
                mrr_list.append(1.0/rank); break
        else:
            mrr_list.append(0.0)

        for k in TOP_K:
            hit = int(correct_id in ranked[:k])
            recalls[k].append(hit)
            precs[k].append(hit/k)

    mrr = np.mean(mrr_list)
    print(f"\n  MRR: {mrr:.4f}")
    print(f"  {'K':<5} {'R@K':<10} {'P@K'}")
    for k in TOP_K:
        print(f"  {k:<5} {np.mean(recalls[k]):<10.4f} {np.mean(precs[k]):.4f}")
    print(f"{'='*60}")

    return {
        'Model': label, 'MRR': round(mrr, 4), 'MAP': round(mrr, 4),
        **{f'R@{k}': round(np.mean(recalls[k]), 4) for k in TOP_K},
        **{f'P@{k}': round(np.mean(precs[k]),   4) for k in TOP_K},
    }

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    first_dev  = next(model.base_model.parameters()).device
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for batch in pbar:
        q_ids  = batch['q_ids'].to(first_dev)
        q_mask = batch['q_mask'].to(first_dev)
        c_ids  = batch['c_ids'].to(first_dev)
        c_mask = batch['c_mask'].to(first_dev)
        loss, _, _ = model(q_ids, q_mask, c_ids, c_mask)
        if loss.dim() > 0: loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'temp': f'{model.temperature.item():.4f}'})
        del q_ids, q_mask, c_ids, c_mask, loss
    torch.cuda.empty_cache()
    return total_loss / len(loader)

def get_val_loss(model, loader, epoch):
    model.eval()
    total_loss = 0.0
    first_dev  = next(model.base_model.parameters()).device
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for batch in pbar:
            q_ids  = batch['q_ids'].to(first_dev)
            q_mask = batch['q_mask'].to(first_dev)
            c_ids  = batch['c_ids'].to(first_dev)
            c_mask = batch['c_mask'].to(first_dev)
            loss, _, _ = model(q_ids, q_mask, c_ids, c_mask)
            if loss.dim() > 0: loss = loss.mean()
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    torch.cuda.empty_cache()
    return total_loss / len(loader)

# ════════════════════════════════════════════════════════════
# PART 1: ZERO-SHOT EVALUATION
# No LoRA, no projection — raw pretrained weights only (v2 fix)
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  PART 1: Zero-Shot Evaluation (no LoRA, no projection)")
print(f"{'='*60}")

zs_base  = load_base_model()
zs_model = SaulLMZeroShot(zs_base)
print("✓ Zero-shot model ready (bare pretrained weights, last-token pool → L2-norm)")

all_results = []
all_results.append(compute_pool_metrics(zs_model, "SaulLM Zero-Shot — Paragraph"))

del zs_model, zs_base
torch.cuda.empty_cache()
import gc; gc.collect()

# ════════════════════════════════════════════════════════════
# PART 2: FINE-TUNING
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  PART 2: Fine-Tuning")
print(f"{'='*60}")

from peft import LoraConfig, get_peft_model

ft_base  = load_base_model()
ft_base  = get_peft_model(ft_base, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="FEATURE_EXTRACTION",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"]))
model = SaulLMDualEncoder(ft_base, embedding_dim=EMBEDDING_DIM, temp_init=TEMP_INIT)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {total_params:,}")
print(f"  Temperature init: {model.temperature.item():.4f} (learnable)")

train_ds     = CitationDataset(train_df, tokenizer, MAX_LENGTH)
val_ds       = CitationDataset(val_df,   tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
optimizer    = AdamW([p for p in model.parameters() if p.requires_grad],
                     lr=LEARNING_RATE, weight_decay=0.01)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
print(f"  Total steps: {total_steps} | Warmup steps: {warmup_steps}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

best_val_loss   = float('inf')
best_epoch      = 0
no_improve      = 0       # early stopping counter

for epoch in range(EPOCHS):
    try:
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        vl_loss = get_val_loss(model, val_loader, epoch)
        print(f"Epoch {epoch+1}: Train Loss={tr_loss:.4f} | Val Loss={vl_loss:.4f} "
              f"| Temp={model.temperature.item():.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch    = epoch + 1
            no_improve    = 0
            model.base_model.save_pretrained(f"{OUTPUT_DIR}/best_lora")
            torch.save(model.projection.state_dict(), f"{OUTPUT_DIR}/best_projection.pt")
            torch.save({'temperature': model.temperature.data}, f"{OUTPUT_DIR}/best_temperature.pt")
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP})")
            if no_improve >= EARLY_STOP:
                print(f"\n  ⏹ Early stopping triggered after epoch {epoch+1} "
                      f"(no improvement for {EARLY_STOP} epochs)")
                break
    except Exception as e:
        print(f"⚠️ Epoch {epoch+1} error: {e}")
        import traceback; traceback.print_exc()
        print(f"  best_val_loss so far: {best_val_loss}")
        print(f"  best_epoch so far: {best_epoch}")
        try:
            model.base_model.save_pretrained(f"{OUTPUT_DIR}/emergency_epoch{epoch+1}")
            print(f"  ✓ Emergency checkpoint saved")
        except Exception as save_err:
            print(f"  ✗ Emergency save also failed: {save_err}")
        continue

print(f"\nTraining complete. Best epoch: {best_epoch} | val_loss={best_val_loss:.4f}")

# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION — load best checkpoint
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  PART 3: Fine-Tuned Evaluation")
print(f"  Loading best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f})")
print(f"{'='*60}")

best_lora_path = f"{OUTPUT_DIR}/best_lora"
if not os.path.exists(os.path.join(best_lora_path, "adapter_config.json")):
    print(f"\n⚠️ WARNING: No best_lora checkpoint found at {best_lora_path}")
    print(f"  This likely means training crashed before saving any checkpoint.")
    print(f"  Checking for emergency checkpoints...")
    
    # Look for any emergency checkpoint
    emergency_paths = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith("emergency_")])
    if emergency_paths:
        best_lora_path = f"{OUTPUT_DIR}/{emergency_paths[-1]}"
        print(f"  Found emergency checkpoint: {best_lora_path}")
    else:
        print(f"  No checkpoints found at all. Skipping fine-tuned evaluation.")
        print(f"\n  ZERO-SHOT RESULTS ONLY:")
        print(f"  MRR: (see zero-shot results above)")
        print(f"\n{'='*60}")
        print(f"Finished (zero-shot only — no fine-tuned model available)")
        print(f"{'='*60}")
        import sys; sys.exit(0)

from peft import PeftModel
ft_eval_base  = load_base_model()
ft_eval_base  = PeftModel.from_pretrained(ft_eval_base, best_lora_path)
ft_eval_model = SaulLMDualEncoder(ft_eval_base, embedding_dim=EMBEDDING_DIM, temp_init=TEMP_INIT)
ft_eval_model.projection.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_projection.pt"))
temp_state = torch.load(f"{OUTPUT_DIR}/best_temperature.pt")
ft_eval_model.temperature.data = temp_state['temperature']
print(f"✓ Best model loaded (temp={ft_eval_model.temperature.item():.4f})")

all_results.append(compute_pool_metrics(ft_eval_model, "SaulLM Fine-Tuned — Paragraph"))

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print("  SAULLM PARAGRAPH ABLATION V5 FINAL RESULTS")
print("  Fact + Scrubbed Paragraph → Cited Case | Pool: 1 correct + 999 random | 9979 pools")
print(f"{'='*60}")
summary = pd.DataFrame(all_results).set_index('Model')
print(summary.to_string())
summary.to_csv(f"{OUTPUT_DIR}/saullm_paragraph_v5_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/saullm_paragraph_v5_results.csv")
