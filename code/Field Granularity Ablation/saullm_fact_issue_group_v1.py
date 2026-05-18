"""
SG-LegalCite Stage 2 All Fields — SaulLM-7B (Mistral-based) v4
Task:   [FACT] Extract of Facts + [ISSUE_GROUP] Issue Group
        → Cited Case
Pool:   stage2_fact_issue_group_pools_v1.json (9979 pools)
Split:  By URL (same as ALL other models for fair comparison)

v9 change: PYTHONPATH points to ~/shannon/packages (the actual local package cache, Python 3.12 compatible).

Training setup IDENTICAL to single-stage v3 — only the query construction differs.

Ablation purpose:
  Tests whether Issue Group alone (coarsest doctrinal category) achieves similar retrieval
  gain compared to fact+principle (SaulLM single-stage = 38.2 MRR).
  Compared against single-stage (fact+principle) on identical pools.

Decoder-specific (necessary differences, documented):
  QLoRA: 4-bit NF4 + LoRA (r=16, alpha=32, dropout=0.1)
  device_map="auto" (multi-GPU sharding)
  Last-token pooling + linear projection head (fine-tuned only)
  Gradient checkpointing

Zero-shot evaluation (v2 fix — Kai Dong):
  No LoRA adapters, no projection head.
  Raw pretrained weights only: last-token pool → L2-normalise.
  This ensures zero-shot reflects the model's true pretrained
  embedding geometry, consistent with encoder zero-shot runs.

SaulLM-specific (inherited from single-stage v3):
  flash_attn mock + blocker (broken system library)
  ATTN_IMPL="sdpa" (Mistral-based, no rope_scaling patch needed)
  No tokenizer fallback needed (SaulLM tokenizer works)

To run: qsub run_saullm_fact_issue_group_v1.pbs
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 1 A100 GPU (run alongside saullm_fact_issue_v1)

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
# INSTALL PACKAGES
# ============================================================================
# v12: SKIP pip install — packages already on disk in ~/.local
# (compute nodes have no internet anyway, so install was a no-op
#  except for accidentally pulling in incompatible triton/torch versions)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)")

ATTN_IMPL = "sdpa"

# ============================================================================
# CONFIG — IDENTICAL to single-stage v3, only POOL_PATH and OUTPUT_DIR differ
# ============================================================================
MODEL_NAME     = "Equall/Saul-Instruct-v1"
CSV_PATH       = "./COMBINED_ALL_CASES_FINAL_V2.csv"
POOL_PATH      = "./stage2_fact_issue_group_pools_v1.json"
LOOKUP_PATH    = "./stage2_case_lookup.json"
OUTPUT_DIR     = "./citation_rec_saullm_fact_issue_group_v1"

# --- Identical to ALL other models ---
BATCH_SIZE     = 64
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
print(f"SaulLM-7B Fact+Issue Group v1 — [FACT] [ISSUE_GROUP] → Cited Case")
print(f"{'='*60}")
print(f"Model:      {MODEL_NAME}")
print(f"Pool:        stage2_fact_issue_group_pools_v1.json (9979)")
print(f"Split:      by URL (identical to all models)")
print(f"Batch:      {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LEARNING_RATE}")
print(f"Warmup:     {WARMUP_RATIO} | Temp: learnable init={TEMP_INIT}")
print(f"Early Stop: patience={EARLY_STOP} (stops if no val_loss improvement)")
print(f"QLoRA:      r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Attention:  {ATTN_IMPL}")
print(f"GPUs:       1x A100 (run concurrently with fact+issue) (v10 — PYTHONPATH chains .local + shannon/packages)")

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\nLoading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, encoding='latin-1', on_bad_lines='skip')
print(f"Raw rows: {len(df)}")

df_clean = df[['URL', 'Extract of Facts', 'Issue Group', 'Issue',
               'Key Principles Illustrated', 'Cited Case']].copy()
df_clean.columns = ['url', 'fact', 'issue_group', 'issue', 'principle', 'cited_case']
df_clean = df_clean.dropna(subset=['fact', 'issue_group', 'issue', 'principle', 'cited_case', 'url'])
df_clean = df_clean[df_clean['fact'].str.len() > 20]
df_clean = df_clean[df_clean['issue_group'].str.len() > 2]
df_clean = df_clean[df_clean['issue'].str.len() > 5]
df_clean = df_clean[df_clean['principle'].str.len() > 20]
df_clean = df_clean[df_clean['cited_case'].str.len() > 3]
df_clean = df_clean[~df_clean['fact'].str.contains(
    'ERROR|not available|Insufficient|CONTENT_BLOCKED', case=False, na=False)]
print(f"Clean rows: {len(df_clean)}")

# Build all-fields query: [FACT] {fact} [ISSUE_GROUP] {issue_group} [ISSUE] {issue} [PRINCIPLE] {principle}
df_clean['query'] = ('[FACT] '         + df_clean['fact'].astype(str) +
                     ' [ISSUE_GROUP] ' + df_clean['issue_group'].astype(str))

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
    print(f"  Loading {MODEL_NAME} with {ATTN_IMPL}...")
    t0 = time.time()
    try:
        m = AutoModel.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                      device_map="auto", trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation=ATTN_IMPL)
    except Exception as e:
        print(f"  ⚠️ {ATTN_IMPL} failed ({e}), falling back to eager")
        m = AutoModel.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                      device_map="auto", trust_remote_code=True,
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
        # Query: [FACT] fact [ISSUE_GROUP] issue_group
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
    print(f"  Query: [FACT] fact [ISSUE_GROUP] issue_group")
    print(f"  Pool: 1 correct + 999 random | 9979 pools")
    print(f"{'='*60}")

    pool_ids        = list(pools.keys())
    unique_queries  = list(set(
        '[FACT] '         + pools[fid]['fact_text'] +
        ' [ISSUE_GROUP] ' + pools[fid]['issue_group_text']
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
        query_str  = ('[FACT] '         + p['fact_text'] +
                      ' [ISSUE_GROUP] ' + p['issue_group_text'])
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
all_results.append(compute_pool_metrics(zs_model, "SaulLM Zero-Shot — All Fields"))

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
        try:
            model.base_model.save_pretrained(f"{OUTPUT_DIR}/emergency_epoch{epoch+1}")
            print(f"  ✓ Emergency checkpoint saved")
        except: pass
        continue

print(f"\nTraining complete. Best epoch: {best_epoch} | val_loss={best_val_loss:.4f}")

# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION — load best checkpoint
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  PART 3: Fine-Tuned Evaluation")
print(f"  Loading best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f})")
print(f"{'='*60}")

from peft import PeftModel
ft_eval_base  = load_base_model()
ft_eval_base  = PeftModel.from_pretrained(ft_eval_base, f"{OUTPUT_DIR}/best_lora")
ft_eval_model = SaulLMDualEncoder(ft_eval_base, embedding_dim=EMBEDDING_DIM, temp_init=TEMP_INIT)
ft_eval_model.projection.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_projection.pt"))
temp_state = torch.load(f"{OUTPUT_DIR}/best_temperature.pt")
ft_eval_model.temperature.data = temp_state['temperature']
print(f"✓ Best model loaded (temp={ft_eval_model.temperature.item():.4f})")

all_results.append(compute_pool_metrics(ft_eval_model, "SaulLM Fine-Tuned — All Fields"))

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print("  SAULLM ALL FIELDS v4 FINAL RESULTS")
print("  [FACT] [ISSUE_GROUP] [ISSUE] [PRINCIPLE] → Cited Case")
print("  Pool: 1 correct + 999 random | 9979 pools")
print(f"{'='*60}")
summary = pd.DataFrame(all_results).set_index('Model')
print(summary.to_string())
summary.to_csv(f"{OUTPUT_DIR}/saullm_fact_issue_group_v1_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/saullm_fact_issue_group_v1_results.csv")
