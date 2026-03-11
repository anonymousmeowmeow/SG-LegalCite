# ============================================================
# SG-LegalCite Stage 2 Direct Baseline — AdaptLLM law-LLM
# Task: Facts → Cited Case directly (no key principles)
# Model: AdaptLLM/law-LLM (LLaMA-1-7B + legal domain continued pre-training)
# Reference: AdaptLLM: Adapting Large Language Models via Reading Comprehension (ICLR 2024)
# Pool: SAME as all models — stage2_direct_candidate_pools_v2.json
# 9942 pools — one per (fact, cited_case) pair
# Zero-Shot + Fine-Tuned (QLoRA)
#
# NOTE: device_map="auto" is used instead of DataParallel — required for QLoRA
#       (4-bit quantized models cannot be wrapped with nn.DataParallel).
#       All other settings identical to Legal BERT / SBERT / Custom Legal-BERT /
#       Legal-Longformer / Pile-of-Law / SAILER / Legal-en-RoBERTa Large.
# ============================================================

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

import subprocess
import sys
import time

print("=" * 60)
print("INSTALLING REQUIRED PACKAGES")
print("=" * 60)
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install',
    'transformers==4.42.4', 'accelerate', 'sentencepiece',
    'bitsandbytes', 'numpy<2.0', 'peft==0.11.1', '-q'
])

# Try flash-attn (OK if it fails)
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'flash-attn', '--no-build-isolation', '-q'], timeout=300)
    FLASH_ATTN_AVAILABLE = True
    print("✓ Flash Attention 2 installed")
except Exception as e:
    FLASH_ATTN_AVAILABLE = False
    print(f"⚠️ Flash Attention not available, using SDPA: {e}")

print("✓ Packages installed")

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModel, BitsAndBytesConfig,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

ATTN_IMPLEMENTATION = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa"

# ── CONFIG — same as all other models except where noted ──────
MODEL_NAME    = "AdaptLLM/law-LLM"
POOL_PATH     = "./stage2_direct_candidate_pools_v2.json"
LOOKUP_PATH   = "./stage2_case_lookup.json"
OUTPUT_DIR    = "./citation_rec_adaptllm_direct_v1"
MAX_LENGTH    = 512
BATCH_SIZE    = 64        # same as all models
EPOCHS        = 10        # same as all models
LEARNING_RATE = 2e-5      # same as all models
WARMUP_RATIO  = 0.1       # same as all models
EMBEDDING_DIM = 4096      # LLaMA-1-7B hidden size
# QLoRA settings — required for 7B model
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.1
TOP_K         = [1, 5, 10, 20]
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nModel: {MODEL_NAME}")
print(f"Architecture: LLaMA-1-7B + legal domain continued pre-training")
print(f"Fine-tuning: QLoRA (4-bit) + LoRA r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Warmup: {WARMUP_RATIO}")
print(f"Temperature: learnable, init=0.07 (same as all models)")
print(f"Pooling: last-token (decoder-style)")
print(f"device_map=auto (required for QLoRA — cannot use DataParallel with 4-bit)")
print(f"Note: All other settings identical to encoder models for fair comparison")

# ── LOAD DATA ─────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv('COMBINED_ALL_CASES_FINAL_V2.csv', encoding='latin-1', on_bad_lines='skip')
df_clean = df[['URL', 'Extract of Facts', 'Key Principles Illustrated', 'Cited Case']].copy()
df_clean.columns = ['url', 'fact', 'principle', 'cited_case']
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean['fact'].str.len() > 20]
df_clean = df_clean[df_clean['principle'].str.len() > 20]
df_clean = df_clean[df_clean['cited_case'].str.len() > 3]
df_clean = df_clean[~df_clean['fact'].str.contains(
    'ERROR|not available|Insufficient|CONTENT_BLOCKED', case=False, na=False)]
print(f"Clean rows: {len(df_clean)}")

# Split by URL — same as all other models
unique_urls = df_clean['url'].unique()
train_urls, temp_urls = train_test_split(unique_urls, test_size=0.2, random_state=42)
val_urls,   test_urls = train_test_split(temp_urls,  test_size=0.5, random_state=42)
train_df = df_clean[df_clean['url'].isin(train_urls)].reset_index(drop=True)
val_df   = df_clean[df_clean['url'].isin(val_urls)].reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── LOAD POOLS ─────────────────────────────────────────────────
print("\nLoading direct V2 candidate pools (same as all models)...")
with open(POOL_PATH,   'r') as f: pools  = json.load(f)
with open(LOOKUP_PATH, 'r') as f: lookup = json.load(f)
id_to_case = {int(k): v for k, v in lookup.items()}
print(f"✓ Loaded {len(pools)} pools")
print(f"✓ Loaded {len(id_to_case)} unique cited cases")

# ── TOKENIZER ─────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")


# ── MODEL ──────────────────────────────────────────────────────
def load_base_model(adapter_path=None):
    """Load AdaptLLM with QLoRA. Optionally load saved LoRA adapter."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"  Loading base model: {MODEL_NAME}")
    print(f"  Attention: {ATTN_IMPLEMENTATION}")
    t0 = time.time()
    try:
        base = AutoModel.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
    except Exception as e:
        print(f"  ⚠️ {ATTN_IMPLEMENTATION} failed, falling back to eager: {e}")
        base = AutoModel.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    print(f"  ✓ Base model loaded in {time.time()-t0:.1f}s")

    base.gradient_checkpointing_enable()
    base = prepare_model_for_kbit_training(base)

    if adapter_path is not None:
        print(f"  Loading LoRA adapter from: {adapter_path}")
        base = PeftModel.from_pretrained(base, adapter_path)
    else:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        base = get_peft_model(base, lora_config)
        print("  ✓ LoRA configured")
        base.print_trainable_parameters()
    return base


class AdaptLLMDualEncoder(nn.Module):
    """Dual encoder using AdaptLLM law-LLM for citation recommendation.
    - Last-token pooling (decoder-style, as in v9)
    - Linear projection head (same as v9)
    - Learnable temperature init=0.07 (same as all other models)
    - QLoRA: base model stays 4-bit; only LoRA adapters + projection are trained
    """
    def __init__(self, base_model, embedding_dim=4096):
        super().__init__()
        self.base_model  = base_model
        self.projection  = nn.Linear(embedding_dim, embedding_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))  # learnable, same as all models
        # Move projection to first GPU
        first_device = next(iter(base_model.hf_device_map.values())) \
            if hasattr(base_model, 'hf_device_map') else device
        self.projection = self.projection.to(first_device)
        print(f"  Projection head on: {first_device}")
        print(f"  Temperature: learnable, init=0.07")

    def get_last_token_embedding(self, hidden_states, attention_mask):
        batch_size       = hidden_states.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(batch_size, device=hidden_states.device),
                             sequence_lengths]

    def encode(self, input_ids, attention_mask):
        # Input to first GPU
        first_device = next(self.base_model.parameters()).device
        input_ids      = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True, return_dict=True)
        pooled = self.get_last_token_embedding(outputs.hidden_states[-1], attention_mask)
        pooled = pooled.to(self.projection.weight.device)
        return F.normalize(self.projection(pooled), p=2, dim=-1)

    def forward(self, query_input_ids, query_attention_mask,
                candidate_input_ids, candidate_attention_mask):
        q_emb  = self.encode(query_input_ids,     query_attention_mask)
        c_emb  = self.encode(candidate_input_ids, candidate_attention_mask)
        sim    = torch.matmul(q_emb, c_emb.T) / self.temperature
        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        loss   = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss, q_emb, c_emb


# ── DATASET ────────────────────────────────────────────────────
class DirectDataset(Dataset):
    """Query: Fact | Candidate: Cited Case name"""
    def __init__(self, df, tokenizer, max_length=512):
        self.pairs      = df[['fact', 'cited_case']].drop_duplicates().reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        def tok(text):
            return self.tokenizer(str(text), truncation=True, max_length=self.max_length,
                                  padding='max_length', return_tensors='pt')
        q = tok(row['fact'])
        c = tok(row['cited_case'])
        return {
            'query_input_ids':          q['input_ids'].squeeze(),
            'query_attention_mask':     q['attention_mask'].squeeze(),
            'candidate_input_ids':      c['input_ids'].squeeze(),
            'candidate_attention_mask': c['attention_mask'].squeeze(),
        }


# ── METRICS ────────────────────────────────────────────────────
def compute_metrics(ranked_ids, correct_id, k_values):
    mrr = 0.0
    for rank, cid in enumerate(ranked_ids, 1):
        if cid == correct_id:
            mrr = 1.0 / rank
            break
    result = {'MRR': mrr, 'MAP': mrr}
    for k in k_values:
        hit              = int(correct_id in ranked_ids[:k])
        result[f'R@{k}'] = hit
        result[f'P@{k}'] = hit / k
        result[f'F1@{k}'] = (2*(hit/k)*hit)/((hit/k)+hit) if hit > 0 else 0.0
    return result


# ── ENCODE HELPER ──────────────────────────────────────────────
def encode_texts(model, texts, batch_size=32):
    """Encode a list of texts to embeddings. Uses smaller batch for 7B model."""
    all_embs = []
    model.eval()
    first_device = next(model.base_model.parameters()).device
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="    Encoding", leave=False):
            batch = texts[i:i+batch_size]
            enc   = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                              padding='max_length', return_tensors='pt')
            ids   = enc['input_ids'].to(first_device)
            mask  = enc['attention_mask'].to(first_device)
            emb   = model.encode(ids, mask)
            all_embs.append(emb.cpu().float().numpy())
            if i % (batch_size * 20) == 0:
                torch.cuda.empty_cache()
    return np.vstack(all_embs)


# ── EVALUATION ON POOLS ────────────────────────────────────────
def evaluate_pools(model, label):
    print(f"\n{'='*60}")
    print(f"  {label} — AdaptLLM Direct Baseline")
    print(f"  Query: Fact | Pool: 1 correct + 999 random | 9942 pools")
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

    all_metrics = {f'R@{k}': [] for k in TOP_K}
    all_metrics.update({f'P@{k}': [] for k in TOP_K})
    all_metrics.update({f'F1@{k}': [] for k in TOP_K})
    mrr_list = []

    for fid in pool_ids:
        pool_data  = pools[fid]
        correct_id = pool_data['correct_case_id']
        pool_ids_  = pool_data['pool']
        pool_rows  = [cid_to_row[cid] for cid in pool_ids_]
        pool_embs  = case_embs[pool_rows]
        query_emb  = fact_embs[fact_to_idx[pool_data['fact_text']]]
        sims       = np.dot(pool_embs, query_emb)
        ranked_ids = [pool_ids_[r] for r in np.argsort(sims)[::-1]]

        m = compute_metrics(ranked_ids, correct_id, TOP_K)
        mrr_list.append(m['MRR'])
        for k in TOP_K:
            all_metrics[f'R@{k}'].append(m[f'R@{k}'])
            all_metrics[f'P@{k}'].append(m[f'P@{k}'])
            all_metrics[f'F1@{k}'].append(m[f'F1@{k}'])

    print(f"\n  MRR: {np.mean(mrr_list):.4f} | MAP: {np.mean(mrr_list):.4f}")
    print(f"  {'K':<5} {'Recall@K':<12} {'Precision@K':<14} {'F1@K'}")
    for k in TOP_K:
        print(f"  {k:<5} {np.mean(all_metrics[f'R@{k}']):<12.4f} "
              f"{np.mean(all_metrics[f'P@{k}']):<14.4f} "
              f"{np.mean(all_metrics[f'F1@{k}']):.4f}")
    print(f"{'='*60}")

    return {
        'Model': label,
        'MRR':  round(np.mean(mrr_list), 4),
        'MAP':  round(np.mean(mrr_list), 4),
        **{f'R@{k}':  round(np.mean(all_metrics[f'R@{k}']),  4) for k in TOP_K},
        **{f'P@{k}':  round(np.mean(all_metrics[f'P@{k}']),  4) for k in TOP_K},
        **{f'F1@{k}': round(np.mean(all_metrics[f'F1@{k}']), 4) for k in TOP_K},
    }


# ════════════════════════════════════════════════════════════
# PART 1: ZERO-SHOT EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 1: Zero-Shot AdaptLLM — Direct Baseline")
print("="*60)

zs_base  = load_base_model(adapter_path=None)
zs_model = AdaptLLMDualEncoder(zs_base, embedding_dim=EMBEDDING_DIM)
print(f"✓ Zero-shot model ready")

all_results = []
result = evaluate_pools(zs_model, "AdaptLLM Zero-Shot")
all_results.append(result)

del zs_model, zs_base
torch.cuda.empty_cache()
import gc; gc.collect()


# ════════════════════════════════════════════════════════════
# PART 2: FINE-TUNING
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2: Fine-Tuning AdaptLLM — Facts → Cited Case")
print("="*60)

train_dataset = DirectDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = DirectDataset(val_df,   tokenizer, MAX_LENGTH)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
print(f"  Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")
print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

ft_base  = load_base_model(adapter_path=None)
ft_model = AdaptLLMDualEncoder(ft_base, embedding_dim=EMBEDDING_DIM)
first_device = next(ft_model.base_model.parameters()).device

# Optimizer — only trainable params (LoRA + projection + temperature)
trainable_params = [p for p in ft_model.parameters() if p.requires_grad]
print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

optimizer    = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
print(f"  Total steps: {total_steps} | Warmup steps: {warmup_steps}")

best_val_loss = float('inf')
history       = []

for epoch in range(EPOCHS):
    # Train
    ft_model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        q_ids  = batch['query_input_ids']
        q_mask = batch['query_attention_mask']
        c_ids  = batch['candidate_input_ids']
        c_mask = batch['candidate_attention_mask']
        optimizer.zero_grad()
        loss, _, _ = ft_model(q_ids, q_mask, c_ids, c_mask)
        if loss.dim() > 0: loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    avg_train = epoch_loss / len(train_loader)
    torch.cuda.empty_cache()

    # Validate
    ft_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            q_ids  = batch['query_input_ids']
            q_mask = batch['query_attention_mask']
            c_ids  = batch['candidate_input_ids']
            c_mask = batch['candidate_attention_mask']
            loss, _, _ = ft_model(q_ids, q_mask, c_ids, c_mask)
            if loss.dim() > 0: loss = loss.mean()
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)
    torch.cuda.empty_cache()

    history.append({'epoch': epoch+1, 'train_loss': avg_train, 'val_loss': avg_val})
    print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        # Save LoRA adapter + projection
        ft_model.base_model.save_pretrained(f"{OUTPUT_DIR}/best_lora")
        torch.save(ft_model.projection.state_dict(), f"{OUTPUT_DIR}/best_projection.pt")
        torch.save({'temperature': ft_model.temperature.data},
                   f"{OUTPUT_DIR}/best_temperature.pt")
        print(f"  ✓ Best model saved (val_loss={avg_val:.4f})")

pd.DataFrame(history).to_csv(f"{OUTPUT_DIR}/history.csv", index=False)
print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")

del ft_model, ft_base
torch.cuda.empty_cache()
gc.collect()


# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3: Fine-Tuned AdaptLLM — Direct Baseline")
print("="*60)

eval_base  = load_base_model(adapter_path=f"{OUTPUT_DIR}/best_lora")
eval_model = AdaptLLMDualEncoder(eval_base, embedding_dim=EMBEDDING_DIM)
# Restore projection + temperature
eval_model.projection.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_projection.pt"))
temp_state = torch.load(f"{OUTPUT_DIR}/best_temperature.pt")
eval_model.temperature.data = temp_state['temperature']
print(f"✓ Fine-tuned model loaded (temp={eval_model.temperature.item():.4f})")

result = evaluate_pools(eval_model, "AdaptLLM Fine-Tuned")
all_results.append(result)


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  ADAPTLLM DIRECT BASELINE FINAL RESULTS")
print("  Zero-Shot vs Fine-Tuned | Facts → Case (no principles)")
print("  Pool: 1 correct + 999 random | 9942 pools (same as all models)")
print("="*60)
summary_df = pd.DataFrame(all_results).set_index('Model')
print(summary_df.to_string())
summary_df.to_csv(f"{OUTPUT_DIR}/adaptllm_direct_v1_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/adaptllm_direct_v1_results.csv")
