# ============================================================
# SG-LegalCite Stage 2 Direct Baseline — Legal-Longformer
# Task: [FACT] Fact + [PRINCIPLE] Ground Truth Principle → Cited Case
# Model: lexlms/legal-longformer-base (~148M params)
# Pre-training: LeXFiles (~19B tokens / 689GB from 6 legal systems)
# Pool: SAME as all models — stage2_single_stage_pools.json
# 9979 pools — one per (fact, principle, cited_case) triple
# Zero-Shot + Fine-Tuned
# ============================================================

import subprocess
import sys

print("Installing required packages...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'transformers==4.42.4', 'numpy<2.0', 'accelerate', '-q'])
print("✓ Packages installed")

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
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

# ── CONFIG — identical to Legal BERT / SBERT / Custom Legal-BERT ──────
MODEL_NAME    = "lexlms/legal-longformer-base"
POOL_PATH     = "./stage2_single_stage_pools.json"
LOOKUP_PATH   = "./stage2_case_lookup.json"
OUTPUT_DIR    = "./citation_rec_longformer_single_stage_v1"
MAX_LENGTH    = 512
BATCH_SIZE    = 64        # same as all models
EPOCHS        = 10        # same as all models
LEARNING_RATE = 2e-5      # same as all models
WARMUP_RATIO  = 0.1       # same as all models
TOP_K         = [1, 5, 10, 20]
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nModel: {MODEL_NAME}")
print(f"Architecture: Longformer-base (12 layers, 768 hidden, ~148M params)")
print(f"Pre-training: LeXFiles (~19B tokens / 689GB from 6 legal systems)")
print(f"Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Warmup: {WARMUP_RATIO}")
print(f"Note: No gradient accumulation — same effective batch as all other models (64)")

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

# Split by URL — same as Legal BERT, SBERT, Custom Legal-BERT
unique_urls = df_clean['url'].unique()
train_urls, temp_urls = train_test_split(unique_urls, test_size=0.2, random_state=42)
val_urls,   test_urls = train_test_split(temp_urls,  test_size=0.5, random_state=42)
train_df = df_clean[df_clean['url'].isin(train_urls)].reset_index(drop=True)
val_df   = df_clean[df_clean['url'].isin(val_urls)].reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── LOAD POOLS ─────────────────────────────────────────────────
print("\nLoading direct V2 candidate pools (same as all models)...")
with open(POOL_PATH,   'r') as f: pools    = json.load(f)
with open(LOOKUP_PATH, 'r') as f: lookup   = json.load(f)
id_to_case = {int(k): v for k, v in lookup.items()}
print(f"✓ Loaded {len(pools)} pools")
print(f"✓ Loaded {len(id_to_case)} unique cited cases")


# ── MODEL ──────────────────────────────────────────────────────
class LegalLongformerDualEncoder(nn.Module):
    """Dual encoder using Legal-Longformer.
    Key difference from BERT models: requires global_attention_mask for CLS token.
    """
    def __init__(self, model_name, pooling='mean'):
        super().__init__()
        self.encoder     = AutoModel.from_pretrained(model_name)
        self.pooling     = pooling
        self.temperature = nn.Parameter(torch.tensor(0.07))
        print(f"  Pooling: {pooling}")
        print(f"  Hidden size: {self.encoder.config.hidden_size}")
        print(f"  Num layers: {self.encoder.config.num_hidden_layers}")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings    = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def encode(self, input_ids, attention_mask):
        # Longformer requires global_attention_mask — global attention on CLS token
        global_attention_mask         = torch.zeros_like(input_ids)
        global_attention_mask[:, 0]   = 1  # CLS token gets global attention
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        if self.pooling == 'mean':
            embeddings = self.mean_pooling(outputs, attention_mask)
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, query_input_ids, query_attention_mask,
                candidate_input_ids, candidate_attention_mask):
        q_emb  = self.encode(query_input_ids,     query_attention_mask)
        c_emb  = self.encode(candidate_input_ids, candidate_attention_mask)
        sim    = torch.matmul(q_emb, c_emb.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        loss   = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss, q_emb, c_emb


# ── DATASET ────────────────────────────────────────────────────
class DirectDataset(Dataset):
    """Query: [FACT] Fact [PRINCIPLE] Principle | Candidate: Cited Case name"""
    def __init__(self, df, tokenizer, max_length=512):
        self.pairs      = df[['fact', 'principle', 'cited_case']].drop_duplicates().reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        def tok(text):
            return self.tokenizer(text, truncation=True, max_length=self.max_length,
                                  padding='max_length', return_tensors='pt')
        q = tok(f"[FACT] {row['fact']} [PRINCIPLE] {row['principle']}")
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
def encode_texts(model, tokenizer, texts, batch_size=128):
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="    Encoding", leave=False):
            batch = texts[i:i+batch_size]
            enc   = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                              padding='max_length', return_tensors='pt')
            ids   = enc['input_ids'].to(device)
            mask  = enc['attention_mask'].to(device)
            if hasattr(model, 'module'):
                emb = model.module.encode(ids, mask)
            else:
                emb = model.encode(ids, mask)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


# ── EVALUATION ON POOLS ────────────────────────────────────────
def evaluate_pools(model, tokenizer, label):
    print(f"\n{'='*60}")
    print(f"  {label} — Legal-Longformer Single Stage")
    print(f"  Query: [FACT]+[PRINCIPLE] | Pool: 1 correct + 999 random | 9979 pools")
    print(f"{'='*60}")

    pool_ids     = list(pools.keys())
    unique_queries = list(set(pools[fid]['query_text'] for fid in pool_ids))
    query_to_idx   = {q: i for i, q in enumerate(unique_queries)}

    print(f"  Encoding {len(unique_queries)} unique queries (fact + principle)...")
    fact_embs = encode_texts(model, tokenizer, unique_queries)

    needed_case_ids = sorted(set(cid for p in pools.values() for cid in p['pool']))
    case_texts      = [id_to_case[cid] for cid in needed_case_ids]
    cid_to_row      = {cid: idx for idx, cid in enumerate(needed_case_ids)}
    print(f"  Encoding {len(case_texts)} case candidates...")
    case_embs = encode_texts(model, tokenizer, case_texts)

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
        query_emb  = fact_embs[query_to_idx[pool_data['query_text']]]
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
print("  PART 1: Zero-Shot Legal-Longformer — Direct Baseline")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"✓ Tokenizer loaded | Vocab size: {tokenizer.vocab_size}")

zs_model = LegalLongformerDualEncoder(MODEL_NAME, pooling='mean')
if torch.cuda.device_count() > 1:
    zs_model = nn.DataParallel(zs_model)
zs_model = zs_model.to(device)
print(f"✓ Zero-shot model loaded: {MODEL_NAME}")

all_results = []
result = evaluate_pools(zs_model, tokenizer, "Legal-Longformer Zero-Shot")
all_results.append(result)
del zs_model
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# PART 2: FINE-TUNING
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2: Fine-Tuning Legal-Longformer — Facts → Cited Case")
print("="*60)

train_dataset = DirectDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = DirectDataset(val_df,   tokenizer, MAX_LENGTH)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
print(f"  Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")
print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

ft_model = LegalLongformerDualEncoder(MODEL_NAME, pooling='mean')
if torch.cuda.device_count() > 1:
    print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel!")
    ft_model = nn.DataParallel(ft_model)
ft_model = ft_model.to(device)

optimizer    = AdamW(ft_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
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
        q_ids  = batch['query_input_ids'].to(device)
        q_mask = batch['query_attention_mask'].to(device)
        c_ids  = batch['candidate_input_ids'].to(device)
        c_mask = batch['candidate_attention_mask'].to(device)
        optimizer.zero_grad()
        loss, _, _ = ft_model(q_ids, q_mask, c_ids, c_mask)
        if loss.dim() > 0: loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    avg_train = epoch_loss / len(train_loader)

    # Validate
    ft_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            q_ids  = batch['query_input_ids'].to(device)
            q_mask = batch['query_attention_mask'].to(device)
            c_ids  = batch['candidate_input_ids'].to(device)
            c_mask = batch['candidate_attention_mask'].to(device)
            loss, _, _ = ft_model(q_ids, q_mask, c_ids, c_mask)
            if loss.dim() > 0: loss = loss.mean()
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)

    history.append({'epoch': epoch+1, 'train_loss': avg_train, 'val_loss': avg_val})
    print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(ft_model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
        print(f"  ✓ Best model saved (val_loss={avg_val:.4f})")

pd.DataFrame(history).to_csv(f"{OUTPUT_DIR}/history.csv", index=False)
print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")


# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3: Fine-Tuned Legal-Longformer — Direct Baseline")
print("="*60)

eval_model = LegalLongformerDualEncoder(MODEL_NAME, pooling='mean')
if torch.cuda.device_count() > 1:
    eval_model = nn.DataParallel(eval_model)
eval_model = eval_model.to(device)
eval_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt"))
print("✓ Fine-tuned model loaded")

result = evaluate_pools(eval_model, tokenizer, "Legal-Longformer Fine-Tuned")
all_results.append(result)


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  LEGAL-LONGFORMER SINGLE STAGE FINAL RESULTS")
print("  Zero-Shot vs Fine-Tuned | Fact + GT Principle → Case")
print("  Pool: 1 correct + 999 random | 9979 pools (same as all models)")
print("="*60)
summary_df = pd.DataFrame(all_results).set_index('Model')
print(summary_df.to_string())
summary_df.to_csv(f"{OUTPUT_DIR}/longformer_single_stage_v1_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/longformer_single_stage_v1_results.csv")
