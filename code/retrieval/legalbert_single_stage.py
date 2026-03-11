# ============================================================
# SG-LegalCite Single-Stage
# Task: [FACT] Fact [PRINCIPLE] Ground Truth Principle → Cited Case
# Zero-Shot + Fine-Tuned Legal BERT
# Pool: 1 correct case + 999 random negatives
# 7358 pools — one per (fact, principle) pair
# Ground truth principles used directly (no Stage 1 prediction)
# ============================================================

import os
import subprocess
import sys

print("Installing required packages...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'transformers==4.42.4', 'numpy<2.0', 'accelerate', '-q'])
print("✓ Packages installed")

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

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ── CONFIG ────────────────────────────────────────────────────
MODEL_NAME    = "nlpaueb/legal-bert-base-uncased"
POOL_PATH     = "./stage2_single_stage_pools.json"
LOOKUP_PATH   = "./stage2_case_lookup.json"
OUTPUT_DIR    = "./citation_rec_single_stage"
MAX_LENGTH    = 512
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
print("\nLoading single-stage candidate pools...")
with open(POOL_PATH,   'r') as f: pools    = json.load(f)
with open(LOOKUP_PATH, 'r') as f: lookup   = json.load(f)
id_to_case = {int(k): v for k, v in lookup.items()}
print(f"✓ Loaded {len(pools)} pools")
print(f"✓ Loaded {len(id_to_case)} unique cited cases")


# ── MODEL ─────────────────────────────────────────────────────
class LegalBERTDualEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder     = AutoModel.from_pretrained(model_name)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings    = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / \
               torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, input_ids, attention_mask):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, query_input_ids, query_attention_mask,
                candidate_input_ids, candidate_attention_mask):
        q_emb  = self.encode(query_input_ids, query_attention_mask)
        c_emb  = self.encode(candidate_input_ids, candidate_attention_mask)
        sim    = torch.matmul(q_emb, c_emb.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        loss   = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss, q_emb, c_emb


# ── DATASET ───────────────────────────────────────────────────
class SingleStageDataset(Dataset):
    """Training: [FACT] Fact [PRINCIPLE] Ground Truth Principle → Cited Case"""
    def __init__(self, df, tokenizer, max_length=512):
        self.df         = df
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        query_text = f"[FACT] {row['fact']} [PRINCIPLE] {row['principle']}"
        query_enc  = self.tokenizer(query_text, truncation=True, max_length=self.max_length,
                                    padding='max_length', return_tensors='pt')
        case_enc   = self.tokenizer(row['cited_case'], truncation=True, max_length=self.max_length,
                                    padding='max_length', return_tensors='pt')
        return {
            'query_input_ids':          query_enc['input_ids'].squeeze(),
            'query_attention_mask':     query_enc['attention_mask'].squeeze(),
            'candidate_input_ids':      case_enc['input_ids'].squeeze(),
            'candidate_attention_mask': case_enc['attention_mask'].squeeze(),
        }


# ── ENCODE HELPER ─────────────────────────────────────────────
def encode_texts(model, tokenizer, texts, batch_size=64, max_length=512):
    all_embeddings = []
    encode_model   = model.module if hasattr(model, 'module') else model
    encode_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="    Encoding", leave=False):
            batch = texts[i:i+batch_size]
            enc   = tokenizer(batch, truncation=True, max_length=max_length,
                              padding='max_length', return_tensors='pt')
            embs  = encode_model.encode(enc['input_ids'].to(device),
                                        enc['attention_mask'].to(device))
            all_embeddings.append(embs.cpu().numpy())
    return np.vstack(all_embeddings)


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
def evaluate_single_stage(model, tokenizer, label):
    print(f"\n{'='*60}")
    print(f"  {label} — Single Stage")
    print(f"  Query: [FACT] Fact [PRINCIPLE] Ground Truth Principle")
    print(f"  Pool: 1 correct + 999 random | 7358 pools")
    print(f"{'='*60}")

    pool_ids    = list(pools.keys())
    query_texts = [pools[fid]['query_text'] for fid in pool_ids]

    needed_case_ids = sorted(set(cid for p in pools.values() for cid in p['pool']))
    case_texts      = [id_to_case[cid] for cid in needed_case_ids]
    cid_to_row      = {cid: idx for idx, cid in enumerate(needed_case_ids)}

    print(f"  Encoding {len(query_texts)} queries...")
    query_embs = encode_texts(model, tokenizer, query_texts)
    print(f"  Encoding {len(case_texts)} case candidates...")
    case_embs  = encode_texts(model, tokenizer, case_texts)

    all_metrics        = {f'R@{k}': [] for k in TOP_K_EVAL}
    all_metrics.update({f'P@{k}':  [] for k in TOP_K_EVAL})
    all_metrics.update({f'F1@{k}': [] for k in TOP_K_EVAL})
    mrr_list, map_list = [], []

    for i, fid in enumerate(pool_ids):
        pool_data  = pools[fid]
        correct_id = pool_data['correct_case_id']
        pool_ids_  = pool_data['pool']
        pool_rows  = [cid_to_row[cid] for cid in pool_ids_]
        pool_embs  = case_embs[pool_rows]
        query_emb  = query_embs[i]

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
print("  PART 1: Zero-Shot Legal BERT — Single Stage")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
zs_model  = LegalBERTDualEncoder(MODEL_NAME).to(device)
if torch.cuda.device_count() > 1:
    zs_model = nn.DataParallel(zs_model)
print("✓ Zero-shot model loaded (no fine-tuning)")

all_results = []
result      = evaluate_single_stage(zs_model, tokenizer, label="Zero-Shot")
all_results.append(result)


# ════════════════════════════════════════════════════════════
# PART 2: FINE-TUNING
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2: Fine-Tuning — Single Stage")
print("  Training: [FACT] Fact [PRINCIPLE] Ground Truth Principle → Case")
print("="*60)

train_dataset = SingleStageDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = SingleStageDataset(val_df,   tokenizer, MAX_LENGTH)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

ft_model     = LegalBERTDualEncoder(MODEL_NAME).to(device)
if torch.cuda.device_count() > 1:
    ft_model = nn.DataParallel(ft_model)

optimizer    = AdamW(ft_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

best_val_loss = float('inf')
history       = []

for epoch in range(EPOCHS):
    ft_model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        optimizer.zero_grad()
        loss, _, _ = ft_model(
            batch['query_input_ids'].to(device),
            batch['query_attention_mask'].to(device),
            batch['candidate_input_ids'].to(device),
            batch['candidate_attention_mask'].to(device)
        )
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)

    ft_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            loss, _, _ = ft_model(
                batch['query_input_ids'].to(device),
                batch['query_attention_mask'].to(device),
                batch['candidate_input_ids'].to(device),
                batch['candidate_attention_mask'].to(device)
            )
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            val_loss += loss.item()

    avg_val = val_loss / len(val_loader)
    history.append({'epoch': epoch+1, 'train_loss': avg_train, 'val_loss': avg_val})
    print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        save_model    = ft_model.module if hasattr(ft_model, 'module') else ft_model
        torch.save(save_model.state_dict(), f"{OUTPUT_DIR}/single_stage_model.pt")
        print(f"  ✓ Best model saved (val_loss={avg_val:.4f})")

pd.DataFrame(history).to_csv(f"{OUTPUT_DIR}/history.csv", index=False)
print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")


# ════════════════════════════════════════════════════════════
# PART 3: FINE-TUNED EVALUATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3: Fine-Tuned Legal BERT — Single Stage")
print("="*60)

eval_model = LegalBERTDualEncoder(MODEL_NAME).to(device)
state_dict = torch.load(f"{OUTPUT_DIR}/single_stage_model.pt", map_location=device)
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
eval_model.load_state_dict(state_dict)
if torch.cuda.device_count() > 1:
    eval_model = nn.DataParallel(eval_model)
print("✓ Fine-tuned model loaded")

result = evaluate_single_stage(eval_model, tokenizer, label="Fine-Tuned")
all_results.append(result)


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  SINGLE STAGE FINAL RESULTS")
print("  Zero-Shot vs Fine-Tuned")
print("  Query: [FACT] Fact [PRINCIPLE] Ground Truth Principle → Case")
print("  Pool: 1 correct + 999 random | 7358 pools")
print("="*60)
summary_df = pd.DataFrame(all_results).set_index('Model')
print(summary_df.to_string())
summary_df.to_csv(f"{OUTPUT_DIR}/single_stage_results.csv")
print(f"\n✓ Results saved to: {OUTPUT_DIR}/single_stage_results.csv")
