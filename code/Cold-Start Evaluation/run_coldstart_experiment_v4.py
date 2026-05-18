"""
SG-LegalCite Cold-Start Experiment — End-to-End Pipeline
=========================================================
Addresses Reviewer 3's "circular dependency" / "oracle principle" concern.

Pipeline:
  1. Load test-set pools from stage2_single_stage_pools.json
  2. Sample 50 pools deterministically (seed=42)
  3. Run contamination checks on fact summaries
  4. For each fact, call DeepSeek-V3 to predict a legal principle
     (using ONLY the fact — no principle, no cited case exposed)
  5. Load fine-tuned SaulLM-7B from ./citation_rec_saullm_single_stage_v1/
  6. Evaluate the SAME model on the SAME 50 pools under two query conditions:
       - Gold:      [FACT] fact [PRINCIPLE] gold_principle
       - Cold-start: [FACT] fact [PRINCIPLE] predicted_principle
  7. Output side-by-side MRR / R@1 / R@5 / R@10 comparison

Why this design:
  - Same model, same pools → only the principle text differs
  - Gold result is the upper bound (oracle setting)
  - Cold-start result is the realistic setting a lawyer would see
  - Gap between the two quantifies the cost of LLM-predicted vs gold principles

Required env vars:
  DEEPSEEK_API_KEY    your DeepSeek key (set as env var; do NOT hardcode)

Required files (in working directory):
  stage2_single_stage_pools.json
  stage2_case_lookup.json
  citation_rec_saullm_single_stage_v1/best_lora/
  citation_rec_saullm_single_stage_v1/best_projection.pt
  citation_rec_saullm_single_stage_v1/best_temperature.pt

Output files:
  coldstart_predictions.json         predicted principles + flags
  coldstart_contamination_log.txt    flagged facts / predictions
  coldstart_results.csv              MRR/R@k for gold vs cold-start
  coldstart_per_pool_results.csv     per-pool ranks for inspection
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING']        = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']     = 'expandable_segments:True'
os.environ['TORCH_USE_CUDA_DSA']          = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']     = '600'
os.environ['HF_ENDPOINT']                 = 'https://hf-mirror.com'

import sys
import re
import json
import time
import random
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════
# FLASH_ATTN + TRANSFORMER_ENGINE BLOCKERS (inherited from saullm_paragraph_v5)
# ════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ENVIRONMENT SETUP")
print("=" * 60)

try:
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'flash-attn', '-y',
                    '--break-system-packages'], capture_output=True, timeout=30)
except: pass

for mod in [k for k in list(sys.modules.keys()) if 'flash_attn' in k]:
    del sys.modules[mod]
sys.path = [p for p in sys.path if 'flash_attn' not in p]

from types import ModuleType
from importlib.machinery import ModuleSpec
import importlib.abc, importlib.util

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
print("✓ flash_attn + transformer_engine blocked")

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

# DeepSeek client (OpenAI-compatible)
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai is required")
    sys.exit(1)

if not torch.cuda.is_available():
    print("*** ERROR: CUDA not available ***")
    sys.exit(1)
device = torch.device('cuda')
print(f"Device: {device}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)")

# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════
# Inputs
POOL_PATH         = "./stage2_single_stage_pools.json"
LOOKUP_PATH       = "./stage2_case_lookup.json"
CHECKPOINT_DIR    = "./citation_rec_saullm_single_stage_v1"

# Outputs
PREDICTIONS_PATH  = "./coldstart_predictions.json"
CONTAMINATION_LOG = "./coldstart_contamination_log.txt"
RESULTS_CSV       = "./coldstart_results.csv"
PER_POOL_CSV      = "./coldstart_per_pool_results.csv"

# Experiment settings
N_SAMPLES         = 50
RANDOM_SEED       = 42

# DeepSeek settings
# IMPORTANT: replace below with your NEW key after revoking the leaked one.
# Do NOT commit this file to git after pasting the key.
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"   # DeepSeek-V3

# SaulLM-7B settings (must match training)
MODEL_NAME        = "Equall/Saul-Instruct-v1"
EMBEDDING_DIM     = 4096
MAX_LENGTH        = 512
ENCODE_BATCH      = 32
TEMP_INIT         = 0.07
ATTN_IMPL         = "sdpa"
TOP_K             = [1, 5, 10, 20]

# Verify API key set
if not DEEPSEEK_API_KEY:
    print("\n*** ERROR: DEEPSEEK_API_KEY not set ***")
    print("Edit run_coldstart_experiment.py and paste your new key on line 170.")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD TEST POOLS + SAMPLE 50
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 1: LOAD POOLS + SAMPLE 50")
print(f"{'=' * 60}")

with open(POOL_PATH, 'r') as f:
    all_pools = json.load(f)
with open(LOOKUP_PATH, 'r') as f:
    lookup = json.load(f)
id_to_case = {int(k): v for k, v in lookup.items()}
print(f"Total test pools: {len(all_pools)}")

random.seed(RANDOM_SEED)
all_keys      = sorted(all_pools.keys())      # sort for determinism
sampled_keys  = random.sample(all_keys, N_SAMPLES)
print(f"Sampled {N_SAMPLES} pool IDs with seed={RANDOM_SEED}")

# ════════════════════════════════════════════════════════════════════════
# STEP 2: CONTAMINATION CHECKS
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 2: CONTAMINATION CHECKS")
print(f"{'=' * 60}")

CITATION_PATTERN = re.compile(
    r'\[\d{4}\]\s*(SG[A-Z]+|UK[A-Z]+|EW[A-Z]+|NSW[A-Z]+|HK[A-Z]+|AC|WLR|SLR|MLJ)\s*\d+',
    re.IGNORECASE)
V_PATTERN = re.compile(r'\b[A-Z][a-zA-Z\-\']+\s+v[\.\s]+[A-Z][a-zA-Z\-\']+')
CITATION_KEYWORDS = re.compile(
    r'\b(SGCA|SGHC|SGHCF|SGHCR|SGCAI|SLR|MLJ|AC|EWCA|EWHC|UKSC|UKHL)\b',
    re.IGNORECASE)

def check_contamination(text, label, log_entries):
    flags = []
    if CITATION_PATTERN.search(text):  flags.append("neutral citation")
    if V_PATTERN.search(text):         flags.append("'Party v Party'")
    if CITATION_KEYWORDS.search(text): flags.append("citation keyword")
    if flags:
        log_entries.append(f"[{label}] FLAGS: {', '.join(flags)}")
        log_entries.append(f"  Text: {text[:300]}")
        log_entries.append("")
        return False
    return True

log_entries = ["=" * 60, "CONTAMINATION LOG", "=" * 60, ""]
flagged_facts = []
for k in sampled_keys:
    fact = all_pools[k]['fact_text']
    if not check_contamination(fact, f"FACT pool_id={k}", log_entries):
        flagged_facts.append(k)
print(f"Facts checked: {N_SAMPLES}")
print(f"Facts flagged for case-reference patterns: {len(flagged_facts)}")
if flagged_facts:
    print(f"⚠️  See {CONTAMINATION_LOG} for details")

# ════════════════════════════════════════════════════════════════════════
# STEP 3: PREDICT PRINCIPLES WITH DEEPSEEK-V3
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 3: PREDICT PRINCIPLES VIA DEEPSEEK-V3")
print(f"{'=' * 60}")

SYSTEM_PROMPT = (
    "You are an expert legal analyst assisting a Singapore lawyer who needs "
    "to find precedent cases. The lawyer has described their client's situation "
    "but has not yet identified the specific legal authority they need."
)

USER_PROMPT_TEMPLATE = """Task: Given a factual description of a case, predict the single most likely legal principle the lawyer would need precedent authority for. Output the principle in full formal legal phrasing as it might appear in a Singapore judgment.

Rules:
- Output exactly ONE legal principle.
- Use formal legal phrasing (e.g., "The duty of care arises when...", "An adverse inference may be drawn where...").
- Focus on the substantive legal proposition, not procedural rules unless procedure is the central issue.
- Do not name specific cases or cite case names.
- Do not include preamble such as "The principle is..." or "Based on these facts...".
- Output the principle directly as a self-contained statement.

Output format: A single sentence or short paragraph stating the legal principle. No additional text.

Facts: {fact_summary}

Legal Principle:"""

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def predict_principle(fact_text, max_retries=3):
    """SAFETY: only fact_text passed — no principle, no cited case."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(fact_summary=fact_text)}
                ],
                temperature=0.0,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1: time.sleep(2 ** attempt)
            else: return None

predictions  = {}
flagged_preds = []
for i, k in enumerate(sampled_keys):
    fact_text = all_pools[k]['fact_text']
    print(f"  [{i+1}/{N_SAMPLES}] pool_id={k}", end='', flush=True)
    predicted = predict_principle(fact_text)
    if predicted is None:
        print("  ❌ FAILED — skipping")
        continue
    is_clean = check_contamination(predicted, f"PREDICTION pool_id={k}", log_entries)
    if not is_clean:
        flagged_preds.append(k)
        print(f"  ⚠️  flagged")
    else:
        print(f"  ✓")
    predictions[k] = {
        "fact_text":                fact_text,
        "gold_principle":           all_pools[k]['principle_text'],
        "predicted_principle":      predicted,
        "correct_case_name":        all_pools[k]['correct_case_name'],
        "prediction_flagged":       not is_clean,
    }

with open(PREDICTIONS_PATH, 'w') as f:
    json.dump(predictions, f, indent=2)
with open(CONTAMINATION_LOG, 'w') as f:
    f.write('\n'.join(log_entries))
print(f"\n✓ Saved predictions    → {PREDICTIONS_PATH}")
print(f"✓ Saved contamination log → {CONTAMINATION_LOG}")
print(f"Predicted: {len(predictions)}/{N_SAMPLES} | flagged: {len(flagged_preds)}")

# Filter to successfully predicted pools only
final_keys = [k for k in sampled_keys if k in predictions]

# ════════════════════════════════════════════════════════════════════════
# STEP 4: LOAD FINE-TUNED SAULLM-7B FROM CHECKPOINT
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 4: LOAD FINE-TUNED SAULLM-7B")
print(f"{'=' * 60}")

# Tokenizer
print(f"Loading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# Base model (4-bit quantised)
print(f"Loading base model with QLoRA (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True)
t0 = time.time()
try:
    base = AutoModel.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL)
except Exception as e:
    print(f"  ⚠️ {ATTN_IMPL} failed: {e}, falling back to eager")
    base = AutoModel.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager")
base = prepare_model_for_kbit_training(base)
print(f"✓ Base model loaded in {time.time()-t0:.1f}s")

# Load LoRA adapter
print(f"Loading LoRA adapter from {CHECKPOINT_DIR}/best_lora...")
base = PeftModel.from_pretrained(base, f"{CHECKPOINT_DIR}/best_lora")
print(f"✓ LoRA adapter loaded")

# Build dual-encoder wrapper (identical to training)
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

model = SaulLMDualEncoder(base, embedding_dim=EMBEDDING_DIM, temp_init=TEMP_INIT)
model.projection.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_projection.pt"))
temp_state = torch.load(f"{CHECKPOINT_DIR}/best_temperature.pt")
model.temperature.data = temp_state['temperature']
model.eval()
print(f"✓ Full model loaded (temp={model.temperature.item():.4f})")

# ════════════════════════════════════════════════════════════════════════
# STEP 5: ENCODE QUERIES + CANDIDATES, COMPUTE METRICS
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 5: EVALUATE FACT-ONLY vs COLD-START vs GOLD")
print(f"{'=' * 60}")

def encode_texts(texts, label=""):
    embs = []
    first_dev = next(model.base_model.parameters()).device
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), ENCODE_BATCH),
                      desc=f"  Encoding {label}", leave=False):
            batch = texts[i:i+ENCODE_BATCH]
            enc   = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                              padding='max_length', return_tensors='pt')
            emb   = model.encode(enc['input_ids'].to(first_dev),
                                 enc['attention_mask'].to(first_dev))
            embs.append(emb.cpu().float().numpy())
            if i % (ENCODE_BATCH * 20) == 0: torch.cuda.empty_cache()
    return np.vstack(embs)

# Build query lists — THREE conditions on the same 50 pools
fact_only_queries = []
gold_queries      = []
coldstart_queries = []
for k in final_keys:
    pool = all_pools[k]
    pred = predictions[k]['predicted_principle']
    fact_only_queries.append(f"[FACT] {pool['fact_text']}")
    gold_queries.append(     f"[FACT] {pool['fact_text']} [PRINCIPLE] {pool['principle_text']}")
    coldstart_queries.append(f"[FACT] {pool['fact_text']} [PRINCIPLE] {pred}")

# Build candidate universe (union of all candidates across the 50 pools)
needed_case_ids = sorted(set(cid for k in final_keys for cid in all_pools[k]['pool']))
cid_to_row      = {cid: i for i, cid in enumerate(needed_case_ids)}
case_texts      = [id_to_case[cid] for cid in needed_case_ids]
print(f"Unique candidate cases across 50 pools: {len(needed_case_ids)}")

# Encode everything once
print("Encoding fact-only queries...")
fact_only_query_embs = encode_texts(fact_only_queries, "fact-only")
print("Encoding gold queries...")
gold_query_embs      = encode_texts(gold_queries, "gold")
print("Encoding cold-start queries...")
coldstart_query_embs = encode_texts(coldstart_queries, "cold-start")
print("Encoding candidates...")
case_embs            = encode_texts(case_texts, "candidates")

# Compute metrics for both settings
def compute_metrics(query_embs, label):
    mrr_list = []
    recalls  = {k: [] for k in TOP_K}
    per_pool_records = []

    for i, k in enumerate(final_keys):
        pool       = all_pools[k]
        correct_id = pool['correct_case_id']
        p_ids      = pool['pool']
        rows       = [cid_to_row[cid] for cid in p_ids]
        sims       = np.dot(case_embs[rows], query_embs[i])
        ranked     = [p_ids[r] for r in np.argsort(sims)[::-1]]

        rank_of_correct = None
        for rank, cid in enumerate(ranked, 1):
            if cid == correct_id:
                rank_of_correct = rank
                mrr_list.append(1.0/rank)
                break
        else:
            mrr_list.append(0.0)
            rank_of_correct = -1

        for tk in TOP_K:
            recalls[tk].append(int(correct_id in ranked[:tk]))

        per_pool_records.append({
            'setting':         label,
            'pool_id':         k,
            'correct_case':    pool['correct_case_name'],
            'rank':            rank_of_correct,
            'reciprocal_rank': 1.0/rank_of_correct if rank_of_correct > 0 else 0.0,
        })

    return {
        'Setting': label,
        'MRR':     round(np.mean(mrr_list), 4),
        **{f'R@{tk}': round(np.mean(recalls[tk]), 4) for tk in TOP_K},
    }, per_pool_records

print("\nComputing metrics for FACT-ONLY setting...")
fo_metrics,   fo_per_pool   = compute_metrics(fact_only_query_embs, "fact_only")
print("Computing metrics for COLD-START setting...")
cs_metrics,   cs_per_pool   = compute_metrics(coldstart_query_embs, "coldstart")
print("Computing metrics for GOLD setting...")
gold_metrics, gold_per_pool = compute_metrics(gold_query_embs,      "gold")

# ════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE + PRINT COMPARISON
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 6: RESULTS")
print(f"{'=' * 60}")

results_df = pd.DataFrame([fo_metrics, cs_metrics, gold_metrics])
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\n{results_df.to_string(index=False)}")
print(f"\n✓ Results saved → {RESULTS_CSV}")

per_pool_df = pd.DataFrame(fo_per_pool + cs_per_pool + gold_per_pool)
per_pool_df.to_csv(PER_POOL_CSV, index=False)
print(f"✓ Per-pool results saved → {PER_POOL_CSV}")

# Comparison summary — three-way
print(f"\n{'─' * 60}")
print("COMPARISON SUMMARY (same 50 pools, same model)")
print(f"{'─' * 60}")
print(f"  {'Metric':<6}  {'fact-only':<12}  {'cold-start':<12}  {'gold':<12}  {'CS vs FO':<12}  {'Gold vs FO':<12}")
print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
for col in ['MRR'] + [f'R@{tk}' for tk in TOP_K]:
    fo_val   = fo_metrics[col]
    cs_val   = cs_metrics[col]
    gold_val = gold_metrics[col]
    cs_vs_fo   = f"{100*(cs_val-fo_val)/fo_val:+.1f}%"     if fo_val > 0 else "n/a"
    gold_vs_fo = f"{100*(gold_val-fo_val)/fo_val:+.1f}%"   if fo_val > 0 else "n/a"
    print(f"  {col:<6}  {fo_val:<12.4f}  {cs_val:<12.4f}  {gold_val:<12.4f}  {cs_vs_fo:<12}  {gold_vs_fo:<12}")

print(f"\n{'=' * 60}")
print("DONE")
print(f"{'=' * 60}")
