# SG-LegalCite: A Singapore Legal Principle-Case Dataset for Jurisdiction-Aware Citation Recommendation

> **Paper:** *SG-LegalCite: A Singapore Legal Principle-Case Dataset for Jurisdiction-Aware Citation Recommendation*
> 
>
> **Dataset:** [HuggingFace](https://huggingface.co/datasets/anonymousmeowmeow/SG-LegalCite) | **Paper:** [[arXiv / ACL Anthology link]](#)

---

## Overview

SG-LegalCite is the **first legal citation retrieval benchmark for Singapore law** and the **first benchmark with principle-level query annotations** across all existing legal retrieval datasets.

Legal citation recommendation in common-law practice requires retrieving precedents that establish a specific **legal principle** — not merely cases with similar facts. SG-LegalCite operationalises this by formulating retrieval as:

> **[FACT]** *case facts* + **[PRINCIPLE]** *legal principle* → *cited case*

The dataset is extracted from 8,523 Singapore Supreme Court judgments (2000–2025) using a cost-effective LLM pipeline validated by legal experts from NUS Law and SMU Yong Pung How School of Law.

---

## Repository Structure

```
SG-LegalCite/
├── dataset/
│   └── README.md                  # Dataset format, field descriptions, and download link
├── code/
│   ├── extraction/
│   │   ├── 00_Generate_Case_Index.py         # Step 0: Generate master case URL index (2000–2025)
│   │   ├── 01_Extract_Cited_Cases_Batch.py   # Step 1: Extract cited cases + paragraph context
│   │   ├── 02_Deepseek_Chat_Batch.py         # Step 2: Extract Key Principles, Issue, Issue Group
│   │   ├── 03_Fact_Query_Batch.py            # Step 3: Generate lawyer-style Fact_Query summaries
│   │   ├── 04_Final_Concatenation_Batch.py   # Step 4: Add Case Name + Precedential Weight
│   │   ├── prompt_with_paragraphs_FINAL.txt  # 15-shot DeepSeek extraction prompt
│   │   ├── LLM_Selection/
│   │       ├── LLM_Selection_KeyPrinciples_All_Models.ipynb  # LLM comparison evaluation notebook
│   │       ├── LLM_Selection_KeyPrinciples_Results.xlsx      # Accuracy summary (Claude/DeepSeek/GPT-4o)
│   │       ├── Claude Sonnet 4_individual_key_principle_extraction_results/
│   │       ├── DeepSeek-Chat_individual_key_principle_extraction_results/
│   │       └── GPT-4o_individual_key_principle_extraction_results/
│   │   └── Few-Shot Experiments (DeepSeek-Chat)/
│   │       ├── FewShot_Experiments_DeepSeek.ipynb            # Few-shot evaluation notebook
│   │       ├── FewShot_KeyPrinciples_Issue_Results.xlsx      # Accuracy summary (0/5/10/15/20-shot)
│   │       ├── Zero-Shot_individual_issue_extraction_results/
│   │       ├── Zero-Shot_individual_keyprinciple_extraction_results/
│   │       ├── 5-Shot_individual_issue_extraction_results/
│   │       ├── 5-Shot_individual_keyprinciple_extraction_results/
│   │       ├── 10-Shot_individual_issue_extraction_results/
│   │       ├── 10-Shot_individual_keyprinciple_extraction_results/
│   │       ├── 15-Shot_individual_issue_extraction_results/
│   │       ├── 15-Shot_individual_keyprinciple_extraction_results/
│   │       ├── 20-Shot_individual_issue_extraction_results/
│   │       └── 20-Shot_individual_keyprinciple_extraction_results/
│   ├── retrieval/
│   │   ├── generate_stage2_direct_pools_v2.py        # Pool generation: fact-only baseline
│   │   ├── generate_stage2_single_stage_pools_v2.py  # Pool generation: principle-augmented
│   │   ├── sbert_direct_v2v5.py                      # SBERT (fact-only)
│   │   ├── sbert_single_stage_v1.py                  # SBERT (principle-augmented)
│   │   ├── customlegalbert_direct_v1.py               # Custom Legal-BERT (fact-only)
│   │   ├── customlegalbert_single_stage_v1.py        # Custom Legal-BERT (principle-augmented)
│   │   ├── longformer_direct_v1.py                   # Legal-Longformer (fact-only)
│   │   ├── longformer_single_stage_v1.py             # Legal-Longformer (principle-augmented)
│   │   ├── pileoflaw_direct_v1.py                    # Pile-of-Law BERT (fact-only)
│   │   ├── pileoflaw_single_stage_v2.py              # Pile-of-Law BERT (principle-augmented)
│   │   ├── roberta_large_direct_v1.py                # Legal-English-RoBERTa (fact-only)
│   │   ├── roberta_large_single_stage_v1.py          # Legal-English-RoBERTa (principle-augmented)
│   │   ├── sailer_direct_v1.py                       # SAILER (fact-only)
│   │   ├── sailer_single_stage_v1.py                 # SAILER (principle-augmented)
│   │   ├── adaptllm_direct_v1.py                     # AdaptLLM (fact-only)
│   │   ├── adaptllm_single_stage_v2.py               # AdaptLLM (principle-augmented)
│   │   ├── legalbert_stage2_direct_v2.py             # Legal-BERT (fact-only)
│   │   ├── legalbert_single_stage.py                 # Legal-BERT (principle-augmented)
│   │   ├── saullm_direct_v2.py                       # SaulLM-7B (fact-only)
│   │   ├── saullm_single_stage_v1.py                 # SaulLM-7B (principle-augmented)
│   │   ├── lawma_direct_v3.py                        # Lawma-8B (fact-only)
│   │   ├── lawma_single_stage_v2.py                  # Lawma-8B (principle-augmented)
│   │   └── run_*.pbs                                 # PBS job scripts for SUTD HPC cluster
├── img/
│   ├── pipeline.png               # Dataset construction pipeline figure
│   ├── legal_hierarchy.png        # Legal citation conceptual structure
│   └── legal_hierarchy2.png       # Knowledge graph structure
├── .gitignore
└── README.md
```

---

## Dataset Statistics

| Attribute | Value |
|---|---|
| Time Span | 2000–2025 |
| Unique Judgments | 8,494 |
| Principle–Case Pairs | 100,554 |
| Unique Principles | 72,264 |
| Unique Cited Cases | 48,298 |
| Unique Issues | 86,247 |
| Unique Issue Groups | 9,712 |
| Avg. Raw Fact Length | 1,034.4 tokens |
| Avg. Fact Length (post-summary) | 45.1 tokens |
| Avg. Citation Paragraph Length | 1,100.5 tokens |
| Avg. Principle Length | 69.9 tokens |

Each record is a triplet **(f, k, c)**:
- **f** — Factual background of the citing judgment (LLM-summarised to ~45 tokens)
- **k** — Legal principle for which the precedent is cited
- **c** — Cited Singapore Supreme Court case

---

## Dataset Construction Pipeline

![Pipeline](img/pipeline.png)

The pipeline proceeds in five steps:

**Step 0 — Case Index Generation (`00_Generate_Case_Index.py`)**
Probes eLitigation to enumerate valid judgment URLs across all court types (SGHC, SGCA, SGCAI, SGHCF, SGHCR) for 2000–2025. Outputs a master CSV with columns: `Year, Court_Type, Case_Number, URL, Full_Reference`.

**Step 1 — Citation and Context Extraction (`01_Extract_Cited_Cases_Batch.py`)**
Playwright + BeautifulSoup extract cited case names and ±5 surrounding paragraphs from eLitigation HTML. No LLM involvement.

**Step 2 — Citation-Level Principle Extraction (`02_Deepseek_Chat_Batch.py`)**
DeepSeek-Chat (15-shot, T=0) extracts three fields per citation: (1) Key Principles Illustrated, (2) Issue Group, (3) Issue. Uses `prompt_with_paragraphs_FINAL.txt`.

**Step 3 — Judgment-Level Fact Extraction (`03_Fact_Query_Batch.py`)**
Three-tier heading fallback strategy locates the Background/Facts section of each judgment. DeepSeek-Chat (T=0.2, max 512 tokens) compresses the scraped section (~1,034 tokens) into a 2–3 sentence lawyer-style `Fact_Query` (~45 tokens), a 23× compression.

**Step 4 — Final Concatenation (`04_Final_Concatenation_Batch.py`)**
Adds `Case Name` (scraped from eLitigation) and `Current Court Level` (derived from court type). Produces the final dataset CSV.

**Source code:** [`code/extraction/`](code/extraction/)

### LLM Selection (15-shot, n=150 samples)

| Model | Accuracy | Cost/Case |
|---|---|---|
| Claude Sonnet 4 | 91.3% | $0.24 |
| **DeepSeek-Chat** | **86.7%** | **$0.02** |
| GPT-4o | 84.7% | $0.16 |

DeepSeek-Chat was selected for full-scale extraction: 12× cost reduction for a 4.6 pp accuracy difference. Total extraction cost: **SGD 100**.

### Few-Shot Experiments (DeepSeek-Chat)

| Examples | Issue | Key Principles |
|---|---|---|
| 0 (zero-shot) | 74.7% | 86.7% |
| 5 | 76.7% | 84.7% |
| 10 | 79.3% | 85.3% |
| **15** | **80.7%** | **86.7%** |
| 20 | 79.3% | 86.7% |

15-shot standardised for all extractions.

---

## Task Formulation

Citation recommendation is framed as nearest-neighbour retrieval over a Singapore-only candidate pool:

$$c^* = \arg\max_{c \in \mathcal{C}} \, s(q, c)$$

Three query formulations are evaluated under identical conditions:

| Setting | Query | Description |
|---|---|---|
| **Fact-only** (f → c) | `f` | Facts only; mirrors existing benchmarks |
| **Principle-only** (k → c) | `k` | Abstract principle only |
| **Principle-augmented** (f ⊕ k → c) | `[FACT] f [PRINCIPLE] k` | **Proposed formulation** |

---

## Experiments

### 1. Models Evaluated

**Encoder Models**

| Model | Params | Pretraining Corpus |
|---|---|---|
| SBERT | 110M | General (MS-MARCO) |
| Legal-BERT | 110M | 12GB EU/UK/US legal text |
| Custom Legal-BERT | 110M | 37GB US case law |
| Legal-Longformer | 148M | 19GB multi-jurisdictional |
| Pile-of-Law BERT | 340M | 256GB US-focused legal sources |
| SAILER† | 110M | 10M+ Chinese judgments |
| Legal-XLM-LF-base | 208M | 24 EU languages |
| Legal-English-RoBERTa | 337M | LexFiles multi-jurisdictional |

**Decoder Models**

| Model | Params | Pretraining Corpus |
|---|---|---|
| AdaptLLM | 7B | US legal + reading comprehension |
| SaulLM-7B | 7B | US/EU/UK/AU legal data |
| Lawma-8B | 8B | US legal tasks |
| SaulLM-54B | 54B MoE | US/EU/UK/AU + DPO alignment |

> **Jurisdiction gap:** All models were pretrained exclusively on China, US, UK, EU, or Australian legal data — none include Singapore legal text.

### 2. Training Setup

**Source code:** [`code/retrieval/`](code/retrieval/)

Fine-tuning uses symmetric InfoNCE contrastive loss:

$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{k \to c} + \mathcal{L}_{c \to k} \right)$$

**Data split:** 80/10/10 (train/val/test), split by unique judgment URL to prevent data leakage.

**Parameter settings for encoder fine-tuning:**

| Parameter | Base Models (110M) | Large Models (300M+) |
|---|---|---|
| Batch size | 64 | 16–32 (gradient accumulation to effective 32) |
| Learning rate | 2e-5 | 1e-5 |
| Temperature τ | 0.07 (learnable) | 0.07 (learnable) |
| Checkpoint selection | Min. validation loss | Min. validation loss |

**Parameter settings for decoder fine-tuning (QLoRA on A100):**

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Quantisation | 4-bit NF4 |
| Batch size | 8 (gradient accumulation to effective 32) |
| Learning rate | 1e-4 |

### 3. Results

Retrieval performance on SG-LegalCite (1000-way candidate pool, principle-augmented queries):

**Zero-Shot Performance**

| Model | MRR | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| *Encoder Models* | | | | | |
| Legal-BERT | 0.8 | 0.1 | 0.2 | 0.8 | 2.3 |
| SBERT | **2.5** | **0.8** | **2.7** | **4.8** | **7.8** |
| Custom Legal-BERT | 0.8 | 0.1 | 0.2 | 0.8 | 2.2 |
| Legal-Longformer | 0.8 | 0.1 | 0.3 | 0.9 | 2.4 |
| Pile-of-Law BERT | 0.9 | 0.1 | 0.3 | 1.1 | 2.9 |
| SAILER† | 0.8 | 0.1 | 0.6 | 1.1 | 2.2 |
| Legal-en-RoBERTa | 0.8 | 0.1 | 0.3 | 1.2 | 2.8 |
| *Decoder Models* | | | | | |
| AdaptLLM | 0.6 | 0.1 | 0.3 | 0.7 | 1.5 |
| SaulLM-7B | 0.6 | 0.1 | 0.2 | 0.7 | 1.6 |
| Lawma-8B | 0.6 | 0.0 | 0.2 | 0.5 | 1.5 |

**Fine-Tuned Performance (Principle-Augmented, Δ% = relative improvement over fine-tuned fact-only baseline)**

| Model | MRR | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| *Encoder Models* | | | | | |
| Legal-BERT | 14.1 (+128%) | 7.1 (+228%) | 19.6 | 27.5 | 37.5 |
| SBERT | 20.9 (+98%) | 12.9 (+178%) | 27.8 | 36.3 | 45.8 |
| Custom Legal-BERT | 9.2 (+51%) | 4.0 (+84%) | 12.4 | 18.9 | 27.4 |
| Legal-Longformer | 10.0 (+47%) | 4.4 (+70%) | 13.9 | 20.9 | 29.9 |
| Pile-of-Law BERT | 12.4 (+119%) | 5.8 (+193%) | 17.5 | 25.5 | 35.2 |
| SAILER† | 0.9 (+13%) | 0.2 (+150%) | 0.6 | 1.2 | 2.0 |
| Legal-en-RoBERTa | 9.4 (+36%) | 3.8 (+43%) | 13.0 | 20.0 | 29.6 |
| *Decoder Models* | | | | | |
| AdaptLLM | 29.5 (+154%) | 17.9 (+240%) | 41.6 | 53.5 | 64.9 |
| **SaulLM-7B** | **38.2 (+193%)** | **24.4 (+317%)** | **54.2** | **66.7** | **77.2** |
| Lawma-8B | 30.5 (+205%) | 18.2 (+354%) | 43.8 | 56.6 | 68.6 |

> † SAILER's structure-aware pre-training causes embedding collapse on SG legal text; fine-tuning provides no meaningful improvement.

---

## Key Findings

**1. Principle-augmented queries consistently outperform fact-only baselines** across all architectures (up to +354% relative R@1 improvement), confirming that legal principles provide critical discriminative signal for citation retrieval.

**2. Pre-training objective is a stronger determinant of performance than model scale or corpus size.** Despite a 57× difference in pretraining corpus size (12GB → 689GB), all legal-domain encoders converge to ~94% R@1 after fine-tuning.

**3. Standard autoregressive pretraining causes embedding collapse that worsens with scale.** SaulLM-54B (0.2% zero-shot R@1) performs worse than its 7B variant (4.2%) due to more severe embedding collapse (mean pairwise similarity 0.98 vs. 0.95).

**4. Reading comprehension pretraining partially bridges the encoder–decoder gap.** AdaptLLM achieves 77.2% zero-shot R@1 — 15.5 pp below Pile-of-Law BERT but far above other decoders — because its multi-task QA pretraining incidentally learns query–document matching.

**5. After fine-tuning, encoders and decoders reach comparable peak performance** (~94–95% R@1), but decoders require substantially more compute (7–54B vs. 110–340M parameters).

---

## Reproducibility Notes

**Unavailable models:** Two legal LLMs could not be included in our evaluation due to accessibility issues:
- **InternLM-Law** (`internlm/internlm2-law-7b`): HuggingFace weights returned HTTP 401 errors at time of experiments.
- **Lawyer GPT**: No publicly released model weights despite the published paper.

These barriers are noted to highlight reproducibility challenges in legal NLP.

---
## Citation
If you use SG-LegalCite in your work, please cite:
```bibtex
@inproceedings{lee2026sglegalcite,
  title  = {SG-LegalCite: A Singapore Legal Principle-Case Dataset for Jurisdiction-Aware Citation Retrieval},
  author = {Lee, Shannon Yueh Ern and Du, Yingpeng and Feng, Kaidong and Loi, Kelry and Lee, Chloe},
  year   = {2026}
}
```


---

## License

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source judgments are publicly available via the [Singapore eLitigation platform](https://www.elitigation.sg).

Code is released under the MIT License.

---

## Acknowledgements

Expert validation was conducted by legally qualified annotators from the National University of Singapore Faculty of Law and Singapore Management University Yong Pung How School of Law. Experiments were run on SUTD's HPC cluster (A100 GPUs).
