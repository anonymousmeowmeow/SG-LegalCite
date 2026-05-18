# Dataset

SG-LegalCite is a principle-augmented benchmark for legal citation retrieval
in Singapore law, comprising 100,890 case-principle pairs extracted from
8,523 Supreme Court judgments spanning 2000-2025.

The dataset files are hosted on HuggingFace:
https://huggingface.co/datasets/anonymousmeowmeow/SG-LegalCite

## Statistics

| Attribute | Value |
|---|---|
| Total case-principle pairs | 100,890 |
| Unique citing judgments | 8,523 |
| Unique cited cases | 48,478 |
| Unique issues | 86,519 |
| Unique issue groups | 9,748 |
| Time span | 2000–2025 |
| Courts covered | SGCA, SGCAI, SGHC, SGHCF, SGHCR |

Each judgment is uniquely identified by its `judgment_url`, which corresponds
1:1 with the Singapore neutral citation (`Full_Reference`) of the citing
judgment (e.g., `https://www.elitigation.sg/gd/s/2023_SGCA_15` ↔ `[2023] SGCA 15`).

## Format

Each record is a JSON line with the following fields:

```json
{
  "judgment_url": "https://www.elitigation.sg/...",
  "court": "SGCA",
  "year": 2023,
  "fact": "My client was found not yet fit for admission to the Bar...",
  "principle": "Dishonesty is to be distinguished from a lack of academic diligence.",
  "cited_case": "Re Suria Shaik Aziz [2023] 5 SLR 1272",
  "issue": "Whether the stakeholders agree that Mr Foo should be admitted to the Bar.",
  "issue_group": "Admission of Candidate"
}
```

## Files

| File | Records | Description |
|---|---|---|
| `train.jsonl` | 79,950 | Training split (80%) |
| `val.jsonl` | 10,555 | Validation split (10%) |
| `test.jsonl` | 10,385 | Test split (10%) |
| `candidate_pool.jsonl` | 48,478 | All unique cited cases |

The 80/10/10 split is performed at the **judgment level** (by unique
`judgment_url`) to prevent data leakage: all records derived from a single
citing judgment fall into the same split. The split uses `random_state=42`
for reproducibility.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("anonymousmeowmeow/SG-LegalCite")
```

Or load JSONL files directly:

```python
import json

def load_split(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train = load_split('train.jsonl')
val   = load_split('val.jsonl')
test  = load_split('test.jsonl')
pool  = load_split('candidate_pool.jsonl')
```
