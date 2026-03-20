# Dataset

The dataset files are hosted on HuggingFace: https://huggingface.co/datasets/anonymousmeowmeow/SG-LegalCite

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
| `train.jsonl` | ~80,443 | Training split (80%) |
| `val.jsonl` | ~10,055 | Validation split (10%) |
| `test.jsonl` | ~10,056 | Test split (10%) |
| `candidate_pool.jsonl` | 48,298 | All unique cited cases |

Split is by unique judgment URL to prevent data leakage.
