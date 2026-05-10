---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
size_categories:
  - n<1K
tags:
  - mental-health
  - triage
  - synthetic
  - embeddings
  - linear-probe
pretty_name: Mental-Health Support Tier
---

# Mental-Health Support Tier (synthetic)

A small synthetic English-language text-classification dataset that pairs
short user-written messages with one of four **support tiers**:

| Tier            | Meaning                                                                  |
| --------------- | ------------------------------------------------------------------------ |
| `self_help`     | Low-intensity, transient complaints. Habit / lifestyle suggestions help. |
| `peer_support`  | Moderate distress that benefits from talking with friends or community.  |
| `professional`  | Persistent symptoms that warrant a clinician (therapist, GP, etc.).      |
| `crisis`        | Acute risk language that warrants an immediate crisis resource.          |

The dataset was generated for a coursework project on embeddings-based text
classification. It is **not** scraped from real users; every example was
produced from hand-curated templates with slot fillers.

## Files

| File                     | Rows  | Notes                                          |
| ------------------------ | ----- | ---------------------------------------------- |
| `dataset.csv`            | ~1000 | Full set, columns `text`, `label`              |
| `dataset.jsonl`          | ~1000 | Same data with integer labels for `datasets`   |
| `train.csv`              | ~800  | 80 % stratified split                          |
| `test.csv`               | ~200  | 20 % stratified split                          |
| `adversarial_test.csv`   | 20    | Hand-written paraphrases (5 per tier)          |
| `label_map.json`         |  -    | label-name <-> id map                          |

## Loading

```python
from datasets import load_dataset

ds = load_dataset(
    "<your-username>/mental-health-support-tier",
    data_files={"train": "train.csv", "test": "test.csv"},
)
print(ds)
```

## Intended use

The dataset is meant for educational use: building and evaluating small text
classifiers on top of frozen sentence embeddings (linear probes), and showing
end-to-end how dataset → embeddings → classifier → demo fit together.

## Out-of-scope use

This dataset is **not** suitable for training a system that takes real
clinical action. It is too small, fully synthetic, and the label boundaries
were authored by a single person; do not use it to triage real users.

## Ethics

- No real user content is included.
- Crisis-tier templates were written to describe *the need for help*, not to
  contain operational detail.
- Any downstream demo built on this dataset should always surface crisis
  resources independently of the model's prediction.

## License

CC-BY-4.0.
