---
license: mit
language:
  - en
library_name: scikit-learn
tags:
  - text-classification
  - linear-probe
  - sentence-transformers
  - mental-health
base_model: sentence-transformers/all-MiniLM-L6-v2
pipeline_tag: text-classification
---

# Mental-Health Support Tier Classifier

A logistic-regression **linear probe** on top of frozen
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
embeddings. Given a short user message, it predicts one of four support tiers:
`self_help`, `peer_support`, `professional`, `crisis`.

## How it works

```
text  ─►  all-MiniLM-L6-v2 (frozen)  ─►  384-dim vector  ─►  Logistic Regression  ─►  tier
```

The classifier is shipped as a single `joblib` bundle that contains:

- `backend` — `"st"`
- `embedder` — `"sentence-transformers/all-MiniLM-L6-v2"`
- `classifier` — fitted `sklearn.linear_model.LogisticRegression`
- `label_to_id` / `id_to_label` / `label_names`
- `trained_at` — ISO timestamp

## Usage

```python
import joblib, numpy as np
from sentence_transformers import SentenceTransformer

bundle = joblib.load("classifier.joblib")
embedder = SentenceTransformer(bundle["embedder"])
clf = bundle["classifier"]

text = "I've felt numb for almost six months and exercise doesn't help anymore."
vec = embedder.encode([text], normalize_embeddings=True)
proba = clf.predict_proba(vec)[0]
for label, p in zip(bundle["label_names"], proba):
    print(f"{label:14s}  {p:.3f}")
```

## Training

- Base model: `sentence-transformers/all-MiniLM-L6-v2` (frozen, no fine-tuning).
- Classifier: `LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")`.
- Data: `<your-username>/mental-health-support-tier` (synthetic, ~1000 rows).
- Selection: macro-F1 on a 200-row stratified test split, compared against
  calibrated linear SVM and a small MLP.

## Evaluation

Metrics on the held-out test split and on a 20-row hand-written adversarial
set are included in the repo (`metrics.json`, `adversarial_metrics.json`).
The confusion matrix is in `confusion_matrix.png`.

## Intended use

Educational demonstration of the embeddings → linear-probe → demo workflow.
The companion Space at
`<your-username>/mental-health-support-tier-demo` always surfaces crisis
resources regardless of the model's prediction.

## Limitations

- Trained on **synthetic, templated** data. It will degrade on user text that
  is far from the training distribution (slang, code-switching, sarcasm).
- The label boundaries are author-defined, not clinically validated.
- The model has no notion of who is talking — context, history, user intent,
  cultural framing, etc. are all out of scope.
- **Do not** use it for real-world triage decisions.

## License

MIT.
