# Mental-Health Support Tier Triage

-A small embeddings-based text classifier that maps short user-written messages
to one of four **support tiers**:


-A lightweight embeddings-based NLP system that classifies short mental-health support messages into four support tiers using sentence embeddings and machine learning classifiers.

Repository:
https://github.com/sherazahmad634/embeddings-mental-health-triage
Hugging Face dataset:
 https://huggingface.co/datasets/Sherazahmad634/mental-health-support-tier
Hugging Face model:
 https://huggingface.co/Sherazahmad634/mental-health-support-tier-classifier
Hugging Face Space (demo): https://huggingface.co/spaces/Sherazahmad634/mental-health-support-tier-demo



| Tier            | Meaning                                                                  |
| --------------- | ------------------------------------------------------------------------ |
| `self_help`     | Low-intensity, transient complaints. Habit/lifestyle suggestions help.   |
| `peer_support`  | Moderate distress that benefits from talking with friends or the community.  |
| `professional`  | Persistent symptoms that warrant a clinician (therapist, GP, etc.).      |
| `crisis`        | Acute risk language that warrants an immediate crisis resource.          |

The project is the deliverable for the *Embeddings System* assignment. It
covers all four required artefacts:

1. A custom dataset (synthetic, generated from hand-curated templates).
2. An embedding model (`sentence-transformers/all-MiniLM-L6-v2`).
3. A trained classifier (logistic regression on top of frozen embeddings — a
   linear probe — selected from three candidates by macro-F1 on a held-out
   test set).
4. A working Gradio demo that runs locally and on Hugging Face Spaces.

> **Disclaimer.** This is a research/coursework demo, not a diagnostic tool.
> The Gradio app shows crisis resources on every prediction so that a model
> error in the highest-risk tier cannot suppress them.

---

## Quick start

```bash
# 1. install (CPU is fine; first run downloads ~80 MB for the embedder)
pip install -r requirements.txt

# 2. Regenerate the dataset (deterministic, ~1000 examples in 4 balanced classes)
python scripts/generate_dataset.py --per-class 250 --seed 42

# 3. train + evaluate three classifiers; the best one is saved
python scripts/train_classifier.py --backend st            # sentence-transformers
# or, with no network access:
python scripts/train_classifier.py --backend tfidf         # TF-IDF baseline

# 4. sanity-check on hand-written, out-of-template messages
python scripts/evaluate_adversarial.py

# 5. Run the demo locally (http://127.0.0.1:7860)
python app.py
```

## Repository layout

```
.
├── app.py                          # Gradio demo (entrypoint for HF Spaces)
├── data/
│   ├── dataset.csv                 # full set (~1000 rows, text + label)
│   ├── dataset.jsonl               # HF datasets-friendly JSONL
│   ├── train.csv / test.csv        # 80/20 stratified split
│   ├── adversarial_test.csv        # 20 hand-written paraphrases
│   └── label_map.json              # label-name <-> id mapping
├── scripts/
│   ├── generate_dataset.py         # template-based data generator
│   ├── train_classifier.py         # embed + train + eval + save bundle
│   └── evaluate_adversarial.py     # run trained model on out-of-template set
├── models/
│   ├── classifier.joblib           # bundle: classifier + vectorizer + label map
│   ├── metrics.json                # accuracy / macro-F1 per classifier
│   └── adversarial_metrics.json    # accuracy on the hand-written set
├── figures/
│   └── confusion_matrix.png        # confusion matrix of the best classifier
├── deploy/
│   ├── HUGGINGFACE.md              # exact commands for pushing to HF
│   └── GITHUB.md                   # exact commands for pushing to GitHub
├── report.pdf                      # 2-page academic report
├── requirements.txt
└── README.md
```

## How it works

```
text  ─►  sentence-transformers/all-MiniLM-L6-v2  ─►  384-dim vector  ─►  Logistic Regression  ─►  tier
```

- `scripts/generate_dataset.py` produces ~250 examples per tier from
  hand-curated templates with slot fillers. Seeded for full reproducibility.
- `scripts/train_classifier.py` encodes the texts once, then trains three
  classifiers on top of the cached embeddings:
  - logistic regression (selected as the production model),
  - calibrated linear SVM (for a margin-based comparison),
  - a small MLP (one hidden layer of 128 units).
- The script picks the classifier with the best macro-F1 on the held-out test
  split and serialises it to `models/classifier.joblib`. The bundle also
  contains the label map (and the TF-IDF vectorizer if the baseline backend
  was used), so the demo is one `joblib.load` away from running.

### Why "linear probe"?

We freeze the pre-trained embedding model and only train a simple linear
classifier on top of its outputs. This is fast (training fits in a few
seconds), transparent (the weights are inspectable per-class), and produces
strong results when the embedding model already separates the relevant
concepts — which is the case here for the mental-health-vs-not-mental-health
and the urgency axes.

## Results

See `models/metrics.json` for the full breakdown. On the templated 200-row
test split, all three classifiers reach near-ceiling accuracy (the templates
are highly distinctive). The more informative number is the **adversarial
accuracy** on the 20 hand-written paraphrases, which is reported in
`models/adversarial_metrics.json`. The confusion matrix of the best
classifier is rendered in `figures/confusion_matrix.png`.

## Two backends

The training script supports two feature backends through `--backend`:

| Backend | Features                                        | Network needed? | Used for                            |
| ------- | ----------------------------------------------- | --------------- | ----------------------------------- |
| `st`    | `sentence-transformers/all-MiniLM-L6-v2` (384d) | yes (first run) | The actual deliverable              |
| `tfidf` | TF-IDF, 1- and 2-grams, capped at 20k features  | no              | Pipeline verification in sandboxes  |

Both backends produce a `classifier.joblib` with the same shape, so `app.py`
runs against either one without modification.



## Tech Stack

- Python
- Sentence Transformers
- Scikit-learn
- Gradio
- Hugging Face
- Pandas
- NumPy

## Future Work

- Multilingual support
- Real-world anonymized evaluation
- Larger datasets
- Transformer fine-tuning
- Context-aware triage

## Author

Sheraz Ahmad  
Master's Student — AI and Language  
Stockholm University

