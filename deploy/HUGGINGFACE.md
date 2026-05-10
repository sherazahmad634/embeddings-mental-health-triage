# Deploying to Hugging Face

You'll create three artefacts on Hugging Face: a **dataset**, a **model**,
and a **Space** (the demo). The exact names below match what's referenced in
the report — feel free to change `<your-username>` and the slugs to your own.

## 0. One-time setup

```bash
pip install huggingface_hub
huggingface-cli login          # paste a write token from https://huggingface.co/settings/tokens
```

## 1. Push the dataset

```bash
cd embeddings-mental-health-triage

huggingface-cli repo create mental-health-support-tier --type dataset
git lfs install                                        # required for any binary files

# Clone the empty dataset repo and copy the relevant files in
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/<your-username>/mental-health-support-tier hf-dataset
cp data/dataset.csv data/dataset.jsonl data/train.csv data/test.csv data/adversarial_test.csv data/label_map.json hf-dataset/
cp deploy/dataset_card.md hf-dataset/README.md         # see the template at the bottom of this doc

cd hf-dataset
git add . && git commit -m "Initial dataset upload" && git push
cd ..
```

## 2. Push the trained model

The classifier itself is a ~50 KB joblib bundle, no Git LFS needed.

```bash
huggingface-cli repo create mental-health-support-tier-classifier
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/<your-username>/mental-health-support-tier-classifier hf-model
cp models/classifier.joblib models/metrics.json models/adversarial_metrics.json hf-model/
cp figures/confusion_matrix.png hf-model/
cp deploy/model_card.md hf-model/README.md             # see the template at the bottom of this doc

cd hf-model
git add . && git commit -m "Initial model upload" && git push
cd ..
```

## 3. Deploy the Space (the demo)

Create a new Space at https://huggingface.co/new-space:
- **Name:** `mental-health-support-tier-demo`
- **SDK:** Gradio
- **Hardware:** CPU basic (free tier is enough)

Then push the demo files:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/<your-username>/mental-health-support-tier-demo hf-space
cp app.py requirements.txt hf-space/
mkdir -p hf-space/models hf-space/data
cp models/classifier.joblib hf-space/models/
cp data/label_map.json hf-space/data/

# Spaces need an `app.py` at the root and a `README.md` with the right metadata
cat > hf-space/README.md <<'EOF'
---
title: Mental-Health Support Tier Triage
emoji: ❤️‍🩹
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

A small embeddings-based classifier that maps short user-written messages to one of four support tiers (self_help / peer_support / professional / crisis). Crisis resources are surfaced on every request.
EOF

cd hf-space
git add . && git commit -m "Initial space upload" && git push
cd ..
```

Hugging Face will build the Space and expose it at
`https://huggingface.co/spaces/<your-username>/mental-health-support-tier-demo`.

## Optional templates

`deploy/dataset_card.md` and `deploy/model_card.md` are written for you in
this folder so you can drop them straight into the repo as `README.md`.
