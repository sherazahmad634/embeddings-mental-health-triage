# Pushing the project to GitHub

```bash
cd embeddings-mental-health-triage

git init
git add .
git commit -m "Initial commit: embeddings-based support-tier triage"

# Create a new empty repository on github.com first (don't initialise it with
# a README), then run:
git branch -M main
git remote add origin https://github.com/<your-username>/embeddings-mental-health-triage.git
git push -u origin main
```

The submission link for the assignment is the URL of the public GitHub repo:

    https://github.com/<your-username>/embeddings-mental-health-triage

## Required submission contents (already in the repo)

- `README.md` — project overview and usage.
- `report.md` — 2-page academic report.
- `data/` — dataset (CSV + JSONL).
- `scripts/` — generation, training, evaluation.
- `models/classifier.joblib` — trained classifier.
- `app.py` — Gradio demo (the same file ships to the HF Space).
- Links inside `report.md` to the HF dataset, model, and Space.
