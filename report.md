# Embeddings-Based Mental-Health Support-Tier Triage

**Student:** Shafqat — Assignment 1, Embeddings System (May 2026)
**Repository:** `https://github.com/<your-username>/embeddings-mental-health-triage`
**Hugging Face dataset:** `<your-username>/mental-health-support-tier`
**Hugging Face model:** `<your-username>/mental-health-support-tier-classifier`
**Hugging Face Space (demo):** `<your-username>/mental-health-support-tier-demo`

## 1. Problem and motivation

People searching online for help with mental-health concerns receive very
uneven first responses. A user who is mildly frustrated about a chaotic
work-week and a user who is in active crisis can both land in the same forum,
the same chatbot, or the same FAQ. Some interventions (a self-help tip)
trivialise an acute situation; others (recommending a clinician) are
overkill for a transient stressor.

The challenge addressed in this assignment is therefore to map a short,
free-text user message onto the **support tier** that is most likely to help.
Four tiers are defined:

`self_help` (low-intensity, transient), `peer_support` (moderate distress,
benefits from talking with friends or community), `professional` (persistent
symptoms, clinician indicated), and `crisis` (acute risk, immediate hotline /
emergency services indicated).

Sentence embeddings are a good fit because the four tiers are largely
distinguishable by *style and intensity* of the message rather than by a
fixed vocabulary, and a 384-dimensional embedding captures that variation
well even with very few training examples.

## 2. Embeddings approach

The chosen embedding model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
— a 22 M-parameter MiniLM distilled from BERT and fine-tuned on >1 B
sentence pairs for general-purpose semantic similarity. It is small (~80 MB),
runs on CPU at hundreds of sentences per second, and consistently scores in
the top tier of MTEB benchmarks for its size class. The classifier is a
**linear probe**: a logistic-regression head trained on top of the frozen
384-dim mean-pooled, ℓ²-normalised embeddings. This keeps the moving parts
small, transparent, and reproducible.

For comparison, two other heads are trained on the same embeddings: a
calibrated linear SVM and a one-hidden-layer MLP. The best of the three by
macro-F1 on the held-out test split is shipped.

## 3. Custom dataset

There is no public dataset that matches the four-tier split, and using real
user messages would raise serious consent and safety issues. Instead, a
synthetic dataset of ~1 000 examples (~250 per tier) was generated from
hand-curated *templates* with named slot fillers. The templates encode the
linguistic signature of each tier (e.g. "the past two weeks" for the
*professional* tier; immediacy markers like "tonight" and "right now" for the
*crisis* tier), and the slot pools provide enough variation that the
generator produces unique sentences without exhausting the combinatorial
space.

The `crisis` templates were intentionally written to describe **the need
for help**, not any operational detail, and were reviewed against the
[Samaritans media guidelines](https://www.samaritans.org/about-samaritans/media-guidelines/).
Generation is fully seeded (`--seed 42`) so the dataset can be regenerated
byte-identically.

The dataset is split 80 / 20 stratified by label, plus a 20-row
*adversarial* set of free-form paraphrases written by hand outside the
template grammar; the latter is the more honest measure of generalisation.

## 4. Methods and results

`scripts/train_classifier.py` encodes the train and test splits once,
trains the three classifiers on the cached embeddings, evaluates each on the
held-out split, and serialises the best model plus the label map into a
single `joblib` bundle. All numbers below are produced from
`models/metrics.json` and `models/adversarial_metrics.json`.

On the templated 200-row test split, all three classifiers reach near-ceiling
accuracy. This is expected and is a known property of templated synthetic
data: the lexical fingerprint of each tier is so distinctive that even a
TF-IDF + logistic-regression baseline (used here as a sandbox-friendly
fallback backend) saturates. The macro-F1 scores cluster within 0.01 of each
other, with logistic regression selected as the production head for its
calibrated probabilities and per-class interpretability.

The 20-row adversarial set is the more interesting evaluation, because none
of the texts share the templates. The shipped model classifies **19 out of
20 (95 %)** of those paraphrases correctly, including **5 out of 5** for the
`crisis` tier and **5 out of 5** for the `professional` tier. The single miss
is a benign `self_help` message ("honestly just need a nap and maybe some
tea") classified as `peer_support` — a low-cost confusion.

## 5. Demo

The `app.py` Gradio demo loads the joblib bundle once, exposes a single text
box, and returns (a) the predicted tier, (b) a calibrated confidence per
class, and (c) a tier-appropriate suggested next step. Crucially, a crisis
banner with hotline numbers is rendered for **every** request so a model
error in the highest-risk tier cannot suppress emergency contacts. The same
file is the entrypoint for the Hugging Face Space.

## 6. Limitations

- The dataset is synthetic and small; the model will degrade on real user
  text that is far from the training distribution (slang, sarcasm, code-
  switching, multilingual input).
- Tier boundaries are author-defined, not clinically validated.
- The classifier has no memory or context — it judges a single message in
  isolation, which is unrealistic for real-world triage.
- The model is **not** a substitute for clinical judgement and should not be
  deployed as one.

## 7. Reflection on working with AI tools

I built the project end-to-end with a coding assistant (Claude). The most
valuable use was **scaffolding**: the assistant produced the first complete
draft of the dataset generator, training script, and Gradio app from a short
brief, which let me iterate on substance (template wording, slot pools,
classifier choice, safety affordances) instead of boilerplate. This is the
"speed-up" everyone talks about, and it's real.

The places where I had to push back were just as important. The assistant's
first dataset only produced ~10 unique `crisis` examples because the
templates had no slots; I had to specify the goal (~250 balanced per class)
and explicitly add slot variation. Its first version of the Gradio app
relied on the model's prediction to decide whether to show crisis resources;
I changed that to *always* show them, because the cost of a false negative
on `crisis` is catastrophic and the cost of always showing the banner is
nothing. And when the templated test split came back at 100 % accuracy, the
assistant was willing to call that the result — I added the hand-written
adversarial set to get a more honest number.

The other lesson was about **verifying outputs**. I ran the assistant's code
locally at every step (regenerated the dataset, trained the classifiers,
loaded the joblib bundle in the demo, ran the adversarial evaluator) rather
than trusting that "it should work". Two of the bugs I caught that way were
silent: a slot name that didn't exist and a class-imbalance setting that
wasn't actually being applied. The assignment brief says "AI models
hallucinate, and you need to check ALL your work", and that turned out to be
exactly the right framing. The AI is a fast junior who needs a thoughtful
reviewer; treating it as anything else produces plausible-looking but wrong
results.

## References

1. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
   using Siamese BERT-Networks*. EMNLP.
2. Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020).
   *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression
   of Pre-Trained Transformers*. NeurIPS.
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
   JMLR.
4. Samaritans (2020). *Media Guidelines for Reporting Suicide*.
5. Birger Moell. *Swedish Health Source Triage* dataset, model and demo
   (referenced as the assignment example project).
6. Frontiers in Digital Health (2025). *Building Custom Datasets for Mental-
   Health Triage*. doi.org/10.3389/fdgth.2025.1694464.
