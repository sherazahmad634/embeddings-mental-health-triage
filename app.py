"""Gradio demo for the mental-health support-tier classifier.

The app loads the joblib bundle written by ``scripts/train_classifier.py`` and
exposes a single text box. Given a short user message, it returns:

  * the predicted support tier (one of self_help / peer_support /
    professional / crisis),
  * a calibrated confidence per class as a label-confidence widget, and
  * a contextual response with practical next steps.

A persistent crisis-resources banner is shown for *every* prediction. This is
deliberate: a misclassification in the crisis tier is the most costly error,
so we never rely on the model alone to surface emergency contacts.

The app supports two backends transparently. If the bundle was trained with
sentence-transformers, the embedding model is loaded once at startup. If it
was trained with the TF-IDF baseline, the saved vectorizer is reused.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = Path(os.environ.get("CLASSIFIER_BUNDLE", ROOT / "models" / "classifier.joblib"))


# ---------------------------------------------------------------------------
# Static copy
# ---------------------------------------------------------------------------

CRISIS_BANNER = (
    "**If you or someone else is in immediate danger, contact local emergency "
    "services (e.g. 112 in the EU, 911 in the US, 999 in the UK).**  \n"
    "Free 24/7 support: **988** (US Suicide & Crisis Lifeline), "
    "**116 123** (Samaritans, UK & IE), or [findahelpline.com](https://findahelpline.com) "
    "for an international list."
)

TIER_RESPONSES: dict[str, str] = {
    "self_help": (
        "Sounds like a low-intensity, everyday slump. Small habit changes "
        "(sleep, movement, time outside, less screen time) and self-care "
        "routines tend to help here. Keep an eye on it — if it lingers for "
        "weeks, consider moving up a tier."
    ),
    "peer_support": (
        "This sounds like something that gets lighter when you're not carrying "
        "it alone. Talking to a trusted friend, family member, or peer-support "
        "community can help. Online support communities (e.g. r/Anxiety, "
        "7 Cups) can be a good first step."
    ),
    "professional": (
        "What you're describing has lasted long enough or is intense enough "
        "that professional support is likely to help more than self-help or "
        "peer support alone. A therapist, counsellor, or your GP is a good "
        "starting point. Many countries also have low-cost or free tele-therapy "
        "options."
    ),
    "crisis": (
        "**This sounds urgent.** Please reach out to a crisis line or "
        "emergency services right now — you don't have to handle this on your "
        "own. The numbers in the banner above connect you to a trained human, "
        "for free, 24/7."
    ),
}

EXAMPLES: list[list[str]] = [
    ["Work has been chaotic and I'm a little burnt out. Any tips for unwinding in the evenings?"],
    ["My best friend just lost a parent and I have no idea how to show up for her."],
    ["I've felt numb for almost six months and exercise doesn't help anymore. I think I should talk to someone."],
    ["I don't think I can keep myself safe tonight, please tell me where to call."],
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find classifier bundle at {BUNDLE_PATH}. "
            "Run `python scripts/train_classifier.py` first."
        )
    return joblib.load(BUNDLE_PATH)


@lru_cache(maxsize=1)
def load_embedder():
    """Lazily construct the SentenceTransformer once."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(load_bundle()["embedder"])


def featurize(texts: list[str]):
    bundle = load_bundle()
    if bundle["backend"] == "st":
        emb = load_embedder().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)
    if bundle["backend"] == "tfidf":
        return bundle["tfidf_vectorizer"].transform(texts)
    raise ValueError(f"unknown backend: {bundle['backend']}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(message: str):
    message = (message or "").strip()
    if not message:
        return (
            {l: 0.0 for l in load_bundle()["label_names"]},
            "Please enter a short description of how you're feeling.",
        )

    bundle = load_bundle()
    clf = bundle["classifier"]
    id_to_label = bundle["id_to_label"]
    label_names = bundle["label_names"]

    X = featurize([message])
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
    else:
        # Decision function -> softmax (shouldn't happen because we calibrate)
        scores = clf.decision_function(X)[0]
        e = np.exp(scores - scores.max())
        proba = e / e.sum()

    label_to_score = {id_to_label[i]: float(proba[i]) for i in range(len(label_names))}
    pred_idx = int(np.argmax(proba))
    pred_label = id_to_label[pred_idx]
    response = TIER_RESPONSES[pred_label]
    return label_to_score, response


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    bundle_info = ""
    try:
        b = load_bundle()
        bundle_info = (
            f"_Backend: **{b['backend']}**, "
            f"classifier: **{b['classifier_name']}**, "
            f"trained: {b.get('trained_at', 'unknown')}._"
        )
    except FileNotFoundError as e:
        bundle_info = f"⚠️ {e}"

    with gr.Blocks(title="Mental-Health Support Tier Triage") as demo:
        gr.Markdown(
            "# Mental-Health Support Tier Triage\n"
            "Type a short description of how you're feeling. The model will "
            "suggest what kind of support is most likely to help: a self-help "
            "habit change, talking to a peer, seeing a professional, or "
            "contacting a crisis line. **This is a research demo, not a "
            "diagnostic tool.**"
        )
        gr.Markdown(CRISIS_BANNER)
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(
                    label="How are you feeling?",
                    placeholder="e.g. \"Work has been chaotic this week and I'm a little burnt out…\"",
                    lines=4,
                )
                btn = gr.Button("Suggest a support tier", variant="primary")
                gr.Examples(EXAMPLES, inputs=inp, label="Try one of these")
            with gr.Column(scale=2):
                proba_out = gr.Label(label="Predicted support tier (with confidence)", num_top_classes=4)
                response_out = gr.Markdown(label="Suggested next step")

        gr.Markdown(bundle_info)
        gr.Markdown(
            "_The classifier maps text to one of four tiers: "
            "**self_help**, **peer_support**, **professional**, **crisis**. "
            "Crisis resources are surfaced for every prediction so a "
            "misclassification cannot suppress them._"
        )

        btn.click(predict, inputs=inp, outputs=[proba_out, response_out])
        inp.submit(predict, inputs=inp, outputs=[proba_out, response_out])

    return demo


if __name__ == "__main__":
    build_ui().launch()
