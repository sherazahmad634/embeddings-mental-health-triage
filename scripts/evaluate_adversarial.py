"""Evaluate the trained classifier on a small hand-written adversarial test set.

The synthetic training data is highly templated, so any reasonable classifier
trivially scores 100% on the held-out templated test split. The *interesting*
question is how the model behaves on free-form text that doesn't match the
templates. ``data/adversarial_test.csv`` contains 20 short, realistic
paraphrases written by hand (5 per tier).

Usage:
    python3 scripts/evaluate_adversarial.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def featurize(bundle: dict, texts: list[str]) -> np.ndarray:
    """Re-create the feature vectors used by the bundle's classifier."""
    if bundle["backend"] == "st":
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(bundle["embedder"])
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
    if bundle["backend"] == "tfidf":
        return bundle["tfidf_vectorizer"].transform(texts)
    raise ValueError(f"unknown backend: {bundle['backend']}")


def main() -> None:
    bundle = joblib.load(ROOT / "models" / "classifier.joblib")
    df = pd.read_csv(ROOT / "data" / "adversarial_test.csv")
    label_to_id = bundle["label_to_id"]
    id_to_label = bundle["id_to_label"]

    X = featurize(bundle, df["text"].tolist())
    y_true = df["label"].map(label_to_id).values
    y_pred = bundle["classifier"].predict(X)

    n = len(df)
    correct = int((y_pred == y_true).sum())
    print(f"Adversarial accuracy: {correct}/{n} = {correct / n:.2%}")

    # Per-class breakdown
    from collections import defaultdict

    per_class: dict[str, list[int]] = defaultdict(list)
    for true_id, pred_id in zip(y_true, y_pred):
        per_class[id_to_label[int(true_id)]].append(int(true_id == pred_id))
    print("\nPer-class:")
    for label, hits in per_class.items():
        print(f"  {label:14s}  {sum(hits)}/{len(hits)}")

    # Show every prediction with confidence
    print("\nDetail:")
    if hasattr(bundle["classifier"], "predict_proba"):
        proba = bundle["classifier"].predict_proba(X)
    else:
        proba = None
    for i, row in df.iterrows():
        true_label = row["label"]
        pred_label = id_to_label[int(y_pred[i])]
        mark = "✓" if pred_label == true_label else "✗"
        conf = f"  ({proba[i, y_pred[i]] * 100:.1f}%)" if proba is not None else ""
        print(f"  {mark}  pred={pred_label:13s} true={true_label:13s}{conf}  {row['text'][:80]}")

    # Save a small report
    report = {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "per_class": {k: {"correct": sum(v), "total": len(v)} for k, v in per_class.items()},
    }
    out = ROOT / "models" / "adversarial_metrics.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report -> {out}")


if __name__ == "__main__":
    main()
