"""Train and evaluate text classifiers for the mental-health triage dataset.

The pipeline is the same for every model:

    text  ->  feature vector  ->  sklearn classifier  ->  label

Two feature backends are supported:

    --backend st       sentence-transformers/all-MiniLM-L6-v2 (default,
                       what the assignment asks for; downloads the model on
                       first use)
    --backend tfidf    TF-IDF baseline that requires no network access. Useful
                       for verifying the pipeline in sandboxed environments.

Three classifiers are trained and evaluated on the held-out test set:

    Logistic Regression    (linear probe; the model we ship)
    Linear SVM             (margin-based linear baseline)
    Small MLP              (one hidden layer of 128 units)

Outputs (under ../models and ../figures):

    classifier.joblib              best classifier + label map + backend tag
    train_embeddings.npy / .json   cached features (for reproducibility)
    test_embeddings.npy / .json
    metrics.json                   per-classifier accuracy / macro-F1 / report
    confusion_matrix.png           confusion matrix of the best model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"


# ---------------------------------------------------------------------------
# Feature backends
# ---------------------------------------------------------------------------

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_with_sentence_transformers(texts: list[str]) -> np.ndarray:
    """Encode `texts` with sentence-transformers/all-MiniLM-L6-v2.

    Imported lazily so the script remains usable without torch installed.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(ST_MODEL_NAME)
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype(np.float32)


def embed_with_tfidf(train_texts: list[str], test_texts: list[str]):
    """Fit TF-IDF on training texts only, then transform both splits.

    Returns dense numpy arrays plus the fitted vectorizer (so the demo can
    re-use it). This backend is for sandbox verification only.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=20000,
    )
    Xtr = vec.fit_transform(train_texts).astype(np.float32)
    Xte = vec.transform(test_texts).astype(np.float32)
    return Xtr, Xte, vec


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def build_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            C=4.0,
            class_weight="balanced",
            n_jobs=None,
        ),
        # LinearSVC has no predict_proba; wrap with calibration so the demo
        # can show a confidence bar for any model we pick.
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"),
            cv=3,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def evaluate(name: str, clf, X_test, y_test, label_names: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    print(f"  {name:12s}  acc={acc:.4f}  macro_f1={f1:.4f}")
    return {"name": name, "accuracy": acc, "macro_f1": f1, "report": report}


def plot_confusion_matrix(y_true, y_pred, label_names: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (best classifier)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--backend",
        choices=["st", "tfidf"],
        default="st",
        help="Feature backend (st = sentence-transformers; tfidf = sandbox baseline)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    MODELS.mkdir(exist_ok=True, parents=True)
    FIGURES.mkdir(exist_ok=True, parents=True)

    # Load data
    label_map = json.loads((DATA / "label_map.json").read_text())
    label_to_id: dict[str, int] = label_map["label_to_id"]
    id_to_label: dict[str, str] = label_map["id_to_label"]
    label_names = [id_to_label[str(i)] for i in range(len(id_to_label))]

    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    y_train = train_df["label"].map(label_to_id).values
    y_test = test_df["label"].map(label_to_id).values
    print(f"Loaded {len(train_df)} train / {len(test_df)} test rows")
    print(f"Backend: {args.backend}")

    # Compute features
    t0 = time.time()
    extras: dict = {}
    if args.backend == "st":
        X_train = embed_with_sentence_transformers(train_df["text"].tolist())
        X_test = embed_with_sentence_transformers(test_df["text"].tolist())
    else:
        X_train, X_test, vec = embed_with_tfidf(
            train_df["text"].tolist(), test_df["text"].tolist()
        )
        extras["tfidf_vectorizer"] = vec
    print(f"Encoded features in {time.time() - t0:.1f}s "
          f"(train shape={X_train.shape}, test shape={X_test.shape})")

    # Cache embeddings (only for the dense ST backend; tfidf is sparse)
    if args.backend == "st":
        np.save(MODELS / "train_embeddings.npy", X_train)
        np.save(MODELS / "test_embeddings.npy", X_test)

    # Train + evaluate every classifier
    classifiers = build_classifiers()
    results = []
    fitted = {}
    for name, clf in classifiers.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        dt = time.time() - t0
        print(f"Trained {name} in {dt:.2f}s")
        res = evaluate(name, clf, X_test, y_test, label_names)
        res["train_seconds"] = dt
        results.append(res)
        fitted[name] = clf

    # Pick best by macro-F1 (more informative than accuracy for the smaller crisis class)
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nBest classifier: {best['name']}  (macro_f1={best['macro_f1']:.4f})")

    best_clf = fitted[best["name"]]
    y_pred = best_clf.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, label_names, FIGURES / "confusion_matrix.png")

    # Persist artifacts. We bundle everything the demo needs in one joblib file
    # so app.py can stay simple (load -> use).
    import joblib

    bundle = {
        "backend": args.backend,
        "embedder": ST_MODEL_NAME if args.backend == "st" else None,
        "tfidf_vectorizer": extras.get("tfidf_vectorizer"),
        "classifier_name": best["name"],
        "classifier": best_clf,
        "label_to_id": label_to_id,
        "id_to_label": {int(i): l for i, l in id_to_label.items()},
        "label_names": label_names,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    joblib.dump(bundle, MODELS / "classifier.joblib")
    print(f"Saved bundle -> {MODELS / 'classifier.joblib'}")

    # Persist metrics for the report
    (MODELS / "metrics.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "results": results,
                "best": best["name"],
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "label_names": label_names,
            },
            indent=2,
        )
    )
    print(f"Saved metrics -> {MODELS / 'metrics.json'}")


if __name__ == "__main__":
    main()
