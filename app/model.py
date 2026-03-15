"""
app/model.py — Model Loading & Inference Logic
===============================================

This module is the single point of contact between the FastAPI routes and
the trained scikit-learn pipeline. Keeping ML logic isolated here (away from
routing code) makes each layer independently testable and replaceable — i.e.
swapping the model backend requires changes only in this file, not in main.py.
"""

from pathlib import Path

import joblib

# Path to the serialized pipeline produced by train.py
MODEL_PATH = Path(__file__).parent.parent / "model" / "sentiment_pipeline.pkl"

# Module-level variable holding the loaded pipeline.
# Storing it here (rather than loading on each request) means the pipeline
# is loaded exactly once at startup. Inference calls then hit in-memory
# objects, keeping per-request latency in the low milliseconds.
_pipeline = None


def load_model() -> None:
    """
    Load the trained scikit-learn pipeline from disk into module memory.

    Called once during API startup via the lifespan context manager in main.py.
    Failing fast here (FileNotFoundError on missing model) is intentional —
    the server should refuse to start rather than serve 500 errors at request
    time when someone forgets to run train.py first.

    Raises:
        FileNotFoundError : if train.py has not been run yet.
    """
    global _pipeline

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run 'python train.py' to train and save the model first."
        )

    # joblib is the recommended deserializer for scikit-learn objects —
    # it handles large NumPy arrays more efficiently than standard pickle.
    _pipeline = joblib.load(MODEL_PATH)


def predict(text: str) -> dict:
    """
    Run sentiment inference on a single text string.

    Args:
        text : The raw review / comment string to classify.

    Returns:
        dict with keys: text, sentiment, confidence.

    The pipeline internally applies the fitted TF-IDF transform followed by
    Logistic Regression classification in a single call, guaranteeing that
    identical preprocessing is applied at inference as was applied at training.

    predict_proba() returns a probability vector over all classes. We take the
    index of the maximum probability as the predicted class, and its value as
    the confidence score. Because Logistic Regression is well-calibrated,
    a confidence of 0.93 genuinely means the model is ~93% certain.

    Raises:
        RuntimeError : if called before load_model() has been invoked.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Model pipeline is not loaded. "
            "Ensure load_model() is called during application startup."
        )

    # predict_proba expects an iterable of samples; wrap in a list for one sample
    probabilities = _pipeline.predict_proba([text])[0]

    # argmax gives the index of the highest-probability class
    predicted_index = int(probabilities.argmax())

    # pipeline.classes_ maps integer indices back to label strings
    # e.g. classes_ = ["negative", "positive"] (alphabetical order by default)
    sentiment = _pipeline.classes_[predicted_index]
    confidence = round(float(probabilities[predicted_index]), 4)

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """
    Run sentiment inference on a list of texts in a single vectorised pass.

    Args:
        texts : List of raw text strings to classify.

    Returns:
        List of dicts, each with keys: text, sentiment, confidence.

    Batching is significantly more efficient than calling predict() in a loop:
    TF-IDF vectorisation and Logistic Regression both operate on the entire
    input matrix at once using optimised NumPy / SciPy routines, rather than
    building and scoring one sparse vector at a time.

    Raises:
        RuntimeError : if called before load_model() has been invoked.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Model pipeline is not loaded. "
            "Ensure load_model() is called during application startup."
        )

    # Vectorise the full batch in one call — much faster than a Python loop
    probabilities = _pipeline.predict_proba(texts)  # shape: (n_samples, n_classes)
    predicted_indices = probabilities.argmax(axis=1)  # shape: (n_samples,)

    results = []
    for i, text in enumerate(texts):
        sentiment = _pipeline.classes_[predicted_indices[i]]
        confidence = round(float(probabilities[i][predicted_indices[i]]), 4)
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
        })

    return results
