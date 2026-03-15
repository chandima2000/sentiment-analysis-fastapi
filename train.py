"""
train.py — Training Script for the Sentiment Analysis Classifier
=================================================================

Dataset  : IMDB Dataset.csv (50,000 labelled movie reviews)
Model    : scikit-learn Pipeline → TF-IDF Vectorizer + Logistic Regression
Output   : model/sentiment_pipeline.pkl  (joblib-serialized pipeline)
           model/eval_report.txt         (classification report)

Run:
    python train.py

Author decision notes (see inline comments for full explanations):
  - TF-IDF chosen over raw bag-of-words for its IDF weighting benefit
  - Bigrams (ngram_range 1-2) to capture short negations like "not good"
  - Logistic Regression for calibrated probabilities via predict_proba
  - joblib for faster and more memory-efficient sklearn serialization
"""

import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = Path("data") / "IMDB_Dataset.csv"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.pkl"
EVAL_REPORT_PATH = Path("result") / "evaluation_report.txt"


# Fixed seed ensures the same train/test split and model weights every run,
# making results reproducible across machines and reviewers.
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80/20 split is a standard starting point for this data size


# ---------------------------------------------------------------------------
# Step 1 — Load & clean data
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove HTML tags and normalize whitespace from raw review text.
    """

    # Remove HTML tags (e.g. <br />, <b>, </b>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data() -> tuple[list[str], list[str]]:
    """
    Load the IMDB CSV, clean review text, and return parallel lists.

    The dataset is balanced (25,000 positive, 25,000 negative), so no
    class-weighting or oversampling is needed for this baseline.
    """

    if not DATA_PATH.exists():
        print(f" Dataset not found at '{DATA_PATH}'.")
        print("  Place 'IMDB_Dataset.csv' inside the data/ directory.")
        sys.exit(1)

    print(f" Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Validate expected columns exist
    if "review" not in df.columns or "sentiment" not in df.columns:
        print(" CSV must contain 'review' and 'sentiment' columns.")
        sys.exit(1)

    # Apply HTML cleaning to every review
    df["review"] = df["review"].apply(clean_text)

    texts = df["review"].tolist()
    labels = df["sentiment"].tolist()  # already "positive" / "negative"

    pos = labels.count("positive")
    neg = labels.count("negative")
    print(f" Loaded {len(texts):,} reviews  ({pos:,} positive | {neg:,} negative)")
    return texts, labels


# ---------------------------------------------------------------------------
# Step 2 — Build the pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Construct a scikit-learn Pipeline combining TF-IDF and Logistic Regression.

    Why a Pipeline?
    Using Pipeline ensures that the same TfidfVectorizer (with the same fitted
    vocabulary and IDF weights) is always applied at both training and inference
    time. Saving the pipeline as one object eliminates the risk of train-serve
    skew — a common source of silent bugs in ML systems.

    TfidfVectorizer parameters:
      - max_features=50_000 : caps vocabulary to the 50k most frequent terms,
        preventing memory issues while retaining sufficient coverage.
      - ngram_range=(1, 2)  : includes both unigrams and bigrams. Bigrams
        capture sentiment patterns like "not good" or "highly recommend"
        that unigrams alone would miss.
      - sublinear_tf=True   : applies 1+log(tf) scaling, compressing the
        influence of very frequent words and giving rarer, more discriminative
        words proportionally more weight.
      - strip_accents="unicode" : normalizes accented characters for consistency.
      - min_df=2            : ignores terms appearing in fewer than 2 documents,
        removing typos and one-off tokens that bloat the feature matrix.
      - stop_words="english": removes common English words ("the", "is", etc.)
        that carry no sentiment signal, reducing noise in the feature space.

    LogisticRegression parameters:
      - C=1.0     : inverse regularization strength. Default value; generalizes
        well without hyperparameter tuning for this dataset size.
      - max_iter=1000 : ensures convergence on the large TF-IDF feature space
        (default 100 is often insufficient for text classification).
      - solver="lbfgs" : efficient for multi-class problems and large feature
        spaces; memory-friendly compared to "liblinear".
      - random_state=42 : fixed seed for reproducibility.
    """

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=2,
        stop_words="english"
    )

    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


# ---------------------------------------------------------------------------
# Step 3 — Train and evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(texts: list[str], labels: list[str]) -> Pipeline:
    """
    Split data, train the pipeline, and print an evaluation report.

    Why stratified split?
    stratify=labels preserves the class ratio (50/50 here) in both the train
    and test sets. This is best practice — especially important if the dataset
    were imbalanced, and makes results comparable across different runs.

    Returns:
        Trained Pipeline object ready for serialization.
    """

    print(f"\n Splitting data  ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"   Train: {len(X_train):,} samples   |   Test: {len(X_test):,} samples")

    print("\n Training TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print(" Training complete.")

    # Evaluate on held-out test set
    print("\n Evaluation Report (Test Set):")
    print("-" * 55)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Persist evaluation report alongside the model for reference
    MODEL_DIR.mkdir(exist_ok=True)
    EVAL_REPORT_PATH.parent.mkdir(exist_ok=True)  # create result/ if it doesn't exist
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Sentiment Analysis — Evaluation Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Dataset : {DATA_PATH}\n")
        f.write(f"Model   : TF-IDF (max 50k features, bigrams) + Logistic Regression\n")
        f.write(f"Split   : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test (stratified, seed={RANDOM_STATE})\n\n")
        f.write(report)
    print(f" Evaluation report saved → {EVAL_REPORT_PATH}")

    return pipeline


# ---------------------------------------------------------------------------
# Step 4 — Save the trained pipeline
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline) -> None:
    """
    Serialize the trained pipeline to disk using joblib.
    """

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\n Model saved → {MODEL_PATH}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Sentiment Analysis — Training Pipeline")
    print("=" * 55)

    texts, labels = load_data()
    pipeline = train_and_evaluate(texts, labels)
    save_model(pipeline)

    print("\n Done! Start the API with:")
    print("     uvicorn app.main:app --reload\n")
