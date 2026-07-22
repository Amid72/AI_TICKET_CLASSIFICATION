"""
Step 3: Train the ticket classification model.

Usage:
    python src/train.py

Reads data/tickets.csv, trains a TF-IDF + Logistic Regression pipeline,
evaluates it on a held-out test split, and saves the trained pipeline
to models/ticket_classifier.pkl
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from preprocess import clean_series

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ticket_classifier.pkl")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "category"])
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    df["clean_text"] = clean_series(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.2, random_state=42, stratify=df["category"]
    )

    print(f"Training on {len(X_train)} tickets, testing on {len(X_test)} tickets...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("\n=== Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
