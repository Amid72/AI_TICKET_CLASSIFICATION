"""
Step 4: Predict the category of a new support ticket from the command line.

Usage:
    python src/predict.py "My card was charged twice this month"
"""
import os
import sys
import joblib

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ticket_classifier.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_ticket(text: str, model=None):
    if model is None:
        model = load_model()
    cleaned = clean_text(text)
    category = model.predict([cleaned])[0]

    result = {"text": text, "predicted_category": category}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_
        result["confidence"] = round(float(max(probs)), 3)
        result["all_scores"] = {
            cls: round(float(p), 3) for cls, p in zip(classes, probs)
        }
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your ticket text here"')
        sys.exit(1)

    ticket_text = " ".join(sys.argv[1:])
    output = predict_ticket(ticket_text)

    print(f"\nTicket: {output['text']}")
    print(f"Predicted category: {output['predicted_category']}")
    if "confidence" in output:
        print(f"Confidence: {output['confidence']}")
        print("Scores:")
        for cls, score in sorted(output["all_scores"].items(), key=lambda x: -x[1]):
            print(f"  {cls}: {score}")
