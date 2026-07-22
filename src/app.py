"""
Step 5: Flask web app exposing the classifier as an API + simple UI.

Usage:
    python src/app.py
Then open http://127.0.0.1:5000 in your browser.

Endpoints:
    GET  /              -> simple web UI to try the classifier
    POST /api/classify   -> JSON API. Body: {"text": "..."}
    GET  /api/health      -> health check
"""
import os
from flask import Flask, request, jsonify, render_template

from predict import load_model, predict_ticket

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Load model once at startup
try:
    MODEL = load_model()
except FileNotFoundError:
    MODEL = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/api/classify", methods=["POST"])
def classify():
    if MODEL is None:
        return jsonify({
            "error": "Model not trained yet. Run `python src/train.py` first."
        }), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    result = predict_ticket(text, model=MODEL)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
