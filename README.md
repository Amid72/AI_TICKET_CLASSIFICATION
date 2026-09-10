# AI Ticket Classification System
https://amid.pythonanywhere.com

A complete, working project that automatically classifies customer support
tickets into categories (**Billing, Technical, Account, General**) using a
TF-IDF + Logistic Regression model, served through a Flask API with a simple
web UI.

```
ai_ticket_classification/
├── data/
│   └── tickets.csv          # sample labeled ticket dataset
├── src/
│   ├── preprocess.py        # text cleaning utilities
│   ├── train.py              # trains and saves the model
│   ├── predict.py            # CLI prediction script
│   └── app.py                 # Flask API + web UI
├── models/                   # trained model gets saved here
├── templates/
│   └── index.html            # simple web UI
├── static/                   # (empty, for future CSS/JS/images)
├── requirements.txt
└── README.md
```

## Step 1 — Set up your environment

```bash
cd ai_ticket_classification
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2 — Review / expand the dataset

Open `data/tickets.csv`. It has two columns: `text` and `category`.
50 sample tickets across 4 categories are included so the project runs
out of the box. For a real project, replace or add to this with your own
labeled tickets — the more examples per category, the better the model
performs. Aim for at least 50–100 examples per category for solid results.

## Step 3 — Train the model

```bash
cd src
python train.py
```

This will:
- Load and clean the ticket text (`preprocess.py`)
- Split the data into train/test sets
- Train a TF-IDF + Logistic Regression pipeline
- Print accuracy and a classification report
- Save the trained pipeline to `../models/ticket_classifier.pkl`

## Step 4 — Test predictions from the command line

```bash
python predict.py "I was charged twice for my subscription this month"
```

You'll see the predicted category plus a confidence score for each class.

## Step 5 — Run the web app / API

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser to use the web UI, or
call the API directly:

```bash
curl -X POST http://127.0.0.1:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My app keeps crashing when I log in"}'
```

Response:
```json
{
  "text": "My app keeps crashing when I log in",
  "predicted_category": "Technical",
  "confidence": 0.87,
  "all_scores": {
    "Account": 0.04,
    "Billing": 0.02,
    "General": 0.07,
    "Technical": 0.87
  }
}
```

## Step 6 — Next steps to extend the project

- **More data**: swap in your real historical support tickets for better accuracy.
- **More categories**: add new labels to `tickets.csv` — no code changes needed, the model adapts automatically.
- **Better models**: try `RandomForestClassifier`, `LinearSVC`, or fine-tune a transformer (e.g. `distilbert-base-uncased`) via Hugging Face for higher accuracy on larger datasets.
- **Deploy**: containerize `app.py` with Docker and deploy to Render, Railway, or AWS.
- **Database integration**: connect to a real ticketing system (Zendesk, Freshdesk, Jira Service Desk) via their APIs to auto-tag incoming tickets.
- **Retraining pipeline**: schedule periodic retraining as new labeled tickets come in.

## Troubleshooting

- **`ModuleNotFoundError`**: make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.
- **"No trained model found"**: run `python train.py` before `predict.py` or `app.py`.
- **Low accuracy**: add more labeled examples per category in `tickets.csv`, especially for categories the model confuses most (check the classification report from `train.py`).
