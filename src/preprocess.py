"""
Step 2: Text preprocessing utilities.

Cleans raw ticket text before it is fed into the vectorizer/model.
"""
import re
import string


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers/extra whitespace from a ticket string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                         # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                 # collapse whitespace
    return text


def clean_series(series):
    """Apply clean_text to a pandas Series of ticket texts."""
    return series.astype(str).apply(clean_text)
