"""
Sentiment analysis on Amazon product reviews using spaCy + spacytextblob.

What this script does:
1) Loads the Datafiniti Amazon reviews dataset (CSV).
2) Selects the 'reviews.text' column and drops missing values.
3) Preprocesses text (lowercase/strip + removes stopwords using spaCy).
4) Uses TextBlob sentiment (via spacytextblob) to compute polarity.
5) Classifies each review as Positive / Neutral / Negative based on
   polarity.
6) Tests the model on sample reviews and compares similarity between
   two reviews.
"""

from pathlib import Path
import re
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Configuration
CSV_FILENAME = \
 "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
SPACY_MODEL = "en_core_web_md"
REPORT_FILENAME = "sentiment_analysis_report.pdf"

# TextBlob polarity ranges from -1 (very negative) to +1 (very positive).
POS_THRESHOLD = 0.10
NEG_THRESHOLD = -0.10


# Helper functions
def load_spacy_pipeline():
    """
    This function loads spaCy model and adds
    spacytextblob for sentiment.
    """
    nlp = spacy.load(SPACY_MODEL)
    if "spacytextblob" not in nlp.pipe_names:
        nlp.add_pipe("spacytextblob")
    return nlp


def basic_clean_text(text):
    """
    This function does the following:
    - Converts to string safely (handles non-string values)
    - Lowercase text and strips whitespace
    - Collapses repeated whitespace
    """
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_stopwords_spacy(nlp, text):
    """
    This function removes stopwords using spaCy token attributes.
    """
    doc = nlp(text)
    kept_tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if token.is_stop:
            continue
        kept_tokens.append(token.text)
    return " ".join(kept_tokens)


def predict_sentiment(nlp, review_text):
    """
    This function takes a review as input and returns:
    - cleaned_text
    - polarity score
    - sentiment label (Positive / Neutral / Negative)
    """
    cleaned = basic_clean_text(review_text)

    # Stopword removal
    cleaned_no_stops = remove_stopwords_spacy(nlp, cleaned)

    doc = nlp(cleaned_no_stops)
    polarity = doc._.blob.polarity

    if polarity > POS_THRESHOLD:
        label = "Positive"
    elif polarity < NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return cleaned_no_stops, polarity, label


# Load spaCy pipeline
nlp = load_spacy_pipeline()

# Load dataset
csv_path = Path(CSV_FILENAME)
if not csv_path.exists():
    raise FileNotFoundError(
        f"CSV file not found: {csv_path.resolve()}\n"
        f"Place '{CSV_FILENAME}' in the same folder as this script."
    )

df = pd.read_csv(csv_path)

# Clean data by removing missing review text
clean_df = df.dropna(subset=["reviews.text"]).copy()
reviews_data = clean_df["reviews.text"]

print("Rows in original dataset:", len(df))
print("Rows after dropping missing reviews.text:", len(clean_df))

# Run sentiment for a small sample
print("\n--- Sample predictions (first 5 reviews) ---")
sample_indices = clean_df.index[:5]
for idx in sample_indices:
    raw_review = clean_df.loc[idx, "reviews.text"]
    cleaned, polarity, label = predict_sentiment(nlp, raw_review)
    print(f"\nIndex: {idx}")
    print("Raw:", raw_review)
    print("Cleaned (no stopwords):", cleaned)
    print("Polarity:", polarity)
    print("Predicted:", label)

# Run sentiment for ALL reviews
print("\nComputing sentiment for all reviews: ")
polarities = []
labels = []
cleaned_texts = []

for review in reviews_data.tolist():
    cleaned, polarity, label = predict_sentiment(nlp, review)
    cleaned_texts.append(cleaned)
    polarities.append(polarity)
    labels.append(label)

clean_df["cleaned_review"] = cleaned_texts
clean_df["polarity"] = polarities
clean_df["sentiment_label"] = labels

# Distribution summary
label_counts = clean_df["sentiment_label"].value_counts(dropna=False)
polarity_stats = clean_df["polarity"].describe()

print("\n--- Sentiment Distribution ---")
print(label_counts)

print("\n--- Polarity Statistics ---")
print(polarity_stats)

print("\n--- Sentiment label distribution ---")
for k, v in label_counts.items():
    print(k, ":", v)

# Review similarity demo
print("\n--- Similarity demo between two reviews ---")
if len(clean_df) >= 2:
    r1 = clean_df["cleaned_review"].iloc[0]
    r2 = clean_df["cleaned_review"].iloc[1]
    sim = nlp(r1).similarity(nlp(r2))
    print("Review 0:", r1[:120], "...")
    print("Review 1:", r2[:120], "...")
    print("Similarity score:", sim)
