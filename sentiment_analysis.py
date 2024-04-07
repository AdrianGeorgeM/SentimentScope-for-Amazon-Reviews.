import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacytextblob.spacytextblob import SpacyTextBlob
import string

# Initialize spaCy with the English model and add TextBlob for sentiment analysis.
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Define a function to analyze the sentiment of a text using polarity.
# Outputs "Positive", "Neutral", or "Negative" based on the polarity score.
def analyze_sentiment(review_text):
    doc = nlp(review_text)
    polarity = doc._.blob.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    return 'Neutral'

# Function to clean the review text by removing punctuation and stop words.
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word.lower() not in STOP_WORDS])
    return text

# Load a subset of the dataset for testing.
data = pd.read_csv('amazon_product_reviews.csv', low_memory=False, nrows=1000)

# Preprocess the text by converting to string and cleaning.
data['reviews.text'] = data['reviews.text'].astype(str)
data['reviews.text'] = data['reviews.text'].apply(clean_text)

# Apply sentiment analysis to each review.
data['sentiment'] = data['reviews.text'].apply(analyze_sentiment)

# Save the sentiment analysis results.
data.to_csv('sentiment_analysis_output.csv', index=False)

# Test the sentiment analysis with two sample reviews.
sample_reviews = data['reviews.text'].sample(2)
for review in sample_reviews:
    print(f"Review: {review}\nSentiment: {analyze_sentiment(review)}\n")

# Document the model's accuracy with two sample reviews in a text report as per requiremnt from the reviewer of this task
with open('report.txt', 'w') as file:
    file.write("Sentiment Analysis Report\n")
    file.write("----------------------------\n")
    file.write("Sample Reviews and their Sentiment Scores:\n")
    for review in sample_reviews:
        sentiment = analyze_sentiment(review)
        file.write(f"Review: {review}\nSentiment: {sentiment}\n\n")