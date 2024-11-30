import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load your dataset
fake_job_postings = pd.read_csv('fake_job_postings_cleaned.csv')

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer on the text data
count_vectorizer.fit(fake_job_postings['text'])

# Save the vectorizer to a file
joblib.dump(count_vectorizer, 'vectorizer.pkl')

print("Vectorizer saved as vectorizer.pkl")
