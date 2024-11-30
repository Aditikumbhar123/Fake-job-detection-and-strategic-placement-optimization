import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load your dataset
fake_job_postings = pd.read_csv('fake_job_postings_cleaned.csv')

# Prepare the data
X = fake_job_postings['text']
y = fake_job_postings['fraudulent']  # Assuming this is your target column

# Create the CountVectorizer and fit it
count_vectorizer = CountVectorizer(stop_words='english')
X_vectorized = count_vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(count_vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved as model.pkl and vectorizer.pkl")
