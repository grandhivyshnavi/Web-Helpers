import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load your dataset
data = pd.read_csv('C:/SIP/Web Helpers/amazon_product_data.csv')

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['review_body'])
y = data['Sentiment_ebook']

# Create and train the Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X, y)

# Save the model and vectorizer
joblib.dump(nb_model, 'multinomial_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
