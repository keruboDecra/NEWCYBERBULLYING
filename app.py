import streamlit as st
import re
import joblib
import numpy as np
import nltk
nltk.download('wordnet')
# Download NLTK resources
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from nltk.stem import WordNetLemmatizer

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function to predict using the loaded model
def predict_cyberbullying(text):
    try:
        # Load the model and vectorizer
        model = joblib.load('svm_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer')

        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Transform the preprocessed text using the loaded vectorizer
        text_tfidf = Vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = model.predict(text_tfidf)

        return prediction[0]
    except Exception as e:
        st.error(f"Error loading or using the model: {e}")
        return None

# Streamlit UI
st.title('Cyberbullying Detection App')

# Input text box
user_input = st.text_area("Enter a text:", "")

# Check if the user has entered any text
if user_input:
    # Make prediction
    prediction = predict_cyberbullying(user_input)

    # Display the prediction
    if prediction is not None:
        st.write(f"Prediction: {'Cyberbullying' if prediction == 1 else 'Not Cyberbullying'}")
