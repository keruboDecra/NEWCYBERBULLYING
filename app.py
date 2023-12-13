import streamlit as st
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load SVM model and TF-IDF vectorizer
model_path = 'svm_model.joblib'  # Adjust the path if necessary
model = joblib.load(model_path)
vectorizer_path = 'tfidf_vectorizer.joblib'  # Adjust the path if necessary
vectorizer = joblib.load(vectorizer_path)

# Function for text preprocessing
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
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Transform the preprocessed text using the loaded vectorizer
        text_tfidf = vectorizer.transform([preprocessed_text])

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
