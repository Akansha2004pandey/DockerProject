from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import re
import string
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

MAX_LEN = 200  # Match the maxlen used during training

# Initialize FastAPI app
app = FastAPI(title="LSTM Based Sentiment Analysis App", summary="Sentiment Analysis API")

# Load your model
model = load_model("lstm_model.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=5000)

# Preprocessing Functions
def clean_text(text):
    """Clean the text by removing HTML, punctuation, numbers, and stopwords."""
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = text.lower()  # Convert to lowercase

    # Remove punctuation and numbers
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    
    return " ".join(filtered)

def preprocess_text(text):
    """Preprocess text for prediction."""
    cleaned = clean_text(text)  # Clean the text
    tokenizer.fit_on_texts([cleaned])  # Fit tokenizer on the cleaned text (simulating training behavior)
    sequence = tokenizer.texts_to_sequences([cleaned])  # Convert text to sequence
    padded = pad_sequences(sequence, maxlen=200)  # Pad the sequence
    return padded

# Pydantic model for input data
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: InputText):
    """Predict sentiment based on input text."""
    preprocessed = preprocess_text(data.text)  # Preprocess the input text
    prediction = model.predict(preprocessed)[0][0]  # Get the prediction (binary classification)
    
    # Convert prediction to label
    label = "Positive" if prediction > 0.5 else "Negative"
    
    return {
        "text": data.text,
        "cleaned_text": preprocess_text(data.text).tolist(),
        "prediction_score": float(prediction),
        "label": label
    }
