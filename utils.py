
import json
import re
import string
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

MAX_LEN = 100  # Match the maxlen used during training

def load_tokenizer(path="tokenizer.json"):
    with open(path) as f:
        data = json.load(f)
    return tokenizer_from_json(data)

def clean_text(text):
   
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()

    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    
    text = re.sub(r"\s+", " ", text).strip()

    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    
    return " ".join(filtered)

def preprocess_text(text, tokenizer):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    return padded
