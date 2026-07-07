
import re
import string
import pickle
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the trained model and vectorizer
# Ensure these files are in the same directory as app.py or provide full paths
try:
    with open('LogisticRegression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Make sure 'LogisticRegression_model.pkl' and 'vectorizer.pkl' are in the same directory.")
    model = None
    vectorizer = None

# Initialize NLTK components (download if not already present in the environment)
# In a production environment, these downloads should be handled in a setup script or Dockerfile
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing functions (copied from the notebook)
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokenized_text = word_tokenize(cleaned_text)
    no_stopwords_text = remove_stopwords(tokenized_text)
    lemmatized_text = apply_lemmatization(no_stopwords_text)
    return ' '.join(lemmatized_text)

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API! Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded.'}), 500

    data = request.get_json(force=True)
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided for prediction.'}), 400

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Transform the processed text using the loaded TF-IDF vectorizer
    text_vectorized = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_vectorized)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'

    return jsonify({'text': text, 'predicted_sentiment': sentiment})

if __name__ == '__main__':
    # For deployment, consider using a production-ready WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000)
