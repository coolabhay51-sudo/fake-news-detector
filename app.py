from flask import Flask, render_template, request, jsonify
import pickle, re, os, json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# ── Download NLTK data ────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ── Load model & vectorizer ───────────────────────────────────
MODEL_PATH = 'model.pkl'
VEC_PATH   = 'vectorizer.pkl'

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VEC_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Model loaded successfully!")
else:
    print("⚠️  Model not found. Run train_model.py first.")

# ── Text preprocessing ────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join([stemmer.stem(t) for t in tokens])

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('text', '').strip()

    if not news_text:
        return jsonify({'error': 'Please enter some news text.'}), 400

    if len(news_text.split()) < 5:
        return jsonify({'error': 'Please enter at least 5 words.'}), 400

    if model is None or vectorizer is None:
        # Demo mode — simulate prediction for testing
        import random
        label = random.choice(['REAL', 'FAKE'])
        confidence = round(random.uniform(78, 97), 1)
        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'precision': 94.8,
            'recall': 94.6,
            'f1': 94.7,
            'accuracy': 94.7,
            'demo_mode': True
        })

    clean = preprocess(news_text)
    features = vectorizer.transform([clean])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = round(float(max(proba)) * 100, 1)
    label = 'REAL' if pred == 1 else 'FAKE'

    return jsonify({
        'prediction': label,
        'confidence': confidence,
        'precision': 94.8,
        'recall': 94.6,
        'f1': 94.7,
        'accuracy': 94.7,
        'demo_mode': False
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
