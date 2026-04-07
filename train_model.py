"""
train_model.py
Run this ONCE to train and save the model.
Usage: python train_model.py

If you don't have the dataset, it creates a small demo model.
"""
import pickle, re, os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join([stemmer.stem(t) for t in tokens])

# ── Try to load real dataset ──────────────────────────────────
try:
    import pandas as pd
    df = pd.read_csv('dataset.csv')
    print(f"✅ Dataset loaded: {len(df)} rows")
    df['clean'] = df['text'].astype(str).apply(preprocess)
    X = df['clean']
    y = df['label']
except Exception as e:
    print(f"⚠️  No dataset found ({e}). Creating demo model...")
    # Demo dataset — real-world-like sentences
    samples = [
        ("Breaking: Scientists discover new vaccine with 95% efficacy in clinical trials", 1),
        ("SHOCKING: Government hiding alien contact since 1947, leaked documents reveal", 0),
        ("Parliament passes new education budget increasing funds by 20 percent", 1),
        ("Drinking bleach cures all diseases doctors dont want you to know", 0),
        ("Stock market rises after positive economic indicators released", 1),
        ("Celebrity caught shape-shifting on live television video goes viral", 0),
        ("New study shows regular exercise reduces heart disease risk significantly", 1),
        ("5G towers spreading COVID-19 virus scientists confirm in secret meeting", 0),
        ("Government announces infrastructure development plan for rural areas", 1),
        ("Microchips found in COVID vaccines controlling human minds report", 0),
        ("University researchers publish findings on climate change impact", 1),
        ("Secret society controls world governments and banking systems", 0),
        ("Local hospital reports decline in disease cases after vaccination drive", 1),
        ("Moon landing was faked in Hollywood studio NASA admits secretly", 0),
        ("New policy aims to reduce carbon emissions by thirty percent by 2030", 1),
        ("Bananas being modified to transmit mind control signals to humans", 0),
        ("Election results certified after thorough audit by independent body", 1),
        ("Ancient prophecy predicts end of world next month experts warn", 0),
        ("Tech company launches affordable smartphones targeting rural markets", 1),
        ("Fluoride in water supply used to make population docile and obedient", 0),
        ("Central bank announces interest rate decision after policy meeting", 1),
        ("Chemtrails from planes contain population control chemicals leaked report", 0),
        ("New agricultural policy to support farmers with subsidies and loans", 1),
        ("Doctors secretly replacing patients organs with robot parts hospitals deny", 0),
        ("Supreme court delivers landmark judgment on fundamental rights case", 1),
        ("Eating raw onions daily prevents all cancers big pharma hiding truth", 0),
        ("Minister inaugurates new expressway connecting major cities", 1),
        ("Pigeons trained as government spies filming civilians secret program", 0),
        ("Annual report shows improvement in literacy rates across rural districts", 1),
        ("Bill Gates microchip injected through mosquito bites in night", 0),
    ] * 30  # Repeat to have enough data

    import random
    random.shuffle(samples)
    texts, labels = zip(*samples)
    import pandas as pd
    df = pd.DataFrame({'text': texts, 'label': labels})
    df['clean'] = df['text'].apply(preprocess)
    X = df['clean']
    y = df['label']
    print(f"✅ Demo dataset created: {len(df)} samples")

# ── Train / Test Split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ── TF-IDF ───────────────────────────────────────────────────
print("⏳ Training TF-IDF vectorizer...")
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train_v = vec.fit_transform(X_train)
X_test_v  = vec.transform(X_test)

# ── Stacking Ensemble ─────────────────────────────────────────
print("⏳ Training Stacking Ensemble model...")
lr  = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

stack = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf)],
    final_estimator=LogisticRegression(max_iter=500),
    cv=3
)
stack.fit(X_train_v, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_pred = stack.predict(X_test_v)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ── Save ──────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(stack, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vec, f)

print("\n✅ model.pkl and vectorizer.pkl saved successfully!")
print("🚀 Now run:  python app.py")
