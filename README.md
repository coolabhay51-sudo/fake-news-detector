# Fake News Detector 🔍
**MCA Final Year Project | Sandip University | 2024–2026**
Prashant Kumar Prabhat | PRN: 240205221083
Guide: Prof. Ratneshwar Kumar Singh

---

## Apne Computer par Kaise Chalayein (Step by Step)

### Step 1 — Python install karo (agar nahi hai)
https://www.python.org/downloads/ se Python 3.8+ download karo

### Step 2 — Project folder open karo
```
Terminal / Command Prompt open karo
cd fakenews_app
```

### Step 3 — Libraries install karo
```
pip install -r requirements.txt
```

### Step 4 — ML Model train karo (ek baar)
```
python train_model.py
```
Yeh model.pkl aur vectorizer.pkl files banayega.

### Step 5 — Website start karo
```
python app.py
```

### Step 6 — Browser mein kholo
```
http://localhost:5000
```
Mobile mein same WiFi par: http://[aapka-PC-ka-IP]:5000

---

## Internet par Free Deploy karna (Render.com)

1. GitHub par account banao
2. Is project ko GitHub par upload karo
3. render.com par jaao — New Web Service
4. GitHub repo connect karo
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `gunicorn app:app`
7. Deploy karo — link milega jaise: https://fake-news-app.onrender.com

---

## Project Structure
```
fakenews_app/
├── app.py              ← Main Flask application
├── train_model.py      ← ML model training script
├── requirements.txt    ← Python libraries
├── model.pkl           ← Trained model (after training)
├── vectorizer.pkl      ← TF-IDF vectorizer (after training)
├── dataset.csv         ← Your dataset (optional)
└── templates/
    └── index.html      ← Website HTML
```

---

## Real Dataset Use Karna
dataset.csv file banao with columns:
- `text` — news article text
- `label` — 1 for Real, 0 for Fake

LIAR dataset: https://huggingface.co/datasets/liar
FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet
