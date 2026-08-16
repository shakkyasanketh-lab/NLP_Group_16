# src/app.py
from flask import Flask, request, jsonify, render_template
import joblib
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

app = Flask(__name__)

# ── Preprocessing Setup ───────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
keep_words = {'not', 'no', 'never', 'won', 'free', 'urgent', 'verify'}
stop_words = stop_words - keep_words

def preprocess_email(text):
    if not text or text.strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── Attention Layer ───────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, bilstm_output):
        scores  = self.attention(bilstm_output)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(bilstm_output * weights, dim=1)
        return context, weights

# ── BiLSTM Model ──────────────────────────────────────────────
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_size * 2, 64)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(64, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        embedded       = self.dropout(self.embedding(x))
        bilstm_out, _  = self.bilstm(embedded)
        context, attn  = self.attention(bilstm_out)
        out            = self.dropout(context)
        out            = self.relu(self.fc1(out))
        out            = self.sigmoid(self.fc2(out))
        return out.squeeze(), attn

# ── Load Models ───────────────────────────────────────────────
print("Loading models...")
device = torch.device('cpu')  # Use CPU for Flask app

# Load SVM
svm_model = joblib.load('../models/svm_model.pkl')
tfidf     = joblib.load('../models/tfidf_vectorizer.pkl')
print("SVM loaded!")

# Load word2idx tokenizer
with open('../models/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
print("Tokenizer loaded!")

# Load BiLSTM
MAX_WORDS  = 50000
MAX_LEN    = 200
bilstm     = BiLSTMAttention(
    vocab_size=MAX_WORDS, embed_dim=128,
    hidden_size=128, num_layers=2, dropout=0.3
)
bilstm.load_state_dict(
    torch.load('../models/bilstm_attention_best.pt',
               map_location=device)
)
bilstm.eval()
print("BiLSTM loaded!")
print("All models ready!")

# ── Helper Functions ──────────────────────────────────────────
def tokenize_pad(text, word2idx, max_len):
    tokens = str(text).split()
    ids    = [word2idx.get(w, 1) for w in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
    return ids

def get_risk_level(confidence):
    if confidence >= 0.90:   return "CRITICAL", "#dc3545"
    elif confidence >= 0.75: return "HIGH",     "#fd7e14"
    elif confidence >= 0.55: return "MEDIUM",   "#ffc107"
    else:                    return "LOW",       "#28a745"

def get_suspicious_words(text, word2idx):
    phishing_keywords = [
        'url', 'verify', 'account', 'suspended', 'urgent',
        'click', 'password', 'login', 'confirm', 'security',
        'alert', 'winner', 'free', 'money', 'bank', 'credit',
        'update', 'expire', 'limited', 'offer', 'prize'
    ]
    words = text.lower().split()
    found = [w for w in words if w in phishing_keywords]
    return list(set(found))[:8]

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data       = request.get_json()
        email_text = data.get('email', '').strip()

        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400

        # Preprocess
        cleaned = preprocess_email(email_text)

        # ── SVM Prediction ────────────────────────────────────
        svm_tfidf  = tfidf.transform([cleaned])
        svm_pred   = int(svm_model.predict(svm_tfidf)[0])
        svm_conf   = float(svm_model.predict_proba(svm_tfidf)[0][1])

        # ── BiLSTM Prediction ─────────────────────────────────
        ids        = tokenize_pad(cleaned, word2idx, MAX_LEN)
        tensor     = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            bilstm_out, attn = bilstm(tensor)
        bilstm_conf = float(bilstm_out.item())
        bilstm_pred = int(bilstm_conf >= 0.5)

        # ── Final Decision ──────────────────────────────────────
        # Per Section 5.4 of the report, the SVM is the single model
        # selected for deployment (best F1/ROC-AUC, cheapest inference,
        # and unaffected by the BiLSTM's validation-methodology caveat
        # in Section 9.2). BiLSTM is run and shown in the UI purely as
        # a reference signal — it does NOT influence the verdict below.
        verdict     = "PHISHING" if svm_conf >= 0.5 else "SAFE"
        risk, color = get_risk_level(svm_conf)

        # ── Suspicious Words ──────────────────────────────────
        suspicious = get_suspicious_words(cleaned, word2idx)

        return jsonify({
            'verdict':          verdict,
            'risk_level':       risk,
            'risk_color':       color,
            'confidence':       round(svm_conf * 100, 1),
            'svm_confidence':   round(svm_conf * 100, 1),
            'svm_verdict':      'PHISHING' if svm_pred else 'SAFE',
            'bilstm_confidence':round(bilstm_conf * 100, 1),
            'bilstm_verdict':   'PHISHING' if bilstm_pred else 'SAFE',
            'suspicious_words': suspicious,
            'cleaned_text':     cleaned[:200]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)