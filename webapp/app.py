"""
PhishGuard AI — Backend
Neural Nexus | CCS3356 — Group 16

Serves predictions from up to 6 models (one ML + one DL per member).
Each model slot is loaded independently and silently skipped if its files
aren't present yet — so this app works today with just Malitha's 2 models,
and automatically lights up more model cards as teammates push their files.

Required files per member (drop into ./models/):

  Malitha  (spaCy cleaning)
    nb_model.pkl, bow_vectorizer.pkl                        -> Naive Bayes
    cnn_model.keras, cnn_tokenizer.pkl                       -> CNN

  Shakkya  (regex cleaning: lowercase, strip HTML/URLs/emails/numbers)
    svm_model.pkl, svm_tfidf_vectorizer.pkl                  -> SVM
    bilstm_model.keras, bilstm_tokenizer.pkl                 -> BiLSTM+Attention

  Nishen   (NLTK cleaning: punctuation/number strip, NLTK stopwords+lemmatizer)
    logreg_model.pkl, logreg_tfidf_vectorizer.pkl            -> Logistic Regression
    gru_model.keras, gru_tokenizer.pkl                       -> GRU

>>> IMPORTANT: the cleaning functions below for Shakkya's and Nishen's models
    are best-guess reconstructions from the group's project doc descriptions.
    Once they share their ACTUAL preprocessing code, replace clean_regex()
    and clean_nltk() with their real functions, or predictions for their
    models will be inaccurate even though the app "runs".

Run:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python app.py

Then open http://127.0.0.1:5000
"""

import os
import re
import pickle

from flask import Flask, request, jsonify, render_template
import numpy as np
import spacy

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MAX_LEN = 200  # sequence length for all Keras models (CNN/BiLSTM/GRU)

# Registry of every model this app can potentially serve.
# "loaded" flips to True only if its files are found — missing models
# are skipped gracefully rather than crashing the app.
MODELS = {}


def try_load_sklearn(key, owner, label, model_file, vec_file, clean_fn, vec_kind):
    """Load a classic ML model (Naive Bayes / SVM / Logistic Regression)."""
    model_path = os.path.join(MODELS_DIR, model_file)
    vec_path = os.path.join(MODELS_DIR, vec_file)
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        print(f"[skip] {label} — files not found yet ({model_file}, {vec_file})")
        return
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        MODELS[key] = {
            "owner": owner, "label": label, "type": "sklearn",
            "model": model, "vectorizer": vectorizer,
            "clean_fn": clean_fn, "vec_kind": vec_kind,
        }
        print(f"[ok]   {label} loaded ({owner})")
    except Exception as e:
        print(f"[fail] {label} — {e}")


def try_load_keras(key, owner, label, model_file, tok_file, clean_fn):
    """Load a deep learning model (CNN / BiLSTM / GRU)."""
    model_path = os.path.join(MODELS_DIR, model_file)
    tok_path = os.path.join(MODELS_DIR, tok_file)
    if not (os.path.exists(model_path) and os.path.exists(tok_path)):
        print(f"[skip] {label} — files not found yet ({model_file}, {tok_file})")
        return
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        with open(tok_path, "rb") as f:
            tokenizer = pickle.load(f)
        MODELS[key] = {
            "owner": owner, "label": label, "type": "keras",
            "model": model, "tokenizer": tokenizer, "clean_fn": clean_fn,
        }
        print(f"[ok]   {label} loaded ({owner})")
    except Exception as e:
        print(f"[fail] {label} — {e}")


# ---------------------------------------------------------------------------
# Preprocessing — one cleaning style per member, matching their own pipeline
# ---------------------------------------------------------------------------
print("Loading spaCy (Malitha's cleaning) ...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_spacy(text: str) -> str:
    """Malitha: lowercase, strip URLs/emails/punctuation, spaCy stopword removal + lemmatize."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 2]
    return " ".join(tokens)


_NLTK_READY = False
def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


def clean_regex(text: str) -> str:
    """Shakkya (SVM/BiLSTM): lowercase, strip HTML/URLs/emails/numbers/punctuation.
    NOTE: best-guess reconstruction — replace with Shakkya's real function."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_nltk(text: str) -> str:
    """Nishen (LogReg/GRU): NLTK tokenize, stopword removal, WordNet lemmatize.
    NOTE: best-guess reconstruction — replace with Nishen's real function."""
    _ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    stops = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stops and len(t) > 2]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Load whatever model files are actually present
# ---------------------------------------------------------------------------
try_load_sklearn("nb", "Malitha", "Naive Bayes", "nb_model.pkl", "bow_vectorizer.pkl", clean_spacy, "bow")
try_load_keras("cnn", "Malitha", "CNN", "cnn_model.keras", "cnn_tokenizer.pkl", clean_spacy)

try_load_sklearn("svm", "Shakkya", "SVM", "svm_model.pkl", "svm_tfidf_vectorizer.pkl", clean_regex, "tfidf")
try_load_keras("bilstm", "Shakkya", "BiLSTM + Attention", "bilstm_model.keras", "bilstm_tokenizer.pkl", clean_regex)

try_load_sklearn("logreg", "Nishen", "Logistic Regression", "logreg_model.pkl", "logreg_tfidf_vectorizer.pkl", clean_nltk, "tfidf")
try_load_keras("gru", "Nishen", "GRU", "gru_model.keras", "gru_tokenizer.pkl", clean_nltk)

print(f"\n{len(MODELS)}/6 models loaded: {', '.join(m['label'] for m in MODELS.values()) or 'none'}\n")


# ---------------------------------------------------------------------------
# Preprocessing — MUST match the pipeline used during training
# (see PhishGuard_Member3_Pipeline.ipynb, Stage 2)
# ---------------------------------------------------------------------------
def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def spacy_preprocess(text: str) -> str:
    cleaned = basic_clean(text)
    doc = nlp(cleaned)
    tokens = [
        tok.lemma_
        for tok in doc
        if not tok.is_stop and not tok.is_punct and len(tok.text) > 2
    ]
    return " ".join(tokens)


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


def predict_one(key: str, raw_text: str):
    """Run a single loaded model end-to-end: clean -> vectorize -> predict."""
    entry = MODELS[key]
    cleaned = entry["clean_fn"](raw_text)
    if not cleaned:
        return None

    if entry["type"] == "sklearn":
        vec = entry["vectorizer"].transform([cleaned])
        prob = float(entry["model"].predict_proba(vec)[0][1])
    else:  # keras
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = entry["tokenizer"].texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        prob = float(entry["model"].predict(padded, verbose=0)[0][0])

    return {
        "label": entry["label"],
        "owner": entry["owner"],
        "probability": round(prob, 4),
        "risk": risk_label(prob),
    }, cleaned


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_count=len(MODELS))


@app.route("/models")
def models_status():
    """Lets the frontend know which of the 6 models are actually live."""
    return jsonify({
        key: {"label": v["label"], "owner": v["owner"]} for key, v in MODELS.items()
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    raw_text = (data or {}).get("email_text", "").strip()

    if not raw_text:
        return jsonify({"error": "Please paste an email to scan."}), 400

    if not MODELS:
        return jsonify({"error": "No trained models found in the models/ folder yet."}), 503

    per_model = {}
    preview = ""
    for key in MODELS:
        out = predict_one(key, raw_text)
        if out is None:
            continue
        result, cleaned = out
        per_model[key] = result
        if not preview:
            preview = cleaned  # first successful cleaning, just for display

    if not per_model:
        return jsonify({"error": "Couldn't extract any meaningful text from that input."}), 400

    combined_prob = float(np.mean([m["probability"] for m in per_model.values()]))

    return jsonify({
        "combined": {
            "probability": round(combined_prob, 4),
            "risk": risk_label(combined_prob),
        },
        "models": per_model,
        "cleaned_preview": preview[:200],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
