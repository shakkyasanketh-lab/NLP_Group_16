# PhishGuard AI — Frontend + Backend (Member 3)

## Setup
```bash
pip install -r requirements.txt --break-system-packages
python -m spacy download en_core_web_sm
```

## Add your trained models
Copy these 4 files from your `models/` folder in the repo into this project's `models/` folder:
- nb_model.pkl
- bow_vectorizer.pkl
- cnn_model.keras
- tokenizer.pkl

## Run
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in Chrome and Firefox (Task Guide requires testing both).

## What it does
- Paste any email text into the console box, click "Run Scan".
- Backend runs YOUR exact preprocessing (spaCy: lowercase, URL/email strip, punctuation removal, stopword removal, lemmatisation) then feeds it to both saved models.
- Naive Bayes uses the saved BoW vectorizer; CNN uses the saved tokenizer + padding (maxlen=200), matching training exactly.
- Combined verdict = average of both model probabilities.
- Risk badge/colour: green <40%, amber 40-70%, red >70% (adjust in app.py `risk_label()` if you want different thresholds).

## Folder structure
```
phishguard-app/
├── app.py              # Flask backend, /predict endpoint
├── requirements.txt
├── models/              # put your 4 model files here (not committed if large — see note)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/script.js
```

## Note on git
`cnn_model.keras` and `tokenizer.pkl` may be too big for GitHub's website upload (25MB limit) — 
you already solved this for the `models/` folder in your main repo. For this app folder specifically,
you can either:
  (a) not commit the model files here and just reference the ones in the top-level `models/` folder 
      by adjusting MODELS_DIR in app.py, or
  (b) use git CLI (`git add` + `git push`) which doesn't have the 25MB web-upload limit.
