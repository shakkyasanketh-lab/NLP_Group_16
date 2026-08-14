# Member 2 — Nishen Madawa (CIT-24-01-0256)

**Models:** Logistic Regression (ML) + GRU (DL)
**Branch:** `feature/nishen_CIT-24-01-0256`
**Notebook:** `notebooks/madava.ipynb`

---

## What changed in this revision

| # | Problem in the previous version | Fix |
|---|---|---|
| 1 | The dataset cell used `csv_candidates[0]`, which picked **`Ling.csv` (2,859 rows)** instead of the combined **`phishing_email.csv` (~82,500 rows)**. Every result was computed on 3% of the data, with a 2400/458 class split. | The notebook now searches explicitly for `phishing_email.csv` and raises a clear error if it is missing. |
| 2 | Column handling only looked for `body` / `text_combined`; on the source files the subject line was thrown away. | Handles `text_combined`, `subject`+`body`, and `Email Text`, and coerces `label` to int. |
| 3 | The regularisation parameter `C` was chosen by comparing **test-set** F1 scores, then the same test set was reported as the final result. That is data leakage — the reported scores are optimistic and it is a standard viva question. | `GridSearchCV` with 5-fold CV **inside the training set only**. The test set is evaluated exactly once, at the end. |
| 4 | The deep learning model was missing entirely — `GRU` was imported but never used. The notebook stopped after saving the LR model. | Full GRU section: Keras tokenizer (fitted on training text only), padding to 200, `Embedding(128) → GRU(128) → Dropout(0.3) → Dense(64) → Dense(1, sigmoid)`, class weights, `ModelCheckpoint` on best `val_loss`, `EarlyStopping`, up to 15 epochs. |
| 5 | Class imbalance was not handled in either model. | `class_weight='balanced'` for LR, computed class weights for the GRU. |
| 6 | `madava.ipynb` was committed twice — once in `notebooks/` and once in `models/`. | The copy in `models/` was removed. |
| 7 | Screenshots were saved only to Google Drive, with generic `Screenshot 2026-07-30 ....png` names in the repo, and were produced from the wrong dataset. | The notebook saves the exact filenames the task guide requires, plus a cell that copies them into the repo folders. The old images are kept in `screenshots/archive_ling_subset/` for reference. |
| 8 | No fixed random seeds. | `SEED = 42` applied to `random`, `numpy`, and `tensorflow`. |
| 9 | `Embedding(..., input_length=...)` style code breaks on Keras 3 (current Colab). | Uses an explicit `Input` layer; checkpoint path ends in `.keras` as Keras 3 requires. |

---

## Deliverables produced by the notebook

| File | Task guide requirement |
|---|---|
| `models/lr_model.pkl` | Logistic Regression trained model |
| `models/gru_model.keras` | GRU trained model (best checkpoint) |
| `models/tfidf_member2.pkl` | TF-IDF vectorizer |
| `models/keras_tokenizer_member2.pkl` | Keras tokenizer (needed to run the GRU in the app) |
| `models/m2_results.csv` | Results table for Shakkya |
| `screenshots/m2_class_dist.png` | EDA chart 1 |
| `screenshots/m2_top_words.png` | EDA chart 2 |
| `screenshots/m2_lr_cm.png` | LR confusion matrix |
| `screenshots/m2_lr_roc.png` | LR ROC curve |
| `screenshots/m2_gru_cm.png` | GRU confusion matrix |
| `screenshots/m2_gru_roc.png` | GRU ROC curve |
| `screenshots/m2_gru_loss.png` | GRU training vs validation loss |
| `screenshots/m2_roc_comparison.png` | Both models on one ROC axis |

---

## How to run

1. Open `notebooks/madava.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.** The GRU section needs it.
3. Run every cell top to bottom. You will be asked to upload `kaggle.json` once.
4. Preprocessing on the full 82,500 emails takes roughly 5–12 minutes. GRU training takes roughly 10–25 minutes on a T4.
5. The last code cell copies all outputs into the repo folders so they can be committed.

---

## Note on the model assignment

The group README and the Member 2 task guide both list **LSTM** as this member's
deep learning model; the implemented model is a **GRU**. If the group has agreed
to the change, the README model table should be updated to match, otherwise the
notebook and the documentation contradict each other.

## Note on `requirements.txt`

`pip install -r requirements.txt` currently fails at the repo root: line 1 puts
`torch torchvision torchaudio` on a single line, and `pickle5` does not build on
Python 3.8+. This is a shared file, so it is flagged here rather than changed on
this branch.
