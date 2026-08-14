# Member 2 — Final Results

**Nishen Madawa · CIT-24-01-0256 · Logistic Regression + GRU**
Branch: `feature/nishen_CIT-24-01-0256`

---

## Experimental setup

| Item | Value |
|---|---|
| Dataset | `phishing_email.csv` (Kaggle, naserabdullahalam) — the combined file |
| Emails after cleaning | **82,483** |
| Class balance | 42,891 phishing (52.0%) / 39,592 legitimate (48.0%) |
| Split | 80 / 20 stratified — 65,986 train / 16,497 test |
| Preprocessing | lowercase → remove URLs, emails, digits, punctuation → NLTK tokenize → remove stopwords → WordNet lemmatize |
| ML features | TF-IDF, 10,000 features, unigrams + bigrams, `min_df=2`, fitted on training data only |
| DL features | Keras tokenizer (20,000-word vocabulary, fitted on training data only), sequences padded/truncated to 200 |
| Random seed | 42 (`random`, `numpy`, `tensorflow`) |

---

## Results table — send this to Shakkya

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9853 | 0.9857 | 0.9861 | **0.9859** | **0.9986** |
| GRU | 0.9861 | 0.9850 | 0.9882 | **0.9866** | **0.9978** |

All figures are on the held-out 16,497-email test set, which was used exactly once.

---

## Logistic Regression — detail

- Hyperparameter `C` selected by `GridSearchCV` over `{0.01, 0.1, 1, 10, 100}`, 5-fold stratified CV **on the training set only**. Best: **C = 10.0** (CV F1 = 0.9850).
- `class_weight='balanced'`.
- 10-fold cross-validation F1 on the training set: **0.9832 ± 0.0023** — close to the 0.9859 test F1, so the model is not overfitting.

**Confusion matrix**

|  | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 7,796 (TN) | 123 (FP) |
| **Actual Phishing** | 119 (FN) | 8,459 (TP) |

123 legitimate emails were wrongly flagged; 119 phishing emails slipped through.

**Most influential features** (highest positive coefficients → phishing):
`replica, http, remove, pill, medication, med, love, click, site, account, quality, viagra, life, money, investment`

Towards legitimate: `enron, wrote, thanks, university, vince, louise, opensuse, tony, perl, question, python`

---

## GRU — detail

**Architecture**

```
Input(200)
Embedding(vocab=20000, dim=128)     2,560,000 params
GRU(128)                               99,072 params
Dropout(0.3)
Dense(64, relu)                         8,256 params
Dense(1, sigmoid)                          65 params
--------------------------------------------------
Total                               2,667,393 params
```

- Optimizer Adam (lr = 1e-3), loss binary cross-entropy, batch size 256.
- Class weights computed with `compute_class_weight('balanced', ...)`.
- Trained up to 15 epochs with `ModelCheckpoint` (best `val_loss`) and `EarlyStopping(patience=3, restore_best_weights=True)`.
- **Stopped early at epoch 6; the best epoch was epoch 3** (val_loss 0.0514). Training loss kept falling after epoch 3 while validation loss flattened — that is the overfitting point, and the checkpoint is why the reported model is the epoch-3 one.

| Epoch | Train loss | Val loss | Val accuracy |
|---|---|---|---|
| 1 | 0.6422 | 0.1941 | 0.9306 |
| 2 | 0.0867 | 0.0544 | 0.9821 |
| **3** | **0.0345** | **0.0514** | **0.9844** |
| 4 | 0.0211 | 0.0544 | 0.9833 |
| 5 | 0.0131 | 0.0532 | 0.9859 |
| 6 | 0.0096 | 0.0538 | 0.9842 |

**Confusion matrix**

|  | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 7,790 (TN) | 129 (FP) |
| **Actual Phishing** | 101 (FN) | 8,477 (TP) |

---

## Written analysis (for the report / Week 9 submission)

Logistic Regression on TF-IDF features reached an F1 of 0.9859 with a ROC-AUC of 0.9986 on the held-out test set, and the GRU reached an F1 of 0.9866 with a ROC-AUC of 0.9978. The two models are within 0.07 percentage points of each other on F1, so on this dataset the deep learning model gives no meaningful accuracy advantage over the linear baseline.

The difference is in the error profile. The GRU caught 18 more phishing emails than Logistic Regression (101 false negatives versus 119) at the cost of 6 more false alarms (129 versus 123). For a phishing filter the false negative is the more expensive error — it is the one that reaches the user's inbox — so the GRU is marginally preferable on that axis, though the margin is small enough to be within run-to-run variance.

Logistic Regression's practical advantages are decisive here: it trains in seconds against roughly twelve minutes for the GRU, and its coefficients are directly interpretable, which is what makes the "which words triggered this?" explanation feature of the PhishGuard app possible. Its limitation is that TF-IDF is a bag-of-words representation, so word order is discarded. The GRU reads the email as a sequence and can in principle represent phrasing such as "your account will be suspended" as a pattern rather than three independent tokens, but the near-identical scores suggest phishing emails in this corpus are already separable from vocabulary alone — the sequential context adds little. The GRU is also opaque: there is no coefficient to point at when explaining a decision.

One caveat worth stating: both models score highly partly because the corpus is assembled from distinct sources (Enron for legitimate, Nigerian Fraud and Nazario for phishing), so some of the signal is corpus provenance rather than phishing style — note that the strongest "legitimate" features are `enron`, `vince` and `louise`, which are Enron-specific terms rather than general markers of legitimate email. Performance on genuinely unseen mail would very likely be lower.

---

## Viva preparation — your actual numbers

**What was your best F1 score?**
My GRU achieved an F1 of 0.9866 and my Logistic Regression 0.9859, both on a held-out test set of 16,497 emails.

**Know your confusion matrix — how many false positives, and why?**
My Logistic Regression produced 123 false positives out of 7,919 legitimate emails (1.6%). These are legitimate emails containing vocabulary that is statistically associated with phishing in the training data — marketing language, financial terms, and embedded links. Because I used `class_weight='balanced'`, the model is slightly biased toward catching phishing, which trades a few false positives for fewer missed phishing emails.

**Why did you choose Logistic Regression?**
It is a strong, fast, interpretable baseline for binary text classification and pairs naturally with TF-IDF features. I wanted a baseline to measure the deep learning model against — and in this case it turned out to essentially match it.

**Why GRU rather than LSTM?**
A GRU is a gated recurrent network with two gates (update and reset) instead of the LSTM's three (input, forget, output). It has fewer parameters, trains faster, and typically performs comparably on medium-sized text datasets. Given a 200-token sequence length over 82,000 emails, the training-time saving was worth it.

**How did you choose C?**
With `GridSearchCV` over five values, using 5-fold cross-validation inside the training set only. I deliberately did not select it against the test set, because that would leak test information into model selection and make the reported score optimistic.

**Why did training stop at epoch 6?**
Early stopping with patience 3. Validation loss reached its minimum at epoch 3 and did not improve for the next three epochs, so training halted and the epoch-3 weights were restored. After epoch 3 the training loss kept falling while validation loss did not — the model had started to overfit.

---

## Generated files

| File | Description |
|---|---|
| `models/lr_model.pkl` | Tuned Logistic Regression (C = 10.0) |
| `models/gru_model.keras` | GRU, best checkpoint (epoch 3) |
| `models/tfidf_member2.pkl` | TF-IDF vectorizer |
| `models/keras_tokenizer_member2.pkl` | Keras tokenizer for the GRU |
| `models/m2_results.csv` | Results table |
| `models/lr_results.json`, `models/gru_results.json` | Full metrics |
| `models/gru_history.json` | Per-epoch training history |
| `screenshots/m2_class_dist.png` | Class distribution |
| `screenshots/m2_top_words.png` | Top 50 phishing words |
| `screenshots/m2_lr_cm.png`, `m2_lr_roc.png` | LR confusion matrix and ROC |
| `screenshots/m2_gru_cm.png`, `m2_gru_roc.png`, `m2_gru_loss.png` | GRU confusion matrix, ROC, loss curve |
| `screenshots/m2_roc_comparison.png` | Both models on one ROC axis |
| `reports/m2_run_log.txt`, `reports/m2_gru_log.txt` | Full console output from the training runs |
