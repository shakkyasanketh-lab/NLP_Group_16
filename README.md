# PhishGuard AI 🛡️
### Intelligent Phishing Email Detection & Classification System
**CCS3356 — Natural Language Processing | Sri Lanka Technology Campus**

---

## 👥 Group: Neural Nexus (Group 16)

| Member | Student ID | Branch | Models |
|--------|-----------|--------|--------|
| Shakkya Sanketh (Leader) | CIT-24-01-0573 | feature/shakkya_CIT-24-01-0573 | SVM + BiLSTM+Attention |
| Nishen Madava | CIT-24-01-0256 | feature/nishen_CIT-24-01-0256 | Logistic Regression + GRU |
| Malitha Gayashan | CIT-24-01-0562 | feature/malitha_CIT-24-01-0562 | Naive Bayes + CNN |

---

## 📌 Problem Statement
Phishing emails are the #1 cyber attack vector globally. PhishGuard AI uses 
NLP and machine learning to automatically detect phishing emails, classify 
the attack type, and explain which words triggered the detection.

---

## 📦 Dataset
- **Source:** Phishing Email Dataset — Kaggle (naserabdullahalam)
- **Size:** ~82,500 labelled emails
- **Classes:** Phishing (1) vs Legitimate (0)
- **Combined from:** Enron, Ling-Spam, CEAS-08, Nazario, Nigerian Fraud, SpamAssassin

---

## 🔧 Setup Instructions
```bash
git clone https://github.com/YourUsername/NLP_Group_16.git
cd NLP_Group_16
pip install -r requirements.txt
```

---

## 🚀 How to Run
```bash
cd src
python app.py
# Open browser → http://localhost:5000
```

---

## 📊 Model Summary
| Model | Type | Member | F1 Score |
|-------|------|--------|----------|
| SVM | ML | Shakkya | TBD |
| BiLSTM + Attention | DL | Shakkya | TBD |
| Logistic Regression | ML | Nishen | TBD |
| GRU | DL | Nishen | TBD |
| Naive Bayes | ML | Malitha | TBD |
| CNN | DL | Malitha | TBD |

---

## 📁 Repository Structure
```
NLP_Group_16/
├── data/           ← Dataset files
├── notebooks/      ← Jupyter notebooks per member  
├── src/            ← Python source files + Flask app
├── models/         ← Saved trained models
├── reports/        ← Final report
├── screenshots/    ← App and result screenshots
├── videos/         ← Progress video
├── requirements.txt
└── README.md
```