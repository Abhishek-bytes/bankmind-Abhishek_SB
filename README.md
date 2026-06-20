# 🏦 BankMind — Customer Subscription Predictor
A machine-learning system that predicts whether a bank customer is likely to subscribe
to a term deposit, built for the **BankMind Challenge (Omdena)**.

- **Track B** — ML Engineer: EDA → model comparison → evaluation → EXPLANATION.md


## 🚀 Live Demo


## What's in here

```
bankmind-abhishek/
├── data/
│   └── bank-full.csv              # UCI Bank Marketing dataset (45,211 rows)
├── train_model.py                 # EDA + 6-model ensemble training + evaluation
├── app.py                          # ← Streamlit deployment UI (Track B showcase)
├── model/
│   ├── bankmind_model.pkl         # saved best pipeline (created by train_model.py)
│   └── metrics_summary.json       # all-model metrics (created by train_model.py)
├── notebooks/
│   └── bankmind_walkthrough.ipynb # end-to-end notebook with SHAP section
├── requirements.txt
├── EXPLANATION.md                 # ← required answers for Track B/C graders
└── README.md
```

## A design decision worth flagging up front

The raw dataset has campaign-contact columns: `duration`, `campaign`,
`pdays`, `previous`, `poutcome`, `contact`, `day`, `month`. **None of these
are used by the deployed model.**

Two reasons:

1. **Leakage.** `duration` is the length of the sales call itself — average
   ~537s for customers who subscribed vs. ~221s for those who didn't. That's
   only known *after* the RM has already called the customer, so a model
   that depends on it can't actually be used to decide who to call.
2. **The business problem.** This is meant to be a pre-contact recommender —
   "which customer should an RM call, and about what" — so the model only
   uses information available *before* any contact: `age`, `job`, `marital`,
   `education`, `default`, `balance`, `housing`, `loan`.

This does mean the model's recall/precision numbers are more modest than
you'd get by including `duration` — that's the realistic trade-off, not a
bug. See `EXPLANATION.md` Q3 and Q4 for more on this.

## 🏃 Quick Start (Local)

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd bankmind-abhishek
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

This runs the complete pipeline:
- **EDA**: Analyzes 45,211 customers, identifies 11.7% subscription rate
- **Training**: Builds 6 ensemble models (Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Voting, Stacking)
- **Evaluation**: Reports accuracy, precision, recall, F1 + classification_report
- **Predictions**: Shows 5 sample predictions from test set
- **Output**: Saves best model → `model/bankmind_model.pkl`

Expected runtime: ~5-10 minutes. Console output shows detailed progress.

### 3. View Interactive Dashboard
```bash
streamlit run streamlit_app.py
```

Opens `http://localhost:8501` with 4 tabs:
- **🔮 Prediction** — Enter customer profile, see P(yes) + key driving factors
- **📊 Leaderboard** — Compare all 6 models by F1 score
- **📈 Feature Importance** — Visualize which features matter most
- **ℹ️ About** — EDA insights, model architecture, data dictionary

---

## ✅ Track B Requirements Met

| Requirement | Status | Evidence |
|---|---|---|
| EDA (class distribution, missing values) | ✅ | `run_eda()` shows 11.7% imbalance clearly |
| Train & compare models (2+) | ✅ | **6 models** trained (exceeds spec) |
| Proper evaluation (accuracy, precision, recall, F1, classification_report) | ✅ | All metrics printed + leaderboard by F1 |
| 5 sample predictions (2+ yes, 2+ no) | ✅ | Real customers with features + outcomes |
| EXPLANATION.md answers | ✅ | All 5 Track B questions answered |
| Saved model (.pkl) | ✅ | `model/bankmind_model.pkl` ready for deployment |

---

## 📊 Model Performance

**Best Model**: Random Forest

| Metric | Value | Interpretation |
|---|---|---|
| **Accuracy** | 0.7141 | (But misleading for imbalanced data — see Q4 in EXPLANATION.md) |
| **Precision** | 0.2223 | Of predicted 'yes', 22% were actually correct |
| **Recall** | 0.5775 | Of actual 'yes' customers, we found 58% |
| **F1-score** | 0.3210 | Harmonic mean — THE KEY METRIC (why F1 > accuracy) |

**Why F1 wins**: Accuracy trap would give 88.3% accuracy if model always predicted "no" (useless).
F1 balances finding subscribers (recall) vs. not wasting RM time (precision).

**Leaderboard**:
```
Random Forest           0.7141    0.2223    0.5775    0.3210  ⭐ BEST
Stacking Ensemble       0.6994    0.2161    0.5974    0.3174
Voting Ensemble         0.6988    0.2147    0.5926    0.3152
Gradient Boosting       0.7009    0.2142    0.5832    0.3133
Extra Trees             0.6568    0.1933    0.6096    0.2936
Logistic Regression     0.6224    0.1800    0.6267    0.2797
```

---

## 📚 Key Documentation

- **[EXPLANATION.md](EXPLANATION.md)** — Answers to all Track B/C evaluation questions
- **[requirements.txt](requirements.txt)** — All Python dependencies
- **[train_model.py](train_model.py)** — Well-commented ML pipeline code
- **[streamlit_app.py](streamlit_app.py)** — Interactive dashboard UI

