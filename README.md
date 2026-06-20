# 🏦 BankMind — Customer Subscription Predictor

A machine-learning system that predicts whether a bank customer is likely to subscribe to different products , built for the **BankMind Challenge (Omdena)**.

- **Track B** — ML Engineer: EDA → model comparison → evaluation → `EXPLANATION.md`

# Live Deployment Link 
 https://bankmind-abhishek.streamlit.app/

## What's in here

```
bankmind-abhishek/
├── data/
│   └── bank-full.csv              # UCI Bank Marketing dataset (45,211 rows)
├── train_model.py                 # EDA + 6-model ensemble training + evaluation
├── app.py                         # Streamlit dashboard UI
├── model/
│   ├── bankmind_model.pkl         # saved best pipeline (created by train_model.py)
│   └── metrics_summary.json       # all-model metrics (created by train_model.py)
├── notebooks/
│   └── bankmind_walkthrough.ipynb # end-to-end notebook with SHAP section
├── requirements.txt
├── EXPLANATION.md                 # required answers for Track B graders
└── README.md
```


## 🏃 How to Run (Local)

### 1. Clone & install
```bash
git clone https://github.com/Abhishek-bytes/bankmind-Abhishek_SB.git
cd bankmind-Abhishek_SB
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
Runs EDA, trains 6 models (Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Voting, Stacking), prints evaluation metrics, and saves the best pipeline to `model/bankmind_model.pkl`.


### 3. Launch the dashboard
```bash
python -m streamlit run app.py
```
Opens `http://localhost:8501` with prediction, leaderboard, feature importance, and about tabs.

## Best Model

**Random Forest** — Accuracy 0.7141, Precision 0.2223, Recall 0.5775, **F1 0.3210** (key metric for this imbalanced dataset — see `EXPLANATION.md` Q4).

## Docs

- **[EXPLANATION.md](EXPLANATION.md)** — answers to Track Bevaluation questions
- **[requirements.txt](requirements.txt)** — dependencies
- **[train_model.py](train_model.py)** — ML pipeline
- **[app.py](app.py)** — Streamlit dashboard
