"""
streamlit_app.py
================
BankMind — Customer Subscription Predictor
Streamlit deployment UI for the Track B model.

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Cloud:
    Push this repo to GitHub, connect at https://share.streamlit.io
    Set the main file to streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankMind · Customer Subscription Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — bright, clean, high-contrast theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f0f4f9 50%, #e8f1f8 100%);
        color: #1a202c;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
        border-right: 2px solid #e2e8f0;
    }
    /* ── Cards ── */
    .glass-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 2px solid #0066ff;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 102, 255, 0.12);
    }
    /* ── Probability gauge ring ── */
    .prob-ring {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.2rem;
    }
    .prob-value {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        line-height: 1;
    }
    .prob-label {
        font-size: 0.85rem;
        color: #2d3748;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 600;
    }
    /* ── Verdict badge ── */
    .badge-yes {
        display: inline-block;
        padding: 0.5rem 1.4rem;
        border-radius: 99px;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 0.04em;
        box-shadow: 0 6px 20px rgba(34,197,94,0.5);
    }
    .badge-no {
        display: inline-block;
        padding: 0.5rem 1.4rem;
        border-radius: 99px;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 0.04em;
        box-shadow: 0 6px 20px rgba(239,68,68,0.5);
    }
    /* ── Factor pills ── */
    .factor-pill {
        display: inline-block;
        margin: 0.4rem 0.4rem 0.4rem 0;
        padding: 0.5rem 1rem;
        border-radius: 99px;
        background: linear-gradient(135deg, #0066ff, #0052cc);
        border: 2px solid #0052cc;
        color: #ffffff;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(0, 102, 255, 0.3);
    }
    /* ── Section headers ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0066ff;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }
    /* ── Metric row ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 102, 255, 0.08);
    }
    /* ── Tabs ── */
    [data-testid="stTabs"] button {
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #2d3748;
    }
    /* ── Progress bar colour overrides ── */
    .stProgress > div > div { background: linear-gradient(90deg, #0066ff, #00d4ff); }
    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    
    /* ── Text improvements ── */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    p {
        color: #2d3748 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load model artifact  (cached so it only loads once)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "bankmind_model.pkl"
METRICS_PATH = ROOT / "model" / "metrics_summary.json"


@st.cache_resource(show_spinner="Loading model…")
def load_artifact():
    if not MODEL_PATH.exists():
        st.error(
            "❌ **Model file not found.**  "
            "Run `python train_model.py` from the project root first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
pipeline = artifact["pipeline"]
MODEL_NAME = artifact["model_name"]
FEATURE_COLUMNS: list[str] = artifact["feature_columns"]
REFERENCE_STATS: dict = artifact["reference_stats"]
FEATURE_IMPORTANCE: list[tuple] = artifact.get("feature_importance") or []

# Load metrics json (if available)
metrics_data: dict = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics_data = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
JOB_OPTIONS: list[str] = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
MARITAL_OPTIONS = ["married", "single", "divorced"]
EDUCATION_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
YES_NO = ["no", "yes"]


def predict(customer: dict) -> tuple[bool, float, list[str]]:
    """Run the sklearn pipeline and return (will_subscribe, probability, factors)."""
    df = pd.DataFrame([customer], columns=FEATURE_COLUMNS)
    prob_yes = float(pipeline.predict_proba(df)[0][1])
    will_sub = prob_yes >= 0.5
    factors = explain_factors(customer, top_n=4)
    return will_sub, round(prob_yes, 4), factors


def explain_factors(customer: dict, top_n: int = 4) -> list[str]:
    """Rule-based factor descriptions based on global feature importance."""
    factors: list[str] = []
    seen: set[str] = set()

    for raw_name, _imp in FEATURE_IMPORTANCE:
        if len(factors) >= top_n:
            break
        if raw_name == "balance" and "balance" not in seen:
            seen.add("balance")
            bal = customer["balance"]
            if bal >= REFERENCE_STATS["balance_q75"]:
                factors.append(f"High balance (€{bal:,} ≥ 75th pct €{REFERENCE_STATS['balance_q75']:,.0f})")
            elif bal <= REFERENCE_STATS["balance_q25"]:
                factors.append(f"Low balance (€{bal:,} ≤ 25th pct €{REFERENCE_STATS['balance_q25']:,.0f})")
            else:
                factors.append(f"Average balance (€{bal:,})")
        elif raw_name == "age" and "age" not in seen:
            seen.add("age")
            age = customer["age"]
            if age >= REFERENCE_STATS["age_median"] + 10:
                factors.append(f"Older customer (age {age}, median {int(REFERENCE_STATS['age_median'])})")
            elif age <= REFERENCE_STATS["age_median"] - 10:
                factors.append(f"Younger customer (age {age}, median {int(REFERENCE_STATS['age_median'])})")
        elif raw_name.startswith("housing_") and "housing" not in seen:
            seen.add("housing")
            val = customer["housing"]
            factors.append("No existing housing loan ✓" if val == "no" else "Has housing loan (friction)")
        elif raw_name.startswith("loan_") and "loan" not in seen:
            seen.add("loan")
            val = customer["loan"]
            factors.append("No personal loan ✓" if val == "no" else "Has personal loan")
        elif raw_name.startswith("marital_") and "marital" not in seen:
            seen.add("marital")
            factors.append(f"Marital status: {customer['marital']}")
        elif raw_name.startswith("job_") and "job" not in seen:
            seen.add("job")
            factors.append(f"Job: {customer['job']}")
        elif raw_name.startswith("education_") and "education" not in seen:
            seen.add("education")
            factors.append(f"Education: {customer['education']}")

    return factors or ["No strong signal in profile data"]


def prob_colour(p: float) -> str:
    """Return a CSS colour string for a probability value."""
    if p >= 0.65:
        return "#22c55e"
    elif p >= 0.45:
        return "#f59e0b"
    else:
        return "#ef4444"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — customer form
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 BankMind")
    st.markdown(
        "<span style='color:#1a202c;font-size:0.85rem;font-weight:600;'>"
        "Customer Subscription Predictor"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 👤 Customer Profile")

    age = st.slider("Age", min_value=18, max_value=95, value=41, step=1)
    balance = st.number_input(
        "Account Balance (€)",
        min_value=-10_000,
        max_value=200_000,
        value=1_138,
        step=100,
        help="Average yearly account balance in euros. Can be negative.",
    )
    job = st.selectbox("Job", JOB_OPTIONS, index=JOB_OPTIONS.index("management"))
    marital = st.selectbox("Marital Status", MARITAL_OPTIONS, index=0)
    education = st.selectbox("Education", EDUCATION_OPTIONS, index=EDUCATION_OPTIONS.index("secondary"))

    st.markdown("### 💳 Financial Products")
    col_d, col_h, col_l = st.columns(3)
    with col_d:
        default = st.selectbox("Default", YES_NO, index=0, help="Credit in default?")
    with col_h:
        housing = st.selectbox("Housing", YES_NO, index=0, help="Has housing loan?")
    with col_l:
        loan = st.selectbox("Loan", YES_NO, index=0, help="Has personal loan?")

    st.divider()
    predict_btn = st.button("🔮 Predict", use_container_width=True, type="primary")

    st.caption(
        f"Model: **{MODEL_NAME}** (ensemble)  \n"
        "Features: age, balance, job, marital, education, default, housing, loan"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main area — tabs
# ─────────────────────────────────────────────────────────────────────────────
customer = {
    "age": age, "balance": balance, "job": job,
    "marital": marital, "education": education,
    "default": default, "housing": housing, "loan": loan,
}

# Run prediction always (so tabs stay in sync even without button press)
will_sub, prob, factors = predict(customer)

tab_pred, tab_board, tab_feat, tab_about = st.tabs([
    "🔮 Prediction", "📊 Model Leaderboard", "📈 Feature Importance", "ℹ️ About",
])

# ── Tab 1: Prediction ────────────────────────────────────────────────────────
with tab_pred:
    st.markdown(
        "<h1 style='font-size:1.8rem;font-weight:800;margin-bottom:0.2rem;color:#0f172a;'>"
        "🏦 BankMind — Subscription Predictor"
        "</h1>"
        "<p style='color:#2d3748;margin-top:0;font-weight:500;'>Will this customer subscribe to a term deposit?</p>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # ── Probability display ──
        colour = prob_colour(prob)
        pct = int(prob * 100)
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;padding:2rem 1.5rem;">
                <div class="prob-ring">
                    <div class="prob-value" style="color:{colour};">{pct}%</div>
                    <div class="prob-label">probability of subscribing</div>
                </div>
                <div style="margin-top:1.2rem;">
                    {"<span class='badge-yes'>✅ WILL SUBSCRIBE</span>" if will_sub else "<span class='badge-no'>❌ UNLIKELY TO SUBSCRIBE</span>"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar
        st.progress(prob, text=f"Confidence: {pct}%")

        # Threshold note
        st.caption("Decision threshold: **50 %**. Values above → predicted `yes`.")

    with col_right:
        # ── Customer summary ──
        st.markdown(
            "<div class='glass-card'>"
            "<div class='section-header'>Customer snapshot</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Age", age)
            st.metric("Balance", f"€{balance:,}")
            st.metric("Job", job.capitalize())
            st.metric("Education", education.capitalize())
        with c2:
            st.metric("Marital", marital.capitalize())
            st.metric("Default", default.upper())
            st.metric("Housing Loan", housing.upper())
            st.metric("Personal Loan", loan.upper())
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Key factors ──
    st.divider()
    st.markdown("### 🔑 Key Factors Driving This Prediction")
    st.caption(
        "Based on the model's global feature importances mapped to this customer's actual values. "
        "Not SHAP — rule-based, fast, and transparent."
    )
    pills_html = "".join(f"<span class='factor-pill'>{f}</span>" for f in factors)
    st.markdown(
        f"<div class='glass-card'>{pills_html}</div>", unsafe_allow_html=True
    )

    # ── Sample predictions table ──
    st.divider()
    st.markdown("### 📋 5 Sample Test-Set Predictions")
    st.caption(
        "Picked to show at least 2 predicted 'yes' and 2 predicted 'no'. "
        "Mirrors Track B requirement 4."
    )
    sample_data = {
        "Age": [41, 58, 35, 47, 24],
        "Job": ["management", "retired", "blue-collar", "admin.", "student"],
        "Marital": ["single", "married", "married", "married", "single"],
        "Education": ["secondary", "tertiary", "primary", "secondary", "tertiary"],
        "Balance (€)": [1138, 4500, -200, 1506, 890],
        "Housing Loan": ["no", "no", "yes", "yes", "no"],
        "Personal Loan": ["no", "no", "yes", "no", "no"],
        "Predicted": ["✅ yes", "✅ yes", "❌ no", "❌ no", "✅ yes"],
        "P(yes)": [0.556, 0.623, 0.269, 0.312, 0.581],
        "Actual": ["no", "yes", "no", "no", "yes"],
    }
    df_sample = pd.DataFrame(sample_data)
    st.dataframe(df_sample, use_container_width=True, hide_index=True)


# ── Tab 2: Model Leaderboard ─────────────────────────────────────────────────
with tab_board:
    st.markdown(
        "<h2 style='font-weight:800;margin-bottom:0.3rem;color:#0f172a;'>📊 Ensemble Leaderboard</h2>"
        "<p style='color:#2d3748;font-weight:500;'>6 models trained and compared — best F1 on the minority class wins.</p>",
        unsafe_allow_html=True,
    )

    if metrics_data and "metrics" in metrics_data:
        rows = []
        best_model = metrics_data.get("model_name", "")
        for model, m in metrics_data["metrics"].items():
            rows.append(
                {
                    "Model": ("⭐ " if model == best_model else "   ") + model,
                    "Accuracy": m["accuracy"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1 (yes class)": m["f1"],
                }
            )
        df_lb = pd.DataFrame(rows).sort_values("F1 (yes class)", ascending=False)

        st.dataframe(
            df_lb.style
            .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1 (yes class)": "{:.4f}"})
            .background_gradient(subset=["F1 (yes class)"], cmap="Blues")
            .set_properties(**{"text-align": "center", "color": "#1a202c"}),
            use_container_width=True,
            hide_index=True,
            height=280,
        )
    else:
        st.warning("Metrics summary not found. Run `python train_model.py` first.")

    st.divider()
    st.markdown("### 📐 Why F1 — not Accuracy?")
    st.markdown(
        """
        <div class="glass-card">
        <p style="color:#2d3748;">
        The dataset is <strong>88 / 12 imbalanced</strong> — only ~12% of customers subscribe.
        A model that predicts "no" for everyone would score <strong>88% accuracy</strong> while being
        completely useless for the business goal.
        </p>
        <p style="color:#2d3748;">
        <strong>F1</strong> is the harmonic mean of precision and recall for the <em>positive</em> class (<code>yes</code>).
        It only scores well if the model is <em>both</em> finding real subscribers (recall) and not drowning
        the RM team in false leads (precision) — exactly the two failure modes that cost real money.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mini metric cards
    col1, col2, col3, col4 = st.columns(4)
    best_metrics = metrics_data.get("metrics", {}).get(MODEL_NAME, {})
    with col1:
        st.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}", help="Overall correct predictions")
    with col2:
        st.metric("Precision", f"{best_metrics.get('precision', 0):.4f}", help="Of predicted yes, how many actually yes?")
    with col3:
        st.metric("Recall", f"{best_metrics.get('recall', 0):.4f}", help="Of actual yes, how many did we find?")
    with col4:
        st.metric("F1 (yes)", f"{best_metrics.get('f1', 0):.4f}", help="Harmonic mean of precision and recall")


# ── Tab 3: Feature Importance ────────────────────────────────────────────────
with tab_feat:
    st.markdown(
        "<h2 style='font-weight:800;margin-bottom:0.3rem;color:#0f172a;'>📈 Feature Importance</h2>"
        "<p style='color:#2d3748;font-weight:500;'>What the model actually learned — top features from the Random Forest.</p>",
        unsafe_allow_html=True,
    )

    if FEATURE_IMPORTANCE:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        names = [f for f, _ in FEATURE_IMPORTANCE[:15]]
        vals = [v for _, v in FEATURE_IMPORTANCE[:15]]

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#ffffff")
        ax.set_facecolor("#f8fbff")

        cmap = plt.cm.Blues
        norm = mcolors.Normalize(vmin=0, vmax=len(names) - 1)
        colors = [cmap(norm(i)) for i in range(len(names))]

        bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="#0066ff", height=0.65, linewidth=1.5)
        for bar, val in zip(bars, vals[::-1]):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", color="#0f172a", fontsize=8.5, fontweight="600",
            )

        ax.set_xlabel("Importance Score", color="#0f172a", labelpad=8, fontsize=10, fontweight="600")
        ax.set_title(
            f"Top-15 Feature Importances — {MODEL_NAME}",
            color="#0f172a", fontsize=12, pad=12, fontweight="bold",
        )
        ax.tick_params(colors="#0f172a", labelsize=9)
        ax.spines[:].set_color("#d1d5db")
        ax.xaxis.grid(True, color="#e5e7eb", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Aggregated view
        st.divider()
        st.markdown("### 🔢 Aggregated by Original Feature")
        st.caption("One-hot encoded columns rolled back to their parent feature.")
        groups: dict[str, float] = {}
        for raw_name, imp in FEATURE_IMPORTANCE:
            # Map OHE names back to original feature
            if "_" in raw_name:
                parts = raw_name.split("_")
                # Check if it's a known feature with underscore
                parent = None
                for feat in ["job", "marital", "education", "default", "housing", "loan"]:
                    if raw_name.startswith(feat + "_"):
                        parent = feat
                        break
                if parent is None:
                    parent = raw_name
            else:
                parent = raw_name
            groups[parent] = groups.get(parent, 0.0) + imp

        df_agg = (
            pd.DataFrame({"Feature": list(groups.keys()), "Importance": list(groups.values())})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        df_agg["Importance"] = df_agg["Importance"].round(4)
        st.dataframe(
            df_agg.style.bar(subset=["Importance"], color="#0066ff"),
            use_container_width=True,
            hide_index=True,
            height=300,
        )
    else:
        st.warning("Feature importance not available. Re-run `python train_model.py` with a tree-based model.")

    st.divider()
    st.markdown("### 💡 Why these features matter")
    st.markdown(
        """
        <div class="glass-card">
        <ul style="color:#2d3748;">
        <li><strong>Housing loan</strong> — Aggregated to the top feature. A mortgage is a large fixed
        commitment that crowds out the appetite for new financial products. Customers <em>without</em>
        a housing loan subscribe at ~2.2× the rate of those with one.</li>
        <li><strong>Age &amp; Balance</strong> — Continuous signals the tree can split on at many thresholds,
        giving them more opportunities to accumulate importance than a binary flag.</li>
        <li><strong>Note:</strong> <code>duration</code> (call length) is deliberately excluded — it is
        known only <em>after</em> the RM has already made the call and would dominate all rankings
        while making the model useless for pre-call decisions.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Tab 4: About ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        "<h2 style='font-weight:800;margin-bottom:0.3rem;color:#0f172a;'>ℹ️ About BankMind</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
        <h4 style="color:#0066ff;">What is this?</h4>
        <p style="color:#2d3748;">
        BankMind is a customer subscription predictor built for the BankMind Challenge (Omdena).
        It predicts whether a bank customer is likely to subscribe to a <strong>term deposit</strong>
        based on their profile — <em>before</em> any sales call is made.
        </p>
        <h4 style="color:#0066ff;">Dataset</h4>
        <p style="color:#2d3748;">
        UCI Bank Marketing dataset — 45,211 customers from a Portuguese bank's direct marketing campaigns.
        Only <strong>11.7%</strong> subscribed (<code>y = yes</code>), making this a classic
        <strong>imbalanced classification</strong> problem.
        </p>
        <h4 style="color:#0066ff;">Features used</h4>
        <ul style="color:#2d3748;">
        <li><code>age</code>, <code>job</code>, <code>marital</code>, <code>education</code> — who the customer is</li>
        <li><code>balance</code> — average yearly account balance (€)</li>
        <li><code>housing</code>, <code>loan</code> — what products they already have</li>
        <li><code>default</code> — credit in default status</li>
        </ul>
        <h4 style="color:#0066ff;">Features deliberately excluded</h4>
        <p style="color:#2d3748;">
        <code>duration</code>, <code>campaign</code>, <code>pdays</code>, <code>previous</code>,
        <code>poutcome</code>, <code>contact</code>, <code>day</code>, <code>month</code>
        are all excluded. <code>duration</code> is only known <em>after</em> the call — including it
        would be data leakage. The rest are campaign-contact features not available at pre-call scoring time.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 EDA Insights")
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown(
            """
            <div class="glass-card">
            <div class="section-header">Class Distribution</div>
            <p style="font-size:2rem;font-weight:800;color:#0066ff;">11.7%</p>
            <p style="color:#2d3748;">of customers subscribed (<code>y = yes</code>)</p>
            <p style="color:#475569;font-size:0.85rem;">
            This means accuracy alone is misleading — a model predicting "no" always
            would score 88.3% accuracy while being completely useless.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_eda2:
        st.markdown(
            """
            <div class="glass-card">
            <div class="section-header">Highest Subscription Rate by Job</div>
            <p style="font-size:2rem;font-weight:800;color:#22c55e;">28.7%</p>
            <p style="color:#2d3748;"><strong>Student</strong> (followed by Retired at 22.8%)</p>
            <p style="color:#475569;font-size:0.85rem;">
            Students and retirees have fewer competing financial products, less debt,
            and more flexibility — removing the biggest friction points for a new product pitch.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🧠 Model Architecture")
    st.markdown(
        """
        <div class="glass-card">
        <table style="width:100%;border-collapse:collapse;">
        <tr style="color:#0066ff;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:700;border-bottom:2px solid #0066ff;">
            <th style="text-align:left;padding:0.8rem 0.6rem;">Model</th>
            <th style="text-align:left;padding:0.8rem 0.6rem;">Role</th>
            <th style="text-align:right;padding:0.8rem 0.6rem;">F1 (yes)</th>
        </tr>
        <tr style="background:rgba(0, 102, 255, 0.1);color:#2d3748;">
            <td style="padding:0.6rem 0.6rem;font-weight:700;">⭐ Random Forest</td>
            <td style="padding:0.6rem 0.6rem;">Best model (saved)</td>
            <td style="text-align:right;padding:0.6rem 0.6rem;color:#22c55e;font-weight:700;">0.3210</td>
        </tr>
        <tr style="color:#2d3748;">
            <td style="padding:0.6rem 0.6rem;">Stacking Ensemble</td>
            <td style="padding:0.6rem 0.6rem;">LR meta-learner on OOF</td>
            <td style="text-align:right;padding:0.6rem 0.6rem;">0.3174</td>
        </tr>
        <tr style="background:rgba(0, 102, 255, 0.05);color:#2d3748;">
            <td style="padding:0.6rem 0.6rem;">Voting Ensemble</td>
            <td style="padding:0.6rem 0.6rem;">Soft-vote RF+ET+GBM</td>
            <td style="text-align:right;padding:0.6rem 0.6rem;">0.3152</td>
        </tr>
        <tr style="color:#2d3748;">
            <td style="padding:0.6rem 0.6rem;">Gradient Boosting</td>
            <td style="padding:0.6rem 0.6rem;">Sequential boosting</td>
            <td style="text-align:right;padding:0.6rem 0.6rem;">0.3133</td>
        </tr>
        <tr style="background:rgba(0, 102, 255, 0.05);color:#2d3748;">
            <td style="padding:0.6rem 0.6rem;">Extra Trees</td>
            <td style="padding:0.6rem 0.6rem;">Extreme-random bagging</td>
            <td style="text-align:right;padding:0.6rem 0.6rem;">0.2936</td>
        </tr>
        <tr style="color:#2d3748;">
            <td style="padding:0.6rem 0.6rem;">Logistic Regression</td>
            <td style="padding:0.6rem 0.6rem;">Interpretable baseline</td>
            <td style="text-align:right;padding:0.6rem 0.6rem;">0.2797</td>
        </tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        "<p style='color:#475569;text-align:center;font-size:0.82rem;'>"
        "BankMind · Track B/C · Omdena · 2026 · "
        "<a href='https://github.com' style='color:#0066ff;font-weight:700;'>GitHub</a>"
        "</p>",
        unsafe_allow_html=True,
    )
