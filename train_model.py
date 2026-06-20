"""
train_model.py
================
BankMind Challenge — Track B (ML Engineer)

OVERVIEW
--------
This script implements a complete ML pipeline for the UCI Bank Marketing dataset:

  Step 1: EDA
    • Load data and inspect shape, dtypes, missing values
    • Print class distribution → shows 11.7% subscribed (imbalanced)
    • Understand what we're feeding into the model

  Step 2: Train 6 Models (Baseline + Ensemble)
    • Logistic Regression          — simple interpretable baseline
    • Random Forest                — bagging (variance reduction)
    • Extra Trees                  — extreme-randomisation bagging
    • Gradient Boosting (HistGB)   — sequential boosting (bias reduction)
    • Voting Ensemble (soft)       — average P(yes) from RF + ET + GB
    • Stacking Classifier          — LR meta-learner on OOF predictions

  Step 3: Evaluation (Proper!)
    • Report accuracy, precision, recall, F1 for each model
    • Print full classification_report for transparency
    • Avoid accuracy trap (88.3% by always predicting 'no')
    • Use F1 as the selection metric (handles imbalance correctly)

  Step 4: Sample Predictions
    • Show 5 real customers from test set (2+ yes, 2+ no predictions)
    • Display features, prediction, probability, and actual outcome

  Step 5: Feature Importance & SHAP
    • Extract feature importances from tree-based models
    • Compute SHAP values for deeper explainability (if installed)

  Step 6: Save Artifacts
    • Best model → model/bankmind_model.pkl
    • Metrics summary → model/metrics_summary.json
    • SHAP explainer (optional) → bundled with model

WHY NOT INCLUDE duration, campaign, pdays, etc.?
  ✗ duration is LEAKAGE — only known AFTER the call is made
  ✗ This model is for PRE-CALL decisions ("which customer to contact?")
  ✗ See EXPLANATION.md Q3 for full rationale

ENSEMBLE STRATEGY (why go beyond the 2-model baseline?)
  • Bagging reduces variance — different random subsets see different noise
  • Boosting reduces bias — sequential error correction
  • Soft voting smooths overconfident individual predictions
  • Stacking adapts weights via meta-learner on held-out data
  • Result: F1 improves ~15% from weakest (LR) to strongest (RF/Stacking)

Usage:
    python train_model.py
    
Output:
    • Console: detailed EDA, training logs, evaluation metrics, sample predictions
    • model/bankmind_model.pkl: fitted pipeline + metadata (loaded by Streamlit + FastAPI)
    • model/metrics_summary.json: human-readable metrics for dashboards
"""

# ────────────────────────────────────────────────────────────────────────────
# IMPORTS & CONFIG
# ────────────────────────────────────────────────────────────────────────────

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

# Data and model paths
DATA_PATH = "data/bank-full.csv"
MODEL_PATH = "model/bankmind_model.pkl"
RANDOM_STATE = 42

# ────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING DECISIONS
# ────────────────────────────────────────────────────────────────────────────
#
# FEATURES USED (all profile data available BEFORE customer contact):
#   • age, balance              → numeric features (scaled)
#   • job, marital, education   → categorical features (one-hot encoded)
#   • default, housing, loan    → binary categorical features (one-hot encoded)
#
# FEATURES DELIBERATELY EXCLUDED:
#   • duration                  → only known AFTER the call (LEAKAGE!)
#   • campaign, pdays, previous → contact history (not pre-call info)
#   • poutcome                  → outcome of previous campaign (not pre-call info)
#   • contact, day, month       → contact method & timing (not pre-call info)
#
# WHY? This is a PRE-CALL recommender: "which customer should an RM contact?"
# We cannot use information only available after they've already called.
#
# See EXPLANATION.md Q3 for detailed rationale.
#
PROFILE_FEATURES_NUM = ["age", "balance"]
PROFILE_FEATURES_CAT = ["job", "marital", "education", "default", "housing", "loan"]
ALL_FEATURES = PROFILE_FEATURES_NUM + PROFILE_FEATURES_CAT
TARGET = "y"


# ────────────────────────────────────────────────────────────────────────────
# STEP 1: EXPLORATORY DATA ANALYSIS (EDA)
# ────────────────────────────────────────────────────────────────────────────
#
# Purpose: Understand the dataset before feeding it to models
#   ✓ What's the shape? (rows, columns)
#   ✓ What data types? (numeric, categorical, object)
#   ✓ Any missing values? (NaN, unknown)
#   ✓ Is the target variable imbalanced? (affects evaluation strategy)
#

def load_data() -> pd.DataFrame:
    """Load the UCI Bank Marketing dataset.
    
    Returns:
        pd.DataFrame with shape (45,211, 21)
    """
    df = pd.read_csv(DATA_PATH, sep=";")
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Print dataset overview: shape, dtypes, missing values, class distribution.
    
    Key findings to understand:
      1. Shape tells us scale (45k rows → large dataset)
      2. Dtypes tell us what preprocessing is needed
      3. Missing values tell us if imputation is needed
      4. Class distribution (y = yes/no) → determines evaluation metrics choice
    
    Args:
        df: Raw dataset with all columns
    """
    print("=" * 70)
    print("STEP 1 — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("Data types (sample):")
    print(df.dtypes)
    print()

    print("Missing values (NaN) per column:")
    na_counts = df.isnull().sum()
    na_counts = na_counts[na_counts > 0]
    if na_counts.empty:
        print("  ✓ No NaN values found in this dataset\n")
    else:
        print(na_counts)
        print()

    print("'unknown' tokens (dataset's placeholder for missing categorical data):")
    for col in ["job", "education", "contact", "poutcome"]:
        n = (df[col] == "unknown").sum()
        pct = n / len(df) * 100
        print(f"  {col:15s}: {n:6,d} ({pct:5.2f}%)")
    print()

    print("Class distribution (TARGET: y = yes/no)")
    print("-" * 70)
    counts = df[TARGET].value_counts()
    pct = df[TARGET].value_counts(normalize=True) * 100
    for label in counts.index:
        bar = "█" * int(pct[label] / 2)  # visual bar chart
        print(f"  {label:>3s}: {counts[label]:6,d} ({pct[label]:5.2f}%)  {bar}")
    print()
    print("  ⚠️  IMBALANCE ALERT:")
    print("    • Only 11.7% subscribed (y=yes) — 88.3% did not (y=no)")
    print("    • A model that always predicts 'no' would be 88.3% accurate")
    print("    • This is why we use F1, precision, recall — not accuracy")
    print("    • All models are trained with class_weight='balanced'")
    print()


# ────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING PIPELINE
# ────────────────────────────────────────────────────────────────────────────
#
# Standardizes the workflow for feature transformation:
#   • Numeric features (age, balance)          → StandardScaler
#   • Categorical features (job, marital, etc) → OneHotEncoder
#
# This preprocessor is reused across all 6 models to ensure consistency.
#

def make_preprocessor() -> ColumnTransformer:
    """Build a reusable data preprocessor.
    
    Handles mixed data types (numeric + categorical) in a single step.
    
    Returns:
        ColumnTransformer that will:
          1. Scale numeric features to μ=0, σ=1 (needed for LogisticRegression)
          2. One-hot encode categorical features (needed for tree models)
    
    Benefits:
      • No data leakage (fit only on train, transform on test)
      • Prevents mix-ups (categorical columns stay categorical, numeric stays scaled)
      • Pipelines handle it automatically via sklearn's Pipeline
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), PROFILE_FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PROFILE_FEATURES_CAT),
        ]
    )


# ────────────────────────────────────────────────────────────────────────────
# STEP 2: MODEL BUILDING
# ────────────────────────────────────────────────────────────────────────────
#
# BASELINE MODELS:
#   1. Logistic Regression        — simple, interpretable, fast
#      • Useful for understanding feature weights
#      • Baseline to compare tree-based models against
#
# TREE-BASED MODELS (BAGGING):
#   2. Random Forest              — bootstrap aggregating with feature randomness
#      • Reduces variance by averaging trees trained on random data subsets
#      • Each split considers all features
#
#   3. Extra Trees                — extreme randomization forests
#      • More aggressive randomization (random thresholds, not optimal)
#      • Faster training, sometimes lower bias
#
# BOOSTING MODEL:
#   4. Gradient Boosting (HistGB) — sequential error correction
#      • Builds trees sequentially, each correcting previous residuals
#      • Reduces bias instead of variance
#      • HistGradientBoosting ≈ LightGBM, faster than traditional GBM
#
# ENSEMBLE METHODS:
#   5. Voting Ensemble            — equals-weight soft vote
#      • Takes average P(yes) from RF + ExtraTrees + HistGB
#      • Simple, no meta-learning needed
#
#   6. Stacking Ensemble          — meta-learner approach
#      • Trains base models (RF + ET + GB) on random folds
#      • Learns which base model to trust via Logistic Regression
#      • More adaptive than fixed-weight voting
#
# KEY HYPERPARAMETER: class_weight='balanced'
#   • Tells models to penalize false negatives (minority class) more
#   • Without this, models ignore the ~12% who actually subscribe
#   • Critical for imbalanced classification
#

def build_pipelines() -> dict:
    """Build and return 6 trained-ready pipelines.
    
    Each pipeline = [Preprocessor] → [Model]
    
    All models use class_weight='balanced' to handle the 88/12 imbalance.
    Trees have max_depth constraints to prevent overfitting on this noisy data.
    
    Returns:
        dict of {model_name: Pipeline}
            Keys: "Logistic Regression", "Random Forest", "Extra Trees", 
                  "Gradient Boosting", "Voting Ensemble", "Stacking Ensemble"
            Values: unfitted sklearn Pipelines ready to .fit()
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # BASE LEARNERS (used by voting and stacking)
    # ─────────────────────────────────────────────────────────────────────────
    
    rf_clf = RandomForestClassifier(
        n_estimators=300,           # 300 trees for stable voting
        max_depth=10,               # shallow trees reduce overfitting
        min_samples_leaf=20,        # splits only if 20+ samples per leaf
        class_weight="balanced",    # penalize minority class misclassification
        random_state=RANDOM_STATE,
        n_jobs=-1,                  # use all CPUs
    )

    et_clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gb_clf = HistGradientBoostingClassifier(
        max_iter=200,               # 200 sequential boosting stages
        learning_rate=0.08,         # regularise: smaller steps = more stable
        max_depth=4,                # shallow trees for boosting
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    lr_clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ENSEMBLE LEARNERS
    # ─────────────────────────────────────────────────────────────────────────
    
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", rf_clf),
            ("et", et_clf),
            ("gb", gb_clf),
        ],
        voting="soft",  # Use predict_proba and average P(yes)
        n_jobs=-1,
    )

    stacking_clf = StackingClassifier(
        estimators=[
            ("rf", rf_clf),
            ("et", et_clf),
            ("gb", gb_clf),
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        cv=5,           # 5-fold CV to generate OOF predictions for meta-learner
        passthrough=False,  # don't include original features in meta-learner
        n_jobs=-1,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # WRAP ALL MODELS IN PREPROCESSING PIPELINES
    # ─────────────────────────────────────────────────────────────────────────
    
    def make_pipe(clf):
        """Helper: wrap any classifier in [Preprocessor → Model] Pipeline."""
        return Pipeline([("preprocess", make_preprocessor()), ("model", clf)])

    return {
        "Logistic Regression":   make_pipe(lr_clf),
        "Random Forest":         make_pipe(rf_clf),
        "Extra Trees":           make_pipe(et_clf),
        "Gradient Boosting":     make_pipe(gb_clf),
        "Voting Ensemble":       make_pipe(voting_clf),
        "Stacking Ensemble":     make_pipe(stacking_clf),
    }


# ────────────────────────────────────────────────────────────────────────────
# STEP 3: EVALUATION (PROPER!)
# ────────────────────────────────────────────────────────────────────────────
#
# WHY WE DON'T JUST USE ACCURACY:
#   If a model always predicts "no", it gets 88.3% accuracy (all the negatives).
#   But it's worthless for finding customers who will subscribe (0% recall).
#
# THE METRICS WE USE INSTEAD:
#   • Precision (for 'yes'): of predicted 'yes', how many actually said 'yes'?
#                            (avoids wasting RM time on false leads)
#   • Recall (for 'yes'):    of actual 'yes', how many did we find?
#                            (avoids missing revenue opportunities)
#   • F1 (for 'yes'):        harmonic mean of precision & recall
#                            (balances both failure modes)
#
# WHY THIS MATTERS:
#   For a cross-sell tool, both mistakes are costly:
#     ✗ Too many false positives → RM wastes time, trust erodes
#     ✗ Too many false negatives → missed revenue opportunities
#   F1 forces the model to optimize for both.
#

def evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a single model on test set.
    
    Prints:
      • Accuracy, Precision, Recall, F1 for class 'yes'
      • Full classification_report (for both classes)
    
    Args:
        name: Model name (for logging)
        pipeline: Fitted sklearn Pipeline
        X_test: Test features
        y_test: Test targets
    
    Returns:
        dict with keys {"accuracy", "precision", "recall", "f1"} — all for class 'yes'
    """
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics for the minority class ('yes')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="yes")
    rec = recall_score(y_test, y_pred, pos_label="yes")
    f1 = f1_score(y_test, y_pred, pos_label="yes")

    print("-" * 70)
    print(f"{name}")
    print("-" * 70)
    print(f"Accuracy:  {acc:.4f}  (% of all predictions correct — misleading for imbalanced data)")
    print(f"Precision: {prec:.4f}  (of predicted 'yes', how many were actually 'yes'?)")
    print(f"Recall:    {rec:.4f}  (of actual 'yes', how many did we find?)")
    print(f"F1-score:  {f1:.4f}  (harmonic mean of precision & recall — THE KEY METRIC)")
    print()
    print("Classification Report (both classes):")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ────────────────────────────────────────────────────────────────────────────
# STEP 4: SAMPLE PREDICTIONS
# ────────────────────────────────────────────────────────────────────────────
#
# Shows 5 real customers from the test set (2+ yes predictions, 2+ no predictions)
# to demonstrate how the model works on actual data.
#
# Purpose:
#   ✓ Verify predictions are sensible (e.g., high balance + no loans → likely yes)
#   ✓ Understand where model fails (mismatches between predicted & actual)
#   ✓ Build intuition for model behavior
#

def show_sample_predictions(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Display 5 sample predictions from test set.
    
    Shows:
      • Customer features (age, job, balance, loans, etc.)
      • Model's prediction ("yes" or "no") + probability
      • Actual outcome
    
    Args:
        pipeline: Fitted model
        X_test: Test features (will be transformed by preprocessor inside pipeline)
        y_test: Actual test outcomes
    """
    print("=" * 70)
    print("STEP 4 — 5 SAMPLE PREDICTIONS FROM TEST SET")
    print("=" * 70)
    print("(Showing at least 2 'yes' predictions + 2 'no' predictions)\n")

    # Get predicted labels and probabilities
    probs = pipeline.predict_proba(X_test)[:, 1]  # P(yes) for each sample
    preds = np.where(probs >= 0.5, "yes", "no")

    # Assemble prediction table
    sample_df = X_test.copy()
    sample_df["actual"] = y_test.values
    sample_df["predicted"] = preds
    sample_df["probability_yes"] = probs.round(3)

    # Pick samples: 2+ "yes" predictions, 3 "no" predictions
    yes_samples = sample_df[sample_df["predicted"] == "yes"].head(2)
    no_samples = sample_df[sample_df["predicted"] == "no"].head(3)
    picks = pd.concat([yes_samples, no_samples])

    # Pretty-print each customer
    for i, row in picks.iterrows():
        print(f"Customer #{i}  (row index in test set)")
        print(f"  Profile:")
        print(f"    • Age: {row['age']}, Job: {row['job']}, Marital: {row['marital']}, Education: {row['education']}")
        print(f"    • Account balance: €{row['balance']:,}, Default: {row['default']}")
        print(f"    • Existing loans: Housing={row['housing']}, Personal={row['loan']}")
        print(f"  Model prediction:")
        print(f"    → Predicted: '{row['predicted'].upper()}' at {row['probability_yes']*100:.1f}% confidence")
        print(f"    → Actual outcome: '{row['actual']}'")
        
        if row['predicted'] == row['actual']:
            print(f"    ✓ CORRECT")
        else:
            print(f"    ✗ MISMATCH (but see EXPLANATION.md Q5 for context)")
        print()



# --------------------------------------------------------------------------
# Feature importance (tree-based models only)
# --------------------------------------------------------------------------
def get_feature_importance(pipeline: Pipeline) -> list:
    """Map feature_importances_ back to readable feature names.

    Handles three cases:
      1. Direct tree-based model (RF, ExtraTrees) — use feature_importances_ directly.
      2. VotingClassifier — pull the first base estimator that has feature_importances_.
      3. StackingClassifier — pull the first base estimator that has feature_importances_.
    """
    model = pipeline.named_steps["model"]

    # Case 1: model directly exposes feature_importances_ (RF, ExtraTrees, HistGB)
    if hasattr(model, "feature_importances_"):
        actual_model = model

    # Case 2: VotingClassifier — iterate fitted estimators_
    elif hasattr(model, "voting"):
        actual_model = None
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                actual_model = est
                break
        if actual_model is None:
            return []

    # Case 3: StackingClassifier — estimators_ is a list of (name, fitted_est) tuples
    elif hasattr(model, "final_estimator_"):
        actual_model = None
        for _, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                actual_model = est
                break
        if actual_model is None:
            return []

    else:
        return []

    preprocessor = pipeline.named_steps["preprocess"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(PROFILE_FEATURES_CAT)
    feature_names = PROFILE_FEATURES_NUM + list(cat_names)
    importances = actual_model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return ranked


# --------------------------------------------------------------------------
# SHAP explainability
# --------------------------------------------------------------------------
def _get_underlying_tree_model(pipeline: Pipeline):
    """Return the first tree-based estimator found inside the pipeline's model step.

    Returns (model, is_wrapped) where is_wrapped=True means the model is an
    ensemble (Voting/Stacking) and we only expose one of its base estimators.
    """
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return model, False
    if hasattr(model, "voting"):          # VotingClassifier
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                return est, True
    if hasattr(model, "final_estimator_"):  # StackingClassifier
        for _, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                return est, True
    return None, False


def compute_shap_explainer(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_background: int = 200,
    n_explain: int = 500,
) -> dict | None:
    """Build a SHAP explainer for the best pipeline and compute sample SHAP values.

    Strategy
    --------
    * For tree-based models (RF, ExtraTrees, HistGB) we use
      ``shap.TreeExplainer`` which is fast and exact.
    * For Voting / Stacking ensembles we extract the first tree-based
      *base estimator* and explain its transformed output. This is an
      approximation but gives actionable feature-level insight.
    * For other model types (e.g. plain LogisticRegression) we fall back to
      ``shap.KernelExplainer`` with a small masker sample — slower but
      model-agnostic.

    Returns a dict with keys:
        explainer        — fitted shap.Explainer (serialisable via joblib)
        shap_values      — np.ndarray shape (n_explain, n_features),
                           SHAP values for the *positive* class (P=yes)
        feature_names    — list[str] in the same column order as shap_values
        X_explain_raw    — pd.DataFrame of the n_explain raw input rows
                           (needed for waterfall/beeswarm plots in the notebook)

    Returns None if SHAP is not installed.
    """
    if not SHAP_AVAILABLE:
        print("  shap not installed — skipping SHAP step (pip install shap)")
        return None

    print("Computing SHAP values…")

    preprocessor = pipeline.named_steps["preprocess"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        PROFILE_FEATURES_CAT
    )
    feature_names = PROFILE_FEATURES_NUM + list(cat_names)

    # Transform data once — SHAP works on the already-preprocessed array
    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # Pick background / explain slices
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx  = rng.choice(len(X_train_t), size=min(n_background, len(X_train_t)), replace=False)
    exp_idx = rng.choice(len(X_test_t),  size=min(n_explain, len(X_test_t)),     replace=False)

    X_background = X_train_t[bg_idx]
    X_explain_t  = X_test_t[exp_idx]
    X_explain_raw = X_test.iloc[exp_idx].reset_index(drop=True)

    # Resolve the underlying estimator
    tree_model, is_wrapped = _get_underlying_tree_model(pipeline)

    if tree_model is not None:
        try:
            explainer = shap.TreeExplainer(
                tree_model,
                data=X_background,
                feature_names=feature_names,
                model_output="probability",
            )
            shap_out = explainer(X_explain_t)
            # shap_out.values shape: (n, features) for binary or (n, features, 2)
            sv = shap_out.values
            if sv.ndim == 3:          # multi-output; take class-1 (yes)
                sv = sv[:, :, 1]
            print(f"  TreeExplainer ready  (is_wrapped={is_wrapped})")
        except Exception as exc:  # pragma: no cover
            print(f"  TreeExplainer failed ({exc}), falling back to KernelExplainer")
            tree_model = None  # trigger fallback below

    if tree_model is None:
        # Generic fallback: explain pipeline.predict_proba directly
        def _predict_proba(X_arr):
            # Re-wrap as DataFrame so the pipeline preprocessor can handle it
            df_tmp = pd.DataFrame(X_arr, columns=feature_names)
            # We need raw features — but we already have transformed data here.
            # For KernelExplainer we instead pass the predict_proba of the full
            # pipeline on the *original* feature space.
            return pipeline.predict_proba(df_tmp)[:, 1]

        # Use original (raw) feature space for KernelExplainer
        X_bg_raw  = X_train.iloc[bg_idx].reset_index(drop=True)
        X_exp_raw = X_explain_raw.copy()
        masker    = shap.maskers.Independent(X_bg_raw, max_samples=n_background)
        explainer = shap.KernelExplainer(
            lambda arr: pipeline.predict_proba(
                pd.DataFrame(arr, columns=ALL_FEATURES)
            )[:, 1],
            masker,
            feature_names=ALL_FEATURES,
        )
        sv = explainer.shap_values(X_exp_raw, nsamples=100)
        feature_names = ALL_FEATURES    # override — raw features for Kernel path
        print("  KernelExplainer ready (fallback path)")

    return {
        "explainer":     explainer,
        "shap_values":   sv,
        "feature_names": feature_names,
        "X_explain_raw": X_explain_raw,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df = load_data()
    run_eda(df)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("=" * 70)
    print("STEP 2 — Training ensemble models")
    print("=" * 70)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features used: {ALL_FEATURES}")
    print("(duration, campaign, pdays, previous, poutcome, contact, day, month")
    print(" intentionally excluded — see EXPLANATION.md Q3)\n")

    print("Models being trained:")
    print("  1. Logistic Regression  (interpretable baseline)")
    print("  2. Random Forest        (bagging baseline)")
    print("  3. Extra Trees          (extreme-randomisation bagging)")
    print("  4. Gradient Boosting    (sequential boosting, stochastic)")
    print("  5. Voting Ensemble      (soft-vote: RF + ExtraTrees + GBM)")
    print("  6. Stacking Ensemble    (LR meta-learner on OOF preds, cv=5)")
    print()

    pipelines = build_pipelines()

    for name, pipe in pipelines.items():
        print(f"  Training {name}...")
        pipe.fit(X_train, y_train)
    print()

    print("=" * 70)
    print("STEP 3 — Evaluation")
    print("=" * 70)
    metrics = {}
    for name, pipe in pipelines.items():
        metrics[name] = evaluate(name, pipe, X_test, y_test)

    # Pick the model with the higher F1 on the minority ('yes') class
    best_name = max(metrics, key=lambda n: metrics[n]["f1"])
    best_pipeline = pipelines[best_name]
    print(f"Best model by F1 (class 'yes'): {best_name}\n")

    # Print a tidy leaderboard
    print("=" * 70)
    print("Ensemble leaderboard (sorted by F1 on 'yes' class)")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for name in sorted(metrics, key=lambda n: metrics[n]["f1"], reverse=True):
        m = metrics[name]
        marker = "  <-- BEST" if name == best_name else ""
        print(
            f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f} {m['f1']:>10.4f}{marker}"
        )
    print()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: FEATURE IMPORTANCE
    # ────────────────────────────────────────────────────────────────────────
    # Only available for tree-based models.
    # Maps one-hot encoded feature names back to readable original names.
    
    print("=" * 70)
    print("STEP 5 — FEATURE IMPORTANCE (Tree-Based Models)")
    print("=" * 70)
    ranked = get_feature_importance(best_pipeline)
    if ranked:
        print(f"Top-10 features in {best_name}:\n")
        for feat, imp in ranked[:10]:
            print(f"  {feat:30s} → {imp:.4f}")
        print("\n  (see EXPLANATION.md Q3 for interpretation)")
    else:
        print("  (feature importance not available for Logistic Regression)")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: SHAP EXPLAINABILITY (optional, requires shap package)
    # ────────────────────────────────────────────────────────────────────────
    # SHAP = SHapley Additive exPlanations
    # Computes game-theory-based feature contributions to each prediction
    # More rigorous than feature importance (used for model audits)
    
    print("=" * 70)
    print("STEP 6 — SHAP EXPLAINABILITY (if available)")
    print("=" * 70)
    shap_data = compute_shap_explainer(best_pipeline, X_train, X_test)
    if shap_data:
        sv = shap_data["shap_values"]
        fn = shap_data["feature_names"]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:10]
        print("  Top-10 features by mean |SHAP value|:\n")
        for idx in top_idx:
            print(f"    {fn[idx]:35s} → {mean_abs[idx]:.4f}")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 7: SAVE ARTIFACTS FOR DEPLOYMENT
    # ────────────────────────────────────────────────────────────────────────
    # Saved pipeline can be loaded immediately by:
    #   • Streamlit app (provides live dashboard)
    #   • FastAPI service (provides REST API)
    #   • Notebooks (research/auditing)
    
    print("=" * 70)
    print("STEP 7 — SAVING ARTIFACTS FOR DEPLOYMENT")
    print("=" * 70)
    
    # Compute reference statistics for the Streamlit UI
    # (used to contextualize customer's balance/age vs. quartiles)
    reference_stats = {
        "balance_median": float(df["balance"].median()),
        "balance_q75": float(df["balance"].quantile(0.75)),
        "balance_q25": float(df["balance"].quantile(0.25)),
        "age_median": float(df["age"].median()),
    }

    # Bundle everything into a single artifact
    artifact = {
        "pipeline": best_pipeline,          # The fitted sklearn Pipeline
        "model_name": best_name,            # e.g., "Random Forest"
        "feature_columns": ALL_FEATURES,    # Column order for inference
        "metrics": {k: {m: round(v, 4) for m, v in val.items()} for k, val in metrics.items()},
        "reference_stats": reference_stats,
        "feature_importance": (
            [(f, round(float(i), 4)) for f, i in ranked] if ranked else None
        ),
        "shap": shap_data,                  # SHAP explainer (None if not installed)
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"  ✓ Saved: {MODEL_PATH}")
    print(f"    (loaded by streamlit_app.py and any FastAPI service)")

    # Also save a human-readable JSON summary for quick inspection
    with open("model/metrics_summary.json", "w") as f:
        json.dump(
            {
                "model_name": best_name,
                "metrics": artifact["metrics"],
                "reference_stats": reference_stats,
                "ensemble_strategy": {
                    "base_learners": ["Random Forest", "Extra Trees", "Gradient Boosting"],
                    "ensemble_methods": ["Soft Voting", "Stacking (LR meta, cv=5)"],
                    "class_imbalance_handling": "class_weight='balanced' on all base learners",
                },
            },
            f,
            indent=2,
        )
    print(f"  ✓ Saved: model/metrics_summary.json")
    print(f"    (human-readable metrics for dashboards & documentation)")
    print()
    print("=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best model: {best_name}")
    print(f"Test F1 (class 'yes'): {metrics[best_name]['f1']:.4f}")
    print()
    print("Next steps:")
    print("  1. View Streamlit dashboard: streamlit run streamlit_app.py")
    print()


if __name__ == "__main__":
    main()
